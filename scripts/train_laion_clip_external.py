#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.regraph_vlm import ReGraphVLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LAION-fMRI external ROI+CLIP validation models.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--subject-a", required=True)
    parser.add_argument("--subject-b", required=True)
    parser.add_argument("--file-template", default="{subject}_laion_visual_roi_scalar4.pt")
    parser.add_argument("--repetitions", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--model",
        choices=["roi_mlp_clip", "gated_roi_transformer_clip", "fusion_clip"],
        default="gated_roi_transformer_clip",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--clip-temperature", type=float, default=0.07)
    parser.add_argument("--lambda-clip", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch_threads() -> None:
    threads = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("OMP_NUM_THREADS")
    if threads:
        torch.set_num_threads(max(1, int(threads)))
        torch.set_num_interop_threads(max(1, min(2, int(threads))))


def load_subject(data_dir: Path, subject: str, file_template: str) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    payload = torch.load(data_dir / file_template.format(subject=subject), map_location="cpu", weights_only=False)
    labels = np.asarray(payload["image_label"], dtype=str)
    reps = payload["repetition"].numpy().astype(np.int16)
    x = payload["x"].float()
    if "clip" not in payload:
        raise KeyError(f"Missing CLIP embeddings in {data_dir / file_template.format(subject=subject)}")
    clip = payload["clip"].float()
    return labels, reps, x, clip


def split_labels(labels: list[str], seed: int, train_frac: float, val_frac: float) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(labels, dtype=object)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(math.floor(n * train_frac))
    n_val = int(math.floor(n * val_frac))
    return {
        "train": sorted(shuffled[:n_train].tolist()),
        "val": sorted(shuffled[n_train : n_train + n_val].tolist()),
        "test": sorted(shuffled[n_train + n_val :].tolist()),
    }


def index_by_label_rep(labels: np.ndarray, reps: np.ndarray) -> dict[tuple[str, int], int]:
    return {(str(label), int(rep)): idx for idx, (label, rep) in enumerate(zip(labels, reps))}


class PositivePairDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        clip_a: torch.Tensor,
        idx_a: dict[tuple[str, int], int],
        idx_b: dict[tuple[str, int], int],
        labels: list[str],
        repetitions: list[int],
    ) -> None:
        self.x_a = x_a
        self.x_b = x_b
        self.clip_a = clip_a
        self.pairs = [(idx_a[(label, rep)], idx_b[(label, rep)]) for label in labels for rep in repetitions]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ia, ib = self.pairs[index]
        return self.x_a[ia], self.x_b[ib], self.clip_a[ia]


def standardize(
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    idx_a: dict[tuple[str, int], int],
    idx_b: dict[tuple[str, int], int],
    train_labels: list[str],
    repetitions: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    train_idx_a = [idx_a[(label, rep)] for label in train_labels for rep in repetitions]
    train_idx_b = [idx_b[(label, rep)] for label in train_labels for rep in repetitions]
    train = torch.cat([x_a[train_idx_a], x_b[train_idx_b]], dim=0)
    mean = train.mean(dim=(0, 1), keepdim=True)
    std = train.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    return (x_a - mean) / std, (x_b - mean) / std


def make_model(args: argparse.Namespace, clip_dim: int) -> ReGraphVLM:
    if args.model == "roi_mlp_clip":
        graph_encoder = "roi_mlp"
        readout = "flat"
    elif args.model == "fusion_clip":
        graph_encoder = "fusion"
        readout = "gated_flat"
    else:
        graph_encoder = "roi_transformer_noadj"
        readout = "gated_flat"
    return ReGraphVLM(
        n_nodes=180,
        node_feature_dim=4,
        clip_dim=clip_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        readout=readout,
        roi_id_mode="normal",
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        graph_encoder=graph_encoder,
        num_subjects=5,
    )


def contrastive_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = z_a @ z_b.T / temperature
    target = torch.arange(z_a.shape[0], device=z_a.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def clip_loss(model: ReGraphVLM, z_a: torch.Tensor, z_b: torch.Tensor, clip: torch.Tensor, temperature: float) -> torch.Tensor:
    brain = torch.cat([z_a, z_b], dim=0)
    image = model.encode_image(torch.cat([clip, clip], dim=0))
    logits = brain @ image.T / temperature
    target = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def rank_metrics(scores: np.ndarray) -> tuple[float, float]:
    ranks = []
    for i, row in enumerate(scores):
        order = np.argsort(-row, kind="mergesort")
        ranks.append(int(np.where(order == i)[0][0]) + 1)
    ranks = np.asarray(ranks)
    return float((ranks <= 5).mean()), float((1.0 / ranks).mean())


@torch.no_grad()
def encode_blocks(
    model: ReGraphVLM,
    x: torch.Tensor,
    clip: torch.Tensor,
    indices: list[int],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    brain_out = []
    image_out = []
    for start in range(0, len(indices), batch_size):
        idx = torch.tensor(indices[start : start + batch_size], dtype=torch.long)
        xb = x[idx].to(device)
        cb = clip[idx].to(device)
        brain_out.append(model.encode_brain(xb, None, None).cpu().numpy())
        image_out.append(model.encode_image(cb).cpu().numpy())
    return np.concatenate(brain_out, axis=0), np.concatenate(image_out, axis=0)


@torch.no_grad()
def evaluate(
    model: ReGraphVLM,
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    clip_a: torch.Tensor,
    idx_a: dict[tuple[str, int], int],
    idx_b: dict[tuple[str, int], int],
    labels: list[str],
    repetitions: list[int],
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    y_all = []
    score_all = []
    pair_r5 = []
    pair_mrr = []
    image_r5 = []
    image_mrr = []
    brain_r5 = []
    brain_mrr = []
    for rep in repetitions:
        ia = [idx_a[(label, rep)] for label in labels]
        ib = [idx_b[(label, rep)] for label in labels]
        za, zi = encode_blocks(model, x_a, clip_a, ia, device, batch_size)
        zb, _ = encode_blocks(model, x_b, clip_a, ib, device, batch_size)

        brain_scores = za @ zb.T
        target = np.eye(len(labels), dtype=np.int8)
        y_all.append(target.reshape(-1))
        score_all.append(brain_scores.reshape(-1))
        r5, mrr = rank_metrics(brain_scores)
        r5_rev, mrr_rev = rank_metrics(brain_scores.T)
        pair_r5.extend([r5, r5_rev])
        pair_mrr.extend([mrr, mrr_rev])

        a_img = za @ zi.T
        b_img = zb @ zi.T
        img_a_r5, img_a_mrr = rank_metrics(a_img)
        img_b_r5, img_b_mrr = rank_metrics(b_img)
        image_r5.extend([img_a_r5, img_b_r5])
        image_mrr.extend([img_a_mrr, img_b_mrr])

        img_a = zi @ za.T
        img_b = zi @ zb.T
        brain_a_r5, brain_a_mrr = rank_metrics(img_a)
        brain_b_r5, brain_b_mrr = rank_metrics(img_b)
        brain_r5.extend([brain_a_r5, brain_b_r5])
        brain_mrr.extend([brain_a_mrr, brain_b_mrr])

    y = np.concatenate(y_all)
    score = np.concatenate(score_all)
    return {
        "AUROC": float(roc_auc_score(y, score)),
        "AUPRC": float(average_precision_score(y, score)),
        "R@5": float(np.mean(pair_r5)),
        "MRR": float(np.mean(pair_mrr)),
        "image_R@5": float(np.mean(image_r5)),
        "image_MRR": float(np.mean(image_mrr)),
        "brain_R@5": float(np.mean(brain_r5)),
        "brain_MRR": float(np.mean(brain_mrr)),
        "chance_R@5": float(5.0 / len(labels)),
        "chance_AUPRC": float(y.mean()),
    }


def main() -> None:
    args = parse_args()
    configure_torch_threads()
    set_seed(args.seed)
    out_dir = args.out_dir / args.model / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_a, reps_a, x_a, clip_a = load_subject(args.data_dir, args.subject_a, args.file_template)
    labels_b, reps_b, x_b, _ = load_subject(args.data_dir, args.subject_b, args.file_template)
    idx_a = index_by_label_rep(labels_a, reps_a)
    idx_b = index_by_label_rep(labels_b, reps_b)
    common_labels = sorted({label for label in labels_a if all((label, rep) in idx_a and (label, rep) in idx_b for rep in args.repetitions)})
    splits = split_labels(common_labels, args.seed, args.train_frac, args.val_frac)
    x_a, x_b = standardize(x_a, x_b, idx_a, idx_b, splits["train"], args.repetitions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(args, clip_dim=int(clip_a.shape[1])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_ds = PositivePairDataset(x_a, x_b, clip_a, idx_a, idx_b, splits["train"], args.repetitions)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    best = {"epoch": -1, "val_AUROC": -1.0}
    best_state = None
    stale = 0
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xa, xb, ci in loader:
            xa = xa.to(device)
            xb = xb.to(device)
            ci = ci.to(device)
            za = model.encode_brain(xa, None, None)
            zb = model.encode_brain(xb, None, None)
            loss = contrastive_loss(za, zb, args.temperature)
            if args.lambda_clip != 0.0:
                loss = loss + args.lambda_clip * clip_loss(model, za, zb, ci, args.clip_temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
        val = evaluate(model, x_a, x_b, clip_a, idx_a, idx_b, splits["val"], args.repetitions, device, args.batch_size)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val["AUROC"] > best["val_AUROC"]:
            best = {"epoch": epoch, "val_AUROC": val["AUROC"]}
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    val = evaluate(model, x_a, x_b, clip_a, idx_a, idx_b, splits["val"], args.repetitions, device, args.batch_size)
    test = evaluate(model, x_a, x_b, clip_a, idx_a, idx_b, splits["test"], args.repetitions, device, args.batch_size)
    summary = {
        "model": args.model,
        "seed": args.seed,
        "subject_a": args.subject_a,
        "subject_b": args.subject_b,
        "n_images_total": len(common_labels),
        "n_train_images": len(splits["train"]),
        "n_val_images": len(splits["val"]),
        "n_test_images": len(splits["test"]),
        "n_nodes": int(x_a.shape[1]),
        "node_feature_dim": int(x_a.shape[2]),
        "clip_dim": int(clip_a.shape[1]),
        "lambda_clip": float(args.lambda_clip),
        "best_epoch": best["epoch"],
        "elapsed_seconds": float(time.time() - start),
        **{f"val_{k}": v for k, v in val.items()},
        **{f"test_{k}": v for k, v in test.items()},
    }
    torch.save({"model_state": model.state_dict(), "args": vars(args), "splits": splits, "summary": summary}, out_dir / "best.pt")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
