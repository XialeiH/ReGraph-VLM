#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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

from models.bnt_encoder import BNTTokenEncoder
from models.regraph_vlm import RoiMLPEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNeuroMod visual-ROI external cross-subject model.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/cneuromod_things/visual_roi_scalar4_smoke"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/cneuromod_things/visual_roi_scalar4_smoke/trained_external"),
    )
    parser.add_argument("--subject-a", default="sub-01")
    parser.add_argument("--subject-b", default="sub-02")
    parser.add_argument("--file-template", default="{subject}_cneuromod_visual_roi_scalar4.pt")
    parser.add_argument("--repetitions", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--model", choices=["roi_mlp", "roi_transformer_gated"], default="roi_transformer_gated")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_subject(data_dir: Path, subject: str, file_template: str) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    pt = torch.load(data_dir / file_template.format(subject=subject), map_location="cpu")
    labels = np.asarray(pt["image_label"], dtype=str)
    reps = pt["repetition"].numpy().astype(np.int16)
    x = pt["x"].float()
    return labels, reps, x


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


class PairDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        idx_a: dict[tuple[str, int], int],
        idx_b: dict[tuple[str, int], int],
        labels: list[str],
        repetitions: list[int],
    ) -> None:
        self.x_a = x_a
        self.x_b = x_b
        self.pairs = [(idx_a[(label, rep)], idx_b[(label, rep)]) for label in labels for rep in repetitions]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ia, ib = self.pairs[index]
        return self.x_a[ia], self.x_b[ib]


def make_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "roi_mlp":
        return RoiMLPEncoder(180, 4, args.hidden_dim, args.embedding_dim, args.dropout)
    return BNTTokenEncoder(
        n_nodes=180,
        in_dim=4,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        readout="gated_flat",
        roi_id_mode="normal",
        use_graph_bias=False,
    )


def contrastive_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = z_a @ z_b.T / temperature
    target = torch.arange(z_a.shape[0], device=z_a.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def rank_metrics(scores: np.ndarray) -> tuple[float, float]:
    ranks = []
    for i, row in enumerate(scores):
        order = np.argsort(-row)
        ranks.append(int(np.where(order == i)[0][0]) + 1)
    ranks = np.asarray(ranks)
    return float((ranks <= 5).mean()), float((1.0 / ranks).mean())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    idx_a: dict[tuple[str, int], int],
    idx_b: dict[tuple[str, int], int],
    labels: list[str],
    repetitions: list[int],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    y_all = []
    score_all = []
    r5s = []
    mrrs = []
    for rep in repetitions:
        ia = torch.tensor([idx_a[(label, rep)] for label in labels], dtype=torch.long)
        ib = torch.tensor([idx_b[(label, rep)] for label in labels], dtype=torch.long)
        za = model(x_a[ia].to(device), None).cpu().numpy()
        zb = model(x_b[ib].to(device), None).cpu().numpy()
        scores = za @ zb.T
        target = np.eye(len(labels), dtype=np.int8)
        y_all.append(target.reshape(-1))
        score_all.append(scores.reshape(-1))
        r5, mrr = rank_metrics(scores)
        r5_rev, mrr_rev = rank_metrics(scores.T)
        r5s.extend([r5, r5_rev])
        mrrs.extend([mrr, mrr_rev])
    y = np.concatenate(y_all)
    score = np.concatenate(score_all)
    return {
        "AUROC": float(roc_auc_score(y, score)),
        "AUPRC": float(average_precision_score(y, score)),
        "R@5": float(np.mean(r5s)),
        "MRR": float(np.mean(mrrs)),
        "chance_R@5": float(5.0 / len(labels)),
        "chance_AUPRC": float(y.mean()),
    }


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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = args.out_dir / args.model / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_a, reps_a, x_a = load_subject(args.data_dir, args.subject_a, args.file_template)
    labels_b, reps_b, x_b = load_subject(args.data_dir, args.subject_b, args.file_template)
    idx_a = index_by_label_rep(labels_a, reps_a)
    idx_b = index_by_label_rep(labels_b, reps_b)
    common_labels = sorted({label for label in labels_a if all((label, rep) in idx_a and (label, rep) in idx_b for rep in args.repetitions)})
    splits = split_labels(common_labels, args.seed, args.train_frac, args.val_frac)
    x_a, x_b = standardize(x_a, x_b, idx_a, idx_b, splits["train"], args.repetitions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_ds = PairDataset(x_a, x_b, idx_a, idx_b, splits["train"], args.repetitions)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    best = {"epoch": -1, "val_AUROC": -1.0}
    best_state = None
    stale = 0
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xa, xb in loader:
            xa = xa.to(device)
            xb = xb.to(device)
            loss = contrastive_loss(model(xa, None), model(xb, None), args.temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        val = evaluate(model, x_a, x_b, idx_a, idx_b, splits["val"], args.repetitions, device)
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
    val = evaluate(model, x_a, x_b, idx_a, idx_b, splits["val"], args.repetitions, device)
    test = evaluate(model, x_a, x_b, idx_a, idx_b, splits["test"], args.repetitions, device)
    summary = {
        "model": args.model,
        "seed": args.seed,
        "subject_a": args.subject_a,
        "subject_b": args.subject_b,
        "n_images_total": len(common_labels),
        "n_train_images": len(splits["train"]),
        "n_val_images": len(splits["val"]),
        "n_test_images": len(splits["test"]),
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
