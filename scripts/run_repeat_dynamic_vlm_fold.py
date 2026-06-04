#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.regraph_vlm import ReGraphVLM
from scripts.run_regraph_vlm_fold import (
    auroc,
    average_precision,
    balanced_accuracy,
    best_threshold,
    clip_alignment_from_embeddings,
    grouped_retrieval,
    load_adjacency,
    pair_infonce_from_embeddings,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train repeat-dynamic ROI-token VLM without replacing the ROI Transformer.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument(
        "--pair-dataset-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_allfold"),
    )
    parser.add_argument(
        "--sequence-dataset-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_allfold"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/repeat_dynamic_lnn"),
    )
    parser.add_argument("--base-encoder", choices=["roi_transformer_noadj", "roi_mlp"], default="roi_transformer_noadj")
    parser.add_argument("--temporal-module", choices=["gru", "cfc"], default="cfc")
    parser.add_argument("--lambda-clip", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--clip-temperature", type=float, default=0.07)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-eval-details", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ordered_x_seq(seq: dict[str, Any]) -> torch.Tensor:
    x_seq = seq["x_seq"].float()
    repeat_seq = [int(r) for r in seq["repeat_seq"].tolist()]
    if len(repeat_seq) != 3:
        raise ValueError(f"Repeat-dynamic model requires strict T=3 sequences, got repeats={repeat_seq}")
    out = torch.empty_like(x_seq)
    for source_idx, repeat_idx in enumerate(repeat_seq):
        if repeat_idx < 1 or repeat_idx > 3:
            raise ValueError(f"Unexpected repeat index {repeat_idx}; expected 1, 2, or 3")
        out[repeat_idx - 1] = x_seq[source_idx]
    return out


def load_sequences(fold_dir: Path, split: str) -> list[dict[str, Any]]:
    return torch.load(fold_dir / f"{split}_sequences.pt", map_location="cpu", weights_only=False)


def sequence_lookup(sequence_fold_dir: Path) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[tuple[int, int], torch.Tensor], dict[int, torch.Tensor]]:
    by_subject_image: dict[tuple[int, int], dict[str, Any]] = {}
    train_by_image_subject: dict[tuple[int, int], torch.Tensor] = {}
    train_global: dict[int, list[torch.Tensor]] = {}
    for split in ["train", "val", "test"]:
        for seq in load_sequences(sequence_fold_dir, split):
            subject = int(seq["subject"])
            image_id = int(seq["nsdId"])
            x_seq = ordered_x_seq(seq)
            by_subject_image[(subject, image_id)] = {"x_seq": x_seq, "clip": seq["clip"].float().clone(), "split": split}
            if split == "train":
                train_by_image_subject[(image_id, subject)] = x_seq
                train_global.setdefault(image_id, []).append(x_seq)
    train_global_mean = {image_id: torch.stack(rows, dim=0).mean(dim=0) for image_id, rows in train_global.items()}
    return by_subject_image, train_by_image_subject, train_global_mean


def train_reference_sequence(
    image_id: int,
    exclude_subject: int | None,
    train_by_image_subject: dict[tuple[int, int], torch.Tensor],
    train_global: dict[int, torch.Tensor],
) -> torch.Tensor:
    if exclude_subject is None:
        try:
            return train_global[image_id]
        except KeyError as exc:
            raise KeyError(f"Missing train reference sequence for image {image_id}") from exc
    rows = [x for (candidate_image, subject), x in train_by_image_subject.items() if candidate_image == image_id and subject != exclude_subject]
    if not rows:
        raise KeyError(f"Missing subject-excluded train reference sequence for image {image_id}, exclude_subject={exclude_subject}")
    return torch.stack(rows, dim=0).mean(dim=0)


class RepeatDynamicPairDataset(Dataset):
    def __init__(
        self,
        pairs: list[dict[str, Any]],
        split: str,
        by_subject_image: dict[tuple[int, int], dict[str, Any]],
        train_by_image_subject: dict[tuple[int, int], torch.Tensor],
        train_global: dict[int, torch.Tensor],
    ) -> None:
        self.pairs = pairs
        self.split = split
        self.by_subject_image = by_subject_image
        self.train_by_image_subject = train_by_image_subject
        self.train_global = train_global
        self.ref_cache: dict[tuple[int, int | None], torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pair = self.pairs[idx]
        subject_1 = int(pair.get("subject_1", pair.get("subject", 0)))
        subject_2 = int(pair.get("subject_2", pair.get("reference_subject", 0)))
        image_1 = int(pair["nsdId_1"])
        image_2 = int(pair["nsdId_2"])
        anchor = self.by_subject_image.get((subject_1, image_1))
        if anchor is None:
            raise KeyError(f"Missing anchor sequence for subject={subject_1}, image={image_1}")
        exclude_subject = subject_1 if self.split == "train" else None
        cache_key = (image_2, exclude_subject)
        ref_seq = self.ref_cache.get(cache_key)
        if ref_seq is None:
            ref_seq = train_reference_sequence(image_2, exclude_subject, self.train_by_image_subject, self.train_global)
            self.ref_cache[cache_key] = ref_seq
        return {
            "x1_seq": anchor["x_seq"].float(),
            "x2_seq": ref_seq.float(),
            "clip_1": pair["clip_1"].float(),
            "clip_2": pair["clip_2"].float(),
            "same_image": torch.tensor(float(pair["same_image"]), dtype=torch.float32),
            "subject_1": subject_1,
            "subject_2": subject_2,
            "nsdId_1": image_1,
            "nsdId_2": image_2,
            "repeat_1": int(pair["repeat_1"]),
            "repeat_2": int(pair["repeat_2"]),
        }


def collate_dynamic_pairs(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "x1_seq": torch.stack([item["x1_seq"] for item in batch]),
        "x2_seq": torch.stack([item["x2_seq"] for item in batch]),
        "clip_1": torch.stack([item["clip_1"] for item in batch]),
        "clip_2": torch.stack([item["clip_2"] for item in batch]),
        "same_image": torch.stack([item["same_image"] for item in batch]),
        "subject_1": torch.tensor([int(item["subject_1"]) for item in batch], dtype=torch.int64),
        "subject_2": torch.tensor([int(item["subject_2"]) for item in batch], dtype=torch.int64),
        "nsdId_1": torch.tensor([int(item["nsdId_1"]) for item in batch], dtype=torch.int64),
        "nsdId_2": torch.tensor([int(item["nsdId_2"]) for item in batch], dtype=torch.int64),
        "repeat_1": torch.tensor([int(item["repeat_1"]) for item in batch], dtype=torch.int64),
        "repeat_2": torch.tensor([int(item["repeat_2"]) for item in batch], dtype=torch.int64),
    }


class CfcTemporalModule(nn.Module):
    """Small CfC-style continuous-time recurrent cell over repeat embeddings."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(dim, dim)
        self.hidden_proj = nn.Linear(dim, dim, bias=False)
        self.dt_proj = nn.Linear(1, dim)
        self.gate = nn.Linear(dim * 2 + 1, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, u: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        h = torch.zeros(u.shape[0], u.shape[2], dtype=u.dtype, device=u.device)
        states: list[torch.Tensor] = []
        for step in range(u.shape[1]):
            dt = delta_t[:, step : step + 1].to(dtype=u.dtype, device=u.device)
            candidate = torch.tanh(self.input_proj(u[:, step]) + self.hidden_proj(h) + self.dt_proj(dt))
            gate = torch.sigmoid(self.gate(torch.cat([u[:, step], h, dt], dim=-1)))
            h = self.norm(gate * candidate + (1.0 - gate) * h)
            h = self.dropout(h)
            states.append(h)
        return torch.stack(states, dim=1)


class RepeatDynamicVLM(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        node_feature_dim: int,
        clip_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
        base_encoder: str,
        temporal_module: str,
        num_heads: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        readout = "gated_flat" if base_encoder == "roi_transformer_noadj" else "flat"
        self.base = ReGraphVLM(
            n_nodes=n_nodes,
            node_feature_dim=node_feature_dim,
            clip_dim=clip_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
            readout=readout,
            roi_id_mode="normal",
            num_heads=num_heads,
            num_layers=num_layers,
            graph_encoder=base_encoder,
        )
        self.temporal_module_name = temporal_module
        if temporal_module == "gru":
            self.temporal = nn.GRU(embedding_dim, embedding_dim, num_layers=1, batch_first=True)
        elif temporal_module == "cfc":
            self.temporal = CfcTemporalModule(embedding_dim, dropout)
        else:
            raise ValueError(f"Unsupported temporal module: {temporal_module}")
        self.temporal_proj = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def encode_brain_sequence(
        self,
        x_seq: torch.Tensor,
        repeat_index: torch.Tensor,
        adjacency: torch.Tensor,
        subject: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps = x_seq.shape[:2]
        flat_x = x_seq.flatten(0, 1)
        flat_subject = subject[:, None].expand(batch, steps).reshape(-1)
        u = self.base.encode_brain(flat_x, adjacency, flat_subject).view(batch, steps, -1)
        if self.temporal_module_name == "gru":
            states, _ = self.temporal(u)
        else:
            delta_t = torch.ones(batch, steps, dtype=u.dtype, device=u.device)
            states = self.temporal(u, delta_t)
        gather_index = (repeat_index.long().clamp(1, steps) - 1).view(batch, 1, 1).expand(-1, 1, states.shape[-1])
        selected = states.gather(dim=1, index=gather_index).squeeze(1)
        return F.normalize(self.temporal_proj(selected), dim=-1)

    def encode_image(self, clip: torch.Tensor) -> torch.Tensor:
        return self.base.encode_image(clip)

    def pair_logits(
        self,
        x1_seq: torch.Tensor,
        x2_seq: torch.Tensor,
        adjacency: torch.Tensor,
        subject_1: torch.Tensor,
        subject_2: torch.Tensor,
        repeat_1: torch.Tensor,
        repeat_2: torch.Tensor,
    ) -> torch.Tensor:
        z1 = self.encode_brain_sequence(x1_seq, repeat_1, adjacency, subject_1)
        z2 = self.encode_brain_sequence(x2_seq, repeat_2, adjacency, subject_2)
        cos = (z1 * z2).sum(dim=-1)
        return self.base.log_scale.exp().clamp(max=100.0) * cos + self.base.bias


def collect_pair_scores(
    model: RepeatDynamicVLM,
    loader: DataLoader,
    adjacency: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model.pair_logits(
                batch["x1_seq"],
                batch["x2_seq"],
                adjacency,
                batch["subject_1"],
                batch["subject_2"],
                batch["repeat_1"],
                batch["repeat_2"],
            )
            loss = F.binary_cross_entropy_with_logits(logits, batch["same_image"])
            labels.append(batch["same_image"].cpu().numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())
            total_loss += float(loss.item()) * batch["same_image"].numel()
            total_n += batch["same_image"].numel()
    return np.concatenate(labels), np.concatenate(scores), total_loss / max(total_n, 1)


def repeat_retrieval_metrics(
    model: RepeatDynamicVLM,
    pairs: list[dict[str, Any]],
    dataset: RepeatDynamicPairDataset,
    adjacency: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    positives = [idx for idx, pair in enumerate(pairs) if int(pair["same_image"]) == 1]
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(positives), batch_size):
            indices = positives[start : start + batch_size]
            chunk = [dataset[idx] for idx in indices]
            batch = collate_dynamic_pairs(chunk)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            z1 = model.encode_brain_sequence(batch["x1_seq"], batch["repeat_1"], adjacency, batch["subject_1"]).cpu().numpy()
            z2 = model.encode_brain_sequence(batch["x2_seq"], batch["repeat_2"], adjacency, batch["subject_2"]).cpu().numpy()
            for idx, item in enumerate(chunk):
                rows.append(
                    {
                        "subject": int(item["subject_1"]),
                        "repeat_1": int(item["repeat_1"]),
                        "repeat_2": int(item["repeat_2"]),
                        "nsdId_1": int(item["nsdId_1"]),
                        "nsdId_2": int(item["nsdId_2"]),
                        "z1": z1[idx],
                        "z2": z2[idx],
                    }
                )
    return grouped_retrieval(rows, query_key="z1", candidate_key="z2")


def brain_image_metrics(
    model: RepeatDynamicVLM,
    pairs: list[dict[str, Any]],
    dataset: RepeatDynamicPairDataset,
    adjacency: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    positives = [idx for idx, pair in enumerate(pairs) if int(pair["same_image"]) == 1]
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(positives), batch_size):
            indices = positives[start : start + batch_size]
            chunk = [dataset[idx] for idx in indices]
            batch = collate_dynamic_pairs(chunk)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            zb = model.encode_brain_sequence(batch["x1_seq"], batch["repeat_1"], adjacency, batch["subject_1"]).cpu().numpy()
            zi = model.encode_image(batch["clip_1"]).cpu().numpy()
            for idx, item in enumerate(chunk):
                rows.append(
                    {
                        "subject": int(item["subject_1"]),
                        "repeat_1": int(item["repeat_1"]),
                        "repeat_2": int(item["repeat_2"]),
                        "nsdId_1": int(item["nsdId_1"]),
                        "nsdId_2": int(item["nsdId_1"]),
                        "brain": zb[idx],
                        "image": zi[idx],
                    }
                )
    brain_to_image = grouped_retrieval(rows, query_key="brain", candidate_key="image")
    image_to_brain = grouped_retrieval(rows, query_key="image", candidate_key="brain")
    return {
        "image_R@1": brain_to_image["r1"],
        "image_R@5": brain_to_image["r5"],
        "image_R@10": brain_to_image["r10"],
        "image_MRR": brain_to_image["mrr"],
        "image_median_rank": brain_to_image["median_rank"],
        "brain_R@1": image_to_brain["r1"],
        "brain_R@5": image_to_brain["r5"],
        "brain_R@10": image_to_brain["r10"],
        "brain_MRR": image_to_brain["mrr"],
        "brain_median_rank": image_to_brain["median_rank"],
    }


def evaluate_pairs(
    model: RepeatDynamicVLM,
    pairs: list[dict[str, Any]],
    dataset: RepeatDynamicPairDataset,
    threshold: float,
    adjacency: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_dynamic_pairs)
    labels, scores, loss = collect_pair_scores(model, loader, adjacency, device)
    preds = (scores >= threshold).astype(np.int64)
    repeat_ret = repeat_retrieval_metrics(model, pairs, dataset, adjacency, device, batch_size)
    bi_ret = brain_image_metrics(model, pairs, dataset, adjacency, device, batch_size)
    return {
        "test_loss": loss,
        "AUROC": auroc(labels, scores),
        "AUPRC": average_precision(labels, scores),
        "balanced_accuracy": balanced_accuracy(labels, preds),
        "R@1": repeat_ret["r1"],
        "R@5": repeat_ret["r5"],
        "R@10": repeat_ret["r10"],
        "MRR": repeat_ret["mrr"],
        **bi_ret,
        "n_test_pairs": int(len(pairs)),
    }


def write_eval_details(
    model: RepeatDynamicVLM,
    pairs: list[dict[str, Any]],
    dataset: RepeatDynamicPairDataset,
    adjacency: torch.Tensor,
    device: torch.device,
    batch_size: int,
    output_dir: Path,
) -> None:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_dynamic_pairs)
    score_rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model.pair_logits(
                batch["x1_seq"],
                batch["x2_seq"],
                adjacency,
                batch["subject_1"],
                batch["subject_2"],
                batch["repeat_1"],
                batch["repeat_2"],
            )
            scores = torch.sigmoid(logits).cpu().numpy()
            labels = batch["same_image"].cpu().numpy()
            for idx in range(len(scores)):
                score_rows.append(
                    {
                        "subject": int(batch["subject_1"][idx].cpu()),
                        "subject_2": int(batch["subject_2"][idx].cpu()),
                        "nsdId_1": int(batch["nsdId_1"][idx].cpu()),
                        "nsdId_2": int(batch["nsdId_2"][idx].cpu()),
                        "repeat_1": int(batch["repeat_1"][idx].cpu()),
                        "repeat_2": int(batch["repeat_2"][idx].cpu()),
                        "same_image": int(labels[idx]),
                        "score": float(scores[idx]),
                    }
                )
    if score_rows:
        with (output_dir / "test_pair_scores.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(score_rows[0].keys()))
            writer.writeheader()
            writer.writerows(score_rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    root = args.root.resolve()
    pair_fold_dir = root / args.pair_dataset_root / args.fold
    sequence_fold_dir = root / args.sequence_dataset_root / args.fold
    encoder_name = f"{args.base_encoder}_{args.temporal_module}_repeat_dynamic_clip"
    output_dir = root / args.output_root / encoder_name / f"lambda_{args.lambda_clip:g}" / args.fold / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    by_subject_image, train_by_image_subject, train_global = sequence_lookup(sequence_fold_dir)
    train_pairs = torch.load(pair_fold_dir / "train_pairs.pt", map_location="cpu", weights_only=False)
    val_pairs = torch.load(pair_fold_dir / "val_pairs.pt", map_location="cpu", weights_only=False)
    test_pairs = torch.load(pair_fold_dir / "test_pairs.pt", map_location="cpu", weights_only=False)

    train_dataset = RepeatDynamicPairDataset(train_pairs, "train", by_subject_image, train_by_image_subject, train_global)
    val_dataset = RepeatDynamicPairDataset(val_pairs, "val", by_subject_image, train_by_image_subject, train_global)
    test_dataset = RepeatDynamicPairDataset(test_pairs, "test", by_subject_image, train_by_image_subject, train_global)
    sample = train_dataset[0]
    n_nodes = int(sample["x1_seq"].shape[1])
    node_dim = int(sample["x1_seq"].shape[2])
    clip_dim = int(sample["clip_1"].shape[0])
    adjacency = torch.from_numpy(load_adjacency(pair_fold_dir, "no_adjacency", args.seed)).to(device)

    model = RepeatDynamicVLM(
        n_nodes=n_nodes,
        node_feature_dim=node_dim,
        clip_dim=clip_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        base_encoder=args.base_encoder,
        temporal_module=args.temporal_module,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_dynamic_pairs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_dynamic_pairs)

    best_val = -float("inf")
    best_epoch = -1
    best_threshold_value = 0.5
    bad_epochs = 0
    curve: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_n = 0
        for batch in train_loader:
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            z1 = model.encode_brain_sequence(batch["x1_seq"], batch["repeat_1"], adjacency, batch["subject_1"])
            z2 = model.encode_brain_sequence(batch["x2_seq"], batch["repeat_2"], adjacency, batch["subject_2"])
            cos = (z1 * z2).sum(dim=-1)
            logits = model.base.log_scale.exp().clamp(max=100.0) * cos + model.base.bias
            bce = F.binary_cross_entropy_with_logits(logits, batch["same_image"])
            pos = batch["same_image"] > 0.5
            nce = pair_infonce_from_embeddings(z1, z2, pos, args.temperature)
            clip_loss = clip_alignment_from_embeddings(model, z1, z2, batch["clip_1"], batch["clip_2"], args.clip_temperature)
            loss = bce + nce + args.lambda_clip * clip_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += float(loss.item()) * batch["same_image"].numel()
            train_n += batch["same_image"].numel()

        val_labels, val_scores, val_loss = collect_pair_scores(model, val_loader, adjacency, device)
        val_auroc = auroc(val_labels, val_scores)
        threshold, val_bal = best_threshold(val_labels, val_scores)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_n, 1),
            "val_loss": val_loss,
            "val_auroc": val_auroc,
            "val_balanced_accuracy": val_bal,
            "val_threshold": threshold,
        }
        curve.append(row)
        print(json.dumps(row), flush=True)
        if val_auroc > best_val:
            best_val = float(val_auroc)
            best_epoch = epoch
            best_threshold_value = threshold
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, output_dir / "checkpoint.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    eval_metrics = evaluate_pairs(model, test_pairs, test_dataset, best_threshold_value, adjacency, device, args.batch_size)
    if args.save_eval_details:
        write_eval_details(model, test_pairs, test_dataset, adjacency, device, args.batch_size, output_dir)

    metrics = {
        "model": "repeat_dynamic_vlm",
        "graph_encoder": args.base_encoder,
        "base_encoder": args.base_encoder,
        "temporal_module": args.temporal_module,
        "readout": "gated_flat" if args.base_encoder == "roi_transformer_noadj" else "flat",
        "fold": args.fold,
        "seed": args.seed,
        "lambda_clip": args.lambda_clip,
        "lambda_cross": 0.0,
        "lambda_subject_adv": 0.0,
        "adjacency_mode": "no_adjacency",
        "best_val_metric": best_val,
        "best_epoch": best_epoch,
        **eval_metrics,
        "n_nodes": n_nodes,
        "node_feature_dim": node_dim,
        "clip_dim": clip_dim,
        "status": "ok",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / "learning_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
