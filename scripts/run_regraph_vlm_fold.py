#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.regraph_vlm import ReGraphVLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ReGraph-VLM v0 on same-image repeat matching.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip"))
    parser.add_argument("--extra-test-dataset-root", type=Path, default=None)
    parser.add_argument("--extra-test-name", default="extra")
    parser.add_argument("--output-root", type=Path, default=Path("preproc_v0/repetition_familiarity/results/regraph_vlm"))
    parser.add_argument(
        "--graph-encoder",
        default="bnt_token_flat",
        choices=[
            "bnt_token_flat",
            "edge_bias_bnt",
            "edge_bias_graph_bnt",
            "graph_bnt",
            "regraph_graph",
            "roi_transformer_noadj",
            "roi_mlp",
            "gated_roi_mlp",
            "token_mlp",
            "gated_token_mlp",
            "mindeye2_shared",
            "umbrae_subject",
            "fusion",
        ],
    )
    parser.add_argument("--loss", default="bce_infonce_clip", choices=["bce_infonce_clip"])
    parser.add_argument("--lambda-clip", type=float, default=0.0)
    parser.add_argument(
        "--lambda-cross",
        type=float,
        default=0.0,
        help="Auxiliary cross-subject brain-brain InfoNCE weight. On cross-subject pairs this uses the same positive-pair structure as repeat InfoNCE.",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--clip-temperature", type=float, default=0.07)
    parser.add_argument(
        "--lambda-subject-adv",
        type=float,
        default=0.0,
        help="MindLink-style gradient-reversal subject-adversarial loss weight.",
    )
    parser.add_argument("--num-subjects", type=int, default=8)
    parser.add_argument("--graph-bias-scale", type=float, default=1.0)
    parser.add_argument("--attention-bias-scale", type=float, default=1.0)
    parser.add_argument("--attention-adjacency-scale", type=float, default=0.1)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--readout", default="flat", choices=["flat", "gated_flat"])
    parser.add_argument("--roi-id-mode", default="normal", choices=["normal", "none", "shuffled"])
    parser.add_argument(
        "--adjacency-mode",
        default="default",
        choices=[
            "default",
            "topk20_corr",
            "dense_corr",
            "identity",
            "random",
            "shuffled",
            "no_adjacency",
            "anatomical",
        ],
        help="Phase 3 graph-ablation adjacency. default uses adjacency.npy.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load an existing checkpoint from the output directory and write final metrics without training.",
    )
    parser.add_argument("--save-eval-details", action="store_true", help="Write pair scores and retrieval ranks for bootstrap analysis.")
    return parser.parse_args()


class ClipPairDataset(Dataset):
    def __init__(self, pairs: list[dict[str, Any]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pair = self.pairs[idx]
        subject_1 = int(pair.get("subject_1", pair.get("subject", 0)))
        subject_2 = int(pair.get("subject_2", pair.get("reference_subject", 0)))
        return {
            "x1": pair["x1"].float(),
            "x2": pair["x2"].float(),
            "clip_1": pair["clip_1"].float(),
            "clip_2": pair["clip_2"].float(),
            "same_image": torch.tensor(float(pair["same_image"]), dtype=torch.float32),
            "subject": subject_1,
            "subject_1": subject_1,
            "subject_2": subject_2,
            "nsdId_1": int(pair["nsdId_1"]),
            "nsdId_2": int(pair["nsdId_2"]),
            "repeat_1": int(pair["repeat_1"]),
            "repeat_2": int(pair["repeat_2"]),
        }


def collate_pairs(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "x1": torch.stack([item["x1"] for item in batch]),
        "x2": torch.stack([item["x2"] for item in batch]),
        "clip_1": torch.stack([item["clip_1"] for item in batch]),
        "clip_2": torch.stack([item["clip_2"] for item in batch]),
        "same_image": torch.stack([item["same_image"] for item in batch]),
        "subject": torch.tensor([int(item["subject"]) for item in batch], dtype=torch.int64),
        "subject_1": torch.tensor([int(item["subject_1"]) for item in batch], dtype=torch.int64),
        "subject_2": torch.tensor([int(item["subject_2"]) for item in batch], dtype=torch.int64),
        "nsdId_1": torch.tensor([int(item["nsdId_1"]) for item in batch], dtype=torch.int64),
        "nsdId_2": torch.tensor([int(item["nsdId_2"]) for item in batch], dtype=torch.int64),
        "repeat_1": torch.tensor([int(item["repeat_1"]) for item in batch], dtype=torch.int64),
        "repeat_2": torch.tensor([int(item["repeat_2"]) for item in batch], dtype=torch.int64),
    }


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata_average(scores)
    pos_rank_sum = float(ranks[labels == 1].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    precision = tp / np.arange(1, len(labels) + 1)
    return float((precision * sorted_labels).sum() / n_pos)


def balanced_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    preds = preds.astype(np.int64)
    pos = labels == 1
    neg = labels == 0
    tpr = float((preds[pos] == 1).mean()) if pos.any() else float("nan")
    tnr = float((preds[neg] == 0).mean()) if neg.any() else float("nan")
    return float((tpr + tnr) / 2.0)


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(scores)
    if len(candidates) > 5000:
        candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 5000)))
    best_threshold_value = float(candidates[0])
    best_bal = -1.0
    for threshold in candidates:
        bal = balanced_accuracy(labels, (scores >= threshold).astype(np.int64))
        if bal > best_bal:
            best_bal = float(bal)
            best_threshold_value = float(threshold)
    return best_threshold_value, best_bal


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    adj = np.abs(adjacency.astype(np.float32)).copy()
    np.fill_diagonal(adj, 1.0)
    degree = adj.sum(axis=1)
    inv_sqrt = np.where(degree > 0, degree ** -0.5, 0.0).astype(np.float32)
    return (inv_sqrt[:, None] * adj) * inv_sqrt[None, :]


def load_adjacency(fold_dir: Path, mode: str, seed: int) -> np.ndarray:
    base = np.load(fold_dir / "adjacency.npy").astype(np.float32)
    n_nodes = int(base.shape[0])

    if mode == "default":
        return normalize_adjacency(base)
    if mode == "topk20_corr":
        path = fold_dir / "adjacency_topk20_corr.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing top-k correlation adjacency: {path}")
        return normalize_adjacency(np.load(path).astype(np.float32))
    if mode == "dense_corr":
        path = fold_dir / "adjacency_dense_corr.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing dense correlation adjacency: {path}")
        return normalize_adjacency(np.load(path).astype(np.float32))
    if mode == "identity":
        return np.eye(n_nodes, dtype=np.float32)
    if mode == "no_adjacency":
        return np.zeros((n_nodes, n_nodes), dtype=np.float32)
    if mode == "anatomical":
        for name in ["adjacency_anatomical.npy", "anatomical_adjacency.npy"]:
            path = fold_dir / name
            if path.exists():
                return normalize_adjacency(np.load(path).astype(np.float32))
        raise FileNotFoundError(f"Missing anatomical adjacency in {fold_dir}")

    rng = np.random.default_rng(seed)
    if mode == "shuffled":
        perm = rng.permutation(n_nodes)
        return normalize_adjacency(base[perm][:, perm])
    if mode == "random":
        # Preserve the undirected edge count of the base graph but randomize endpoints.
        upper = np.triu(np.abs(base) > 0, k=1)
        edge_count = int(upper.sum())
        candidates = np.array(np.triu_indices(n_nodes, k=1)).T
        chosen = candidates[rng.choice(len(candidates), size=edge_count, replace=False)]
        adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        weights = rng.uniform(0.5, 1.0, size=edge_count).astype(np.float32)
        adj[chosen[:, 0], chosen[:, 1]] = weights
        adj[chosen[:, 1], chosen[:, 0]] = weights
        return normalize_adjacency(adj)
    raise ValueError(f"Unknown adjacency mode: {mode}")


def pair_infonce_loss(model: ReGraphVLM, batch: dict[str, torch.Tensor], adjacency: torch.Tensor, temperature: float) -> torch.Tensor:
    pos = batch["same_image"] > 0.5
    if int(pos.sum().item()) < 2:
        return batch["x1"].sum() * 0.0
    z1 = model.encode_brain(batch["x1"][pos], adjacency, batch["subject_1"][pos])
    z2 = model.encode_brain(batch["x2"][pos], adjacency, batch["subject_2"][pos])
    logits = (z1 @ z2.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def clip_alignment_loss(model: ReGraphVLM, batch: dict[str, torch.Tensor], adjacency: torch.Tensor, temperature: float) -> torch.Tensor:
    xb = torch.cat([batch["x1"], batch["x2"]], dim=0)
    ci = torch.cat([batch["clip_1"], batch["clip_2"]], dim=0)
    subjects = torch.cat([batch["subject_1"], batch["subject_2"]], dim=0)
    zb = model.encode_brain(xb, adjacency, subjects)
    zi = model.encode_image(ci)
    logits = (zb @ zi.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def collect_pair_scores(model: ReGraphVLM, loader: DataLoader, adjacency: torch.Tensor, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model.pair_logits(batch["x1"], batch["x2"], adjacency, batch["subject_1"], batch["subject_2"])
            loss = F.binary_cross_entropy_with_logits(logits, batch["same_image"])
            probs = torch.sigmoid(logits)
            labels.append(batch["same_image"].cpu().numpy())
            scores.append(probs.cpu().numpy())
            total_loss += float(loss.item()) * batch["same_image"].numel()
            total_n += batch["same_image"].numel()
    return np.concatenate(labels), np.concatenate(scores), total_loss / max(total_n, 1)


def repeat_retrieval_metrics(model: ReGraphVLM, pairs: list[dict[str, Any]], adjacency: torch.Tensor, device: torch.device, batch_size: int) -> dict[str, float]:
    positives = [pair for pair in pairs if int(pair["same_image"]) == 1]
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(positives), batch_size):
            chunk = positives[start : start + batch_size]
            x1 = torch.stack([pair["x1"].float() for pair in chunk]).to(device)
            x2 = torch.stack([pair["x2"].float() for pair in chunk]).to(device)
            s1 = torch.tensor([int(pair.get("subject_1", pair.get("subject", 0))) for pair in chunk], dtype=torch.long, device=device)
            s2 = torch.tensor([int(pair.get("subject_2", pair.get("reference_subject", 0))) for pair in chunk], dtype=torch.long, device=device)
            z1 = model.encode_brain(x1, adjacency, s1).cpu().numpy()
            z2 = model.encode_brain(x2, adjacency, s2).cpu().numpy()
            for idx, pair in enumerate(chunk):
                rows.append(
                    {
                        "subject": int(pair["subject"]),
                        "repeat_1": int(pair["repeat_1"]),
                        "repeat_2": int(pair["repeat_2"]),
                        "nsdId_1": int(pair["nsdId_1"]),
                        "nsdId_2": int(pair["nsdId_2"]),
                        "z1": z1[idx],
                        "z2": z2[idx],
                    }
                )
    return grouped_retrieval(rows, query_key="z1", candidate_key="z2")


def grouped_retrieval(rows: list[dict[str, Any]], query_key: str, candidate_key: str) -> dict[str, float]:
    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((int(row["subject"]), int(row["repeat_1"]), int(row["repeat_2"])), []).append(row)
    recall1 = recall5 = recall10 = 0
    rr: list[float] = []
    ranks: list[int] = []
    n_queries = 0
    for group_rows in groups.values():
        group_rows = sorted(group_rows, key=lambda row: int(row["nsdId_1"]))
        q = np.stack([row[query_key] for row in group_rows], axis=0)
        c = np.stack([row[candidate_key] for row in group_rows], axis=0)
        candidate_ids = np.array([int(row["nsdId_2"]) for row in group_rows])
        true_ids = np.array([int(row["nsdId_1"]) for row in group_rows])
        scores = q @ c.T
        for idx in range(scores.shape[0]):
            order = np.argsort(-scores[idx], kind="mergesort")
            pos = np.where(candidate_ids[order] == true_ids[idx])[0]
            if len(pos) == 0:
                continue
            rank = int(pos[0]) + 1
            recall1 += int(rank <= 1)
            recall5 += int(rank <= 5)
            recall10 += int(rank <= 10)
            rr.append(1.0 / rank)
            ranks.append(rank)
            n_queries += 1
    if n_queries == 0:
        return {"r1": float("nan"), "r5": float("nan"), "r10": float("nan"), "mrr": float("nan"), "median_rank": float("nan")}
    return {
        "r1": float(recall1 / n_queries),
        "r5": float(recall5 / n_queries),
        "r10": float(recall10 / n_queries),
        "mrr": float(np.mean(rr)),
        "median_rank": float(np.median(ranks)),
        "n_queries": int(n_queries),
    }


def retrieval_rank_rows(rows: list[dict[str, Any]], query_key: str, candidate_key: str, mode: str) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((int(row["subject"]), int(row["repeat_1"]), int(row["repeat_2"])), []).append(row)
    out: list[dict[str, Any]] = []
    for group, group_rows in groups.items():
        group_rows = sorted(group_rows, key=lambda row: int(row["nsdId_1"]))
        q = np.stack([row[query_key] for row in group_rows], axis=0)
        c = np.stack([row[candidate_key] for row in group_rows], axis=0)
        candidate_ids = np.array([int(row["nsdId_2"]) for row in group_rows])
        true_ids = np.array([int(row["nsdId_1"]) for row in group_rows])
        scores = q @ c.T
        for idx in range(scores.shape[0]):
            order = np.argsort(-scores[idx], kind="mergesort")
            pos = np.where(candidate_ids[order] == true_ids[idx])[0]
            if len(pos) == 0:
                continue
            rank = int(pos[0]) + 1
            out.append(
                {
                    "mode": mode,
                    "subject": group[0],
                    "repeat_1": group[1],
                    "repeat_2": group[2],
                    "nsdId": int(true_ids[idx]),
                    "rank": rank,
                    "reciprocal_rank": 1.0 / rank,
                    "hit1": int(rank <= 1),
                    "hit5": int(rank <= 5),
                    "hit10": int(rank <= 10),
                    "n_candidates": int(len(candidate_ids)),
                }
            )
    return out


def brain_image_retrieval_metrics(model: ReGraphVLM, pairs: list[dict[str, Any]], adjacency: torch.Tensor, device: torch.device, batch_size: int) -> dict[str, float]:
    positives = [pair for pair in pairs if int(pair["same_image"]) == 1]
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(positives), batch_size):
            chunk = positives[start : start + batch_size]
            x1 = torch.stack([pair["x1"].float() for pair in chunk]).to(device)
            clip1 = torch.stack([pair["clip_1"].float() for pair in chunk]).to(device)
            s1 = torch.tensor([int(pair.get("subject_1", pair.get("subject", 0))) for pair in chunk], dtype=torch.long, device=device)
            zb = model.encode_brain(x1, adjacency, s1).cpu().numpy()
            zi = model.encode_image(clip1).cpu().numpy()
            for idx, pair in enumerate(chunk):
                rows.append(
                    {
                        "subject": int(pair["subject"]),
                        "repeat_1": int(pair["repeat_1"]),
                        "repeat_2": int(pair["repeat_2"]),
                        "nsdId_1": int(pair["nsdId_1"]),
                        "nsdId_2": int(pair["nsdId_1"]),
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


def write_eval_details(
    model: ReGraphVLM,
    pairs: list[dict[str, Any]],
    adjacency: torch.Tensor,
    device: torch.device,
    batch_size: int,
    output_dir: Path,
    prefix: str = "test",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(ClipPairDataset(pairs), batch_size=batch_size, shuffle=False, collate_fn=collate_pairs)
    score_rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model.pair_logits(batch["x1"], batch["x2"], adjacency, batch["subject_1"], batch["subject_2"])
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
        with (output_dir / f"{prefix}_pair_scores.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(score_rows[0].keys()))
            writer.writeheader()
            writer.writerows(score_rows)

    positives = [pair for pair in pairs if int(pair["same_image"]) == 1]
    repeat_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(positives), batch_size):
            chunk = positives[start : start + batch_size]
            x1 = torch.stack([pair["x1"].float() for pair in chunk]).to(device)
            x2 = torch.stack([pair["x2"].float() for pair in chunk]).to(device)
            c1 = torch.stack([pair["clip_1"].float() for pair in chunk]).to(device)
            s1 = torch.tensor([int(pair.get("subject_1", pair.get("subject", 0))) for pair in chunk], dtype=torch.long, device=device)
            s2 = torch.tensor([int(pair.get("subject_2", pair.get("reference_subject", 0))) for pair in chunk], dtype=torch.long, device=device)
            z1 = model.encode_brain(x1, adjacency, s1).cpu().numpy()
            z2 = model.encode_brain(x2, adjacency, s2).cpu().numpy()
            zi = model.encode_image(c1).cpu().numpy()
            for idx, pair in enumerate(chunk):
                common = {
                    "subject": int(pair.get("subject_1", pair.get("subject", 0))),
                    "repeat_1": int(pair["repeat_1"]),
                    "repeat_2": int(pair["repeat_2"]),
                    "nsdId_1": int(pair["nsdId_1"]),
                    "nsdId_2": int(pair["nsdId_2"]),
                }
                repeat_rows.append({**common, "z1": z1[idx], "z2": z2[idx]})
                image_rows.append({**common, "brain": z1[idx], "image": zi[idx], "nsdId_2": int(pair["nsdId_1"])})
    rank_rows = (
        retrieval_rank_rows(repeat_rows, "z1", "z2", "repeat")
        + retrieval_rank_rows(image_rows, "brain", "image", "brain_to_image")
        + retrieval_rank_rows(image_rows, "image", "brain", "image_to_brain")
    )
    if rank_rows:
        with (output_dir / f"{prefix}_retrieval_ranks.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rank_rows[0].keys()))
            writer.writeheader()
            writer.writerows(rank_rows)


def evaluate_pairs(
    model: ReGraphVLM,
    pairs: list[dict[str, Any]],
    threshold: float,
    adjacency: torch.Tensor,
    device: torch.device,
    batch_size: int,
    prefix: str = "",
) -> dict[str, Any]:
    loader = DataLoader(ClipPairDataset(pairs), batch_size=batch_size, shuffle=False, collate_fn=collate_pairs)
    labels, scores, loss = collect_pair_scores(model, loader, adjacency, device)
    preds = (scores >= threshold).astype(np.int64)
    repeat_ret = repeat_retrieval_metrics(model, pairs, adjacency, device, batch_size)
    bi_ret = brain_image_retrieval_metrics(model, pairs, adjacency, device, batch_size)
    base = {
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
    if not prefix:
        return base
    return {f"{prefix}_{key}": value for key, value in base.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    root = args.root.resolve()
    fold_dir = (root / args.dataset_root / args.fold).resolve()
    encoder_dir = f"{args.graph_encoder}_clip" if args.readout == "flat" else f"{args.graph_encoder}_{args.readout}_clip"
    if args.adjacency_mode != "default":
        encoder_dir = f"{encoder_dir}_adj_{args.adjacency_mode}"
    lambda_dir = f"lambda_{args.lambda_clip:g}"
    if args.lambda_cross != 0.0:
        lambda_dir = f"{lambda_dir}_cross_{args.lambda_cross:g}"
    if args.lambda_subject_adv != 0.0:
        lambda_dir = f"{lambda_dir}_subjadv_{args.lambda_subject_adv:g}"
    output_dir = root / args.output_root / encoder_dir / lambda_dir / args.fold / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    train_pairs = torch.load(fold_dir / "train_pairs.pt", map_location="cpu", weights_only=False)
    val_pairs = torch.load(fold_dir / "val_pairs.pt", map_location="cpu", weights_only=False)
    test_pairs = torch.load(fold_dir / "test_pairs.pt", map_location="cpu", weights_only=False)
    extra_test_pairs = None
    if args.extra_test_dataset_root is not None:
        extra_fold_dir = (root / args.extra_test_dataset_root / args.fold).resolve()
        extra_test_pairs = torch.load(extra_fold_dir / "test_pairs.pt", map_location="cpu", weights_only=False)
    sample = train_pairs[0]
    n_nodes = int(sample["x1"].shape[0])
    node_dim = int(sample["x1"].shape[1])
    clip_dim = int(sample["clip_1"].shape[0])
    adjacency = torch.from_numpy(load_adjacency(fold_dir, args.adjacency_mode, args.seed)).to(device)

    train_loader = DataLoader(ClipPairDataset(train_pairs), batch_size=args.batch_size, shuffle=True, collate_fn=collate_pairs)
    val_loader = DataLoader(ClipPairDataset(val_pairs), batch_size=args.batch_size, shuffle=False, collate_fn=collate_pairs)
    test_loader = DataLoader(ClipPairDataset(test_pairs), batch_size=args.batch_size, shuffle=False, collate_fn=collate_pairs)

    model = ReGraphVLM(
        n_nodes=n_nodes,
        node_feature_dim=node_dim,
        clip_dim=clip_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        readout=args.readout,
        roi_id_mode=args.roi_id_mode,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        graph_encoder=args.graph_encoder,
        num_subjects=args.num_subjects,
        graph_bias_scale=args.graph_bias_scale,
        attention_bias_scale=args.attention_bias_scale,
        attention_adjacency_scale=args.attention_adjacency_scale,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = -float("inf")
    best_epoch = -1
    best_threshold_value = 0.5
    bad_epochs = 0
    curve: list[dict[str, Any]] = []

    if not args.eval_only:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            train_n = 0
            for batch in train_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                opt.zero_grad(set_to_none=True)
                logits = model.pair_logits(batch["x1"], batch["x2"], adjacency, batch["subject_1"], batch["subject_2"])
                bce = F.binary_cross_entropy_with_logits(logits, batch["same_image"])
                nce = pair_infonce_loss(model, batch, adjacency, args.temperature)
                cross_nce = pair_infonce_loss(model, batch, adjacency, args.temperature)
                clip_loss = clip_alignment_loss(model, batch, adjacency, args.clip_temperature)
                adv_loss = batch["x1"].sum() * 0.0
                if args.lambda_subject_adv > 0:
                    subject_labels = (batch["subject_1"].long() - 1).clamp(min=0, max=args.num_subjects - 1)
                    subject_logits = model.subject_logits(
                        batch["x1"],
                        adjacency,
                        batch["subject_1"],
                        reverse_scale=1.0,
                    )
                    adv_loss = F.cross_entropy(subject_logits, subject_labels)
                loss = bce + nce + args.lambda_cross * cross_nce + args.lambda_clip * clip_loss + args.lambda_subject_adv * adv_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
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
    load_result = model.load_state_dict(checkpoint["model"], strict=False)
    missing = [key for key in load_result.missing_keys if not key.startswith("subject_classifier.")]
    unexpected = [key for key in load_result.unexpected_keys if not key.startswith("subject_classifier.")]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint state mismatch. missing={missing}, unexpected={unexpected}")
    if args.eval_only:
        val_labels, val_scores, val_loss = collect_pair_scores(model, val_loader, adjacency, device)
        best_val = auroc(val_labels, val_scores)
        best_threshold_value, _ = best_threshold(val_labels, val_scores)
        best_epoch = int(checkpoint.get("epoch", -1))
        curve.append(
            {
                "epoch": best_epoch,
                "train_loss": float("nan"),
                "val_loss": val_loss,
                "val_auroc": best_val,
                "val_balanced_accuracy": balanced_accuracy(val_labels, (val_scores >= best_threshold_value).astype(np.int64)),
                "val_threshold": best_threshold_value,
            }
        )
    eval_metrics = evaluate_pairs(model, test_pairs, best_threshold_value, adjacency, device, args.batch_size)
    if args.save_eval_details:
        write_eval_details(model, test_pairs, adjacency, device, args.batch_size, output_dir, prefix="test")
    if extra_test_pairs is not None:
        eval_metrics.update(
            evaluate_pairs(
                model,
                extra_test_pairs,
                best_threshold_value,
                adjacency,
                device,
                args.batch_size,
                prefix=args.extra_test_name,
            )
        )
        if args.save_eval_details:
            write_eval_details(model, extra_test_pairs, adjacency, device, args.batch_size, output_dir, prefix=args.extra_test_name)

    metrics = {
        "model": "regraph_vlm_v0",
        "graph_encoder": args.graph_encoder,
        "fold": args.fold,
        "seed": args.seed,
        "lambda_clip": args.lambda_clip,
        "lambda_cross": args.lambda_cross,
        "lambda_subject_adv": args.lambda_subject_adv,
        "readout": args.readout,
        "roi_id_mode": args.roi_id_mode,
        "adjacency_mode": args.adjacency_mode,
        "num_subjects": args.num_subjects,
        "graph_bias_scale": args.graph_bias_scale,
        "attention_bias_scale": args.attention_bias_scale,
        "attention_adjacency_scale": args.attention_adjacency_scale,
        "loss": args.loss,
        "clip_temperature": args.clip_temperature,
        "temperature": args.temperature,
        "best_val_metric": best_val,
        "best_epoch": best_epoch,
        **eval_metrics,
        "n_nodes": n_nodes,
        "node_feature_dim": node_dim,
        "clip_dim": clip_dim,
        "status": "ok",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    curve_path = output_dir / ("eval_only_curve.csv" if args.eval_only else "learning_curve.csv")
    with curve_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
