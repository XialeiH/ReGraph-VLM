#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch


SIMILARITY_NAMES = [
    "pearson_flat",
    "cosine_flat",
    "pearson_mean_beta",
    "roiwise_avg_corr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raw similarity baselines for strict T=3 repeat pair matching.")
    parser.add_argument("--root", type=Path, required=True, help="v0_shared_unit root.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/repeat_pair_similarity_baseline"),
    )
    parser.add_argument("--folds", nargs="+", default=["fold_01", "fold_04"])
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    av = a.astype(np.float64).reshape(-1)
    bv = b.astype(np.float64).reshape(-1)
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    return float(np.dot(av, bv) / denom) if denom > 0 else 0.0


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    av = a.astype(np.float64).reshape(-1)
    bv = b.astype(np.float64).reshape(-1)
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    return float(np.dot(av, bv) / denom) if denom > 0 else 0.0


def pearson_flat(x1: torch.Tensor, x2: torch.Tensor) -> float:
    return safe_pearson(x1.numpy(), x2.numpy())


def cosine_flat(x1: torch.Tensor, x2: torch.Tensor) -> float:
    return safe_cosine(x1.numpy(), x2.numpy())


def pearson_mean_beta(x1: torch.Tensor, x2: torch.Tensor) -> float:
    return safe_pearson(x1[:, 0].numpy(), x2[:, 0].numpy())


def roiwise_avg_corr(x1: torch.Tensor, x2: torch.Tensor) -> float:
    a = x1.numpy().astype(np.float64)
    b = x2.numpy().astype(np.float64)
    vals = []
    for roi_idx in range(a.shape[0]):
        vals.append(safe_pearson(a[roi_idx], b[roi_idx]))
    return float(np.mean(vals)) if vals else 0.0


SIMILARITIES: dict[str, Callable[[torch.Tensor, torch.Tensor], float]] = {
    "pearson_flat": pearson_flat,
    "cosine_flat": cosine_flat,
    "pearson_mean_beta": pearson_mean_beta,
    "roiwise_avg_corr": roiwise_avg_corr,
}


def rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
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
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    ranks = np.arange(1, len(labels) + 1)
    precision = tp / ranks
    return float((precision * sorted_labels).sum() / n_pos)


def balanced_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    preds = preds.astype(np.int64)
    pos = labels == 1
    neg = labels == 0
    tpr = float((preds[pos] == 1).mean()) if pos.any() else float("nan")
    tnr = float((preds[neg] == 0).mean()) if neg.any() else float("nan")
    return float((tpr + tnr) / 2.0)


def accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    return float((labels.astype(np.int64) == preds.astype(np.int64)).mean()) if len(labels) else float("nan")


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    if len(labels) == 0:
        return 0.0, float("nan"), float("nan")
    candidates = np.unique(scores)
    if len(candidates) > 5000:
        quantiles = np.linspace(0.0, 1.0, 5000)
        candidates = np.unique(np.quantile(scores, quantiles))
    best_thr = float(candidates[0])
    best_bal = -1.0
    best_acc = -1.0
    for thr in candidates:
        preds = (scores >= thr).astype(np.int64)
        bal = balanced_accuracy(labels, preds)
        acc = accuracy(labels, preds)
        if bal > best_bal or (math.isclose(bal, best_bal) and acc > best_acc):
            best_thr = float(thr)
            best_bal = float(bal)
            best_acc = float(acc)
    return best_thr, best_bal, best_acc


def metrics_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (scores >= threshold).astype(np.int64)
    return {
        "accuracy": accuracy(labels, preds),
        "balanced_accuracy": balanced_accuracy(labels, preds),
    }


def score_pairs(pairs: list[dict[str, object]], sim_name: str) -> tuple[np.ndarray, np.ndarray]:
    labels = np.array([int(pair["same_image"]) for pair in pairs], dtype=np.int64)
    x1 = torch.stack([pair["x1"] for pair in pairs]).numpy().astype(np.float64)  # type: ignore[list-item]
    x2 = torch.stack([pair["x2"] for pair in pairs]).numpy().astype(np.float64)  # type: ignore[list-item]

    if sim_name == "pearson_flat":
        a = x1.reshape(x1.shape[0], -1)
        b = x2.reshape(x2.shape[0], -1)
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        scores = np.divide(np.sum(a * b, axis=1), denom, out=np.zeros_like(denom), where=denom > 0)
    elif sim_name == "cosine_flat":
        a = x1.reshape(x1.shape[0], -1)
        b = x2.reshape(x2.shape[0], -1)
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        scores = np.divide(np.sum(a * b, axis=1), denom, out=np.zeros_like(denom), where=denom > 0)
    elif sim_name == "pearson_mean_beta":
        a = x1[:, :, 0]
        b = x2[:, :, 0]
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        scores = np.divide(np.sum(a * b, axis=1), denom, out=np.zeros_like(denom), where=denom > 0)
    elif sim_name == "roiwise_avg_corr":
        a = x1 - x1.mean(axis=2, keepdims=True)
        b = x2 - x2.mean(axis=2, keepdims=True)
        denom = np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2)
        roi_scores = np.divide(np.sum(a * b, axis=2), denom, out=np.zeros_like(denom), where=denom > 0)
        scores = roi_scores.mean(axis=1)
    else:
        raise ValueError(f"Unknown similarity: {sim_name}")
    return labels, scores.astype(np.float64)


def matrix_similarity(query: np.ndarray, candidates: np.ndarray, sim_name: str) -> np.ndarray:
    if sim_name == "pearson_mean_beta":
        query = query[:, :, 0]
        candidates = candidates[:, :, 0]
    elif sim_name == "roiwise_avg_corr":
        out = np.zeros((query.shape[0], candidates.shape[0]), dtype=np.float64)
        for roi_idx in range(query.shape[1]):
            q = query[:, roi_idx, :].astype(np.float64)
            c = candidates[:, roi_idx, :].astype(np.float64)
            q = q - q.mean(axis=1, keepdims=True)
            c = c - c.mean(axis=1, keepdims=True)
            denom = np.linalg.norm(q, axis=1, keepdims=True) * np.linalg.norm(c, axis=1, keepdims=True).T
            scores = q @ c.T
            scores = np.divide(scores, denom, out=np.zeros_like(scores), where=denom > 0)
            out += scores
        return out / query.shape[1]
    else:
        query = query.reshape(query.shape[0], -1)
        candidates = candidates.reshape(candidates.shape[0], -1)

    query = query.astype(np.float64)
    candidates = candidates.astype(np.float64)
    if sim_name.startswith("pearson"):
        query = query - query.mean(axis=1, keepdims=True)
        candidates = candidates - candidates.mean(axis=1, keepdims=True)
    q_norm = np.linalg.norm(query, axis=1, keepdims=True)
    c_norm = np.linalg.norm(candidates, axis=1, keepdims=True)
    denom = q_norm * c_norm.T
    scores = query @ candidates.T
    return np.divide(scores, denom, out=np.zeros_like(scores), where=denom > 0)


def retrieval_metrics_from_pairs(pairs: list[dict[str, object]], sim_name: str) -> dict[str, float]:
    positives = [pair for pair in pairs if int(pair["same_image"]) == 1]
    groups: dict[tuple[int, int, int], list[dict[str, object]]] = {}
    for pair in positives:
        key = (int(pair["subject"]), int(pair["repeat_1"]), int(pair["repeat_2"]))
        groups.setdefault(key, []).append(pair)

    reciprocal_ranks: list[float] = []
    recall1 = 0
    recall5 = 0
    n_queries = 0
    for group_pairs in groups.values():
        group_pairs = sorted(group_pairs, key=lambda row: int(row["nsdId_1"]))
        query = torch.stack([row["x1"] for row in group_pairs]).numpy()  # type: ignore[list-item]
        candidates = torch.stack([row["x2"] for row in group_pairs]).numpy()  # type: ignore[list-item]
        candidate_ids = np.array([int(row["nsdId_2"]) for row in group_pairs])
        true_ids = np.array([int(row["nsdId_1"]) for row in group_pairs])
        scores = matrix_similarity(query, candidates, sim_name)
        for row_idx in range(scores.shape[0]):
            order = np.argsort(-scores[row_idx], kind="mergesort")
            hit_positions = np.where(candidate_ids[order] == true_ids[row_idx])[0]
            if len(hit_positions) == 0:
                continue
            rank = int(hit_positions[0]) + 1
            reciprocal_ranks.append(1.0 / rank)
            recall1 += int(rank <= 1)
            recall5 += int(rank <= 5)
            n_queries += 1

    if n_queries == 0:
        return {"recall_at_1": float("nan"), "recall_at_5": float("nan"), "mrr": float("nan"), "n_queries": 0}
    return {
        "recall_at_1": float(recall1 / n_queries),
        "recall_at_5": float(recall5 / n_queries),
        "mrr": float(np.mean(reciprocal_ranks)),
        "n_queries": int(n_queries),
    }


def load_pairs(root: Path, dataset_root: Path, fold: str, split: str) -> list[dict[str, object]]:
    path = root / dataset_root / fold / f"{split}_pairs.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing pair dataset: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    thresholds: dict[tuple[str, str], float] = {}
    for fold in args.folds:
        for sim_name in SIMILARITY_NAMES:
            if "val" not in args.splits:
                raise ValueError("The val split is required to select a threshold.")
            val_pairs = load_pairs(root, args.dataset_root, fold, "val")
            val_labels, val_scores = score_pairs(val_pairs, sim_name)
            threshold, val_bal, val_acc = best_threshold(val_labels, val_scores)
            thresholds[(fold, sim_name)] = threshold

            for split in args.splits:
                pairs = load_pairs(root, args.dataset_root, fold, split)
                labels, scores = score_pairs(pairs, sim_name)
                binary = metrics_at_threshold(labels, scores, threshold)
                retrieval = retrieval_metrics_from_pairs(pairs, sim_name)
                rows.append(
                    {
                        "fold": fold,
                        "split": split,
                        "similarity": sim_name,
                        "n_pairs": int(len(pairs)),
                        "n_positive": int(labels.sum()),
                        "n_negative": int(len(labels) - labels.sum()),
                        "auroc": auroc(labels, scores),
                        "auprc": average_precision(labels, scores),
                        "val_selected_threshold": threshold,
                        "val_balanced_accuracy_at_selected_threshold": val_bal,
                        "val_accuracy_at_selected_threshold": val_acc,
                        "accuracy": binary["accuracy"],
                        "balanced_accuracy": binary["balanced_accuracy"],
                        "recall_at_1": retrieval["recall_at_1"],
                        "recall_at_5": retrieval["recall_at_5"],
                        "mrr": retrieval["mrr"],
                        "n_retrieval_queries": retrieval["n_queries"],
                        "score_mean": float(scores.mean()),
                        "score_std": float(scores.std()),
                        "positive_score_mean": float(scores[labels == 1].mean()),
                        "negative_score_mean": float(scores[labels == 0].mean()),
                    }
                )

    fieldnames = [
        "fold",
        "split",
        "similarity",
        "n_pairs",
        "n_positive",
        "n_negative",
        "auroc",
        "auprc",
        "val_selected_threshold",
        "val_balanced_accuracy_at_selected_threshold",
        "val_accuracy_at_selected_threshold",
        "accuracy",
        "balanced_accuracy",
        "recall_at_1",
        "recall_at_5",
        "mrr",
        "n_retrieval_queries",
        "score_mean",
        "score_std",
        "positive_score_mean",
        "negative_score_mean",
    ]
    summary_path = out_dir / "repeat_pair_similarity_baseline_summary.csv"
    write_csv(summary_path, rows, fieldnames)
    (out_dir / "repeat_pair_similarity_baseline_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"summary_csv": str(summary_path), "n_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
