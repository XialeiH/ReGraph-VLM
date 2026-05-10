#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw similarity baseline for cross-subject same-image matching.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/cross_subject_raw_similarity"),
    )
    parser.add_argument("--folds", nargs="+", default=["fold_01", "fold_04"])
    return parser.parse_args()


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


def to_vector(x: torch.Tensor, method: str) -> np.ndarray:
    if method == "pearson_mean_beta":
        return x.float()[:, 0].flatten().numpy()
    return x.float().flatten().numpy()


def zscore(y: np.ndarray) -> np.ndarray:
    std = float(y.std())
    if std < 1e-8:
        return y * 0.0
    return (y - float(y.mean())) / std


def similarity_score(pair: dict[str, Any], method: str) -> float:
    x1 = to_vector(pair["x1"], method)
    x2 = to_vector(pair["x2"], method)
    if method.startswith("pearson"):
        x1 = zscore(x1)
        x2 = zscore(x2)
    denom = float(np.linalg.norm(x1) * np.linalg.norm(x2))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(x1, x2) / denom)


def grouped_retrieval(positives: list[dict[str, Any]], scores: np.ndarray) -> dict[str, float]:
    groups: dict[tuple[int, int, int], list[tuple[dict[str, Any], float]]] = {}
    for pair, score in zip(positives, scores):
        key = (int(pair["subject"]), int(pair["repeat_1"]), int(pair["repeat_2"]))
        groups.setdefault(key, []).append((pair, float(score)))
    r1 = r5 = r10 = 0
    rr = []
    ranks = []
    n = 0
    for rows in groups.values():
        rows = sorted(rows, key=lambda item: int(item[0]["nsdId_1"]))
        candidate_ids = np.array([int(pair["nsdId_2"]) for pair, _ in rows])
        true_ids = np.array([int(pair["nsdId_1"]) for pair, _ in rows])
        group_scores = np.array([score for _, score in rows])
        for idx in range(len(rows)):
            order = np.argsort(-group_scores, kind="mergesort")
            pos = np.where(candidate_ids[order] == true_ids[idx])[0]
            if len(pos) == 0:
                continue
            rank = int(pos[0]) + 1
            r1 += int(rank <= 1)
            r5 += int(rank <= 5)
            r10 += int(rank <= 10)
            rr.append(1.0 / rank)
            ranks.append(rank)
            n += 1
    return {
        "R@1": float(r1 / n) if n else float("nan"),
        "R@5": float(r5 / n) if n else float("nan"),
        "R@10": float(r10 / n) if n else float("nan"),
        "MRR": float(np.mean(rr)) if rr else float("nan"),
        "median_rank": float(np.median(ranks)) if ranks else float("nan"),
        "n_queries": int(n),
    }


def evaluate_split(fold_dir: Path, split: str, method: str) -> dict[str, Any]:
    pairs = torch.load(fold_dir / f"{split}_pairs.pt", map_location="cpu", weights_only=False)
    labels = np.array([int(pair["same_image"]) for pair in pairs], dtype=np.int64)
    scores = np.array([similarity_score(pair, method) for pair in pairs], dtype=np.float64)
    positives = [pair for pair in pairs if int(pair["same_image"]) == 1]
    pos_scores = np.array([similarity_score(pair, method) for pair in positives], dtype=np.float64)
    ret = grouped_retrieval(positives, pos_scores)
    return {
        "split": split,
        "AUROC": auroc(labels, scores),
        "AUPRC": average_precision(labels, scores),
        **ret,
        "n_pairs": int(len(pairs)),
        "n_positive": int(labels.sum()),
        "n_negative": int(len(labels) - labels.sum()),
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    dataset_root = root / args.dataset_root
    output_root = root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    methods = ["pearson_flat", "cosine_flat", "pearson_mean_beta"]
    for fold in args.folds:
        fold_dir = dataset_root / fold
        for method in methods:
            for split in ["val", "test"]:
                row = evaluate_split(fold_dir, split, method)
                row["fold"] = fold
                row["model"] = f"raw_{method}"
                rows.append(row)
    df = pd.DataFrame(rows)
    out_csv = output_root / "cross_subject_raw_similarity_summary.csv"
    out_json = output_root / "cross_subject_raw_similarity_summary.json"
    df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print({"summary_csv": str(out_csv), "n_rows": int(len(df))})


if __name__ == "__main__":
    main()
