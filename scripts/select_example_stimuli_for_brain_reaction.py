#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch


FEATURES = {"mean_beta": 0, "std_beta": 1, "q90_beta": 2, "positive_fraction": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select representative stimulus-specific brain reaction examples.")
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--clip_dataset_root", type=Path, default=None)
    parser.add_argument("--results_csv", type=Path, default=None)
    parser.add_argument("--folds", nargs="+", default=["fold_01", "fold_04"])
    parser.add_argument(
        "--criteria",
        nargs="+",
        default=["high_stability", "strong_delta", "model_success", "model_failure"],
    )
    parser.add_argument("--n_per_criterion", type=int, default=2)
    parser.add_argument("--feature", choices=sorted(FEATURES), default="mean_beta")
    parser.add_argument("--out_csv", type=Path, required=True)
    return parser.parse_args()


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(a.dot(b) / denom)


def load_sequences(dataset_root: Path, folds: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_dir = dataset_root / fold
        for split in ["train", "val", "test"]:
            path = fold_dir / f"{split}_sequences.pt"
            if not path.exists():
                continue
            seqs = torch.load(path, map_location="cpu", weights_only=False)
            for seq in seqs:
                item = dict(seq)
                item["fold"] = fold
                item["split"] = split
                rows.append(item)
    return rows


def cross_subject_similarity(rows: list[dict[str, Any]], feature_idx: int) -> dict[tuple[str, int], float]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["fold"]), int(row["nsdId"])), []).append(row)
    out: dict[tuple[str, int], float] = {}
    for key, seqs in grouped.items():
        if len(seqs) < 2:
            out[key] = float("nan")
            continue
        vals = []
        for repeat_idx in range(3):
            xs = [seq["x_seq"][repeat_idx, :, feature_idx].float().numpy() for seq in seqs]
            for i in range(len(xs)):
                for j in range(i + 1, len(xs)):
                    vals.append(corr(xs[i], xs[j]))
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def compute_rows(seqs: list[dict[str, Any]], feature_idx: int) -> list[dict[str, Any]]:
    cs = cross_subject_similarity(seqs, feature_idx)
    rows = []
    for seq in seqs:
        x_all = seq["x_seq"].float().numpy()
        x = x_all[:, :, feature_idx]
        sim12 = corr(x_all[0], x_all[1])
        sim13 = corr(x_all[0], x_all[2])
        sim23 = corr(x_all[1], x_all[2])
        same_sim = float(np.mean([sim12, sim13, sim23]))
        delta31 = float(np.linalg.norm(x[2] - x[0]) / np.sqrt(x.shape[1]))
        key = (str(seq["fold"]), int(seq["nsdId"]))
        rows.append(
            {
                "criterion": "",
                "fold": str(seq["fold"]),
                "split": str(seq["split"]),
                "subject": int(seq["subject"]),
                "nsdId": int(seq["nsdId"]),
                "repeat_pair": "1-2;1-3;2-3",
                "same_repeat_similarity": same_sim,
                "sim12": sim12,
                "sim13": sim13,
                "sim23": sim23,
                "delta31_magnitude": delta31,
                "cross_subject_similarity": cs.get(key, float("nan")),
                "model_rank": "",
                "model_score": "",
                "notes": "",
            }
        )
    return rows


def take_unique(candidates: list[dict[str, Any]], n: int, used: set[tuple[str, int, int]], criterion: str, note: str) -> list[dict[str, Any]]:
    out = []
    for row in candidates:
        key = (str(row["fold"]), int(row["subject"]), int(row["nsdId"]))
        if key in used:
            continue
        item = dict(row)
        item["criterion"] = criterion
        item["notes"] = note
        out.append(item)
        used.add(key)
        if len(out) >= n:
            break
    return out


def main() -> None:
    args = parse_args()
    seqs = load_sequences(args.dataset_root, args.folds)
    if not seqs:
        raise FileNotFoundError(f"No sequences found under {args.dataset_root}")
    rows = compute_rows(seqs, FEATURES[args.feature])
    selected: list[dict[str, Any]] = []
    used: set[tuple[str, int, int]] = set()

    for criterion in args.criteria:
        if criterion == "high_stability":
            candidates = sorted(rows, key=lambda r: float(r["same_repeat_similarity"]), reverse=True)
            note = "selected by high within-subject repeat similarity"
        elif criterion == "strong_delta":
            candidates = sorted(rows, key=lambda r: float(r["delta31_magnitude"]), reverse=True)
            note = "selected by large repeat3-repeat1 ROI response change"
        elif criterion == "model_success":
            candidates = sorted(
                rows,
                key=lambda r: (
                    -np.inf if np.isnan(float(r["cross_subject_similarity"])) else float(r["cross_subject_similarity"]),
                    float(r["same_repeat_similarity"]),
                ),
                reverse=True,
            )
            note = "model-rank file unavailable; using high cross-subject ROI similarity as success proxy"
        elif criterion == "model_failure":
            candidates = sorted(
                rows,
                key=lambda r: (
                    np.inf if np.isnan(float(r["cross_subject_similarity"])) else float(r["cross_subject_similarity"]),
                    -float(r["same_repeat_similarity"]),
                ),
            )
            note = "model-rank file unavailable; using low cross-subject ROI similarity as failure proxy"
        else:
            continue
        selected.extend(take_unique(candidates, args.n_per_criterion, used, criterion, note))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "criterion",
        "fold",
        "split",
        "subject",
        "nsdId",
        "repeat_pair",
        "same_repeat_similarity",
        "sim12",
        "sim13",
        "sim23",
        "delta31_magnitude",
        "cross_subject_similarity",
        "model_rank",
        "model_score",
        "notes",
    ]
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    print({"out_csv": str(args.out_csv), "n_selected": len(selected)})


if __name__ == "__main__":
    main()
