#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit held-out fold difficulty from serialized cross-subject pairs.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--performance-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-label", default="Gated ReGraph+CLIP")
    return parser.parse_args()


def flat_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(float)
    b = b.reshape(-1).astype(float)
    if np.std(a) == 0 or np.std(b) == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def session_entropy(sessions: list[int]) -> float:
    if not sessions:
        return math.nan
    _, counts = np.unique(np.asarray(sessions), return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def summarize_fold(fold_dir: Path) -> dict[str, Any]:
    pairs = torch.load(fold_dir / "test_pairs.pt", map_location="cpu")
    fold = fold_dir.name
    labels: list[int] = []
    pair_corrs: list[float] = []
    positive_anchor_by_image: dict[int, dict[int, np.ndarray]] = {}
    positive_sessions: list[int] = []
    ref_counts: list[int] = []

    for row in pairs:
        x1 = row["x1"].numpy()
        x2 = row["x2"].numpy()
        label = int(row["same_image"])
        labels.append(label)
        pair_corrs.append(flat_corr(x1, x2))
        if "n_ref_subjects" in row:
            ref_counts.append(int(row["n_ref_subjects"]))
        if label == 1:
            nsdid = int(row["nsdId_1"])
            repeat = int(row["repeat_1"])
            positive_anchor_by_image.setdefault(nsdid, {})[repeat] = x1
            positive_sessions.append(int(row.get("session_1", -1)))

    repeat_corrs: list[float] = []
    complete_repeat_images = 0
    for repeat_map in positive_anchor_by_image.values():
        repeats = sorted(repeat_map)
        if len(repeats) == 3:
            complete_repeat_images += 1
        for i, r1 in enumerate(repeats):
            for r2 in repeats[i + 1 :]:
                repeat_corrs.append(flat_corr(repeat_map[r1], repeat_map[r2]))

    labels_arr = np.asarray(labels)
    corr_arr = np.asarray(pair_corrs, dtype=float)
    valid = np.isfinite(corr_arr)
    raw_auc = float(roc_auc_score(labels_arr[valid], corr_arr[valid])) if len(np.unique(labels_arr[valid])) == 2 else math.nan
    same = corr_arr[(labels_arr == 1) & valid]
    diff = corr_arr[(labels_arr == 0) & valid]

    qc_path = fold_dir / "dataset_qc.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8")) if qc_path.exists() else {}
    test_subject = qc.get("test_subject", "")
    if not test_subject and pairs:
        test_subject = f"subj{int(pairs[0]['subject']):02d}"

    return {
        "fold": fold,
        "test_subject": test_subject,
        "n_test_pairs": len(pairs),
        "n_positive_pairs": int((labels_arr == 1).sum()),
        "n_test_images": len(positive_anchor_by_image),
        "n_complete_repeat_images": complete_repeat_images,
        "anchor_repeat_corr_mean": float(np.nanmean(repeat_corrs)),
        "anchor_repeat_corr_std": float(np.nanstd(repeat_corrs, ddof=1)),
        "raw_pair_corr_auc": raw_auc,
        "raw_pair_corr_same_mean": float(np.nanmean(same)),
        "raw_pair_corr_diff_mean": float(np.nanmean(diff)),
        "raw_pair_corr_gap": float(np.nanmean(same) - np.nanmean(diff)),
        "n_anchor_sessions": int(len(set(positive_sessions))),
        "anchor_session_entropy": session_entropy(positive_sessions),
        "mean_ref_subjects": float(np.mean(ref_counts)) if ref_counts else math.nan,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, fold_rows: pd.DataFrame, corr_rows: pd.DataFrame, model_label: str) -> None:
    lines = [
        "# Fold difficulty QC",
        "",
        "This audit summarizes held-out-subject difficulty using only serialized test pairs and final per-fold model scores.",
        "",
        "## Fold summary",
        "",
    ]
    display_cols = [
        "fold",
        "test_subject",
        "n_test_images",
        "anchor_repeat_corr_mean",
        "raw_pair_corr_auc",
        "raw_pair_corr_gap",
        "anchor_session_entropy",
        "AUROC",
        "R@5",
        "brain_R@5",
    ]
    lines.append(fold_rows[display_cols].to_markdown(index=False, floatfmt=".4f"))
    lines.extend(
        [
            "",
            "## Spearman correlations with final model performance",
            "",
            f"Model: {model_label}. Positive correlations mean the QC metric increases with fold-level performance.",
            "",
            corr_rows.to_markdown(index=False, floatfmt=".4f"),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_rows = [summarize_fold(path) for path in sorted(args.dataset_root.glob("fold_*")) if path.is_dir()]
    fold_df = pd.DataFrame(fold_rows)

    perf = pd.read_csv(args.performance_csv)
    perf = perf[perf["model_label"] == args.model_label].copy()
    merged = fold_df.merge(perf, on="fold", how="left")

    qc_metrics = [
        "anchor_repeat_corr_mean",
        "raw_pair_corr_auc",
        "raw_pair_corr_gap",
        "anchor_session_entropy",
        "n_test_images",
        "mean_ref_subjects",
    ]
    perf_metrics = ["AUROC", "R@5", "brain_R@5"]
    corr_rows = []
    for qc_metric in qc_metrics:
        for perf_metric in perf_metrics:
            corr = merged[[qc_metric, perf_metric]].corr(method="spearman").iloc[0, 1]
            corr_rows.append({"qc_metric": qc_metric, "performance_metric": perf_metric, "spearman_r": corr})
    corr_df = pd.DataFrame(corr_rows)

    write_csv(args.output_dir / "fold_difficulty_qc.csv", merged.to_dict("records"))
    write_csv(args.output_dir / "fold_difficulty_qc_correlations.csv", corr_df.to_dict("records"))
    write_markdown(args.output_dir / "fold_difficulty_qc.md", merged, corr_df, args.model_label)
    print((args.output_dir / "fold_difficulty_qc.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
