#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


MODEL_LABELS = {
    "roi_mlp_clip": "ROI-MLP+CLIP",
    "roi_transformer_noadj_gated_flat_clip": "No-adj gated ROI Transformer+CLIP",
    "bnt_token_flat_gated_flat_clip": "Gated ReGraph/BNT+CLIP",
}

METRICS = ["AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "brain_R@5", "brain_MRR"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize single-reference session-matched cross-subject controls.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def find_metrics(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_root.glob("*/*/fold_*/seed_*/metrics.json")):
        with path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        model_key = path.parts[-5]
        lambda_key = path.parts[-4]
        fold = path.parts[-3]
        seed = path.parts[-2].replace("seed_", "")
        row = {
            "model_key": model_key,
            "model": MODEL_LABELS.get(model_key, model_key),
            "lambda": lambda_key,
            "fold": fold,
            "seed": int(seed),
            "metrics_path": str(path),
        }
        for metric in METRICS:
            row[metric] = float(metrics.get(metric, np.nan))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> str:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "nan"
    if arr.size == 1:
        return f"{arr.mean():.4f}"
    return f"{arr.mean():.4f} +/- {arr.std(ddof=1):.4f}"


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in rows}):
        subset = [row for row in rows if row["model"] == model]
        summary = {"model": model, "n": len(subset)}
        for metric in METRICS:
            summary[metric] = mean_std([float(row[metric]) for row in subset])
        out.append(summary)
    return out


def write_markdown(path: Path, summary_rows: list[dict[str, Any]], all_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Single-reference session-matched cross-subject control",
        "",
        "This control replaces the training-subject average reference graph with a single reference subject trial. Positive and negative examples share the same anchor trial, same reference subject, same repeat index, and exactly matched reference session.",
        "",
        "## Summary",
        "",
    ]
    columns = ["model", "n", *METRICS]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in summary_rows:
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    lines.extend(["", "## Per-fold rows", ""])
    fold_columns = ["model", "fold", "seed", *METRICS]
    lines.append("| " + " | ".join(fold_columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(fold_columns)) + " |")
    for row in all_rows:
        values = []
        for column in fold_columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = find_metrics(args.results_root)
    if not rows:
        raise SystemExit(f"No metrics.json files found under {args.results_root}")
    summary_rows = summarize(rows)
    write_csv(args.output_dir / "single_ref_matched_all_runs.csv", rows)
    write_csv(args.output_dir / "single_ref_matched_summary.csv", summary_rows)
    write_markdown(args.output_dir / "single_ref_matched_summary.md", summary_rows, rows)
    print((args.output_dir / "single_ref_matched_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
