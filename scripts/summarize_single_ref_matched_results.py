#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy may be unavailable on minimal installs.
    stats = None


MODEL_LABELS = {
    "roi_mlp_clip": "ROI-MLP+CLIP",
    "roi_transformer_noadj_gated_flat_clip": "No-adj gated ROI Transformer+CLIP",
    "bnt_token_flat_gated_flat_clip": "Gated ReGraph/BNT+CLIP",
}

METRICS = ["AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "brain_R@5", "brain_MRR"]
PAIRWISE_COMPARISONS = [
    ("Gated ReGraph/BNT+CLIP", "ROI-MLP+CLIP"),
    ("No-adj gated ROI Transformer+CLIP", "ROI-MLP+CLIP"),
    ("Gated ReGraph/BNT+CLIP", "No-adj gated ROI Transformer+CLIP"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize single-reference session-matched cross-subject controls.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="single_ref_matched")
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


def paired_tests(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model = {
        model: {(row["fold"], row["seed"]): row for row in rows if row["model"] == model}
        for model in {row["model"] for row in rows}
    }
    out: list[dict[str, Any]] = []
    for model_a, model_b in PAIRWISE_COMPARISONS:
        common_keys = sorted(set(by_model.get(model_a, {})) & set(by_model.get(model_b, {})))
        for metric in METRICS:
            diffs = np.asarray(
                [float(by_model[model_a][key][metric]) - float(by_model[model_b][key][metric]) for key in common_keys],
                dtype=float,
            )
            diffs = diffs[np.isfinite(diffs)]
            n = int(diffs.size)
            mean_diff = float(diffs.mean()) if n else math.nan
            std_diff = float(diffs.std(ddof=1)) if n > 1 else math.nan
            sem = float(std_diff / math.sqrt(n)) if n > 1 else math.nan
            ci95 = float(1.96 * sem) if n > 1 else math.nan
            if stats is not None and n > 1 and std_diff > 0:
                p_value = float(stats.ttest_1samp(diffs, popmean=0.0).pvalue)
            else:
                p_value = math.nan
            out.append(
                {
                    "comparison": f"{model_a} - {model_b}",
                    "metric": metric,
                    "n": n,
                    "mean_diff": mean_diff,
                    "std_diff": std_diff,
                    "ci95_half_width": ci95,
                    "paired_t_p": p_value,
                }
            )
    return out


def format_float(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.4g}" if abs(value) < 0.001 and value != 0 else f"{value:.4f}"
    return str(value)


def write_markdown(
    path: Path,
    summary_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
) -> None:
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
    lines.extend(["", "## Paired fold-by-seed differences", ""])
    test_columns = ["comparison", "metric", "n", "mean_diff", "std_diff", "ci95_half_width", "paired_t_p"]
    lines.append("| " + " | ".join(test_columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(test_columns)) + " |")
    for row in test_rows:
        lines.append("| " + " | ".join(format_float(row[column]) for column in test_columns) + " |")
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
    test_rows = paired_tests(rows)
    prefix = args.output_prefix
    write_csv(args.output_dir / f"{prefix}_all_runs.csv", rows)
    write_csv(args.output_dir / f"{prefix}_summary.csv", summary_rows)
    write_csv(args.output_dir / f"{prefix}_pairwise_tests.csv", test_rows)
    write_markdown(args.output_dir / f"{prefix}_summary.md", summary_rows, test_rows, rows)
    print((args.output_dir / f"{prefix}_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
