#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ReGraph-VLM v0 runs.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("preproc_v0/repetition_familiarity/results/regraph_vlm"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = (args.root / args.results_dir).resolve()
    rows = []
    for path in sorted(results_dir.glob("**/metrics.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["metrics_path"] = str(path)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No metrics.json files found under {results_dir}")
    df = pd.DataFrame(rows)
    wanted = [
        "model",
        "graph_encoder",
        "fold",
        "seed",
        "lambda_clip",
        "AUROC",
        "AUPRC",
        "balanced_accuracy",
        "R@1",
        "R@5",
        "R@10",
        "MRR",
        "image_R@1",
        "image_R@5",
        "image_R@10",
        "image_MRR",
        "image_median_rank",
        "brain_R@1",
        "brain_R@5",
        "brain_R@10",
        "brain_MRR",
        "brain_median_rank",
        "best_val_metric",
        "best_epoch",
        "status",
        "metrics_path",
    ]
    cols = [col for col in wanted if col in df.columns] + [col for col in df.columns if col not in wanted]
    df = df[cols].sort_values([c for c in ["lambda_clip", "fold", "seed"] if c in df.columns])
    out = results_dir / "regraph_vlm_summary.csv"
    df.to_csv(out, index=False)
    print({"summary_csv": str(out), "n_rows": int(len(df))})


if __name__ == "__main__":
    main()
