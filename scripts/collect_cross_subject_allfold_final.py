#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect final all-fold cross-subject ReGraph-VLM metrics.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/cross_subject_allfold_final"),
    )
    return parser.parse_args()


def read_metrics(base: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(base.glob("**/metrics.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["metrics_path"] = str(path)
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    results = root / "preproc_v0/repetition_familiarity/results"
    sources = [
        results / "cross_subject_allfold_seed11",
        results / "cross_subject_gated_allfold_seed11",
        root / args.output_dir,
    ]
    rows: list[dict[str, object]] = []
    for source in sources:
        if source.exists():
            rows.extend(read_metrics(source))
    if not rows:
        raise FileNotFoundError(f"No metrics found in sources: {[str(s) for s in sources]}")

    df = pd.DataFrame(rows)
    if "readout" not in df.columns:
        df["readout"] = pd.NA
    if "lambda_cross" not in df.columns:
        df["lambda_cross"] = 0.0
    if "metrics_path" in df.columns:
        path_text = df["metrics_path"].astype(str)
        missing_readout = df["readout"].isna()
        df.loc[missing_readout & path_text.str.contains("gated_flat", regex=False), "readout"] = "gated_flat"
        df.loc[missing_readout & ~path_text.str.contains("gated_flat", regex=False), "readout"] = "flat"
    df["lambda_cross"] = df["lambda_cross"].fillna(0.0)
    keep = [
        "model",
        "graph_encoder",
        "readout",
        "fold",
        "seed",
        "lambda_clip",
        "lambda_cross",
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
    cols = [col for col in keep if col in df.columns] + [col for col in df.columns if col not in keep]
    df = df[cols]
    key_cols = [c for c in ["graph_encoder", "readout", "lambda_clip", "lambda_cross", "fold", "seed"] if c in df.columns]
    if key_cols:
        df = df.sort_values("metrics_path").drop_duplicates(key_cols, keep="last")
    sort_cols = [c for c in ["graph_encoder", "readout", "lambda_clip", "fold", "seed"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "regraph_vlm_summary.csv"
    df.to_csv(out_csv, index=False)
    print({"out_csv": str(out_csv), "n_rows": int(len(df))})


if __name__ == "__main__":
    main()
