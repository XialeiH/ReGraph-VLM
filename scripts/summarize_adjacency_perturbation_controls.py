#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize adjacency perturbation control evaluations.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/phase4_adjacency_perturbation"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = args.root.resolve() / args.input_dir
    frames = [pd.read_csv(path) for path in sorted(in_dir.glob("fold_*/seed_*/adjacency_perturbation_metrics.csv"))]
    if not frames:
        raise FileNotFoundError(f"No adjacency perturbation metrics found under {in_dir}")
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(in_dir / "adjacency_perturbation_all_runs.csv", index=False)

    metrics = [m for m in ["AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "brain_R@5", "brain_MRR"] if m in df.columns]
    summary = df.groupby("control_mode").agg(
        n=("AUROC", "count"),
        **{f"{metric}_mean": (metric, "mean") for metric in metrics},
        **{f"{metric}_std": (metric, "std") for metric in metrics},
    )
    summary = summary.reset_index()
    baseline = summary[summary["control_mode"].eq("default")]
    if len(baseline) == 1:
        base = baseline.iloc[0]
        for metric in metrics:
            summary[f"{metric}_drop_vs_default"] = float(base[f"{metric}_mean"]) - summary[f"{metric}_mean"]
    summary.to_csv(in_dir / "adjacency_perturbation_summary.csv", index=False)
    (in_dir / "adjacency_perturbation_summary.md").write_text(
        summary.to_markdown(index=False, floatfmt=".4f"),
        encoding="utf-8",
    )
    print({"summary": str(in_dir / "adjacency_perturbation_summary.csv"), "n_rows": len(df)})


if __name__ == "__main__":
    main()
