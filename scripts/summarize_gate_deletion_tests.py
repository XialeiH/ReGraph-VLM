#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize gate deletion test results.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/interpretability/gates/deletion_tests"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    in_dir = root / args.input_dir
    rows = []
    for path in sorted(in_dir.glob("fold_*/seed_*/gate_deletion_metrics.csv")):
        rows.append(pd.read_csv(path))
    if not rows:
        raise FileNotFoundError(f"No deletion metrics found under {in_dir}")
    df = pd.concat(rows, ignore_index=True)
    base_cols = ["fold", "seed"]
    metrics = [m for m in ["AUROC", "AUPRC", "R@5", "MRR", "brain_R@5", "brain_MRR"] if m in df.columns]
    base = df[df["mode"] == "baseline"][base_cols + metrics].rename(columns={m: f"baseline_{m}" for m in metrics})
    merged = df.merge(base, on=base_cols, how="left")
    for m in metrics:
        merged[f"{m}_drop"] = merged[f"baseline_{m}"] - merged[m]
    out_dir = in_dir.parent
    merged.to_csv(out_dir / "gate_deletion_curves.csv", index=False)
    summary = (
        merged[merged["mode"] != "baseline"]
        .groupby(["mode", "k"])
        .agg(**{f"{m}_drop_mean": (f"{m}_drop", "mean") for m in metrics}, n=("AUROC", "count"))
        .reset_index()
    )
    summary.to_csv(out_dir / "gate_deletion_test_summary.csv", index=False)
    (out_dir / "gate_deletion_test_summary.md").write_text(summary.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")
    print({"out": str(out_dir / "gate_deletion_test_summary.csv"), "n_rows": len(summary)})


if __name__ == "__main__":
    main()
