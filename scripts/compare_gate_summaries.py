#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ROI gate summaries from two gated ROI-token models.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--a-summary", type=Path, required=True)
    parser.add_argument("--b-summary", type=Path, required=True)
    parser.add_argument("--a-name", default="model_a")
    parser.add_argument("--b-name", default="model_b")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def top_overlap(a: pd.DataFrame, b: pd.DataFrame, k: int) -> float:
    aa = set(a.sort_values("gate_mean", ascending=False).head(k)["roi_id"].astype(int))
    bb = set(b.sort_values("gate_mean", ascending=False).head(k)["roi_id"].astype(int))
    return len(aa & bb) / max(1, k)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    a = pd.read_csv(root / args.a_summary)
    b = pd.read_csv(root / args.b_summary)
    merged = a[["roi_id", "gate_mean"]].rename(columns={"gate_mean": f"{args.a_name}_gate_mean"}).merge(
        b[["roi_id", "gate_mean"]].rename(columns={"gate_mean": f"{args.b_name}_gate_mean"}),
        on="roi_id",
        how="inner",
    )
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "gate_summary_roiwise_comparison.csv", index=False)
    x = merged[f"{args.a_name}_gate_mean"]
    y = merged[f"{args.b_name}_gate_mean"]
    rows = [
        {
            "comparison": f"{args.a_name}_vs_{args.b_name}",
            "n_rois": int(len(merged)),
            "pearson_corr": float(x.corr(y, method="pearson")),
            "spearman_corr": float(x.corr(y, method="spearman")),
            "top10_overlap": top_overlap(a, b, 10),
            "top20_overlap": top_overlap(a, b, 20),
            "top40_overlap": top_overlap(a, b, 40),
        }
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "gate_summary_comparison.csv", index=False)
    (out_dir / "gate_summary_comparison.md").write_text(summary.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")
    print({"out": str(out_dir / "gate_summary_comparison.csv"), **rows[0]})


if __name__ == "__main__":
    main()
