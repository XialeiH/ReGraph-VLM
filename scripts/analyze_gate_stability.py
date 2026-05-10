#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze stability of gated ReGraph ROI gates.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--gate-csv",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/interpretability/gates/roi_gate_values.csv"),
    )
    return parser.parse_args()


def top_overlap(a: set[int], b: set[int]) -> float:
    return len(a & b) / max(1, min(len(a), len(b)))


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    gate_csv = root / args.gate_csv
    df = pd.read_csv(gate_csv)
    pivot = df.pivot_table(index="roi_id", columns=["fold", "seed"], values="gate_mean")
    cols = list(pivot.columns)
    corr_rows = []
    for a, b in combinations(cols, 2):
        corr_rows.append(
            {
                "checkpoint_a": f"{a[0]}_seed{a[1]}",
                "checkpoint_b": f"{b[0]}_seed{b[1]}",
                "pearson_corr": float(pivot[a].corr(pivot[b], method="pearson")),
                "spearman_corr": float(pivot[a].corr(pivot[b], method="spearman")),
            }
        )
    corr = pd.DataFrame(corr_rows)
    top_rows = []
    for k in [10, 20, 40]:
        top_sets = {c: set(pivot[c].sort_values(ascending=False).head(k).index.astype(int)) for c in cols}
        vals = [top_overlap(top_sets[a], top_sets[b]) for a, b in combinations(cols, 2)]
        top_rows.append({"top_k": k, "mean_pairwise_overlap": sum(vals) / max(1, len(vals)), "n_pairs": len(vals)})
    out_dir = gate_csv.parent
    corr.to_csv(out_dir / "gate_stability_pairwise.csv", index=False)
    pd.DataFrame(top_rows).to_csv(out_dir / "gate_stability_summary.csv", index=False)
    md = [
        "# Gate Stability",
        "",
        "## Pairwise Correlation Summary",
        corr[["pearson_corr", "spearman_corr"]].describe().to_markdown(floatfmt=".4f"),
        "",
        "## Top-k Overlap",
        pd.DataFrame(top_rows).to_markdown(index=False, floatfmt=".4f"),
    ]
    (out_dir / "gate_stability_summary.md").write_text("\n".join(md), encoding="utf-8")
    print({"out_dir": str(out_dir), "n_pairs": len(corr)})


if __name__ == "__main__":
    main()
