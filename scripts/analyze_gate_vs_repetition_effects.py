#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare gate importance with repetition-effect ROI statistics.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--gate-summary",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/interpretability/gates/roi_gate_summary.csv"),
    )
    parser.add_argument(
        "--repetition-summary",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/analysis/scalar4_T3/roi_repetition_suppression_group.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    gates = pd.read_csv(root / args.gate_summary)
    rep_path = root / args.repetition_summary
    out_dir = (root / args.gate_summary).parent
    if not rep_path.exists():
        (out_dir / "gate_repetition_overlap.md").write_text(
            f"Repetition summary not found: {rep_path}\n", encoding="utf-8"
        )
        print({"status": "missing_repetition_summary", "path": str(rep_path)})
        return
    rep = pd.read_csv(rep_path)
    roi_col = "roi_id" if "roi_id" in rep.columns else "roi"
    rep = rep.rename(columns={roi_col: "roi_id"})
    merged = gates.merge(rep, on="roi_id", how="inner")
    rows = []
    for col in ["mean_delta_21", "mean_delta_31", "group_mean_delta_21", "group_mean_delta_31"]:
        if col in merged.columns:
            rows.append(
                {
                    "effect_column": col,
                    "spearman_gate_vs_abs_effect": float(merged["gate_mean"].corr(merged[col].abs(), method="spearman")),
                    "pearson_gate_vs_abs_effect": float(merged["gate_mean"].corr(merged[col].abs(), method="pearson")),
                }
            )
    for k in [10, 20, 40]:
        top_gate = set(merged.sort_values("gate_mean", ascending=False).head(k)["roi_id"].astype(int))
        for col in [c for c in ["fdr_q_value", "q_value", "q"] if c in merged.columns]:
            sig = set(merged[merged[col] < 0.05]["roi_id"].astype(int))
            rows.append({"effect_column": f"top{k}_overlap_{col}<0.05", "spearman_gate_vs_abs_effect": len(top_gate & sig), "pearson_gate_vs_abs_effect": len(sig)})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "gate_repetition_overlap.csv", index=False)
    (out_dir / "gate_repetition_overlap.md").write_text(out.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")
    print({"out": str(out_dir / "gate_repetition_overlap.csv"), "n_rows": len(out)})


if __name__ == "__main__":
    main()
