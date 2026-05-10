#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot gated ReGraph ROI importance figures.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--gate-summary",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/interpretability/gates/roi_gate_summary.csv"),
    )
    parser.add_argument(
        "--gate-values",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/interpretability/gates/roi_gate_values.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("preproc_v0/repetition_familiarity/results/final_figures"))
    return parser.parse_args()


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out = root / args.output_dir
    summary = pd.read_csv(root / args.gate_summary)
    values = pd.read_csv(root / args.gate_values)

    top = summary.sort_values("gate_mean", ascending=False).head(20).sort_values("gate_mean")
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.barh(top["roi_id"].astype(str), top["gate_mean"], xerr=top["gate_std_across_checkpoints"].fillna(0), color="#355070")
    ax.set_xlabel("Mean gate value")
    ax.set_ylabel("HCP-MMP ROI id")
    ax.set_title("Top gated ReGraph ROIs")
    save(fig, out / "gate_top_rois_barplot")

    pivot = values.pivot_table(index="roi_id", columns=["fold", "seed"], values="gate_mean")
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    im = ax.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title("Gate values across fold/seed checkpoints")
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("ROI id")
    fig.colorbar(im, ax=ax, label="gate")
    save(fig, out / "gate_stability_heatmap")
    print({"output_dir": str(out)})


if __name__ == "__main__":
    main()
