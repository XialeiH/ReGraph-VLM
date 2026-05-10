#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze per-held-out-subject performance from final summaries.")
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    results = root / "preproc_v0/repetition_familiarity/results"
    out_dir = results / "error_analysis"
    fig_dir = results / "final_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results / "cross_subject_allfold_final/regraph_vlm_summary.csv")
    labels = {
        ("roi_mlp", "flat", 2.0): "ROI-MLP+CLIP",
        ("bnt_token_flat", "flat", 2.0): "Flat ReGraph+CLIP",
        ("bnt_token_flat", "gated_flat", 2.0): "Gated ReGraph+CLIP",
    }
    rows = []
    for (enc, readout, lam), label in labels.items():
        x = df[(df["graph_encoder"] == enc) & (df["readout"] == readout) & (df["lambda_clip"] == lam)]
        for fold, g in x.groupby("fold"):
            row = {"model_label": label, "fold": fold, "n": len(g)}
            for metric in ["AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "brain_R@5", "brain_MRR"]:
                if metric in g:
                    row[metric] = float(g[metric].mean())
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "per_subject_performance.csv", index=False)
    metrics = ["AUROC", "R@5", "brain_R@5"]
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        for label, g in out.groupby("model_label"):
            g = g.sort_values("fold")
            ax.plot(g["fold"], g[metric], marker="o", label=label)
        ax.set_title(f"Per-fold performance: {metric}")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8)
        save(fig, fig_dir / f"per_subject_performance_{metric.replace('@','at')}")
    print({"out": str(out_dir / "per_subject_performance.csv"), "n_rows": len(out)})


if __name__ == "__main__":
    main()
