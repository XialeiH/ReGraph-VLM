#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage 2 analysis figures PDF from analysis CSV outputs.")
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--prototype-root", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png-dir", type=Path, default=None)
    return parser.parse_args()


def save_page(fig: plt.Figure, pdf: PdfPages, png_dir: Path | None, page_idx: int) -> None:
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    if png_dir is not None:
        png_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_dir / f"analysis_stage_figures_page_{page_idx:02d}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_sample_entropies(prototype_root: Path) -> np.ndarray:
    values = []
    for fold_dir in sorted(path for path in prototype_root.iterdir() if path.is_dir() and path.name.startswith("fold_")):
        assignments = np.load(fold_dir / "prototype_assignments.npy").astype(np.float32)
        entropy = -(assignments * np.log(np.clip(assignments, 1e-12, None))).sum(axis=1)
        values.append(entropy)
    return np.concatenate(values, axis=0)


def page_usage(analysis_dir: Path, prototype_root: Path, pdf: PdfPages, png_dir: Path | None, page_idx: int) -> None:
    usage = pd.read_csv(analysis_dir / "unit_usage_summary.csv").sort_values("usage_fraction", ascending=False)
    entropies = load_sample_entropies(prototype_root)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    axes[0, 0].bar(np.arange(len(usage)), usage["usage_fraction"].to_numpy(), color="#28536b")
    axes[0, 0].set_title("Per-Unit Top-1 Usage Fraction")
    axes[0, 0].set_xlabel("Unit rank")
    axes[0, 0].set_ylabel("Usage fraction")

    axes[0, 1].hist(usage["mean_activation"].to_numpy(), bins=20, color="#c2948a", edgecolor="black")
    axes[0, 1].set_title("Mean Activation Across Units")
    axes[0, 1].set_xlabel("Mean activation")
    axes[0, 1].set_ylabel("Count")

    axes[1, 0].hist(entropies, bins=25, color="#8fb996", edgecolor="black")
    axes[1, 0].set_title("Per-Sample Assignment Entropy")
    axes[1, 0].set_xlabel("Entropy")
    axes[1, 0].set_ylabel("Count")

    top_rows = usage.head(8)[["unit_id", "usage_fraction", "top1_count", "top5_count", "dead_or_not"]].copy()
    top_rows["usage_fraction"] = top_rows["usage_fraction"].map(lambda x: f"{x:.4f}")
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Usage Summary Snapshot")
    table = axes[1, 1].table(
        cellText=top_rows.values,
        colLabels=top_rows.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)

    save_page(fig, pdf, png_dir, page_idx)


def page_consistency(analysis_dir: Path, pdf: PdfPages, png_dir: Path | None, page_idx: int) -> None:
    cons = pd.read_csv(analysis_dir / "cross_subject_unit_consistency.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    for ax, metric in zip(axes[0], ["cosine", "correlation"]):
        subset = cons[cons["metric"] == metric]
        proto = subset[subset["representation_type"] == "prototype"]["gap"].to_numpy()
        b4 = subset[subset["representation_type"] == "b4_hidden"]["gap"].to_numpy()
        ax.boxplot([proto, b4], labels=["prototype", "b4_hidden"])
        ax.set_title(f"Same-vs-Diff Gap ({metric})")
        ax.set_ylabel("Gap")
        ax.axhline(0.0, color="black", linewidth=1, linestyle="--")

    for ax, metric in zip(axes[1], ["cosine", "correlation"]):
        subset = cons[cons["metric"] == metric].copy()
        pivot = subset.pivot(index="held_out_subject", columns="representation_type", values="gap").reset_index()
        x = np.arange(len(pivot))
        width = 0.35
        ax.bar(x - width / 2, pivot["prototype"], width=width, label="prototype", color="#28536b")
        ax.bar(x + width / 2, pivot["b4_hidden"], width=width, label="b4_hidden", color="#c2948a")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot["held_out_subject"], rotation=30)
        ax.set_title(f"Per-Fold Gap by Held-Out Subject ({metric})")
        ax.set_ylabel("Gap")
        ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
        ax.legend(frameon=False)

    save_page(fig, pdf, png_dir, page_idx)


def page_selectivity(analysis_dir: Path, pdf: PdfPages, png_dir: Path | None, page_idx: int) -> None:
    selectivity = pd.read_csv(analysis_dir / "unit_selectivity_summary.csv")
    examples = pd.read_csv(analysis_dir / "unit_top_images_examples.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    axes[0, 0].hist(selectivity["cross_subject_recurrence_score"], bins=20, color="#8fb996", edgecolor="black")
    axes[0, 0].set_title("Cross-Subject Recurrence Score")
    axes[0, 0].set_xlabel("Recurrence score")
    axes[0, 0].set_ylabel("Count")

    axes[0, 1].hist(selectivity["subject_coverage_top_images"], bins=np.arange(0.5, 8.6, 1), color="#c2948a", edgecolor="black")
    axes[0, 1].set_title("Subject Coverage in Top-20 Images")
    axes[0, 1].set_xlabel("Number of subjects")
    axes[0, 1].set_ylabel("Count")

    top_units = selectivity.sort_values("cross_subject_recurrence_score", ascending=False).head(6)
    axes[1, 0].bar(top_units["unit_id"].astype(str), top_units["cross_subject_recurrence_score"], color="#28536b")
    axes[1, 0].set_title("Most Recurrent Units")
    axes[1, 0].set_xlabel("Unit ID")
    axes[1, 0].set_ylabel("Recurrence score")

    table_rows = []
    for unit_id in top_units["unit_id"].tolist()[:4]:
        subset = examples[examples["unit_id"] == unit_id].head(5)
        table_rows.append(
            [
                int(unit_id),
                ",".join(subset["nsdId"].astype(str).tolist()),
                ",".join(subset["subject"].astype(str).tolist()),
            ]
        )
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Representative Top-Image Indices")
    table = axes[1, 1].table(
        cellText=table_rows,
        colLabels=["unit_id", "top nsdId", "subjects"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)

    save_page(fig, pdf, png_dir, page_idx)


def main() -> None:
    args = parse_args()
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    png_dir = args.output_png_dir
    with PdfPages(args.output_pdf) as pdf:
        page_usage(args.analysis_dir, args.prototype_root, pdf, png_dir, 1)
        page_consistency(args.analysis_dir, pdf, png_dir, 2)
        page_selectivity(args.analysis_dir, pdf, png_dir, 3)


if __name__ == "__main__":
    main()
