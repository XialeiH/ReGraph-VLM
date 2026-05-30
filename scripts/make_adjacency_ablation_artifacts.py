#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["AUROC", "AUPRC", "R@5", "MRR", "brain_R@5"]
MODEL_LABELS = {
    "main_roi_mlp": "ROI-MLP+CLIP",
    "main_noadj_gated": "No-adj gated ROI Transformer+CLIP",
    "main_gated_regraph": "Gated ReGraph/BNT+CLIP",
}
MODEL_ORDER = ["main_roi_mlp", "main_noadj_gated", "main_gated_regraph"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create final adjacency-ablation tables and figure.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_model_comparison"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def latex_table(table: pd.DataFrame) -> str:
    rows = [
        "\\begin{table}[!htbp]",
        "  \\centering",
        "  \\caption{Adjacency ablation for the main all-fold cross-subject setting. The no-adjacency gated ROI Transformer is statistically indistinguishable from the final BNT/ReGraph ROI-token variant, while both outperform ROI-MLP+CLIP.}",
        "  \\label{tab:adjacency_ablation}",
        "  \\resizebox{\\linewidth}{!}{%",
        "  \\begin{tabular}{lccccc}",
        "    \\toprule",
        "    Model & AUROC & AUPRC & R@5 & MRR & Brain R@5 \\\\",
        "    \\midrule",
    ]
    for _, row in table.iterrows():
        values = " & ".join(row[metric] for metric in METRICS)
        rows.append(f"    {row['Model']} & {values} \\\\")
    rows.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}%",
            "  }",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(rows)


def plot_figure(summary: pd.DataFrame, out_dir: Path) -> None:
    colors = {
        "ROI-MLP+CLIP": "#b9b9b9",
        "No-adj gated ROI Transformer+CLIP": "#7fc7c9",
        "Gated ReGraph/BNT+CLIP": "#2f7fbd",
    }
    panels = [
        ("AUROC", (0.79, 0.84)),
        ("R@5", (0.065, 0.105)),
        ("brain_R@5", (0.075, 0.115)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    for ax, (metric, ylim) in zip(axes, panels):
        means = [summary.loc[m, f"{metric}_mean"] for m in MODEL_ORDER]
        sems = [summary.loc[m, f"{metric}_std"] / np.sqrt(summary.loc[m, "n"]) for m in MODEL_ORDER]
        labels = [MODEL_LABELS[m] for m in MODEL_ORDER]
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=sems, capsize=3, color=[colors[label] for label in labels], edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["ROI-MLP", "No-adj\nROI-T", "Gated\nReGraph"], fontsize=8)
        ax.set_ylim(*ylim)
        ax.set_ylabel(metric.replace("brain_R@5", "Brain R@5"))
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        y = ylim[1] - (ylim[1] - ylim[0]) * 0.08
        ax.plot([0, 1], [y, y], color="black", linewidth=0.8)
        ax.text(0.5, y + (ylim[1] - ylim[0]) * 0.015, "***", ha="center", va="bottom", fontsize=10)
        ax.plot([1, 2], [y - (ylim[1] - ylim[0]) * 0.08, y - (ylim[1] - ylim[0]) * 0.08], color="black", linewidth=0.8)
        ax.text(1.5, y - (ylim[1] - ylim[0]) * 0.065, "n.s.", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(w_pad=1.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "png", "svg"]:
        fig.savefig(out_dir / f"adjacency_ablation.{suffix}", dpi=300, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    comparison_dir = root / args.comparison_dir
    out_dir = root / args.out_dir
    fig_dir = root / args.figure_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(comparison_dir / "model_candidate_metric_summary.csv").set_index("model_setting")

    table_rows = []
    for key in MODEL_ORDER:
        row = {"Model": MODEL_LABELS[key], "n": int(summary.loc[key, "n"])}
        for metric in METRICS:
            row[metric] = fmt_mean_std(float(summary.loc[key, f"{metric}_mean"]), float(summary.loc[key, f"{metric}_std"]))
        table_rows.append(row)
    table = pd.DataFrame(table_rows)
    table.to_csv(out_dir / "table_adjacency_ablation.csv", index=False)
    (out_dir / "table_adjacency_ablation_latex.txt").write_text(latex_table(table), encoding="utf-8")
    plot_figure(summary, fig_dir)
    print(
        {
            "csv": str(out_dir / "table_adjacency_ablation.csv"),
            "latex": str(out_dir / "table_adjacency_ablation_latex.txt"),
            "figure": str(fig_dir / "adjacency_ablation.pdf"),
        }
    )


if __name__ == "__main__":
    main()
