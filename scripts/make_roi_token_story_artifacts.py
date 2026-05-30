#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONTROL_LABELS = {
    "baseline": "No-adj gated ROI-T",
    "roi_order_shuffle": "ROI-order shuffle",
    "zero_roi_embedding": "Zero ROI embedding",
    "uniform_gate": "Uniform gate",
    "random_fixed_gate": "Random fixed gate",
}

PERTURB_LABELS = {
    "default": "Default top-k",
    "topk20_corr": "Top-k corr",
    "dense_corr": "Dense corr",
    "identity": "Identity",
    "no_adjacency": "No adjacency",
    "shuffled": "Shuffled",
    "random": "Random",
    "edge_dropout_10": "Drop 10%",
    "edge_dropout_30": "Drop 30%",
    "edge_dropout_50": "Drop 50%",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create final ROI-token story artifacts for the AAAI-style framing.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def metric_cols(df: pd.DataFrame, metrics: list[str]) -> list[str]:
    return [m for m in metrics if f"{m}_mean" in df.columns]


def fmt(mean: float, std: float | None = None) -> str:
    if std is None or pd.isna(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def compact_table(df: pd.DataFrame, label_col: str, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        out = {"Condition": row[label_col], "n": int(row["n"])}
        for metric in metrics:
            out[metric] = fmt(float(row[f"{metric}_mean"]), float(row[f"{metric}_std"]))
        rows.append(out)
    return pd.DataFrame(rows)


def write_latex_table(path: Path, table: pd.DataFrame, caption: str, label: str) -> None:
    metric_cols_ = [c for c in table.columns if c not in {"Condition", "n"}]
    lines = [
        "\\begin{table}[!htbp]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        "  \\resizebox{\\linewidth}{!}{%",
        f"  \\begin{{tabular}}{{l{'c' * (len(metric_cols_) + 1)}}}",
        "    \\toprule",
        "    Condition & n & " + " & ".join(metric_cols_) + " \\\\",
        "    \\midrule",
    ]
    for _, row in table.iterrows():
        values = " & ".join(str(row[c]) for c in metric_cols_)
        lines.append(f"    {row['Condition']} & {int(row['n'])} & {values} \\\\")
    lines.extend(["    \\bottomrule", "  \\end{tabular}%", "  }", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_roi_token_controls(df: pd.DataFrame, out_dir: Path) -> None:
    order = ["baseline", "roi_order_shuffle", "zero_roi_embedding", "uniform_gate", "random_fixed_gate"]
    df = df.set_index("control_mode").loc[order].reset_index()
    colors = ["#2f7fbd", "#c4c4c4", "#9fd2d5", "#b9a6d8", "#8fb8d8"]
    panels = [("AUROC", (0.52, 0.86)), ("R@5", (0.0, 0.105)), ("brain_R@5", (0.0, 0.115))]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
    labels = [CONTROL_LABELS[x] for x in df["control_mode"]]
    x = np.arange(len(labels))
    for ax, (metric, ylim) in zip(axes, panels):
        means = df[f"{metric}_mean"].to_numpy(float)
        sems = df[f"{metric}_std"].to_numpy(float) / np.sqrt(df["n"].to_numpy(float))
        ax.bar(x, means, yerr=sems, capsize=3, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Baseline", "Order\nshuffle", "Zero\nROI emb.", "Uniform\ngate", "Random\ngate"], fontsize=8)
        ax.set_ylim(*ylim)
        ax.set_ylabel(metric.replace("brain_R@5", "Brain R@5"))
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(w_pad=1.0)
    for suffix in ["pdf", "png", "svg"]:
        fig.savefig(out_dir / f"roi_token_controls.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_adjacency_perturbation(df: pd.DataFrame, out_dir: Path) -> None:
    order = ["default", "no_adjacency", "identity", "shuffled", "random", "dense_corr", "edge_dropout_50"]
    df = df.set_index("control_mode").loc[order].reset_index()
    panels = [("AUROC", (0.49, 0.58)), ("R@5", (0.0, 0.025)), ("brain_R@5", (0.0, 0.02))]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
    x = np.arange(len(df))
    for ax, (metric, ylim) in zip(axes, panels):
        means = df[f"{metric}_mean"].to_numpy(float)
        sems = df[f"{metric}_std"].to_numpy(float) / np.sqrt(df["n"].to_numpy(float))
        ax.bar(x, means, yerr=sems, capsize=3, color="#9fb7ca", edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([PERTURB_LABELS[m] for m in df["control_mode"]], rotation=35, ha="right", fontsize=8)
        ax.set_ylim(*ylim)
        ax.set_ylabel(metric.replace("brain_R@5", "Brain R@5"))
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(w_pad=1.0)
    for suffix in ["pdf", "png", "svg"]:
        fig.savefig(out_dir / f"adjacency_perturbation.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def edge_followup_table(results: Path) -> pd.DataFrame:
    rows = []
    sources = [
        ("Main all-fold edge-bias", results / "phase32_edge_bias_allfold_3seed/phase32_edge_bias_allfold_3seed_group_summary.csv"),
        ("Hard-negative edge-bias", results / "phase33_edge_bias_hardneg/phase33_edge_bias_hardneg_group_summary.csv"),
        ("Held-out-image edge-bias", results / "phase33_edge_bias_heldout/phase33_edge_bias_heldout_group_summary.csv"),
    ]
    for label, path in sources:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0]
        rows.append(
            {
                "Condition": label,
                "n": int(row["n"]),
                "AUROC": fmt(float(row["AUROC_mean"]), float(row["AUROC_std"])),
                "AUPRC": fmt(float(row["AUPRC_mean"]), float(row["AUPRC_std"])),
                "R@5": fmt(float(row["R@5_mean"]), float(row["R@5_std"])),
                "Brain R@5": fmt(float(row["brain_R@5_mean"]), float(row["brain_R@5_std"])),
            }
        )
    return pd.DataFrame(rows)


def write_story_md(path: Path, controls: pd.DataFrame, perturb: pd.DataFrame, edge_table: pd.DataFrame) -> None:
    baseline = controls[controls["control_mode"].eq("baseline")].iloc[0]
    shuffle = controls[controls["control_mode"].eq("roi_order_shuffle")].iloc[0]
    zero = controls[controls["control_mode"].eq("zero_roi_embedding")].iloc[0]
    uniform = controls[controls["control_mode"].eq("uniform_gate")].iloc[0]
    random_gate = controls[controls["control_mode"].eq("random_fixed_gate")].iloc[0]
    default = perturb[perturb["control_mode"].eq("default")].iloc[0]
    no_adj = perturb[perturb["control_mode"].eq("no_adjacency")].iloc[0]
    random_adj = perturb[perturb["control_mode"].eq("random")].iloc[0]
    lines = [
        "# Final ROI-Token / AAAI Framing Summary",
        "",
        "## Main Finding",
        "",
        "The strongest defensible claim is that fixed anatomical ROI-token layout, transformer interactions, gated readout, and CLIP alignment drive cross-subject fMRI retrieval. Explicit static adjacency is not the source of the gain.",
        "",
        "## Evidence",
        "",
        f"- No-adj gated baseline: AUROC {baseline['AUROC_mean']:.4f}, R@5 {baseline['R@5_mean']:.4f}, brain R@5 {baseline['brain_R@5_mean']:.4f}.",
        f"- ROI-order shuffle collapses performance: AUROC {shuffle['AUROC_mean']:.4f}, R@5 {shuffle['R@5_mean']:.4f}, brain R@5 {shuffle['brain_R@5_mean']:.4f}.",
        f"- Zeroing learned ROI embeddings barely changes performance: AUROC {zero['AUROC_mean']:.4f}, brain R@5 {zero['brain_R@5_mean']:.4f}.",
        f"- Uniform gates reduce performance: AUROC {uniform['AUROC_mean']:.4f}, brain R@5 {uniform['brain_R@5_mean']:.4f}.",
        f"- Random fixed gates reduce performance: AUROC {random_gate['AUROC_mean']:.4f}, brain R@5 {random_gate['brain_R@5_mean']:.4f}.",
        f"- The weak static-adjacency checkpoint is nearly insensitive to removing/randomizing adjacency: default AUROC {default['AUROC_mean']:.4f}, no-adj AUROC {no_adj['AUROC_mean']:.4f}, random-adj AUROC {random_adj['AUROC_mean']:.4f}.",
        "",
        "## Edge Follow-Up",
        "",
    ]
    if edge_table.empty:
        lines.append("No edge-bias follow-up summary was found.")
    else:
        lines.extend(edge_table.to_markdown(index=False).splitlines())
    lines.extend(
        [
            "",
            "## AAAI Framing",
            "",
            "Frame the method as a gated fixed-order ROI-token Transformer-VLM. Use graph terminology only in the sense that ROI tokens are anatomically fixed brain graph nodes and relations are learned by attention. Avoid claiming that static Pearson adjacency is the performance driver.",
            "",
            "Recommended claim: fixed anatomical ROI-token structure plus gated ROI-preserving readout and image-semantic alignment improves cross-subject natural-image fMRI retrieval over ROI-MLP and graph/GNN baselines.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    results = root / args.results_dir
    out_dir = root / args.out_dir
    fig_dir = root / args.figure_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    controls = pd.read_csv(results / "phase3d_roi_token_controls/roi_token_control_summary.csv")
    controls["Condition"] = controls["control_mode"].map(CONTROL_LABELS).fillna(controls["control_mode"])
    control_metrics = metric_cols(controls, ["AUROC", "AUPRC", "R@5", "MRR", "brain_R@5"])
    control_table = compact_table(controls, "Condition", control_metrics)
    control_table.to_csv(out_dir / "table_roi_token_controls.csv", index=False)
    write_latex_table(
        out_dir / "table_roi_token_controls_latex.txt",
        control_table,
        "ROI-token and gate controls for the no-adjacency gated ROI Transformer. ROI-order shuffling causes the largest collapse, while zeroing learned ROI embeddings has negligible effect.",
        "tab:roi_token_controls",
    )
    plot_roi_token_controls(controls, fig_dir)

    perturb = pd.read_csv(results / "phase4_adjacency_perturbation/adjacency_perturbation_summary.csv")
    perturb["Condition"] = perturb["control_mode"].map(PERTURB_LABELS).fillna(perturb["control_mode"])
    perturb_metrics = metric_cols(perturb, ["AUROC", "AUPRC", "R@5", "MRR", "brain_R@5"])
    perturb_table = compact_table(perturb, "Condition", perturb_metrics)
    perturb_table.to_csv(out_dir / "table_adjacency_perturbation.csv", index=False)
    write_latex_table(
        out_dir / "table_adjacency_perturbation_latex.txt",
        perturb_table,
        "Adjacency perturbation diagnostic for the static-adjacency gated checkpoint. Randomizing, dropping, or removing edges changes performance only minimally, indicating weak reliance on the fixed adjacency.",
        "tab:adjacency_perturbation",
    )
    plot_adjacency_perturbation(perturb, fig_dir)

    edge_table = edge_followup_table(results)
    edge_table.to_csv(out_dir / "table_edge_bias_followup.csv", index=False)
    if not edge_table.empty:
        write_latex_table(
            out_dir / "table_edge_bias_followup_latex.txt",
            edge_table,
            "Learned edge-bias follow-up results. The edge-bias variants do not improve over the no-adjacency gated ROI-token model.",
            "tab:edge_bias_followup",
        )

    write_story_md(out_dir / "aaai_roi_token_story_summary.md", controls, perturb, edge_table)
    print(
        {
            "roi_token_table": str(out_dir / "table_roi_token_controls.csv"),
            "adjacency_perturbation_table": str(out_dir / "table_adjacency_perturbation.csv"),
            "edge_bias_table": str(out_dir / "table_edge_bias_followup.csv"),
            "story": str(out_dir / "aaai_roi_token_story_summary.md"),
        }
    )


if __name__ == "__main__":
    main()
