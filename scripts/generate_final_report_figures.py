#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final ReGraph-VLM report figures.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_figures"),
    )
    return parser.parse_args()


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def grouped_mean(df: pd.DataFrame, filt: dict[str, object], metric: str) -> float:
    x = df.copy()
    for k, v in filt.items():
        x = x[x[k] == v]
    return float(x[metric].mean()) if not x.empty and metric in x.columns else float("nan")


def barplot(labels: list[str], values: list[float], title: str, ylabel: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.35), 4), constrained_layout=True)
    ax.bar(labels, values, color=["#5b8e7d", "#c9a227", "#d95d39", "#355070", "#6d597a"][: len(labels)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(values):
        if pd.notna(v):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    save(fig, out)


def model_overview(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax.axis("off")
    boxes = [
        (0.06, 0.62, "Natural image"),
        (0.28, 0.62, "Frozen CLIP\nimage encoder"),
        (0.53, 0.62, "Image projection\nz_img"),
        (0.06, 0.20, "Trial-level\nROI graph"),
        (0.28, 0.20, "Fixed-order ROI\nTransformer"),
        (0.53, 0.20, "Gated-flatten\nbrain embedding z_brain"),
        (0.78, 0.40, "Contrastive\nalignment + retrieval"),
    ]
    for x, y, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.17, 0.18, fill=False, linewidth=1.8))
        ax.text(x + 0.085, y + 0.09, text, ha="center", va="center", fontsize=10)
    arrows = [((0.23, 0.71), (0.28, 0.71)), ((0.45, 0.71), (0.53, 0.71)), ((0.23, 0.29), (0.28, 0.29)), ((0.45, 0.29), (0.53, 0.29)), ((0.70, 0.71), (0.78, 0.50)), ((0.70, 0.29), (0.78, 0.45))]
    for a, b in arrows:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.6))
    ax.set_title("Gated ReGraph-VLM overview", fontsize=14)
    save(fig, out)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    results = root / "preproc_v0/repetition_familiarity/results"
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    model_overview(out / "figure_model_overview")

    allfold = read(results / "cross_subject_allfold_final/regraph_vlm_summary.csv")
    if allfold.empty:
        allfold = read(results / "cross_subject_gated_allfold_seed11/regraph_vlm_summary.csv")
    seed11 = read(results / "cross_subject_allfold_seed11/regraph_vlm_summary.csv")
    raw = read(results / "cross_subject_raw_similarity_allfold/cross_subject_raw_similarity_summary.csv")

    labels = ["Raw", "ROI-MLP+CLIP", "Flat ReGraph+CLIP", "Gated ReGraph+CLIP"]
    if not seed11.empty or not allfold.empty:
        joined = pd.concat([seed11, allfold], ignore_index=True)
        for metric in ["AUROC", "R@5", "brain_R@5"]:
            values = [
                float(raw[raw["model"] == "raw_pearson_flat"][metric].mean()) if metric in raw.columns and not raw.empty else float("nan"),
                grouped_mean(joined, {"graph_encoder": "roi_mlp", "lambda_clip": 2.0}, metric),
                grouped_mean(joined, {"graph_encoder": "bnt_token_flat", "readout": "flat", "lambda_clip": 2.0}, metric),
                grouped_mean(joined, {"graph_encoder": "bnt_token_flat", "readout": "gated_flat", "lambda_clip": 2.0}, metric),
            ]
            barplot(labels, values, f"Cross-subject main result: {metric}", metric, out / f"figure_cross_subject_{metric.replace('@','at')}")

    held = read(results / "heldout_image/regraph_vlm_summary.csv")
    held_rand = read(results / "heldout_image_random_embedding/regraph_vlm_summary.csv")
    if not held.empty and not held_rand.empty:
        for metric in ["image_R@5", "brain_R@5", "MRR"]:
            values = [
                grouped_mean(held, {"graph_encoder": "bnt_token_flat", "readout": "gated_flat"}, metric),
                grouped_mean(held_rand, {"graph_encoder": "bnt_token_flat", "readout": "gated_flat"}, metric),
            ]
            barplot(["Real CLIP", "Random embedding"], values, f"Held-out image: {metric}", metric, out / f"figure_heldout_clip_vs_random_{metric.replace('@','at')}")

    hard = {
        "ROI-MLP+CLIP": read(results / "cross_subject_hardneg/roi_mlp/regraph_vlm_summary.csv"),
        "Flat ReGraph+CLIP": read(results / "cross_subject_hardneg/bnt_flat/regraph_vlm_summary.csv"),
        "Gated ReGraph+CLIP": read(results / "cross_subject_hardneg/bnt_gated/regraph_vlm_summary.csv"),
    }
    if all(not df.empty for df in hard.values()):
        for metric in ["AUROC", "R@5", "brain_R@5"]:
            values = [float(df[metric].mean()) for df in hard.values()]
            barplot(list(hard), values, f"Hard-negative result: {metric}", metric, out / f"figure_hard_negative_{metric.replace('@','at')}")

    print({"output_dir": str(out), "n_files": len(list(out.glob('*')))})


if __name__ == "__main__":
    main()
