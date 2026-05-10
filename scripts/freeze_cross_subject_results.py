#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze main cross-subject ReGraph-VLM results.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--model-summary",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/cross_subject_regraph_vlm/regraph_vlm_summary.csv"),
    )
    parser.add_argument(
        "--raw-summary",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/cross_subject_raw_similarity/cross_subject_raw_similarity_summary.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/frozen_cross_subject"),
    )
    return parser.parse_args()


def fmt(mean: float, std: float) -> str:
    if pd.isna(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    model_path = root / args.model_summary
    raw_path = root / args.raw_summary
    out_dir = root / args.output_root
    out_dir.mkdir(parents=True, exist_ok=True)

    model_df = pd.read_csv(model_path)
    raw_df = pd.read_csv(raw_path)
    raw_test = raw_df[raw_df["split"] == "test"].copy()

    metrics = [
        "AUROC",
        "AUPRC",
        "R@1",
        "R@5",
        "R@10",
        "MRR",
        "image_R@5",
        "image_MRR",
        "brain_R@5",
        "brain_MRR",
    ]
    for col in metrics:
        if col in model_df.columns:
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
        if col in raw_test.columns:
            raw_test[col] = pd.to_numeric(raw_test[col], errors="coerce")

    rows: list[dict[str, object]] = []
    raw_metric_map = {m: m for m in ["AUROC", "AUPRC", "R@1", "R@5", "R@10", "MRR"] if m in raw_test.columns}
    raw_row: dict[str, object] = {"model": "Raw Pearson", "graph_encoder": "raw_pearson_flat", "lambda": "none", "n": int(len(raw_test))}
    for out_metric, in_metric in raw_metric_map.items():
        raw_row[f"{out_metric}_mean"] = float(raw_test[in_metric].mean())
        raw_row[f"{out_metric}_std"] = float(raw_test[in_metric].std(ddof=1))
    rows.append(raw_row)

    names = {
        ("roi_mlp", 0.0): "ROI-MLP",
        ("roi_mlp", 2.0): "ROI-MLP+CLIP",
        ("bnt_token_flat", 0.0): "BNT/ReGraph",
        ("bnt_token_flat", 2.0): "BNT/ReGraph+CLIP",
    }
    for (encoder, lam), group in model_df.groupby(["graph_encoder", "lambda_clip"], sort=True):
        row: dict[str, object] = {
            "model": names.get((str(encoder), float(lam)), f"{encoder}_lambda_{lam:g}"),
            "graph_encoder": str(encoder),
            "lambda": float(lam),
            "n": int(len(group)),
        }
        for metric in metrics:
            if metric in group.columns:
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)

    out = pd.DataFrame(rows)
    preferred = ["model", "graph_encoder", "lambda", "n"]
    metric_cols: list[str] = []
    for metric in metrics:
        metric_cols.extend([f"{metric}_mean", f"{metric}_std"])
    out = out[[c for c in preferred + metric_cols if c in out.columns]]
    out_csv = out_dir / "cross_subject_main_summary.csv"
    out.to_csv(out_csv, index=False)

    md_lines = ["# Cross-Subject Main Summary", "", out.to_markdown(index=False), ""]
    lookup = {str(row["model"]): row for row in rows}
    comparisons: dict[str, dict[str, float | None]] = {}
    target = lookup.get("BNT/ReGraph+CLIP")
    for baseline_name in ["ROI-MLP+CLIP", "BNT/ReGraph", "Raw Pearson"]:
        base = lookup.get(baseline_name)
        if not target or not base:
            continue
        comp: dict[str, float | None] = {}
        for metric in metrics:
            key = f"{metric}_mean"
            if key in target and key in base and pd.notna(target[key]) and pd.notna(base[key]) and float(base[key]) != 0:
                comp[metric] = float(target[key]) / float(base[key])
            else:
                comp[metric] = None
        comparisons[f"BNT/ReGraph+CLIP_vs_{baseline_name}"] = comp
    md_lines.extend(["## Relative Improvements", "", "```json", json.dumps(comparisons, indent=2), "```", ""])
    (out_dir / "cross_subject_main_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    latex_cols = ["model", "n", "AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "brain_R@5", "brain_MRR"]
    latex_rows = []
    for _, row in out.iterrows():
        item = {"model": row["model"], "n": int(row["n"])}
        for metric in latex_cols[2:]:
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            if mean_key in row and pd.notna(row[mean_key]):
                item[metric] = fmt(float(row[mean_key]), float(row[std_key]) if std_key in row else float("nan"))
            else:
                item[metric] = "--"
        latex_rows.append(item)
    latex_df = pd.DataFrame(latex_rows, columns=latex_cols)
    (out_dir / "cross_subject_main_summary_latex.txt").write_text(latex_df.to_latex(index=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "rows": int(len(out)), "status": "ok"}, indent=2))


if __name__ == "__main__":
    main()
