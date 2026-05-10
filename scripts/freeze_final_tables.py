#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze final ReGraph-VLM result tables.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    return parser.parse_args()


def read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def summarize(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty or "AUROC" not in df.columns:
        return pd.DataFrame()
    group_cols = [c for c in ["graph_encoder", "readout", "lambda_clip", "lambda_cross", "model"] if c in df.columns]
    if not group_cols:
        group_cols = ["model"]
    metrics = ["AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "image_MRR", "brain_R@5", "brain_MRR"]
    agg = {"n": ("AUROC", "count")}
    for metric in metrics:
        if metric in df.columns:
            agg[f"{metric}_mean"] = (metric, "mean")
            agg[f"{metric}_std"] = (metric, "std")
    out = df.groupby(group_cols, dropna=False).agg(**agg).reset_index()
    out.insert(0, "table", label)
    return out


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    if not df.empty:
        path.with_name(path.stem + "_latex.txt").write_text(df.to_latex(index=False, float_format="%.4f"), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    results = root / "preproc_v0/repetition_familiarity/results"
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    sources = {
        "table_static_baselines": results / "cross_subject_raw_similarity_allfold/cross_subject_raw_similarity_summary.csv",
        "table_cross_subject_seed11": results / "cross_subject_allfold_seed11/regraph_vlm_summary.csv",
        "table_gated_allfold_seed11": results / "cross_subject_gated_allfold_seed11/regraph_vlm_summary.csv",
        "table_gated_controls_graphonly": results
        / "regraph_readout_ablation/gated_flatten_graph_only/regraph_vlm_summary.csv",
        "table_gated_controls_shuffled": results
        / "regraph_readout_ablation/gated_clip_controls/shuffled_clip/regraph_vlm_summary.csv",
        "table_gated_controls_random": results
        / "regraph_readout_ablation/gated_clip_controls/random_embedding/regraph_vlm_summary.csv",
        "table_heldout_image": results / "heldout_image/regraph_vlm_summary.csv",
        "table_heldout_image_random": results / "heldout_image_random_embedding/regraph_vlm_summary.csv",
        "table_hard_negative_roi": results / "cross_subject_hardneg/roi_mlp/regraph_vlm_summary.csv",
        "table_hard_negative_bnt_flat": results / "cross_subject_hardneg/bnt_flat/regraph_vlm_summary.csv",
        "table_hard_negative_bnt_gated": results / "cross_subject_hardneg/bnt_gated/regraph_vlm_summary.csv",
        "table_cross_subject_infonce": results / "cross_subject_cross_infonce/regraph_vlm_summary.csv",
        "table_architecture_ablation_gated": results / "regraph_readout_ablation/gated_flatten/regraph_vlm_summary.csv",
        "table_architecture_ablation_fusion": results / "regraph_fusion/regraph_vlm_summary.csv",
        "table_allfold_final": results / "cross_subject_allfold_final/regraph_vlm_summary.csv",
        "table_hard_negative_allfold": results / "cross_subject_hardneg_allfold_seed11/regraph_vlm_summary.csv",
        "table_heldout_final": results / "heldout_image_final/regraph_vlm_summary.csv",
        "table_phase2_sota_graph_baselines": results / "phase2_sota_graph_baselines/regraph_vlm_summary.csv",
    }
    compact_rows = []
    for name, path in sources.items():
        df = read(path)
        if df.empty:
            continue
        df.to_csv(out / f"{name}.csv", index=False)
        compact = summarize(df, name)
        if not compact.empty:
            write_table(compact, out / f"{name}_summary.csv")
            compact_rows.append(compact)
    if compact_rows:
        master = pd.concat(compact_rows, ignore_index=True)
        write_table(master, out / "master_results_summary.csv")
        md = ["# Master Results Summary", ""]
        md.append(master.to_markdown(index=False, floatfmt=".4f"))
        (out / "master_results_summary.md").write_text("\n".join(md), encoding="utf-8")
    print({"output_dir": str(out), "n_tables": len(compact_rows)})


if __name__ == "__main__":
    main()
