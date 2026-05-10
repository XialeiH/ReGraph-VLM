#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect current ReGraph-VLM result summaries into one frozen directory.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/frozen_current_stage"),
    )
    return parser.parse_args()


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    results_root = root / "preproc_v0/repetition_familiarity/results"

    mapping = {
        "within_subject_summary.csv": results_root / "frozen_current_stage/within_subject_repeat_matching_summary.csv",
        "cross_subject_seed11_summary.csv": results_root / "cross_subject_allfold_seed11/regraph_vlm_summary.csv",
        "cross_subject_controls_shuffled_summary.csv": results_root
        / "cross_subject_clip_controls/shuffled_clip/regraph_vlm_summary.csv",
        "cross_subject_controls_random_summary.csv": results_root
        / "cross_subject_clip_controls/random_embedding/regraph_vlm_summary.csv",
        "architecture_ablation_gated_summary.csv": results_root
        / "regraph_readout_ablation/gated_flatten/regraph_vlm_summary.csv",
        "architecture_ablation_fusion_summary.csv": results_root / "regraph_fusion/regraph_vlm_summary.csv",
        "leakage_qc.json": results_root / "frozen_cross_subject/cross_subject_leakage_qc.json",
        "chance_levels.csv": results_root / "frozen_cross_subject/cross_subject_chance_levels.csv",
    }
    copied = {name: copy_if_exists(src, out / name) for name, src in mapping.items()}

    rows = []
    for label, rel in {
        "cross_subject_seed11": out / "cross_subject_seed11_summary.csv",
        "shuffled_clip": out / "cross_subject_controls_shuffled_summary.csv",
        "random_embedding": out / "cross_subject_controls_random_summary.csv",
        "gated_flatten": out / "architecture_ablation_gated_summary.csv",
        "fusion": out / "architecture_ablation_fusion_summary.csv",
    }.items():
        if not rel.exists():
            continue
        df = pd.read_csv(rel)
        if "AUROC" not in df.columns:
            continue
        group_cols = [col for col in ["graph_encoder", "lambda_clip", "readout"] if col in df.columns]
        grouped = (
            df.groupby(group_cols, dropna=False)
            .agg(
                n=("AUROC", "count"),
                AUROC_mean=("AUROC", "mean"),
                AUPRC_mean=("AUPRC", "mean"),
                R5_mean=("R@5", "mean"),
                MRR_mean=("MRR", "mean"),
                image_R5_mean=("image_R@5", "mean"),
                brain_R5_mean=("brain_R@5", "mean"),
                brain_MRR_mean=("brain_MRR", "mean"),
            )
            .reset_index()
        )
        grouped.insert(0, "source", label)
        rows.append(grouped)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(out / "current_all_results_compact_summary.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({"copied": copied}, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "copied": copied}, indent=2))


if __name__ == "__main__":
    main()
