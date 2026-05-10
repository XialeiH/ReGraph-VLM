#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze current within-subject repeat matching results.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results"),
    )
    return parser.parse_args()


def add_row(rows: list[dict[str, object]], label: str, source: str, group: pd.DataFrame, metric_map: dict[str, str]) -> None:
    row: dict[str, object] = {"model": label, "source": source, "n": int(len(group))}
    for out_name, col in metric_map.items():
        if col in group.columns:
            row[f"{out_name}_mean"] = float(group[col].mean())
            row[f"{out_name}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
    rows.append(row)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    results_root = root / args.results_root
    out_dir = results_root / "final_current_stage"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    frozen = results_root / "frozen_baselines" / "frozen_repeat_pair_baselines.csv"
    if frozen.exists():
        df = pd.read_csv(frozen)
        keep = [
            "ROI MLP + BCE_INFONCE",
            "raw pearson_flat",
            "BNT-token + flat + ROI normal + BCE_INFONCE",
        ]
        for label in keep:
            match = df[df["baseline"].eq(label)]
            if not match.empty:
                row = match.iloc[0].to_dict()
                rows.append(
                    {
                        "model": label,
                        "source": "pre_vlm_within_subject",
                        "n": int(row.get("n_runs", 0)),
                        "AUROC_mean": row.get("auroc_mean"),
                        "AUROC_std": row.get("auroc_std"),
                        "AUPRC_mean": row.get("auprc_mean"),
                        "R@1_mean": row.get("recall_at_1_mean"),
                        "R@5_mean": row.get("recall_at_5_mean"),
                        "MRR_mean": row.get("mrr_mean"),
                    }
                )

    rgv = results_root / "regraph_vlm" / "regraph_vlm_summary.csv"
    if rgv.exists():
        df = pd.read_csv(rgv)
        metric_map = {
            "AUROC": "AUROC",
            "AUPRC": "AUPRC",
            "R@1": "R@1",
            "R@5": "R@5",
            "MRR": "MRR",
            "image_R@5": "image_R@5",
            "image_MRR": "image_MRR",
            "brain_R@5": "brain_R@5",
            "brain_MRR": "brain_MRR",
        }
        for graph_encoder, lambda_clip, label in [
            ("bnt_token_flat", 0.0, "BNT/ReGraph graph-only"),
            ("bnt_token_flat", 2.0, "BNT/ReGraph+CLIP lambda=2.0"),
            ("roi_mlp", 2.0, "ROI-MLP+CLIP lambda=2.0"),
        ]:
            group = df[df["graph_encoder"].eq(graph_encoder) & df["lambda_clip"].eq(lambda_clip)]
            if not group.empty:
                add_row(rows, label, "vlm_within_subject_random", group, metric_map)

    hard = results_root / "regraph_vlm_hardneg" / "regraph_vlm_summary.csv"
    if hard.exists():
        df = pd.read_csv(hard)
        metric_map = {
            "hard_AUROC": "AUROC",
            "hard_AUPRC": "AUPRC",
            "hard_R@5": "R@5",
            "hard_MRR": "MRR",
            "random_AUROC": "random_AUROC",
            "random_R@5": "random_R@5",
            "random_MRR": "random_MRR",
            "brain_R@5": "brain_R@5",
            "brain_MRR": "brain_MRR",
        }
        for graph_encoder, label in [
            ("bnt_token_flat", "BNT/ReGraph+CLIP hardneg lambda=2.0"),
            ("roi_mlp", "ROI-MLP+CLIP hardneg lambda=2.0"),
        ]:
            group = df[df["graph_encoder"].eq(graph_encoder) & df["lambda_clip"].eq(2.0)]
            if not group.empty:
                add_row(rows, label, "vlm_within_subject_hardneg", group, metric_map)

    out = pd.DataFrame(rows)
    out_csv = out_dir / "within_subject_repeat_matching_summary.csv"
    out_md = out_dir / "within_subject_repeat_matching_summary.md"
    out.to_csv(out_csv, index=False)
    out_md.write_text(out.to_markdown(index=False) + "\n", encoding="utf-8")
    print({"csv": str(out_csv), "md": str(out_md), "n_rows": int(len(out))})


if __name__ == "__main__":
    main()
