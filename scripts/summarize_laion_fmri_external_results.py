#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ["test_AUROC", "test_AUPRC", "test_R@5", "test_MRR"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize LAION-fMRI external visual-ROI validation runs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("external_validation/laion_fmri/visual_roi_scalar4_laion/trained_external"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("external_validation/summary"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.root.glob("*_*/**/summary.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["pair"] = path.relative_to(args.root).parts[0]
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No LAION-fMRI summary.csv files found under {args.root}")

    all_runs = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_runs.to_csv(args.out_dir / "laion_fmri_visual_roi_all_runs.csv", index=False)

    grouped_rows = []
    for model, group in all_runs.groupby("model"):
        row = {"model": model, "n": len(group)}
        for metric in METRICS:
            if metric in group:
                row[f"{metric}_mean"] = group[metric].mean()
                row[f"{metric}_std"] = group[metric].std(ddof=1)
        grouped_rows.append(row)
    summary = pd.DataFrame(grouped_rows).sort_values("model")
    summary.to_csv(args.out_dir / "laion_fmri_visual_roi_summary.csv", index=False)
    (args.out_dir / "laion_fmri_visual_roi_summary.md").write_text(
        "# LAION-fMRI External Visual-ROI Validation\n\n"
        "Trial-wise public LAION-fMRI beta maps were summarized with public visual ROI masks, padded to the same 180-token interface, and evaluated with cross-subject same-image retrieval. Values are mean ± std over subject-pair × seed runs.\n\n"
        + summary.to_markdown(index=False, floatfmt=".4f")
        + "\n",
        encoding="utf-8",
    )
    print({"all_runs": len(all_runs), "summary": str(args.out_dir / "laion_fmri_visual_roi_summary.csv")})


if __name__ == "__main__":
    main()
