#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DISPLAY_MODELS = {
    "roi_mlp": "ROI-MLP",
    "roi_transformer_gated": "Gated ROI Transformer",
}
LAION_METRICS = ["test_AUROC", "test_AUPRC", "test_R@5", "test_MRR"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update publication result-table blocks in reports/neurips_report/may23.tex.")
    parser.add_argument("--tex", type=Path, default=Path("reports/neurips_report/may23.tex"))
    parser.add_argument(
        "--single-ref-latex",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables/single_ref_matched_allseed_latex.txt"),
    )
    parser.add_argument(
        "--laion-summary",
        type=Path,
        default=Path("external_validation/summary/laion_fmri_visual_roi_summary.csv"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_latex_rows(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("%")]


def table_bounds(text: str, label: str) -> tuple[int, int]:
    label_pos = text.find(f"\\label{{{label}}}")
    if label_pos < 0:
        raise ValueError(f"Table label not found: {label}")
    begin = text.rfind("\\begin{table}", 0, label_pos)
    end = text.find("\\end{table}", label_pos)
    if begin < 0 or end < 0:
        raise ValueError(f"Could not find table environment for {label}")
    return begin, end + len("\\end{table}")


def replace_table_rows(text: str, label: str, rows: list[str]) -> str:
    if not rows:
        return text
    begin, end = table_bounds(text, label)
    table = text[begin:end]
    mid = table.find("\\midrule")
    bottom = table.find("\\bottomrule")
    if mid < 0 or bottom < 0 or bottom <= mid:
        raise ValueError(f"Could not find midrule/bottomrule in {label}")
    replacement = "\\midrule\n    " + "\n    ".join(rows) + "\n    "
    new_table = table[:mid] + replacement + table[bottom:]
    return text[:begin] + new_table + text[end:]


def metric_cell(mean: float, std: float, bold: bool) -> str:
    inner = f"{mean:.4f}{{\\scriptstyle\\pm{std:.4f}}}"
    if bold:
        return f"$\\mathbf{{{inner}}}$"
    return f"${inner}$"


def laion_rows(path: Path) -> list[str]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    best = {metric: df[f"{metric}_mean"].max() for metric in LAION_METRICS}
    rows = []
    for _, row in df.sort_values("model").iterrows():
        cells = [
            "LAION-fMRI visual ROI",
            DISPLAY_MODELS.get(str(row["model"]), str(row["model"])),
            str(int(row["n"])),
        ]
        for metric in LAION_METRICS:
            mean = float(row[f"{metric}_mean"])
            std = float(row[f"{metric}_std"])
            cells.append(metric_cell(mean, std, mean == best[metric]))
        rows.append(" & ".join(cells) + r" \\")
    return rows


def upsert_external_laion_rows(text: str, rows: list[str]) -> str:
    if not rows:
        return text
    begin, end = table_bounds(text, "tab:external_visual_roi_smoke")
    table = text[begin:end]
    table = re.sub(r"\n\s*LAION-fMRI visual ROI & .+?\\\\", "", table)
    bottom = table.find("\\bottomrule")
    if bottom < 0:
        raise ValueError("Could not find bottomrule in external visual-ROI table")
    insertion = "    " + "\n    ".join(rows) + "\n    "
    table = table[:bottom] + insertion + table[bottom:]
    text = text[:begin] + table + text[end:]
    text = text.replace("from three independent natural-image fMRI datasets", "from four independent natural-image fMRI datasets")
    text = text.replace(
        "CNeuroMod-THINGS, BOLD5000, and THINGS-fMRI summaries",
        "CNeuroMod-THINGS, BOLD5000, THINGS-fMRI, and LAION-fMRI summaries",
    )
    text = text.replace("All three external smoke checks", "All external smoke checks")
    return text


def main() -> None:
    args = parse_args()
    text = args.tex.read_text(encoding="utf-8")
    original = text
    text = replace_table_rows(text, "tab:single_ref_retrained", read_latex_rows(args.single_ref_latex))
    text = upsert_external_laion_rows(text, laion_rows(args.laion_summary))
    if text == original:
        print("No publication table updates applied; required snippet files are missing or empty.")
        return
    if args.dry_run:
        print("Publication table updates are available; dry run did not modify the manuscript.")
        return
    args.tex.write_text(text, encoding="utf-8")
    print(f"Updated {args.tex}")


if __name__ == "__main__":
    main()
