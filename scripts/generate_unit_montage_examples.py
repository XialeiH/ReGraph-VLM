#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate weak-interpretability montage PDF from montage index CSV.")
    parser.add_argument("--montage-index", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.montage_index)
    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)

    grouped = list(df.groupby(["fold", "unit_id"], sort=False))
    with PdfPages(args.output_pdf) as pdf:
        for (fold, unit_id), sub in grouped:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            dominance = sub["roi_dominance_label"].iloc[0]
            concentration = sub["concentration_score"].iloc[0]
            ax.set_title(
                f"Unit Montage Preview\nfold={fold} unit={unit_id} dominant_roi={dominance} concentration={concentration:.3f}",
                fontsize=12,
                pad=16,
            )
            table_df = sub[["rank", "nsdId", "subject", "split", "activation"]].copy()
            table_df["activation"] = table_df["activation"].map(lambda x: f"{x:.4f}")
            table = ax.table(
                cellText=table_df.values,
                colLabels=table_df.columns,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.0, 1.5)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    main()
