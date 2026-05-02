#!/usr/bin/env python3
"""Summarize per-fold baseline metric JSON files into one CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric_name = f"{args.baseline}_metrics.json"
    rows = []

    for fold_dir in sorted(p for p in args.baseline_root.iterdir() if p.is_dir()):
        metric_path = fold_dir / metric_name
        if not metric_path.exists():
            continue
        with metric_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append(
            {
                "fold": metrics["fold"],
                "held_out_subject": metrics["held_out_subject"],
                "top1_acc": metrics["top1_acc"],
                "top5_acc": metrics["top5_acc"],
                "chance_level": metrics["chance_level"],
                "n_test_samples": metrics["n_test_samples"],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fold",
                "held_out_subject",
                "top1_acc",
                "top5_acc",
                "chance_level",
                "n_test_samples",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
