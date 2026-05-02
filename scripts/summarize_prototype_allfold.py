#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize prototype all-fold metrics.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for fold_dir in sorted(path for path in args.root.iterdir() if path.is_dir() and path.name.startswith("fold_")):
        metrics_path = fold_dir / "prototype_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "fold": metrics["fold"],
                "held_out_subject": metrics["held_out_subject"],
                "selection_mode": metrics.get("selection_mode", ""),
                "aggregation_space": metrics.get("aggregation_space", ""),
                "selection_metric": metrics.get("selection_metric", ""),
                "top1_acc": metrics["top1_acc"],
                "top5_acc": metrics["top5_acc"],
                "chance_level": metrics["chance_level"],
                "n_test_samples": metrics["n_test_samples"],
                "n_validation_subjects": metrics.get("n_validation_subjects", ""),
                "n_selected_validation_subjects": metrics.get("n_selected_validation_subjects", ""),
                "per_val_subject_top1_mean": metrics.get("per_val_subject_top1_mean", ""),
                "per_val_subject_top1_std": metrics.get("per_val_subject_top1_std", ""),
                "per_val_subject_top5_mean": metrics.get("per_val_subject_top5_mean", ""),
                "per_val_subject_top5_std": metrics.get("per_val_subject_top5_std", ""),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold",
                "held_out_subject",
                "selection_mode",
                "aggregation_space",
                "selection_metric",
                "top1_acc",
                "top5_acc",
                "chance_level",
                "n_test_samples",
                "n_validation_subjects",
                "n_selected_validation_subjects",
                "per_val_subject_top1_mean",
                "per_val_subject_top1_std",
                "per_val_subject_top5_mean",
                "per_val_subject_top5_std",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
