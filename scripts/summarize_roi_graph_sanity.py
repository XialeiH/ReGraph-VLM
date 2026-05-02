#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ROI graph sanity baseline metrics.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(args.root.glob("fold_*/*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model": metrics["model"],
                "fold": metrics["fold"],
                "top1": metrics["top1"],
                "top5": metrics["top5"],
                "chance_level": metrics["chance_level"],
                "best_val_top1": metrics["best_val_top1"],
                "best_val_top5": metrics["best_val_top5"],
                "best_epoch": metrics["best_epoch"],
                "n_nodes": metrics["n_nodes"],
                "node_feature_dim": metrics["node_feature_dim"],
                "n_train": metrics["n_train"],
                "n_val": metrics["n_val"],
                "n_test": metrics["n_test"],
                "status": metrics["status"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "fold",
                "top1",
                "top5",
                "chance_level",
                "best_val_top1",
                "best_val_top5",
                "best_epoch",
                "n_nodes",
                "node_feature_dim",
                "n_train",
                "n_val",
                "n_test",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
