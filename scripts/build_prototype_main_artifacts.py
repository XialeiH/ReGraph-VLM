#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prototype main summary and prototype-vs-B4 comparison tables.")
    parser.add_argument("--prototype-root", type=Path, required=True)
    parser.add_argument("--b4-summary", type=Path, required=True)
    parser.add_argument("--prototype-main-summary", type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    proto_rows: list[dict[str, object]] = []
    for fold_dir in sorted(path for path in args.prototype_root.iterdir() if path.is_dir() and path.name.startswith("fold_")):
        metrics_path = fold_dir / "prototype_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        proto_rows.append(
            {
                "fold": metrics["fold"],
                "held_out_subject": metrics["held_out_subject"],
                "top1_acc": metrics["top1_acc"],
                "top5_acc": metrics["top5_acc"],
                "selected_val_subjects": ";".join(metrics.get("selected_validation_subjects", [])),
                "aggregation_rule": (
                    f"{metrics.get('selection_metric', '')}:top{metrics.get('n_selected_validation_subjects', '')}"
                    f"+{metrics.get('aggregation_space', '')}"
                ),
                "status": "ok",
            }
        )

    write_csv(
        args.prototype_main_summary,
        proto_rows,
        [
            "fold",
            "held_out_subject",
            "top1_acc",
            "top5_acc",
            "selected_val_subjects",
            "aggregation_rule",
            "status",
        ],
    )

    b4_rows = read_csv(args.b4_summary)
    b4_by_fold = {row["fold"]: row for row in b4_rows}
    comparison_rows: list[dict[str, object]] = []
    for row in proto_rows:
        b4_row = b4_by_fold[row["fold"]]
        proto_top1 = float(row["top1_acc"])
        proto_top5 = float(row["top5_acc"])
        b4_top1 = float(b4_row["top1_acc"])
        b4_top5 = float(b4_row["top5_acc"])
        comparison_rows.append(
            {
                "fold": row["fold"],
                "held_out_subject": row["held_out_subject"],
                "b4_top1": b4_top1,
                "proto_top1": proto_top1,
                "delta_top1": proto_top1 - b4_top1,
                "b4_top5": b4_top5,
                "proto_top5": proto_top5,
                "delta_top5": proto_top5 - b4_top5,
            }
        )

    write_csv(
        args.comparison_output,
        comparison_rows,
        [
            "fold",
            "held_out_subject",
            "b4_top1",
            "proto_top1",
            "delta_top1",
            "b4_top5",
            "proto_top5",
            "delta_top5",
        ],
    )


if __name__ == "__main__":
    main()
