#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage 3A smoke results.")
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--prototype-summary", type=Path, default=None)
    parser.add_argument("--b4-summary", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def read_summary_table(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["fold"]: row for row in rows}


def main() -> None:
    args = parse_args()
    proto_rows = read_summary_table(args.prototype_summary)
    b4_rows = read_summary_table(args.b4_summary)

    output_rows: list[dict[str, object]] = []
    fold_dirs = sorted(path for path in args.smoke_root.iterdir() if path.is_dir() and path.name.startswith("fold_"))
    for fold_dir in fold_dirs:
        metrics_path = fold_dir / "light_interaction_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        fold = str(metrics["fold"])
        proto = proto_rows.get(fold, {})
        b4 = b4_rows.get(fold, {})
        proto_top1 = float(proto["top1_acc"]) if proto else float("nan")
        proto_top5 = float(proto["top5_acc"]) if proto else float("nan")
        b4_top1 = float(b4["top1_acc"]) if b4 else float("nan")
        b4_top5 = float(b4["top5_acc"]) if b4 else float("nan")
        output_rows.append(
            {
                "fold": fold,
                "held_out_subject": metrics["held_out_subject"],
                "canonical_validation_subject": metrics["canonical_validation_subject"],
                "top1_acc": f"{float(metrics['top1_acc']):.6f}",
                "top5_acc": f"{float(metrics['top5_acc']):.6f}",
                "prototype_main_top1": f"{proto_top1:.6f}" if proto else "",
                "prototype_main_top5": f"{proto_top5:.6f}" if proto else "",
                "b4_main_top1": f"{b4_top1:.6f}" if b4 else "",
                "b4_main_top5": f"{b4_top5:.6f}" if b4 else "",
                "delta_vs_prototype_top1": f"{float(metrics['top1_acc']) - proto_top1:.6f}" if proto else "",
                "delta_vs_prototype_top5": f"{float(metrics['top5_acc']) - proto_top5:.6f}" if proto else "",
                "delta_vs_b4_top1": f"{float(metrics['top1_acc']) - b4_top1:.6f}" if b4 else "",
                "delta_vs_b4_top5": f"{float(metrics['top5_acc']) - b4_top5:.6f}" if b4 else "",
                "status": metrics.get("status", "ok"),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold",
                "held_out_subject",
                "canonical_validation_subject",
                "top1_acc",
                "top5_acc",
                "prototype_main_top1",
                "prototype_main_top5",
                "b4_main_top1",
                "b4_main_top5",
                "delta_vs_prototype_top1",
                "delta_vs_prototype_top5",
                "delta_vs_b4_top1",
                "delta_vs_b4_top5",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
