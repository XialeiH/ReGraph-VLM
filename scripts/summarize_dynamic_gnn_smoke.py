#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage 4 dynamic GNN smoke results.")
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--prototype-summary", type=Path, required=True)
    parser.add_argument("--interaction-summary", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def read_csv_map(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["fold"]: row for row in rows}


def main() -> None:
    args = parse_args()
    proto_rows = read_csv_map(args.prototype_summary)
    interaction_rows = read_csv_map(args.interaction_summary)
    output_rows: list[dict[str, object]] = []

    for fold_dir in sorted(path for path in args.smoke_root.iterdir() if path.is_dir() and path.name.startswith("fold_")):
        metrics_path = fold_dir / "dynamic_gnn_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        fold = str(metrics["fold"])
        proto = proto_rows.get(fold, {})
        interaction = interaction_rows.get(fold, {})
        dyn_top1 = float(metrics["top1_acc"])
        dyn_top5 = float(metrics["top5_acc"])
        proto_top1 = float(proto["prototype_trial_top1"]) if proto and proto.get("prototype_trial_top1") else float("nan")
        proto_top5 = float(proto["prototype_trial_top5"]) if proto and proto.get("prototype_trial_top5") else float("nan")
        int_top1 = float(interaction["top1_acc"]) if interaction and interaction.get("top1_acc") else float("nan")
        int_top5 = float(interaction["top5_acc"]) if interaction and interaction.get("top5_acc") else float("nan")
        output_rows.append(
            {
                "fold": fold,
                "held_out_subject": metrics["held_out_subject"],
                "top1_acc": f"{dyn_top1:.6f}",
                "top5_acc": f"{dyn_top5:.6f}",
                "prototype_trial_top1": f"{proto_top1:.6f}" if proto else "",
                "prototype_trial_top5": f"{proto_top5:.6f}" if proto else "",
                "interaction_trial_top1": f"{int_top1:.6f}" if interaction else "",
                "interaction_trial_top5": f"{int_top5:.6f}" if interaction else "",
                "delta_vs_prototype_top1": f"{dyn_top1 - proto_top1:.6f}" if proto else "",
                "delta_vs_prototype_top5": f"{dyn_top5 - proto_top5:.6f}" if proto else "",
                "delta_vs_interaction_top1": f"{dyn_top1 - int_top1:.6f}" if interaction else "",
                "delta_vs_interaction_top5": f"{dyn_top5 - int_top5:.6f}" if interaction else "",
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
                "top1_acc",
                "top5_acc",
                "prototype_trial_top1",
                "prototype_trial_top5",
                "interaction_trial_top1",
                "interaction_trial_top5",
                "delta_vs_prototype_top1",
                "delta_vs_prototype_top5",
                "delta_vs_interaction_top1",
                "delta_vs_interaction_top5",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
