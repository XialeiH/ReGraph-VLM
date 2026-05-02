#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage 3B trial-level smoke results.")
    parser.add_argument("--prototype-root", type=Path, required=True)
    parser.add_argument("--interaction-root", type=Path, required=True)
    parser.add_argument("--averaged-prototype-summary", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def read_csv_map(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["fold"]: row for row in rows}


def load_metrics(root: Path, filename: str) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for fold_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("fold_")):
        path = fold_dir / filename
        if not path.exists():
            continue
        output[fold_dir.name] = json.loads(path.read_text(encoding="utf-8"))
    return output


def main() -> None:
    args = parse_args()
    proto = load_metrics(args.prototype_root, "prototype_metrics.json")
    interaction = load_metrics(args.interaction_root, "light_interaction_metrics.json")
    averaged_proto = read_csv_map(args.averaged_prototype_summary)

    folds = sorted(set(proto.keys()) | set(interaction.keys()))
    rows: list[dict[str, object]] = []
    for fold in folds:
        proto_m = proto.get(fold)
        interaction_m = interaction.get(fold)
        avg_row = averaged_proto.get(fold, {})
        row: dict[str, object] = {
            "fold": fold,
            "held_out_subject": (
                proto_m["held_out_subject"] if proto_m else interaction_m["held_out_subject"] if interaction_m else ""
            ),
            "prototype_trial_top1": f"{float(proto_m['top1_acc']):.6f}" if proto_m else "",
            "prototype_trial_top5": f"{float(proto_m['top5_acc']):.6f}" if proto_m else "",
            "interaction_trial_top1": f"{float(interaction_m['top1_acc']):.6f}" if interaction_m else "",
            "interaction_trial_top5": f"{float(interaction_m['top5_acc']):.6f}" if interaction_m else "",
            "averaged_prototype_top1": avg_row.get("top1_acc", ""),
            "averaged_prototype_top5": avg_row.get("top5_acc", ""),
            "delta_interaction_vs_trial_prototype_top1": "",
            "delta_interaction_vs_trial_prototype_top5": "",
            "delta_trial_prototype_vs_averaged_top1": "",
            "delta_trial_prototype_vs_averaged_top5": "",
            "prototype_status": proto_m.get("status", "ok") if proto_m else "missing",
            "interaction_status": interaction_m.get("status", "ok") if interaction_m else "missing",
        }
        if proto_m and interaction_m:
            row["delta_interaction_vs_trial_prototype_top1"] = f"{float(interaction_m['top1_acc']) - float(proto_m['top1_acc']):.6f}"
            row["delta_interaction_vs_trial_prototype_top5"] = f"{float(interaction_m['top5_acc']) - float(proto_m['top5_acc']):.6f}"
        if proto_m and avg_row:
            row["delta_trial_prototype_vs_averaged_top1"] = f"{float(proto_m['top1_acc']) - float(avg_row['top1_acc']):.6f}"
            row["delta_trial_prototype_vs_averaged_top5"] = f"{float(proto_m['top5_acc']) - float(avg_row['top5_acc']):.6f}"
        rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold",
                "held_out_subject",
                "prototype_trial_top1",
                "prototype_trial_top5",
                "interaction_trial_top1",
                "interaction_trial_top5",
                "averaged_prototype_top1",
                "averaged_prototype_top5",
                "delta_interaction_vs_trial_prototype_top1",
                "delta_interaction_vs_trial_prototype_top5",
                "delta_trial_prototype_vs_averaged_top1",
                "delta_trial_prototype_vs_averaged_top5",
                "prototype_status",
                "interaction_status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
