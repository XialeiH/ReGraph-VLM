#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize repeat-state baseline metrics.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/repeat_state_baselines"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.root.resolve() / args.results_root
    rows = []
    for path in sorted(results_root.glob("*/*/fold_*/metrics.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    out_path = results_root / "repeat_state_baseline_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "model",
        "fold",
        "adjacency",
        "control",
        "pool_ratio",
        "auroc",
        "auprc",
        "balanced_accuracy",
        "macro_f1",
        "accuracy",
        "best_val_auroc",
        "best_epoch",
        "n_train",
        "n_val",
        "n_test",
        "status",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (results_root / "repeat_state_baseline_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"summary_csv": str(out_path), "n_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
