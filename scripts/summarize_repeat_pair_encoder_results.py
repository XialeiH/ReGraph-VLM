#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize repeat-pair encoder metrics.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/repeat_pair_encoder_results"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.root.resolve() / args.results_root
    rows = []
    for metrics_path in sorted(results_root.glob("*/*/fold_*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(metrics)
    fieldnames = [
        "model",
        "fold",
        "adjacency",
        "loss_mode",
        "bnt_input_type",
        "readout",
        "roi_id",
        "auroc",
        "auprc",
        "balanced_accuracy",
        "accuracy",
        "recall_at_1",
        "recall_at_5",
        "mrr",
        "best_val_auroc",
        "best_epoch",
        "n_train_pairs",
        "n_val_pairs",
        "n_test_pairs",
        "status",
    ]
    out_path = results_root / "repeat_pair_encoder_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (results_root / "repeat_pair_encoder_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"summary_csv": str(out_path), "n_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
