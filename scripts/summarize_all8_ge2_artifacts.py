#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize all8_ge2_766 fold artifacts.")
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for diag_path in sorted(args.fold_root.glob("fold_*_pca_diagnostics.json")):
        payload = json.loads(diag_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "fold": payload["fold_name"],
                "held_out_subject": payload["held_out_subject"],
                "n_train_samples": int(payload["n_train_samples"]),
                "n_test_samples": int(payload["n_test_samples"]),
                "input_dim": int(payload["canonical_input_dim"]),
                "pca_dim": int(payload["pca_dim_effective"]),
                "explained_variance_cumsum": f"{float(payload['explained_variance_ratio_sum']):.6f}",
                "train_nan_count": int(payload["train_nan_count"]),
                "test_nan_count": int(payload["test_nan_count"]),
                "status": "ok" if int(payload["train_nan_count"]) == 0 and int(payload["test_nan_count"]) == 0 else "check",
            }
        )

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold",
                "held_out_subject",
                "n_train_samples",
                "n_test_samples",
                "input_dim",
                "pca_dim",
                "explained_variance_cumsum",
                "train_nan_count",
                "test_nan_count",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
