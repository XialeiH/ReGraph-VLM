#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report cross-subject retrieval candidate counts and chance levels.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject"),
    )
    parser.add_argument("--folds", nargs="+", default=["fold_01", "fold_04"])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/frozen_cross_subject"),
    )
    return parser.parse_args()


def harmonic_mrr_chance(n: int) -> float:
    if n <= 0:
        return float("nan")
    return sum(1.0 / rank for rank in range(1, n + 1)) / n


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    dataset_root = root / args.dataset_root
    out_dir = root / args.output_root
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for fold in args.folds:
        fold_dir = dataset_root / fold
        for split in ["val", "test"]:
            pairs = torch.load(fold_dir / f"{split}_pairs.pt", map_location="cpu", weights_only=False)
            positives = [p for p in pairs if int(p["same_image"]) == 1]
            candidate_images = sorted({int(p["nsdId_2"]) for p in pairs})
            n = len(candidate_images)
            rows.append(
                {
                    "fold": fold,
                    "split": split,
                    "n_queries": len(positives),
                    "n_candidate_images": n,
                    "n_pairs": len(pairs),
                    "chance_R@1": 1.0 / n if n else float("nan"),
                    "chance_R@5": min(5, n) / n if n else float("nan"),
                    "chance_R@10": min(10, n) / n if n else float("nan"),
                    "chance_MRR_approx": harmonic_mrr_chance(n),
                }
            )
    out_csv = out_dir / "cross_subject_chance_levels.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output_csv": str(out_csv), "rows": len(rows), "status": "ok"}, indent=2))


if __name__ == "__main__":
    main()
