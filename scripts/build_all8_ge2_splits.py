#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ALL_SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 8-fold LOSO splits for all8_ge2_766.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    image_ids = sorted({int(row["nsdId"]) for row in rows})
    splits = []
    for fold_idx, held_out_subject in enumerate(ALL_SUBJECTS, start=1):
        train_subjects = [subject for subject in ALL_SUBJECTS if subject != held_out_subject]
        splits.append(
            {
                "fold_name": f"fold_{fold_idx:02d}",
                "dataset_view": "all8_ge2_766",
                "held_out_subject": held_out_subject,
                "train_subjects": train_subjects,
                "test_subjects": [held_out_subject],
                "n_images": len(image_ids),
                "n_train_samples": len(image_ids) * len(train_subjects),
                "n_test_samples": len(image_ids),
                "image_ids": image_ids,
            }
        )

    output = {
        "dataset_view": "all8_ge2_766",
        "manifest_path": str(args.manifest.resolve()),
        "n_folds": 8,
        "n_images": len(image_ids),
        "subject_order": ALL_SUBJECTS,
        "folds": splits,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
