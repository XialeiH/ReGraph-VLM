#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ALL_SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 8-fold LOSO splits using actual manifest row counts.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-view", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    image_ids = sorted({int(row["nsdId"]) for row in rows})
    splits = []
    for fold_idx, held_out_subject in enumerate(ALL_SUBJECTS, start=1):
        train_subjects = [subject for subject in ALL_SUBJECTS if subject != held_out_subject]
        n_train_samples = sum(1 for row in rows if row["subject"] in train_subjects)
        n_test_samples = sum(1 for row in rows if row["subject"] == held_out_subject)
        splits.append(
            {
                "fold_name": f"fold_{fold_idx:02d}",
                "dataset_view": args.dataset_view,
                "held_out_subject": held_out_subject,
                "train_subjects": train_subjects,
                "test_subjects": [held_out_subject],
                "n_images": len(image_ids),
                "n_train_samples": n_train_samples,
                "n_test_samples": n_test_samples,
                "image_ids": image_ids,
            }
        )

    payload = {
        "dataset_view": args.dataset_view,
        "manifest_path": str(args.manifest.resolve()),
        "n_folds": 8,
        "n_images": len(image_ids),
        "subject_order": ALL_SUBJECTS,
        "folds": splits,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
