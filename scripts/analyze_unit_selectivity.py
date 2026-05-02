#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze prototype unit selectivity from clean main outputs.")
    parser.add_argument("--prototype-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_samples(prototype_root: Path) -> tuple[np.ndarray, list[dict[str, object]]]:
    assignments_all: list[np.ndarray] = []
    sample_rows: list[dict[str, object]] = []
    for fold_dir in sorted(path for path in prototype_root.iterdir() if path.is_dir() and path.name.startswith("fold_")):
        metrics = json.loads((fold_dir / "prototype_metrics.json").read_text(encoding="utf-8"))
        subject = str(metrics["held_out_subject"])
        assignments = np.load(fold_dir / "prototype_assignments.npy").astype(np.float32)
        with (fold_dir / "prototype_predictions.csv").open("r", encoding="utf-8", newline="") as handle:
            predictions = list(csv.DictReader(handle))
        if len(predictions) != assignments.shape[0]:
            raise ValueError(f"Prediction/assignment row mismatch in {fold_dir}")
        assignments_all.append(assignments)
        for row in predictions:
            sample_rows.append(
                {
                    "fold": fold_dir.name,
                    "subject": subject,
                    "nsdId": int(row["nsdId"]),
                }
            )
    return np.concatenate(assignments_all, axis=0), sample_rows


def qualitative_note(n_unique_top_images: int, recurrence_score: float, subject_coverage: int, top_n: int) -> str:
    if recurrence_score >= 0.25 and n_unique_top_images <= int(top_n * 0.7):
        return "recurrent_across_subjects"
    if recurrence_score < 0.1 and n_unique_top_images >= int(top_n * 0.9):
        return "diffuse_or_subject_specific"
    if subject_coverage <= 2:
        return "limited_subject_coverage"
    return "mixed"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments, sample_rows = load_samples(args.prototype_root)
    n_samples, n_units = assignments.shape
    top_examples_rows: list[dict[str, object]] = []
    selectivity_rows: list[dict[str, object]] = []

    for unit_id in range(n_units):
        unit_values = assignments[:, unit_id]
        top_idx = np.argsort(-unit_values)[: args.top_n]
        top_values = unit_values[top_idx]
        top_samples = [sample_rows[int(idx)] for idx in top_idx.tolist()]
        unique_images = sorted({int(sample["nsdId"]) for sample in top_samples})
        subjects = [str(sample["subject"]) for sample in top_samples]
        unique_subjects = sorted(set(subjects))
        image_subject_counts: dict[int, set[str]] = {}
        for sample in top_samples:
            image_subject_counts.setdefault(int(sample["nsdId"]), set()).add(str(sample["subject"]))
        recurrence_score = float(
            np.mean([len(subject_set) / 8.0 for subject_set in image_subject_counts.values()])
        ) if image_subject_counts else 0.0

        for rank, (idx, activation_value) in enumerate(zip(top_idx.tolist(), top_values.tolist()), start=1):
            sample = sample_rows[int(idx)]
            top_examples_rows.append(
                {
                    "unit_id": unit_id,
                    "rank": rank,
                    "nsdId": int(sample["nsdId"]),
                    "subject": str(sample["subject"]),
                    "fold": str(sample["fold"]),
                    "activation_value": float(activation_value),
                }
            )

        selectivity_rows.append(
            {
                "unit_id": unit_id,
                "n_unique_top_images": len(unique_images),
                "cross_subject_recurrence_score": recurrence_score,
                "mean_top_activation": float(np.mean(top_values)) if len(top_values) else 0.0,
                "subject_coverage_top_images": len(unique_subjects),
                "qualitative_note": qualitative_note(
                    n_unique_top_images=len(unique_images),
                    recurrence_score=recurrence_score,
                    subject_coverage=len(unique_subjects),
                    top_n=args.top_n,
                ),
            }
        )

    write_csv(
        args.output_dir / "unit_top_images_examples.csv",
        top_examples_rows,
        ["unit_id", "rank", "nsdId", "subject", "fold", "activation_value"],
    )
    write_csv(
        args.output_dir / "unit_selectivity_summary.csv",
        selectivity_rows,
        [
            "unit_id",
            "n_unique_top_images",
            "cross_subject_recurrence_score",
            "mean_top_activation",
            "subject_coverage_top_images",
            "qualitative_note",
        ],
    )


if __name__ == "__main__":
    main()
