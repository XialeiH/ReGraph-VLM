#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 1 analysis for shared-unit usage and cross-subject consistency.")
    parser.add_argument("--prototype-root", type=Path, required=True)
    parser.add_argument("--b4-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def centered_row_normalize(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=1, keepdims=True)
    return row_normalize(centered)


def load_fold_representation(root: Path, representation_file: str) -> dict[str, dict[str, object]]:
    subject_data: dict[str, dict[str, object]] = {}
    for fold_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("fold_")):
        metrics = json.loads((fold_dir / ("prototype_metrics.json" if "prototype" in representation_file else "b4_metrics.json")).read_text(encoding="utf-8"))
        predictions_path = fold_dir / ("prototype_predictions.csv" if "prototype" in representation_file else "b4_predictions.csv")
        predictions = read_csv(predictions_path)
        reps = np.load(fold_dir / representation_file).astype(np.float32)
        subject = str(metrics["held_out_subject"])
        nsd_ids = np.array([int(row["nsdId"]) for row in predictions], dtype=np.int64)
        subject_data[subject] = {
            "fold": fold_dir.name,
            "subject": subject,
            "nsd_ids": nsd_ids,
            "reps": reps,
        }
    return subject_data


def align_subject_representations(subject_data: dict[str, dict[str, object]]) -> tuple[list[str], np.ndarray, dict[str, str]]:
    subjects = sorted(subject_data.keys())
    common_ids = sorted(set.intersection(*[set(item["nsd_ids"].tolist()) for item in subject_data.values()]))
    aligned = []
    fold_by_subject: dict[str, str] = {}
    for subject in subjects:
        item = subject_data[subject]
        fold_by_subject[subject] = str(item["fold"])
        index = {int(nsd_id): idx for idx, nsd_id in enumerate(item["nsd_ids"].tolist())}
        aligned.append(item["reps"][[index[nsd_id] for nsd_id in common_ids]])
    return subjects, np.stack(aligned, axis=0), fold_by_subject


def compute_unit_usage(prototype_root: Path, output_dir: Path) -> None:
    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    all_assignments: list[np.ndarray] = []
    fold_names: list[str] = []

    for fold_dir in sorted(path for path in prototype_root.iterdir() if path.is_dir() and path.name.startswith("fold_")):
        assignments = np.load(fold_dir / "prototype_assignments.npy").astype(np.float32)
        all_assignments.append(assignments)
        fold_names.append(fold_dir.name)
        top_order = np.argsort(-assignments, axis=1)
        top1 = top_order[:, 0]
        top3 = top_order[:, :3]
        for unit_id in range(assignments.shape[1]):
            fold_rows.append(
                {
                    "fold": fold_dir.name,
                    "unit_id": unit_id,
                    "mean_activation": float(assignments[:, unit_id].mean()),
                    "top1_count": int(np.sum(top1 == unit_id)),
                    "top3_count": int(np.sum(top3 == unit_id)),
                }
            )

    all_concat = np.concatenate(all_assignments, axis=0)
    top_order = np.argsort(-all_concat, axis=1)
    top1 = top_order[:, 0]
    top3 = top_order[:, :3]
    top5 = top_order[:, :5]
    entropy_contrib = -(all_concat * np.log(np.clip(all_concat, 1e-12, None)))
    total_samples = int(all_concat.shape[0])
    for unit_id in range(all_concat.shape[1]):
        top1_count = int(np.sum(top1 == unit_id))
        top3_count = int(np.sum(top3 == unit_id))
        top5_count = int(np.sum(top5 == unit_id))
        summary_rows.append(
            {
                "unit_id": unit_id,
                "mean_activation": float(all_concat[:, unit_id].mean()),
                "top1_count": top1_count,
                "top3_count": top3_count,
                "top5_count": top5_count,
                "usage_fraction": float(top1_count / total_samples),
                "activation_entropy_contrib": float(entropy_contrib[:, unit_id].mean()),
                "dead_or_not": int(top5_count == 0),
            }
        )

    write_csv(
        output_dir / "unit_usage_summary.csv",
        summary_rows,
        [
            "unit_id",
            "mean_activation",
            "top1_count",
            "top3_count",
            "top5_count",
            "usage_fraction",
            "activation_entropy_contrib",
            "dead_or_not",
        ],
    )
    write_csv(
        output_dir / "unit_usage_foldwise.csv",
        fold_rows,
        ["fold", "unit_id", "mean_activation", "top1_count", "top3_count"],
    )


def compute_consistency(prototype_root: Path, b4_root: Path, output_dir: Path) -> None:
    representations = {
        "prototype": load_fold_representation(prototype_root, "prototype_assignments.npy"),
        "b4_hidden": load_fold_representation(b4_root, "b4_hidden.npy"),
    }
    summary_rows: list[dict[str, object]] = []
    detailed_rows: list[dict[str, object]] = []

    for representation_type, subject_data in representations.items():
        subjects, aligned, fold_by_subject = align_subject_representations(subject_data)
        common_ids = sorted(set.intersection(*[set(item["nsd_ids"].tolist()) for item in subject_data.values()]))
        metric_inputs = {
            "cosine": row_normalize(aligned.reshape(-1, aligned.shape[-1])).reshape(aligned.shape),
            "correlation": centered_row_normalize(aligned.reshape(-1, aligned.shape[-1])).reshape(aligned.shape),
        }
        n_images = len(common_ids)

        for metric_name, normalized in metric_inputs.items():
            for anchor_idx, subject in enumerate(subjects):
                same_accumulator = np.zeros(n_images, dtype=np.float64)
                diff_accumulator = np.zeros(n_images, dtype=np.float64)
                other_count = 0
                anchor = normalized[anchor_idx]
                for other_idx, other_subject in enumerate(subjects):
                    if other_idx == anchor_idx:
                        continue
                    other = normalized[other_idx]
                    sim = anchor @ other.T
                    diag = np.diag(sim)
                    diff = (sim.sum(axis=1) - diag) / max(n_images - 1, 1)
                    same_accumulator += diag
                    diff_accumulator += diff
                    other_count += 1
                same_mean = same_accumulator / max(other_count, 1)
                diff_mean = diff_accumulator / max(other_count, 1)

                for nsd_id, same_value, diff_value in zip(common_ids, same_mean.tolist(), diff_mean.tolist()):
                    detailed_rows.append(
                        {
                            "fold": fold_by_subject[subject],
                            "held_out_subject": subject,
                            "nsdId": int(nsd_id),
                            "representation_type": representation_type,
                            "metric": metric_name,
                            "same_image_mean_similarity": float(same_value),
                            "diff_image_mean_similarity": float(diff_value),
                            "gap": float(same_value - diff_value),
                            "n_other_subjects": other_count,
                        }
                    )

                summary_rows.append(
                    {
                        "fold": fold_by_subject[subject],
                        "held_out_subject": subject,
                        "representation_type": representation_type,
                        "metric": metric_name,
                        "same_image_mean_similarity": float(np.mean(same_mean)),
                        "diff_image_mean_similarity": float(np.mean(diff_mean)),
                        "gap": float(np.mean(same_mean) - np.mean(diff_mean)),
                        "n_images": n_images,
                        "n_other_subjects": other_count,
                    }
                )

    write_csv(
        output_dir / "cross_subject_unit_consistency.csv",
        summary_rows,
        [
            "fold",
            "held_out_subject",
            "representation_type",
            "metric",
            "same_image_mean_similarity",
            "diff_image_mean_similarity",
            "gap",
            "n_images",
            "n_other_subjects",
        ],
    )
    write_csv(
        output_dir / "same_vs_diff_image_similarity.csv",
        detailed_rows,
        [
            "fold",
            "held_out_subject",
            "nsdId",
            "representation_type",
            "metric",
            "same_image_mean_similarity",
            "diff_image_mean_similarity",
            "gap",
            "n_other_subjects",
        ],
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compute_unit_usage(args.prototype_root, args.output_dir)
    compute_consistency(args.prototype_root, args.b4_root, args.output_dir)


if __name__ == "__main__":
    main()
