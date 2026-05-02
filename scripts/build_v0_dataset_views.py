#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FULL1000_SUBJECTS = ["subj01", "subj02", "subj05", "subj07"]
ALL_SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Version 0 dataset-view manifests from subject preprocessing outputs.")
    parser.add_argument("--preproc-root", type=Path, required=True, help="Directory containing shared1000_summary.csv and subject outputs.")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_bool_string(value: bool) -> str:
    return "true" if value else "false"


def load_subject_metadata(
    preproc_root: Path,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[int, dict[int, int]]]]:
    per_subject_rows: dict[str, list[dict[str, str]]] = {}
    per_subject_index: dict[str, dict[int, dict[int, int]]] = {}
    for subject in ALL_SUBJECTS:
        rows = read_csv_rows(preproc_root / f"{subject}_shared1000_metadata.csv")
        per_subject_rows[subject] = rows
        rep_maps = {1: {}, 2: {}, 3: {}}
        for idx, row in enumerate(rows):
            nsd_id = int(row["nsdId"])
            reps = int(row["n_repetitions_used"])
            if reps >= 1:
                rep_maps[1][nsd_id] = idx
            if reps >= 2:
                rep_maps[2][nsd_id] = idx
            if reps >= 3:
                rep_maps[3][nsd_id] = idx
        per_subject_index[subject] = rep_maps
    return per_subject_rows, per_subject_index


def build_qc_rows(
    summary_rows: list[dict[str, str]],
    subject_metadata: dict[str, list[dict[str, str]]],
    subject_index: dict[str, dict[int, dict[int, int]]],
) -> list[dict[str, object]]:
    summary_by_subject = {row["subject"]: row for row in summary_rows}
    rows: list[dict[str, object]] = []
    for subject in ALL_SUBJECTS:
        meta_rows = subject_metadata[subject]
        mean_reps = sum(int(row["n_repetitions_used"]) for row in meta_rows) / len(meta_rows)
        rows.append(
            {
                "subject": subject,
                "n_images_ge1": len(subject_index[subject][1]),
                "n_images_ge2": len(subject_index[subject][2]),
                "n_complete_images": len(subject_index[subject][3]),
                "n_usable_rows": int(summary_by_subject[subject]["n_usable_rows"]),
                "mean_reps_per_exported_image": f"{mean_reps:.2f}",
                "has_full_1000": to_bool_string(len(subject_index[subject][3]) == 1000),
                "included_in_all8_ge1_907": to_bool_string(True),
                "included_in_all8_ge2_766": to_bool_string(True),
                "included_in_all8_ge3_515": to_bool_string(True),
                "included_in_full1000_4subj": to_bool_string(subject in FULL1000_SUBJECTS),
            }
        )
    return rows


def build_threshold_manifest(
    preproc_root: Path,
    subject_metadata: dict[str, list[dict[str, str]]],
    subject_index: dict[str, dict[int, dict[int, int]]],
    subjects: list[str],
    min_repetitions: int,
    tag_name: str,
) -> tuple[list[dict[str, object]], list[int]]:
    nsd_sets = [set(subject_index[subject][min_repetitions].keys()) for subject in subjects]
    selected_nsd_ids = sorted(set.intersection(*nsd_sets))

    rows: list[dict[str, object]] = []
    for nsd_id in selected_nsd_ids:
        for subject in subjects:
            sample_index = subject_index[subject][min_repetitions][nsd_id]
            row = subject_metadata[subject][sample_index]
            rows.append(
                {
                    "nsdId": nsd_id,
                    "subject": subject,
                    "feature_path": str((preproc_root / f"{subject}_shared1000_roi_features.npz").resolve()),
                    "metadata_path": str((preproc_root / f"{subject}_shared1000_metadata.csv").resolve()),
                    "sample_index": sample_index,
                    "n_repetitions_used": int(row["n_repetitions_used"]),
                    tag_name: to_bool_string(True),
                    "usable": to_bool_string(True),
                }
            )
    return rows, selected_nsd_ids


def write_dataset_strategy_note(
    preproc_root: Path,
    ge1_nsd_ids: list[int],
    ge2_nsd_ids: list[int],
    ge3_nsd_ids: list[int],
    full_nsd_ids: list[int],
) -> None:
    lines = [
        "# Dataset Strategy Note",
        "",
        "## Updated Facts",
        "",
        f"- All-8-subject usable intersection with `>=1` repetition per sample: `{len(ge1_nsd_ids)}` images.",
        f"- All-8-subject intersection with `>=2` repetitions per sample: `{len(ge2_nsd_ids)}` images.",
        f"- All-8-subject strict complete-average intersection with `3` repetitions per sample: `{len(ge3_nsd_ids)}` images.",
        f"- `full1000_4subj` strict complete-average view: `{len(full_nsd_ids)}` images x `4` subjects = `{len(full_nsd_ids) * 4}` samples.",
        "",
        "## Why The Earlier `core907` Assumption Broke",
        "",
        "- The exported subject metadata include some samples with 1 or 2 repetitions for `subj03`, `subj04`, `subj06`, and `subj08`.",
        "- So `907` is the all-subject intersection under `>=1 repetition`, not under strict complete 3-repeat averaging.",
        "- The strict complete-average all-subject universe is `515`, not `907`.",
        "",
        "## Recommended Next Decision",
        "",
        "- If Version 0 must preserve all 8 subjects while avoiding single-repeat samples, the most practical main view is `all8_ge2_766`.",
        "- If Version 0 must stay strict about complete 3-repeat averages, the main view should be `all8_ge3_515`.",
        "- `full1000_4subj` should remain a secondary sensitivity analysis view, not the main training universe.",
        "",
        "## Why Not Missing-Sample-Tolerant Training Yet",
        "",
        "- Allowing missing subject-image pairs would complicate split-aware normalization, PCA fitting, batching, and cross-subject comparisons too early.",
        "- Version 0 should still prioritize a clean and interpretable shared-unit validation setup before adding missing-data handling.",
        "",
    ]
    (preproc_root / "dataset_strategy_note.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest_summaries(
    preproc_root: Path,
    ge1_nsd_ids: list[int],
    ge2_nsd_ids: list[int],
    ge3_nsd_ids: list[int],
    full_nsd_ids: list[int],
) -> None:
    summary = {
        "all8_ge1_907": {
            "n_unique_images": len(ge1_nsd_ids),
            "n_subjects": 8,
            "n_total_samples": len(ge1_nsd_ids) * 8,
        },
        "all8_ge2_766": {
            "n_unique_images": len(ge2_nsd_ids),
            "n_subjects": 8,
            "n_total_samples": len(ge2_nsd_ids) * 8,
        },
        "all8_ge3_515": {
            "n_unique_images": len(ge3_nsd_ids),
            "n_subjects": 8,
            "n_total_samples": len(ge3_nsd_ids) * 8,
        },
        "full1000_4subj": {
            "n_unique_images": len(full_nsd_ids),
            "n_subjects": 4,
            "n_total_samples": len(full_nsd_ids) * 4,
        },
    }
    (preproc_root / "dataset_view_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    preproc_root = args.preproc_root

    summary_rows = read_csv_rows(preproc_root / "shared1000_summary.csv")
    subject_metadata, subject_index = load_subject_metadata(preproc_root)

    qc_rows = build_qc_rows(summary_rows, subject_metadata, subject_index)
    write_csv(
        preproc_root / "all_subject_preprocessing_qc.csv",
        qc_rows,
        [
            "subject",
            "n_images_ge1",
            "n_images_ge2",
            "n_complete_images",
            "n_usable_rows",
            "mean_reps_per_exported_image",
            "has_full_1000",
            "included_in_all8_ge1_907",
            "included_in_all8_ge2_766",
            "included_in_all8_ge3_515",
            "included_in_full1000_4subj",
        ],
    )

    ge1_rows, ge1_nsd_ids = build_threshold_manifest(
        preproc_root, subject_metadata, subject_index, ALL_SUBJECTS, 1, "is_all8_ge1_907"
    )
    write_csv(
        preproc_root / "all8_ge1_907_manifest.csv",
        ge1_rows,
        [
            "nsdId",
            "subject",
            "feature_path",
            "metadata_path",
            "sample_index",
            "n_repetitions_used",
            "is_all8_ge1_907",
            "usable",
        ],
    )

    ge2_rows, ge2_nsd_ids = build_threshold_manifest(
        preproc_root, subject_metadata, subject_index, ALL_SUBJECTS, 2, "is_all8_ge2_766"
    )
    write_csv(
        preproc_root / "all8_ge2_766_manifest.csv",
        ge2_rows,
        [
            "nsdId",
            "subject",
            "feature_path",
            "metadata_path",
            "sample_index",
            "n_repetitions_used",
            "is_all8_ge2_766",
            "usable",
        ],
    )

    ge3_rows, ge3_nsd_ids = build_threshold_manifest(
        preproc_root, subject_metadata, subject_index, ALL_SUBJECTS, 3, "is_all8_ge3_515"
    )
    write_csv(
        preproc_root / "all8_ge3_515_manifest.csv",
        ge3_rows,
        [
            "nsdId",
            "subject",
            "feature_path",
            "metadata_path",
            "sample_index",
            "n_repetitions_used",
            "is_all8_ge3_515",
            "usable",
        ],
    )

    full_rows, full_nsd_ids = build_threshold_manifest(
        preproc_root, subject_metadata, subject_index, FULL1000_SUBJECTS, 3, "is_full1000_4subj"
    )
    write_csv(
        preproc_root / "full1000_4subj_manifest.csv",
        full_rows,
        [
            "nsdId",
            "subject",
            "feature_path",
            "metadata_path",
            "sample_index",
            "n_repetitions_used",
            "is_full1000_4subj",
            "usable",
        ],
    )

    write_dataset_strategy_note(preproc_root, ge1_nsd_ids, ge2_nsd_ids, ge3_nsd_ids, full_nsd_ids)
    write_manifest_summaries(preproc_root, ge1_nsd_ids, ge2_nsd_ids, ge3_nsd_ids, full_nsd_ids)

    print(
        json.dumps(
            {
                "all_subject_preprocessing_qc_rows": len(qc_rows),
                "all8_ge1_907": {
                    "n_unique_images": len(ge1_nsd_ids),
                    "n_subjects": 8,
                    "n_total_samples": len(ge1_rows),
                },
                "all8_ge2_766": {
                    "n_unique_images": len(ge2_nsd_ids),
                    "n_subjects": 8,
                    "n_total_samples": len(ge2_rows),
                },
                "all8_ge3_515": {
                    "n_unique_images": len(ge3_nsd_ids),
                    "n_subjects": 8,
                    "n_total_samples": len(ge3_rows),
                },
                "full1000_4subj": {
                    "n_unique_images": len(full_nsd_ids),
                    "n_subjects": 4,
                    "n_total_samples": len(full_rows),
                },
            }
        )
    )


if __name__ == "__main__":
    main()
