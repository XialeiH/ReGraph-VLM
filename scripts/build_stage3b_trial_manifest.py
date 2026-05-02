#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Stage 3B trial-level manifest from subject exports.")
    parser.add_argument("--shared-manifest", type=Path, default=Path("preproc_v0/shared1000_manifest.csv"))
    parser.add_argument("--trial-root", type=Path, required=True, help="Stage 3B trial-level output root.")
    parser.add_argument("--dataset-view", type=str, default="all8_trial_ge1_907")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.trial_root.mkdir(parents=True, exist_ok=True)

    shared_manifest = pd.read_csv(args.shared_manifest)
    usable = shared_manifest[shared_manifest["usable"] == True].copy()

    per_subject_sets = {
        subject: set(usable.loc[usable["subject"] == subject, "nsdId"].astype(int).tolist())
        for subject in SUBJECTS
    }
    image_universe = sorted(set.intersection(*per_subject_sets.values()))
    image_universe_path = args.trial_root / f"{args.dataset_view}_image_universe.csv"
    pd.DataFrame({"nsdId": image_universe}).to_csv(image_universe_path, index=False)

    manifest_rows: list[dict[str, object]] = []
    subject_summary_rows: list[dict[str, object]] = []
    for subject in SUBJECTS:
        metadata_path = args.trial_root / f"{subject}_triallevel_shared907_metadata.csv"
        feature_path = args.trial_root / f"{subject}_triallevel_shared907_roi_features.npz"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata for {subject}: {metadata_path}")
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing feature NPZ for {subject}: {feature_path}")

        metadata = pd.read_csv(metadata_path)
        metadata = metadata[metadata["nsdId"].isin(image_universe)].copy()
        metadata = metadata.sort_values(["nsdId", "rep_index_for_subject", "session", "trial_in_session"]).reset_index(drop=True)

        for row in metadata.itertuples(index=False):
            manifest_rows.append(
                {
                    "dataset_view": args.dataset_view,
                    "subject": subject,
                    "nsdId": int(row.nsdId),
                    "session": int(row.session),
                    "trial_in_session": int(row.trial_in_session),
                    "rep_index_for_subject": int(row.rep_index_for_subject),
                    "sample_index": int(row.sample_index),
                    "feature_path": str(feature_path.resolve()),
                    "metadata_path": str(metadata_path.resolve()),
                    "usable": True,
                }
            )

        counts = metadata.groupby("nsdId").size()
        subject_summary_rows.append(
            {
                "subject": subject,
                "n_trial_samples": int(len(metadata)),
                "n_unique_images": int(metadata["nsdId"].nunique()),
                "mean_trials_per_image": float(counts.mean()),
                "min_trials_per_image": int(counts.min()),
                "max_trials_per_image": int(counts.max()),
            }
        )

    manifest_path = args.trial_root / f"{args.dataset_view}_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset_view",
                "subject",
                "nsdId",
                "session",
                "trial_in_session",
                "rep_index_for_subject",
                "sample_index",
                "feature_path",
                "metadata_path",
                "usable",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_path = args.trial_root / f"{args.dataset_view}_subject_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "subject",
                "n_trial_samples",
                "n_unique_images",
                "mean_trials_per_image",
                "min_trials_per_image",
                "max_trials_per_image",
            ],
        )
        writer.writeheader()
        writer.writerows(subject_summary_rows)

    payload = {
        "dataset_view": args.dataset_view,
        "n_subjects": len(SUBJECTS),
        "n_unique_images": len(image_universe),
        "n_total_samples": len(manifest_rows),
        "image_universe_path": str(image_universe_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "subject_summary_path": str(summary_path.resolve()),
    }
    (args.trial_root / f"{args.dataset_view}_manifest_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
