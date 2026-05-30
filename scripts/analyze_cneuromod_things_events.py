#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from itertools import combinations
from pathlib import Path

import pandas as pd


SUB_RE = re.compile(r"(sub-\d+)")
SES_RE = re.compile(r"(ses-\d+)")
RUN_RE = re.compile(r"(run-\d+)")


IMAGE_ID_COLUMNS = (
    "things_image_nr",
    "image_nr",
    "test_image_nr",
    "image_path",
)


def parse_path_ids(path: Path) -> dict[str, str]:
    text = str(path)
    sub = SUB_RE.search(text)
    ses = SES_RE.search(text)
    run = RUN_RE.search(text)
    return {
        "path_subject": sub.group(1) if sub else "",
        "path_session": ses.group(1) if ses else "",
        "path_run": run.group(1) if run else "",
    }


def find_image_column(columns: list[str]) -> str:
    for col in IMAGE_ID_COLUMNS:
        if col in columns:
            return col
    raise ValueError(f"No supported image identifier column found. Columns: {columns}")


def read_events(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    failures = []
    paths = sorted(root.glob("sub-*/ses-*/func/*_events.tsv"))
    for path in paths:
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception as exc:  # Keep going if a DataLad symlink is still unavailable.
            failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        ids = parse_path_ids(path)
        for key, value in ids.items():
            df[key] = value
        df["source_file"] = str(path)
        rows.append(df)
    if not rows:
        raise RuntimeError(f"No readable event TSVs found under {root}")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(failures)


def normalize_subject(values: pd.Series, fallback: pd.Series) -> pd.Series:
    if values.notna().any():
        return values.astype("Int64").astype(str).map(lambda x: f"sub-{int(x):02d}" if x != "<NA>" else "")
    return fallback.astype(str)


def bool_false_mask(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.lower()
    return ~(normalized.isin({"true", "1", "1.0", "yes"}))


def summarize(root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    events, failures = read_events(root)
    image_col = find_image_column(list(events.columns))

    events["subject"] = normalize_subject(events.get("subject_id", pd.Series(dtype="float")), events["path_subject"])
    events["session"] = events.get("session_id", events["path_session"]).astype(str)
    events["run"] = events.get("run_id", events["path_run"]).astype(str)
    events["image_key"] = events[image_col].astype("Int64").astype(str)

    filtered = events.copy()
    if "ran" in filtered.columns:
        filtered = filtered[filtered["ran"].fillna(0).astype(float) == 1.0]
    if "not_for_memory" in filtered.columns:
        filtered = filtered[bool_false_mask(filtered["not_for_memory"])]
    filtered = filtered[filtered["image_key"].notna() & (filtered["image_key"] != "<NA>")]

    subject_summary = []
    for subject, sub_df in filtered.groupby("subject"):
        by_image = sub_df.groupby("image_key")
        count_by_image = by_image.size()
        if "repetition" in sub_df.columns:
            n_reps = by_image["repetition"].nunique(dropna=True)
            strict_t3 = count_by_image[(count_by_image == 3) & (n_reps == 3)]
        else:
            n_reps = pd.Series(index=count_by_image.index, data=pd.NA)
            strict_t3 = count_by_image[count_by_image == 3]
        subject_summary.append(
            {
                "subject": subject,
                "event_files": int(sub_df["source_file"].nunique()),
                "sessions": int(sub_df["session"].nunique()),
                "runs": int(sub_df["run"].nunique()),
                "trials_after_filters": int(len(sub_df)),
                "unique_images": int(count_by_image.shape[0]),
                "strict_t3_images": int(strict_t3.shape[0]),
                "images_with_ge3_trials": int((count_by_image >= 3).sum()),
                "images_with_repetition_1_2_3": int((n_reps == 3).sum()) if "repetition" in sub_df.columns else pd.NA,
            }
        )
    subject_summary_df = pd.DataFrame(subject_summary).sort_values("subject")

    per_subject_image = (
        filtered.groupby(["image_key", "subject"])
        .agg(
            n_trials=("image_key", "size"),
            n_repetitions=("repetition", "nunique") if "repetition" in filtered.columns else ("image_key", "size"),
            sessions=("session", "nunique"),
            runs=("run", "nunique"),
        )
        .reset_index()
    )
    per_subject_image["is_strict_t3"] = (per_subject_image["n_trials"] == 3) & (
        per_subject_image["n_repetitions"] == 3
    )

    strict = per_subject_image[per_subject_image["is_strict_t3"]].copy()
    shared = strict.groupby("image_key")["subject"].agg(lambda x: ",".join(sorted(x))).reset_index()
    shared["n_subjects"] = shared["subject"].str.count(",") + 1
    shared = shared.sort_values(["n_subjects", "image_key"], ascending=[False, True])

    shared_counts = []
    for n_subjects in range(2, int(shared["n_subjects"].max()) + 1 if len(shared) else 2):
        sub = shared[shared["n_subjects"] >= n_subjects]
        positive_pairs = 0
        for subjects in sub["subject"].str.split(","):
            positive_pairs += sum(1 for _ in combinations(subjects, 2)) * 3
        shared_counts.append(
            {
                "minimum_subjects_with_strict_t3": n_subjects,
                "images": int(len(sub)),
                "repeat_matched_cross_subject_positive_pairs": int(positive_pairs),
            }
        )
    shared_counts_df = pd.DataFrame(shared_counts)

    raw_summary = pd.DataFrame(
        [
            {
                "event_root": str(root),
                "image_id_column": image_col,
                "event_paths_total": int(len(list(root.glob("sub-*/ses-*/func/*_events.tsv")))),
                "event_files_readable": int(events["source_file"].nunique()),
                "event_files_failed": int(len(failures)),
                "raw_rows": int(len(events)),
                "rows_after_filters": int(len(filtered)),
                "subjects_after_filters": int(filtered["subject"].nunique()),
                "unique_images_after_filters": int(filtered["image_key"].nunique()),
            }
        ]
    )

    raw_summary.to_csv(out_dir / "cneuromod_things_event_overview.csv", index=False)
    subject_summary_df.to_csv(out_dir / "cneuromod_things_subject_summary.csv", index=False)
    per_subject_image.to_csv(out_dir / "cneuromod_things_per_subject_image_counts.csv", index=False)
    shared.to_csv(out_dir / "cneuromod_things_strict_t3_shared_images.csv", index=False)
    shared_counts_df.to_csv(out_dir / "cneuromod_things_shared_strict_t3_counts.csv", index=False)
    failures.to_csv(out_dir / "cneuromod_things_event_read_failures.csv", index=False)

    lines = [
        "# CNeuroMod-THINGS Event QC",
        "",
        "This is a metadata-only feasibility check for external strict T=3 cross-subject validation.",
        "",
        "## Overview",
        "",
        raw_summary.to_markdown(index=False),
        "",
        "## Per-Subject Counts After Filters",
        "",
        subject_summary_df.to_markdown(index=False),
        "",
        "## Shared Strict-T3 Images",
        "",
        shared_counts_df.to_markdown(index=False) if len(shared_counts_df) else "No shared strict-T3 images found.",
        "",
        "## Filter Policy",
        "",
        "- Keep trials with `ran == 1` when available.",
        "- Exclude trials with `not_for_memory == True` when available.",
        f"- Use `{image_col}` as the canonical image identifier.",
        "- A strict-T3 image has exactly three trials and three distinct repetition labels for a subject.",
        "",
    ]
    if len(failures):
        lines.extend(
            [
                "## Read Failures",
                "",
                f"{len(failures)} event files were not readable, likely because their DataLad content was not fetched yet.",
                "",
            ]
        )
    (out_dir / "cneuromod_things_event_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CNeuroMod-THINGS event TSVs for strict T=3 validation.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/"
            "cneuromod_things/metadata_repo/THINGS/fmriprep/sourcedata/things"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/cneuromod_things/events_qc"),
    )
    args = parser.parse_args()
    summarize(args.root, args.out_dir)
    print(args.out_dir)


if __name__ == "__main__":
    main()
