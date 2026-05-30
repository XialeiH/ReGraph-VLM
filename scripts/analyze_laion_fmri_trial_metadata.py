#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


BASE_URL = "https://laion-fmri.s3.amazonaws.com"
DEFAULT_SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and summarize public LAION-fMRI trial TSV metadata.")
    parser.add_argument("--output-dir", type=Path, default=Path("external_validation/laion_fmri_probe/trial_metadata"))
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--sessions", nargs="+", default=[f"ses-{idx:02d}" for idx in range(1, 31)])
    return parser.parse_args()


def trial_tsv_url(subject: str, session: str) -> str:
    name = f"{subject}_{session}_task-images_desc-SingletrialBetas_trials.tsv"
    return f"{BASE_URL}/derivatives/glmsingle-tedana/{subject}/{session}/func/{name}"


def download(url: str, path: Path) -> bool:
    if path.exists() and path.stat().st_size > 0:
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in {403, 404}:
            return False
        raise
    path.write_bytes(data)
    return True


def read_rows(path: Path, subject: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = []
        for row in reader:
            row["subject"] = subject
            rows.append(row)
        return rows


def summarize(rows: list[dict[str, str]], subjects: list[str]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_subject: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_subject[row["subject"]][row["label"]] += 1

    subject_rows: list[dict[str, object]] = []
    for subject in subjects:
        counts = by_subject[subject]
        repeat_counts = Counter(counts.values())
        subject_rows.append(
            {
                "subject": subject,
                "n_trials": sum(counts.values()),
                "n_unique_labels": len(counts),
                "n_labels_ge2": sum(1 for value in counts.values() if value >= 2),
                "n_labels_ge3": sum(1 for value in counts.values() if value >= 3),
                "n_labels_ge6": sum(1 for value in counts.values() if value >= 6),
                "n_labels_ge12": sum(1 for value in counts.values() if value >= 12),
                "max_repeats": max(counts.values()) if counts else 0,
                "repeat_count_hist": ";".join(f"{key}:{repeat_counts[key]}" for key in sorted(repeat_counts)),
            }
        )

    all_labels = sorted(set().union(*(set(by_subject[subject]) for subject in subjects)))
    cross_rows: list[dict[str, object]] = []
    thresholds = [1, 2, 3, 6, 12]
    for threshold in thresholds:
        labels = [
            label
            for label in all_labels
            if all(by_subject[subject].get(label, 0) >= threshold for subject in subjects)
        ]
        cross_rows.append(
            {
                "criterion": f"label_count_ge{threshold}_in_all_subjects",
                "n_labels": len(labels),
                "example_labels": ", ".join(labels[:10]),
            }
        )
    labels_any_shared = [label for label in all_labels if sum(subject in by_subject and label in by_subject[subject] for subject in subjects) >= 2]
    cross_rows.append(
        {
            "criterion": "label_seen_in_at_least_two_subjects",
            "n_labels": len(labels_any_shared),
            "example_labels": ", ".join(labels_any_shared[:10]),
        }
    )
    return subject_rows, cross_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, subject_rows: list[dict[str, object]], cross_rows: list[dict[str, object]], n_files: int) -> None:
    lines = [
        "# LAION-fMRI Trial Metadata Summary",
        "",
        "This analysis downloads only public SingletrialBetas trial TSV metadata from LAION-fMRI. It does not download raw stimulus images or beta NIfTI files.",
        "",
        f"Downloaded/read trial TSV files: {n_files}",
        "",
        "## Per-subject counts",
        "",
        "| Subject | Trials | Unique labels | Labels >=3 repeats | Labels >=12 repeats | Max repeats |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in subject_rows:
        lines.append(
            f"| {row['subject']} | {row['n_trials']} | {row['n_unique_labels']} | {row['n_labels_ge3']} | {row['n_labels_ge12']} | {row['max_repeats']} |"
        )
    lines.extend(["", "## Cross-subject shared-label counts", "", "| Criterion | Labels | Example labels |", "| --- | ---: | --- |"])
    for row in cross_rows:
        lines.append(f"| {row['criterion']} | {row['n_labels']} | {row['example_labels']} |")
    lines.extend(
        [
            "",
            "Interpretation: labels with at least three repeats in every subject are the most direct candidate set for a strict repeated-image external validation. A full validation still needs ROI extraction from the public single-trial beta maps and careful stimulus/metadata alignment.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    tsv_dir = args.output_dir / "tsv"
    all_rows: list[dict[str, str]] = []
    n_files = 0
    missing: list[str] = []
    for subject in args.subjects:
        for session in args.sessions:
            url = trial_tsv_url(subject, session)
            path = tsv_dir / subject / f"{subject}_{session}_trials.tsv"
            if not download(url, path):
                missing.append(url)
                continue
            n_files += 1
            all_rows.extend(read_rows(path, subject))

    subject_rows, cross_rows = summarize(all_rows, args.subjects)
    write_csv(args.output_dir / "laion_trial_metadata_by_subject.csv", subject_rows)
    write_csv(args.output_dir / "laion_trial_metadata_cross_subject.csv", cross_rows)
    write_markdown(args.output_dir / "laion_trial_metadata_summary.md", subject_rows, cross_rows, n_files)
    if missing:
        (args.output_dir / "missing_trial_tsv_urls.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")
    print((args.output_dir / "laion_trial_metadata_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
