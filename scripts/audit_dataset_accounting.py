#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuditRow:
    item: str
    status: str
    evidence: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit publication dataset split and pair-accounting consistency.")
    parser.add_argument("--tex", type=Path, default=Path("reports/neurips_report/may30.tex"))
    parser.add_argument(
        "--final-tables-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument("--output-prefix", default="dataset_accounting_audit")
    return parser.parse_args()


def ready(ok: bool) -> str:
    return "ready" if ok else "incomplete"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_int(row: dict[str, str], column: str) -> int:
    return int(float(row[column]))


def split_rows_have_source(rows: list[dict[str, str]], source_tex: str, table_label: str) -> bool:
    return bool(rows) and all(source_tex in row.get("source", "") and table_label in row.get("source", "") for row in rows)


def audit_manuscript_exists(tex: Path) -> AuditRow:
    return AuditRow("manuscript exists", "ready" if tex.exists() else "missing", str(tex))


def audit_split_rows(rows: list[dict[str, str]]) -> AuditRow:
    folds = [row.get("fold", "") for row in rows]
    expected = [f"fold_{index:02d}" for index in range(1, 9)]
    ok = folds == expected
    return AuditRow(
        "split fold rows",
        ready(ok),
        "8 ordered held-out folds" if ok else f"found {folds}",
    )


def audit_subject_assignment(rows: list[dict[str, str]]) -> AuditRow:
    tests = [row.get("test_subject", "") for row in rows]
    vals = [row.get("val_subject", "") for row in rows]
    expected = {f"subj{index:02d}" for index in range(1, 9)}
    problems = []
    if set(tests) != expected or len(set(tests)) != 8:
        problems.append(f"test_subjects={tests}")
    if any(test == val for test, val in zip(tests, vals)):
        problems.append("validation subject overlaps test subject")
    return AuditRow(
        "held-out subject assignment",
        ready(not problems),
        "each subject is held out once and validation subject differs from test subject" if not problems else "; ".join(problems),
    )


def audit_sequence_partition(rows: list[dict[str, str]]) -> AuditRow:
    subject_seq = {row["test_subject"]: as_int(row, "test_seq") for row in rows}
    total_seq = sum(subject_seq.values())
    problems = []
    for row in rows:
        fold = row.get("fold", "row")
        test_subject = row["test_subject"]
        val_subject = row["val_subject"]
        train_seq = as_int(row, "train_seq")
        val_seq = as_int(row, "val_seq")
        test_seq = as_int(row, "test_seq")
        if test_seq != subject_seq[test_subject]:
            problems.append(f"{fold}: test_seq mismatch")
        if val_seq != subject_seq[val_subject]:
            problems.append(f"{fold}: val_seq mismatch")
        if train_seq + val_seq + test_seq != total_seq:
            problems.append(f"{fold}: train+val+test != total")
    return AuditRow(
        "strict T=3 sequence partition",
        ready(not problems),
        f"all folds partition {total_seq} strict T=3 sequences" if not problems else "; ".join(problems[:6]),
    )


def audit_pair_counts(rows: list[dict[str, str]]) -> AuditRow:
    problems = []
    for row in rows:
        fold = row.get("fold", "row")
        for split in ("train", "val", "test"):
            seq = as_int(row, f"{split}_seq")
            pairs = as_int(row, f"{split}_pairs")
            if pairs != seq * 6:
                problems.append(f"{fold} {split}: {pairs} != {seq} x 6")
    return AuditRow(
        "strict T=3 pair counts",
        ready(not problems),
        "all pair counts equal sequence counts x 6" if not problems else "; ".join(problems[:6]),
    )


def audit_test_image_counts(rows: list[dict[str, str]]) -> AuditRow:
    problems = [
        f"{row.get('fold', 'row')}: test_imgs != test_seq"
        for row in rows
        if as_int(row, "test_imgs") != as_int(row, "test_seq")
    ]
    return AuditRow(
        "test image counts",
        ready(not problems),
        "test image counts equal held-out strict T=3 sequence counts" if not problems else "; ".join(problems),
    )


def audit_session_rows(rows: list[dict[str, str]]) -> AuditRow:
    splits = [row.get("split", "") for row in rows]
    ok = splits == ["Train", "Val", "Test", "All"]
    return AuditRow(
        "session/order QC rows",
        ready(ok),
        "Train, Val, Test, and All rows present" if ok else f"found {splits}",
    )


def audit_session_balance(rows: list[dict[str, str]]) -> AuditRow:
    problems = []
    for row in rows:
        split = row.get("split", "row")
        pairs = as_int(row, "pairs")
        positive = as_int(row, "positive")
        negative = as_int(row, "negative")
        complete_groups = as_int(row, "complete_groups")
        if positive + negative != pairs:
            problems.append(f"{split}: positive+negative != pairs")
        if positive != negative:
            problems.append(f"{split}: positives != negatives")
        if complete_groups != positive:
            problems.append(f"{split}: complete_groups != positives")
    return AuditRow(
        "session/order positive-negative balance",
        ready(not problems),
        "positive and negative counts are balanced with complete groups" if not problems else "; ".join(problems),
    )


def audit_session_anchor_match(rows: list[dict[str, str]]) -> AuditRow:
    problems = []
    for row in rows:
        split = row.get("split", "row")
        if as_int(row, "problem_groups") != 0:
            problems.append(f"{split}: problem_groups != 0")
        if row.get("anchor_match") != "100%":
            problems.append(f"{split}: anchor_match != 100%")
    return AuditRow(
        "session/order anchor matching",
        ready(not problems),
        "all splits have zero problem groups and 100% anchor matching" if not problems else "; ".join(problems),
    )


def audit_session_totals(split_rows: list[dict[str, str]], session_rows: list[dict[str, str]]) -> AuditRow:
    totals = {
        "Train": sum(as_int(row, "train_pairs") for row in split_rows),
        "Val": sum(as_int(row, "val_pairs") for row in split_rows),
        "Test": sum(as_int(row, "test_pairs") for row in split_rows),
    }
    totals["All"] = totals["Train"] + totals["Val"] + totals["Test"]
    observed = {row["split"]: as_int(row, "pairs") for row in session_rows}
    problems = [f"{split}: observed {observed.get(split)} != {value}" for split, value in totals.items() if observed.get(split) != value]
    return AuditRow(
        "session/order totals match split accounting",
        ready(not problems),
        "session/order pair totals equal split-accounting sums" if not problems else "; ".join(problems),
    )


def audit_source_columns(split_rows: list[dict[str, str]], session_rows: list[dict[str, str]], tex: Path) -> AuditRow:
    tex_path = tex.as_posix()
    split_ok = split_rows_have_source(split_rows, tex_path, "tab:split_accounting")
    session_ok = split_rows_have_source(session_rows, tex_path, "tab:session_order_pair_qc")
    return AuditRow(
        "dataset accounting source columns",
        ready(split_ok and session_ok),
        "split and session QC rows cite manuscript table labels" if split_ok and session_ok else f"split_ok={split_ok}; session_ok={session_ok}",
    )


def audit_manuscript_split_table(tex_text: str, split_rows: list[dict[str, str]], session_rows: list[dict[str, str]]) -> AuditRow:
    missing = []
    for row in split_rows:
        fold = row["fold"].replace("_", "\\_")
        values = [
            fold,
            row["test_subject"].replace("subj", "subj"),
            row["val_subject"].replace("subj", "subj"),
            row["train_seq"],
            row["val_seq"],
            row["test_seq"],
            row["train_pairs"],
            row["val_pairs"],
            row["test_pairs"],
            row["test_imgs"],
        ]
        if not all(value in tex_text for value in values):
            missing.append(row["fold"])
    for row in session_rows:
        values = []
        for column in ("pairs", "positive", "negative", "complete_groups", "problem_groups", "anchor_match"):
            value = str(row[column])
            values.append(value.replace("%", "\\%"))
        if not all(value in tex_text for value in values):
            missing.append(f"session {row['split']}")
    return AuditRow(
        "manuscript dataset table values",
        ready(not missing),
        "split and session/order table values are present in manuscript text" if not missing else "; ".join(missing[:6]),
    )


def audit_manuscript_dataset_claims(tex_text: str) -> AuditRow:
    required = (
        "Strict T=3 repetition dataset",
        "No fold has train--test subject overlap",
        "train--test NSD image-overlap rate is 0",
        "pair counts include one positive and one matched negative per anchor trial",
        "anchor-side session/order control",
        "does not define a single reference session",
        "stricter single-reference session-matched evaluation",
    )
    missing = [fragment for fragment in required if fragment not in tex_text]
    return AuditRow(
        "manuscript dataset-accounting claims",
        ready(not missing),
        "required split/session accounting caveats are stated" if not missing else "; ".join(missing),
    )


def write_outputs(output_dir: Path, output_prefix: str, rows: list[AuditRow]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{output_prefix}.csv"
    md_path = output_dir / f"{output_prefix}.md"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item", "status", "evidence"], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({"item": row.item, "status": row.status, "evidence": row.evidence})
    counts = {status: sum(1 for row in rows if row.status == status) for status in sorted({row.status for row in rows})}
    lines = [
        "# Dataset Accounting Audit",
        "",
        f"Status counts: {counts}",
        "",
        "| Item | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row.item} | {row.status} | {row.evidence} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"), end="")


def main() -> int:
    args = parse_args()
    split_rows = read_csv(args.final_tables_dir / "split_accounting.csv")
    session_rows = read_csv(args.final_tables_dir / "session_order_pair_qc.csv")
    tex_text = read_text(args.tex)
    rows = [
        audit_manuscript_exists(args.tex),
        audit_split_rows(split_rows),
        audit_subject_assignment(split_rows),
        audit_sequence_partition(split_rows),
        audit_pair_counts(split_rows),
        audit_test_image_counts(split_rows),
        audit_session_rows(session_rows),
        audit_session_balance(session_rows),
        audit_session_anchor_match(session_rows),
        audit_session_totals(split_rows, session_rows),
        audit_source_columns(split_rows, session_rows, args.tex),
        audit_manuscript_split_table(tex_text, split_rows, session_rows),
        audit_manuscript_dataset_claims(tex_text),
    ]
    write_outputs(args.final_tables_dir, args.output_prefix, rows)
    return 0 if all(row.status == "ready" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
