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


FINAL_TABLE_FILES = (
    "final_adjacency_ablation_tests.csv",
    "fold_difficulty_qc.csv",
    "model_parameter_counts.csv",
    "publication_paired_stats.csv",
    "session_order_pair_qc.csv",
    "single_ref_matched_allseed_pairwise_tests.csv",
    "single_ref_matched_allseed_summary.csv",
    "single_ref_matched_summary.csv",
    "split_accounting.csv",
    "table_adjacency_ablation.csv",
    "table_adjacency_perturbation.csv",
    "table_allfold_final.csv",
    "table_edge_bias_followup.csv",
    "table_external_visual_roi_smoke.csv",
    "table_gate_confound.csv",
    "table_graph_only.csv",
    "table_hard_negative_allfold.csv",
    "table_heldout_image.csv",
    "table_lowshot_calibration.csv",
    "table_matched_deletion.csv",
    "table_phase2_sota_graph_baselines.csv",
    "table_roi_token_controls.csv",
    "table_within_subject.csv",
)

EXTERNAL_TABLE_FILES = (
    "laion_fmri_visual_roi_all_runs.csv",
    "laion_fmri_visual_roi_pairwise_tests.csv",
    "laion_fmri_visual_roi_summary.csv",
)

COUNT_COLUMNS = {
    "best_epoch",
    "calibration_images",
    "complete_groups",
    "k",
    "n",
    "n_a_gt_b",
    "n_a_lt_b",
    "n_equal",
    "n_images_total",
    "n_pairs",
    "n_roi",
    "n_test_images",
    "n_train_images",
    "n_val_images",
    "negative",
    "node_feature_dim",
    "num_heads",
    "num_layers",
    "pair",
    "pairs",
    "positive",
    "test_imgs",
    "test_pairs",
    "test_seq",
    "train_pairs",
    "train_seq",
    "trainable_parameters",
    "val_pairs",
    "val_seq",
}

ZERO_ALLOWED_COUNT_COLUMNS = {
    "calibration_images",
    "n_a_gt_b",
    "n_a_lt_b",
    "n_equal",
    "problem_groups",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit numeric ranges in publication result CSV artifacts.")
    parser.add_argument(
        "--final-tables-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument("--external-summary-dir", type=Path, default=Path("external_validation/summary"))
    parser.add_argument("--output-prefix", default="result_value_range_audit")
    return parser.parse_args()


def ready(ok: bool) -> str:
    return "ready" if ok else "incomplete"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value.rstrip("%")) if value.endswith("%") else float(value)
    except ValueError:
        return None


def is_probability_column(column: str) -> bool:
    tokens = ("AUROC", "AUPRC", "R@5", "MRR", "chance")
    return any(token in column for token in tokens) and "diff" not in column and "drop" not in column


def is_probability_drop_column(column: str) -> bool:
    return column.endswith("_drop") and any(token in column for token in ("AUROC", "R@5", "MRR"))


def is_correlation_column(column: str) -> bool:
    return column in {"spearman", "partial_spearman", "repeat_corr"}


def is_pvalue_column(column: str) -> bool:
    return column.endswith("_p") or column in {"paired_p", "paired_t_p", "wilcoxon_p"}


def is_nonnegative_column(column: str) -> bool:
    return (
        column.endswith("_std")
        or column.endswith("_sem")
        or column in {"std_diff", "diff_std", "ci95_half_width", "elapsed_seconds"}
    )


def is_difference_column(column: str) -> bool:
    return column in {
        "bootstrap_ci_high",
        "bootstrap_ci_low",
        "ci95_high",
        "ci95_low",
        "mean_diff",
        "mean_diff_a_minus_b",
        "raw_gap",
    }


def is_count_column(column: str) -> bool:
    return column in COUNT_COLUMNS or column in ZERO_ALLOWED_COUNT_COLUMNS


def check_value(column: str, value: str) -> str | None:
    if value == "":
        return None
    parsed = parse_float(value)
    if parsed is None:
        return None
    if parsed != parsed:
        if is_pvalue_column(column):
            return None
        return f"{column}={value} is NaN"
    if is_probability_column(column) and not 0.0 <= parsed <= 1.0:
        return f"{column}={value} outside [0,1]"
    if is_probability_drop_column(column) and not 0.0 <= parsed <= 1.0:
        return f"{column}={value} outside [0,1]"
    if is_correlation_column(column) and not -1.0 <= parsed <= 1.0:
        return f"{column}={value} outside [-1,1]"
    if is_pvalue_column(column) and not 0.0 <= parsed <= 1.0:
        return f"{column}={value} outside [0,1]"
    if is_nonnegative_column(column) and parsed < 0.0:
        return f"{column}={value} is negative"
    if is_difference_column(column) and not -1.0 <= parsed <= 1.0:
        return f"{column}={value} outside [-1,1]"
    if is_count_column(column):
        minimum = 0.0 if column in ZERO_ALLOWED_COUNT_COLUMNS else 1.0
        if parsed < minimum or not parsed.is_integer():
            return f"{column}={value} is not an integer >= {int(minimum)}"
    return None


def check_split_invariants(rows: list[dict[str, str]]) -> str | None:
    for row in rows:
        try:
            train_seq = int(float(row["train_seq"]))
            val_seq = int(float(row["val_seq"]))
            test_seq = int(float(row["test_seq"]))
            train_pairs = int(float(row["train_pairs"]))
            val_pairs = int(float(row["val_pairs"]))
            test_pairs = int(float(row["test_pairs"]))
        except (KeyError, ValueError):
            return "split accounting columns are not parseable integers"
        if train_pairs != train_seq * 6 or val_pairs != val_seq * 6 or test_pairs != test_seq * 6:
            return f"{row.get('fold', 'row')}: pair count does not equal strict T=3 sequences x 6"
    return None


def check_session_invariants(rows: list[dict[str, str]]) -> str | None:
    for row in rows:
        try:
            pairs = int(float(row["pairs"]))
            positive = int(float(row["positive"]))
            negative = int(float(row["negative"]))
            problem_groups = int(float(row["problem_groups"]))
            anchor_match = float(row["anchor_match"].rstrip("%"))
        except (KeyError, ValueError):
            return "session/order QC columns are not parseable"
        if positive + negative != pairs:
            return f"{row.get('split', 'row')}: positive + negative != pairs"
        if problem_groups != 0:
            return f"{row.get('split', 'row')}: problem_groups={problem_groups}"
        if anchor_match != 100.0:
            return f"{row.get('split', 'row')}: anchor_match={anchor_match}%"
    return None


def audit_file(path: Path) -> AuditRow:
    rows = read_csv(path)
    label = path.as_posix()
    if not rows:
        return AuditRow(label, "missing", f"{path} missing or empty")
    checked_values = 0
    for row_index, row in enumerate(rows, start=2):
        for column, value in row.items():
            problem = check_value(column, value.strip())
            if problem:
                return AuditRow(label, "incomplete", f"line {row_index}: {problem}")
            if parse_float(value.strip()) is not None:
                checked_values += 1
    if path.name == "split_accounting.csv":
        problem = check_split_invariants(rows)
        if problem:
            return AuditRow(label, "incomplete", problem)
    if path.name == "session_order_pair_qc.csv":
        problem = check_session_invariants(rows)
        if problem:
            return AuditRow(label, "incomplete", problem)
    return AuditRow(label, "ready", f"{len(rows)} rows, {checked_values} numeric values checked")


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
        "# Result Value Range Audit",
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
    paths = [args.final_tables_dir / name for name in FINAL_TABLE_FILES]
    paths.extend(args.external_summary_dir / name for name in EXTERNAL_TABLE_FILES)
    rows = [audit_file(path) for path in paths]
    write_outputs(args.final_tables_dir, args.output_prefix, rows)
    return 0 if all(row.status == "ready" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
