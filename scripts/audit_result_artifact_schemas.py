#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SchemaSpec:
    path: Path
    min_rows: int
    required_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...] = ()
    required_nonempty_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuditRow:
    item: str
    status: str
    evidence: str


SUMMARY_COLUMNS = (
    "model",
    "n",
    "source",
    "AUROC_mean",
    "AUROC_std",
    "AUPRC_mean",
    "AUPRC_std",
    "R@5_mean",
    "R@5_std",
    "MRR_mean",
    "MRR_std",
    "image_R@5_mean",
    "image_R@5_std",
    "brain_R@5_mean",
    "brain_R@5_std",
    "brain_MRR_mean",
    "brain_MRR_std",
)

SUMMARY_NUMERIC_COLUMNS = (
    "n",
    "AUROC_mean",
    "AUROC_std",
    "AUPRC_mean",
    "AUPRC_std",
    "R@5_mean",
    "R@5_std",
)

SUMMARY_TABLES = {
    "single_ref_matched_allseed_summary.csv": 3,
    "single_ref_matched_summary.csv": 3,
    "table_adjacency_ablation.csv": 3,
    "table_adjacency_perturbation.csv": 5,
    "table_allfold_final.csv": 3,
    "table_edge_bias_followup.csv": 3,
    "table_external_visual_roi_smoke.csv": 8,
    "table_graph_only.csv": 2,
    "table_hard_negative_allfold.csv": 3,
    "table_heldout_image.csv": 4,
    "table_phase2_sota_graph_baselines.csv": 5,
    "table_roi_token_controls.csv": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit publication result artifact CSV schemas.")
    parser.add_argument(
        "--final-tables-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument("--external-summary-dir", type=Path, default=Path("external_validation/summary"))
    parser.add_argument("--output-prefix", default="result_artifact_schema_audit")
    return parser.parse_args()


def ready(ok: bool) -> str:
    return "ready" if ok else "incomplete"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def is_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def specs(final: Path, external: Path) -> list[SchemaSpec]:
    items = [
        SchemaSpec(
            final / name,
            min_rows,
            SUMMARY_COLUMNS,
            SUMMARY_NUMERIC_COLUMNS,
            ("model", "source"),
        )
        for name, min_rows in sorted(SUMMARY_TABLES.items())
    ]
    items.extend(
        [
            SchemaSpec(
                final / "table_within_subject.csv",
                6,
                ("model", "AUROC", "AUPRC", "R@5", "MRR", "source"),
                ("AUROC",),
                ("model", "source"),
            ),
            SchemaSpec(
                final / "table_lowshot_calibration.csv",
                5,
                ("row_label", "calibration_images", "n", "image_R@5_mean", "image_R@5_std", "image_MRR_mean", "image_MRR_std", "source"),
                ("calibration_images", "n", "image_R@5_mean", "image_R@5_std", "image_MRR_mean", "image_MRR_std"),
                ("row_label", "source"),
            ),
            SchemaSpec(
                final / "table_gate_confound.csv",
                3,
                ("row_label", "spearman", "partial_spearman", "n_roi", "source"),
                ("spearman", "partial_spearman", "n_roi"),
                ("row_label", "source"),
            ),
            SchemaSpec(
                final / "table_matched_deletion.csv",
                6,
                ("row_label", "deletion_set", "k", "AUROC_drop", "R@5_drop", "brain_R@5_drop", "source"),
                ("k", "AUROC_drop", "R@5_drop", "brain_R@5_drop"),
                ("row_label", "deletion_set", "source"),
            ),
            SchemaSpec(
                final / "split_accounting.csv",
                8,
                ("fold", "test_subject", "val_subject", "train_seq", "val_seq", "test_seq", "train_pairs", "val_pairs", "test_pairs", "test_imgs", "source"),
                ("train_seq", "val_seq", "test_seq", "train_pairs", "val_pairs", "test_pairs", "test_imgs"),
                ("fold", "source"),
            ),
            SchemaSpec(
                final / "session_order_pair_qc.csv",
                4,
                ("split", "pairs", "positive", "negative", "complete_groups", "problem_groups", "anchor_match", "source"),
                ("pairs", "positive", "negative", "complete_groups", "problem_groups"),
                ("split", "source"),
            ),
            SchemaSpec(
                final / "fold_difficulty_qc.csv",
                8,
                ("fold", "test_subject", "test_seq", "repeat_corr", "raw_AUROC", "raw_gap", "model_AUROC", "brain_R@5", "source"),
                ("test_seq", "repeat_corr", "raw_AUROC", "raw_gap", "model_AUROC", "brain_R@5"),
                ("fold", "source"),
            ),
            SchemaSpec(
                final / "publication_paired_stats.csv",
                100,
                ("setting", "comparison", "metric", "n", "mean_diff", "std_diff", "bootstrap_ci_low", "bootstrap_ci_high", "paired_t_p"),
                ("n", "mean_diff", "std_diff", "bootstrap_ci_low", "bootstrap_ci_high", "paired_t_p"),
                ("setting", "comparison", "metric"),
            ),
            SchemaSpec(
                final / "final_adjacency_ablation_tests.csv",
                16,
                (
                    "comparison",
                    "a_model",
                    "b_model",
                    "metric",
                    "n_pairs",
                    "a_mean",
                    "b_mean",
                    "mean_diff_a_minus_b",
                    "bootstrap_ci_low",
                    "bootstrap_ci_high",
                    "paired_t_p",
                    "wilcoxon_p",
                ),
                ("n_pairs", "a_mean", "b_mean", "mean_diff_a_minus_b", "bootstrap_ci_low", "bootstrap_ci_high", "paired_t_p", "wilcoxon_p"),
                ("comparison", "a_model", "b_model", "metric"),
            ),
            SchemaSpec(
                final / "single_ref_matched_allseed_pairwise_tests.csv",
                21,
                ("comparison", "metric", "n", "mean_diff", "std_diff", "ci95_half_width", "paired_t_p"),
                ("n", "mean_diff", "std_diff", "ci95_half_width", "paired_t_p"),
                ("comparison", "metric"),
            ),
            SchemaSpec(
                final / "model_parameter_counts.csv",
                5,
                ("model", "graph_encoder", "readout", "trainable_parameters", "n_nodes", "node_feature_dim", "clip_dim", "hidden_dim", "embedding_dim", "num_layers", "num_heads", "source"),
                ("trainable_parameters",),
                ("model", "graph_encoder", "readout", "source"),
            ),
            SchemaSpec(
                external / "laion_fmri_visual_roi_summary.csv",
                2,
                ("model", "n", "test_AUROC_mean", "test_AUROC_std", "test_AUPRC_mean", "test_AUPRC_std", "test_R@5_mean", "test_R@5_std", "test_MRR_mean", "test_MRR_std"),
                ("n", "test_AUROC_mean", "test_AUROC_std", "test_AUPRC_mean", "test_AUPRC_std", "test_R@5_mean", "test_R@5_std", "test_MRR_mean", "test_MRR_std"),
                ("model",),
            ),
            SchemaSpec(
                external / "laion_fmri_visual_roi_pairwise_tests.csv",
                4,
                ("comparison", "metric", "n", "mean_diff", "std_diff", "ci95_low", "ci95_high", "paired_p"),
                ("n", "mean_diff", "std_diff", "ci95_low", "ci95_high", "paired_p"),
                ("comparison", "metric"),
            ),
            SchemaSpec(
                external / "laion_fmri_visual_roi_all_runs.csv",
                60,
                (
                    "model",
                    "seed",
                    "subject_a",
                    "subject_b",
                    "n_images_total",
                    "n_train_images",
                    "n_val_images",
                    "n_test_images",
                    "best_epoch",
                    "test_AUROC",
                    "test_AUPRC",
                    "test_R@5",
                    "test_MRR",
                    "pair",
                ),
                ("seed", "n_images_total", "n_train_images", "n_val_images", "n_test_images", "best_epoch", "test_AUROC", "test_AUPRC", "test_R@5", "test_MRR"),
                ("model", "pair"),
            ),
        ]
    )
    return items


def audit_spec(spec: SchemaSpec, root: Path) -> AuditRow:
    header, rows = read_csv(spec.path)
    label = spec.path.as_posix()
    if not header:
        return AuditRow(label, "missing", f"{spec.path} missing or empty")
    missing_columns = [column for column in spec.required_columns if column not in header]
    if missing_columns:
        return AuditRow(label, "incomplete", "missing columns: " + ", ".join(missing_columns))
    if len(rows) < spec.min_rows:
        return AuditRow(label, "incomplete", f"{len(rows)}/{spec.min_rows} minimum rows")
    bad_numeric: list[str] = []
    for row_index, row in enumerate(rows, start=2):
        for column in spec.numeric_columns:
            value = row.get(column, "")
            if value == "" or not is_float(value):
                bad_numeric.append(f"line {row_index} {column}={value or '<empty>'}")
                break
        if bad_numeric:
            break
    if bad_numeric:
        return AuditRow(label, "incomplete", bad_numeric[0])
    bad_empty: list[str] = []
    for row_index, row in enumerate(rows, start=2):
        for column in spec.required_nonempty_columns:
            if not row.get(column, "").strip():
                bad_empty.append(f"line {row_index} {column}=<empty>")
                break
        if bad_empty:
            break
    if bad_empty:
        return AuditRow(label, "incomplete", bad_empty[0])
    rel = spec.path.relative_to(root).as_posix() if spec.path.is_absolute() and spec.path.is_relative_to(root) else label
    return AuditRow(rel, "ready", f"{len(rows)}/{spec.min_rows} minimum rows and required schema present")


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
        "# Result Artifact Schema Audit",
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
    root = Path.cwd()
    rows = [audit_spec(spec, root) for spec in specs(args.final_tables_dir, args.external_summary_dir)]
    write_outputs(args.final_tables_dir, args.output_prefix, rows)
    return 0 if all(row.status == "ready" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
