#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuditRow:
    item: str
    status: str
    evidence: str


EXPECTED_TABLE_SOURCES = {
    "split_accounting.csv": "tab:split_accounting",
    "session_order_pair_qc.csv": "tab:session_order_pair_qc",
    "table_within_subject.csv": "tab:within_subject",
    "table_allfold_final.csv": "tab:cross_subject_main",
    "table_phase2_sota_graph_baselines.csv": "tab:sota_baselines",
    "table_graph_only.csv": "tab:graph_only",
    "table_adjacency_ablation.csv": "tab:adjacency_ablation",
    "table_roi_token_controls.csv": "tab:roi_token_controls",
    "table_adjacency_perturbation.csv": "tab:adjacency_perturbation",
    "table_edge_bias_followup.csv": "tab:edge_bias_followup",
    "single_ref_matched_summary.csv": "tab:single_ref_matched",
    "single_ref_matched_allseed_summary.csv": "tab:single_ref_retrained",
    "table_heldout_image.csv": "tab:heldout",
    "table_hard_negative_allfold.csv": "tab:hardneg",
    "table_lowshot_calibration.csv": "tab:lowshot",
    "table_external_visual_roi_smoke.csv": "tab:external_visual_roi_smoke",
    "table_gate_confound.csv": "tab:gate_confound",
    "table_matched_deletion.csv": "tab:matched_deletion",
    "fold_difficulty_qc.csv": "tab:fold_difficulty",
    "model_parameter_counts.csv": "tab:implementation_details",
}


EXPECTED_AUDIT_ARTIFACTS = {
    "aaai_publication_readiness_audit.csv": 33,
    "bundle_allowlist_audit.csv": 13,
    "citation_integrity_audit.csv": 11,
    "ci_workflow_audit.csv": 10,
    "dataset_accounting_audit.csv": 13,
    "external_data_policy_audit.csv": 6,
    "external_validation_consistency_audit.csv": 11,
    "figure_asset_audit.csv": 9,
    "makefile_targets_audit.csv": 9,
    "manuscript_publication_claims_audit.csv": 55,
    "publication_docs_audit.csv": 51,
    "publication_evidence_manifest_audit.csv": 10,
    "package_metadata_audit.csv": 11,
    "python_syntax_audit.csv": 4,
    "result_artifact_schema_audit.csv": 26,
    "result_value_range_audit.csv": 26,
    "reviewer_response_readiness_audit.csv": 11,
    "table_uncertainty_language_audit.csv": 26,
    "manuscript_table_values_audit.csv": 25,
    "manuscript_stat_claims_audit.csv": 22,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit source/provenance metadata for publication-facing artifacts.")
    parser.add_argument(
        "--final-tables-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument("--source-tex", default="reports/neurips_report/may30.tex")
    parser.add_argument("--output-prefix", default="publication_artifact_provenance_audit")
    return parser.parse_args()


def ready(ok: bool) -> str:
    return "ready" if ok else "incomplete"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def manuscript_labels(source_tex: str) -> set[str]:
    path = Path(source_tex)
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8", errors="replace")
    return set(re.findall(r"\\label\{([^}]+)\}", text))


def audit_table(final_tables_dir: Path, source_tex: str, labels: set[str], csv_name: str, label: str) -> AuditRow:
    path = final_tables_dir / csv_name
    if not path.exists():
        return AuditRow(csv_name, "missing", f"{path} not found")
    if label not in labels:
        return AuditRow(csv_name, "incomplete", f"{source_tex} does not define Table {label}")
    rows = read_csv(path)
    if not rows:
        return AuditRow(csv_name, "incomplete", "empty artifact")
    if "source" not in rows[0]:
        return AuditRow(csv_name, "incomplete", "missing source column")
    missing = []
    for index, row in enumerate(rows, start=1):
        source = row.get("source", "")
        if source_tex not in source or label not in source:
            missing.append(f"row {index}: {source or '<empty>'}")
    return AuditRow(
        csv_name,
        ready(not missing),
        f"{len(rows)} rows cite existing {source_tex}: Table {label}" if not missing else "; ".join(missing[:5]),
    )


def audit_generated_audit(final_tables_dir: Path, csv_name: str, min_rows: int) -> AuditRow:
    path = final_tables_dir / csv_name
    if not path.exists():
        return AuditRow(csv_name, "missing", f"{path} not found")
    rows = read_csv(path)
    if not rows:
        return AuditRow(csv_name, "incomplete", "empty artifact")
    if "status" not in rows[0]:
        return AuditRow(csv_name, "incomplete", "missing status column")
    non_ready = [row for row in rows if row.get("status") != "ready"]
    if non_ready:
        evidence = "; ".join(f"{row.get('item', 'row')}: {row.get('status')}" for row in non_ready[:5])
        return AuditRow(csv_name, "incomplete", evidence)
    status_value = ready(len(rows) >= min_rows)
    evidence = f"{len(rows)}/{min_rows} minimum generated-audit rows ready"
    return AuditRow(csv_name, status_value, evidence)


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
        "# Publication Artifact Provenance Audit",
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
    labels = manuscript_labels(args.source_tex)
    rows = [
        audit_table(args.final_tables_dir, args.source_tex, labels, csv_name, label)
        for csv_name, label in EXPECTED_TABLE_SOURCES.items()
    ]
    rows.extend(
        audit_generated_audit(args.final_tables_dir, csv_name, min_rows)
        for csv_name, min_rows in EXPECTED_AUDIT_ARTIFACTS.items()
    )
    write_outputs(args.final_tables_dir, args.output_prefix, rows)
    return 0 if all(row.status == "ready" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
