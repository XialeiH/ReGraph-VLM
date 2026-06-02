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


EXPECTED_EXTERNAL_ROWS = {
    ("BOLD5000 visual ROI", "ROI-MLP"): 18,
    ("BOLD5000 visual ROI", "Gated ROI Transformer"): 18,
    ("CNeuroMod visual ROI", "ROI-MLP"): 18,
    ("CNeuroMod visual ROI", "Gated ROI Transformer"): 18,
    ("THINGS-fMRI visual ROI", "ROI-MLP"): 9,
    ("THINGS-fMRI visual ROI", "Gated ROI Transformer"): 9,
    ("LAION-fMRI visual ROI", "ROI-MLP"): 30,
    ("LAION-fMRI visual ROI", "Gated ROI Transformer"): 30,
}

EXPECTED_SCAN_DATASETS = [
    "LAION-fMRI",
    "CNeuroMod-THINGS",
    "BOLD5000",
    "THINGS-fMRI",
    "NOD",
]

METRIC_COLUMNS = ["AUROC_mean", "AUPRC_mean", "R@5_mean", "MRR_mean"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit external-validation summary consistency and caveats.")
    parser.add_argument("--tex", type=Path, default=Path("reports/neurips_report/may30.tex"))
    parser.add_argument("--scan", type=Path, default=Path("reports/neurips_report/external_validation_dataset_scan.md"))
    parser.add_argument(
        "--final-tables-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument("--external-summary-dir", type=Path, default=Path("external_validation/summary"))
    parser.add_argument("--output-prefix", default="external_validation_consistency_audit")
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


def split_model_label(label: str) -> tuple[str, str]:
    if " & " not in label:
        return label, ""
    dataset, model = label.split(" & ", 1)
    return dataset.strip(), model.strip()


def audit_scan_exists(scan: Path) -> AuditRow:
    return AuditRow("external-validation scan exists", "ready" if scan.exists() else "missing", str(scan))


def audit_manuscript_external_tables(tex_text: str) -> AuditRow:
    labels = ["tab:external_visual_roi_smoke", "tab:laion_external_pairwise"]
    missing = [label for label in labels if f"\\label{{{label}}}" not in tex_text]
    return AuditRow(
        "manuscript external-validation tables",
        ready(not missing),
        "external smoke and LAION paired-test tables are present in may30.tex" if not missing else "; ".join(missing),
    )


def audit_scan_dataset_coverage(scan_text: str) -> AuditRow:
    missing = [dataset for dataset in EXPECTED_SCAN_DATASETS if dataset not in scan_text]
    return AuditRow(
        "external-validation dataset coverage",
        ready(not missing),
        "LAION-fMRI, CNeuroMod-THINGS, BOLD5000, THINGS-fMRI, and NOD covered" if not missing else "; ".join(missing),
    )


def audit_scan_hpc_policy(scan_text: str) -> AuditRow:
    phrases = [
        "downloaded directly to Shanghai HPC scratch",
        "$REGRAPH_VLM_HPC_ROOT/external_validation",
        "large downloads performed: none",
    ]
    missing = [phrase for phrase in phrases if phrase not in scan_text]
    return AuditRow(
        "external-validation HPC data policy",
        ready(not missing),
        "scan documents HPC scratch-only external data handling" if not missing else "; ".join(missing),
    )


def audit_scan_limitations(scan_text: str) -> AuditRow:
    phrases = [
        "rather than full HCP-MMP 180-ROI validation",
        "external feasibility",
        "not as a replacement for a full external atlas-ROI replication",
        "does not support a universal gated ROI Transformer advantage",
    ]
    missing = [phrase for phrase in phrases if phrase not in scan_text]
    return AuditRow(
        "external-validation limitation language",
        ready(not missing),
        "scan preserves visual-ROI feasibility and non-universal-advantage caveats" if not missing else "; ".join(missing),
    )


def audit_external_table_rows(rows: list[dict[str, str]]) -> AuditRow:
    observed = {split_model_label(row.get("model", "")): int(float(row.get("n", "0"))) for row in rows}
    missing = [f"{dataset} / {model}" for dataset, model in EXPECTED_EXTERNAL_ROWS if (dataset, model) not in observed]
    wrong_n = [
        f"{dataset} / {model}: {observed[(dataset, model)]} != {expected_n}"
        for (dataset, model), expected_n in EXPECTED_EXTERNAL_ROWS.items()
        if (dataset, model) in observed and observed[(dataset, model)] != expected_n
    ]
    extra = [f"{dataset} / {model}" for dataset, model in observed if (dataset, model) not in EXPECTED_EXTERNAL_ROWS]
    problems = missing + wrong_n + extra
    return AuditRow(
        "external visual-ROI table rows",
        ready(not problems and len(rows) == len(EXPECTED_EXTERNAL_ROWS)),
        "8 expected dataset-model rows with expected n values" if not problems and len(rows) == len(EXPECTED_EXTERNAL_ROWS) else "; ".join(problems[:8]),
    )


def audit_external_table_sources(rows: list[dict[str, str]]) -> AuditRow:
    missing = [
        row.get("model", f"row {index}")
        for index, row in enumerate(rows, start=1)
        if "reports/neurips_report/may30.tex" not in row.get("source", "")
        or "tab:external_visual_roi_smoke" not in row.get("source", "")
    ]
    return AuditRow(
        "external visual-ROI table source labels",
        ready(not missing and bool(rows)),
        "all external rows cite may30.tex Table tab:external_visual_roi_smoke" if not missing and rows else "; ".join(missing[:8]),
    )


def audit_external_metric_ranges(rows: list[dict[str, str]]) -> AuditRow:
    problems: list[str] = []
    for row in rows:
        label = row.get("model", "row")
        for column in METRIC_COLUMNS:
            try:
                value = float(row[column])
            except (KeyError, ValueError):
                problems.append(f"{label}: invalid {column}")
                continue
            if not 0.0 <= value <= 1.0:
                problems.append(f"{label}: {column}={value}")
    return AuditRow(
        "external visual-ROI metric ranges",
        ready(not problems and bool(rows)),
        "all external AUROC/AUPRC/R@5/MRR means are in [0,1]" if not problems and rows else "; ".join(problems[:8]),
    )


def audit_external_summary_md(summary_text: str) -> AuditRow:
    phrases = [
        "BOLD5000 visual ROI",
        "CNeuroMod visual ROI",
        "THINGS-fMRI visual ROI",
        "LAION-fMRI visual ROI",
        "not full HCP-MMP 180-ROI external validations",
        "not as full external validation",
    ]
    missing = [phrase for phrase in phrases if phrase not in summary_text]
    return AuditRow(
        "external visual-ROI summary text",
        ready(not missing),
        "summary covers four datasets and preserves feasibility-only caveat" if not missing else "; ".join(missing),
    )


def audit_laion_summary(rows: list[dict[str, str]]) -> AuditRow:
    expected = {"roi_mlp", "roi_transformer_gated"}
    models = {row.get("model", "") for row in rows}
    n_values = {row.get("model", ""): int(float(row.get("n", "0"))) for row in rows}
    problems = []
    if models != expected:
        problems.append(f"models={sorted(models)}")
    for model in expected:
        if n_values.get(model) != 30:
            problems.append(f"{model}: n={n_values.get(model)}")
    return AuditRow(
        "LAION summary rows",
        ready(not problems and len(rows) == 2),
        "ROI-MLP and gated transformer LAION rows have n=30" if not problems and len(rows) == 2 else "; ".join(problems),
    )


def audit_laion_pairwise(rows: list[dict[str, str]]) -> AuditRow:
    expected_metrics = {"test_AUROC", "test_AUPRC", "test_R@5", "test_MRR"}
    metrics = {row.get("metric", "") for row in rows}
    required_columns = {"comparison", "metric", "n", "mean_diff", "ci95_low", "ci95_high", "paired_p"}
    problems = []
    if metrics != expected_metrics:
        problems.append(f"metrics={sorted(metrics)}")
    if rows and not required_columns.issubset(rows[0]):
        problems.append(f"columns={sorted(rows[0])}")
    for row in rows:
        try:
            n_value = int(float(row.get("n", "0")))
            p_value = float(row.get("paired_p", "nan"))
        except ValueError:
            problems.append(f"{row.get('metric', 'row')}: invalid n or paired_p")
            continue
        if n_value != 30:
            problems.append(f"{row.get('metric', 'row')}: n={n_value}")
        if not 0.0 <= p_value <= 1.0:
            problems.append(f"{row.get('metric', 'row')}: paired_p={p_value}")
    return AuditRow(
        "LAION pairwise test rows",
        ready(not problems and len(rows) == 4),
        "four LAION paired-test rows have n=30 and valid p-values" if not problems and len(rows) == 4 else "; ".join(problems[:8]),
    )


def audit_rows(tex: Path, scan: Path, final_tables_dir: Path, external_summary_dir: Path) -> list[AuditRow]:
    tex_text = read_text(tex)
    scan_text = read_text(scan)
    external_rows = read_csv(final_tables_dir / "table_external_visual_roi_smoke.csv")
    summary_text = read_text(external_summary_dir / "external_visual_roi_all4_summary.md")
    laion_summary_rows = read_csv(external_summary_dir / "laion_fmri_visual_roi_summary.csv")
    laion_pairwise_rows = read_csv(external_summary_dir / "laion_fmri_visual_roi_pairwise_tests.csv")
    return [
        audit_scan_exists(scan),
        audit_manuscript_external_tables(tex_text),
        audit_scan_dataset_coverage(scan_text),
        audit_scan_hpc_policy(scan_text),
        audit_scan_limitations(scan_text),
        audit_external_table_rows(external_rows),
        audit_external_table_sources(external_rows),
        audit_external_metric_ranges(external_rows),
        audit_external_summary_md(summary_text),
        audit_laion_summary(laion_summary_rows),
        audit_laion_pairwise(laion_pairwise_rows),
    ]


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
        "# External Validation Consistency Audit",
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
    rows = audit_rows(args.tex, args.scan, args.final_tables_dir, args.external_summary_dir)
    write_outputs(args.final_tables_dir, args.output_prefix, rows)
    return 0 if all(row.status == "ready" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
