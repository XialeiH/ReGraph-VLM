#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TableSpec:
    label: str
    csv_name: str
    metrics: tuple[str, ...]


@dataclass(frozen=True)
class AuditRow:
    item: str
    status: str
    evidence: str


TABLE_SPECS = [
    TableSpec("tab:cross_subject_main", "table_allfold_final.csv", ("AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "brain_R@5")),
    TableSpec("tab:sota_baselines", "table_phase2_sota_graph_baselines.csv", ("AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "brain_R@5")),
    TableSpec("tab:adjacency_ablation", "table_adjacency_ablation.csv", ("AUROC", "AUPRC", "R@5", "MRR", "brain_R@5")),
    TableSpec("tab:roi_token_controls", "table_roi_token_controls.csv", ("AUROC", "AUPRC", "R@5", "MRR", "brain_R@5")),
    TableSpec("tab:adjacency_perturbation", "table_adjacency_perturbation.csv", ("AUROC", "AUPRC", "R@5", "MRR", "brain_R@5")),
    TableSpec("tab:edge_bias_followup", "table_edge_bias_followup.csv", ("AUROC", "AUPRC", "R@5", "brain_R@5")),
    TableSpec("tab:single_ref_matched", "single_ref_matched_summary.csv", ("AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "brain_R@5", "brain_MRR")),
    TableSpec("tab:single_ref_retrained", "single_ref_matched_allseed_summary.csv", ("AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "brain_R@5", "brain_MRR")),
    TableSpec("tab:heldout", "table_heldout_image.csv", ("AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "brain_R@5")),
    TableSpec("tab:hardneg", "table_hard_negative_allfold.csv", ("AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "brain_R@5", "brain_MRR")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit that key may30.tex table numbers match committed CSV artifacts.")
    parser.add_argument("--tex", type=Path, default=Path("reports/neurips_report/may30.tex"))
    parser.add_argument(
        "--final-tables-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument("--output-prefix", default="manuscript_table_values_audit")
    return parser.parse_args()


def ready(ok: bool) -> str:
    return "ready" if ok else "incomplete"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_table_block(tex: str, label: str) -> str:
    label_marker = f"\\label{{{label}}}"
    idx = tex.find(label_marker)
    if idx < 0:
        return ""
    start = tex.rfind("\\begin{table", 0, idx)
    end = tex.find("\\end{table}", idx)
    if start < 0 or end < 0:
        return ""
    return tex[start : end + len("\\end{table}")]


def tex_number(value: str) -> str:
    return f"{float(value):.4f}"


def expected_fragments(row: dict[str, str], metrics: tuple[str, ...]) -> list[str]:
    fragments: list[str] = []
    for metric in metrics:
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        if mean_key in row:
            mean = row.get(mean_key, "")
            if mean == "":
                continue
            fragments.append(tex_number(mean))
            std = row.get(std_key, "")
            if std != "":
                fragments.append(tex_number(std))
            continue

        formatted = row.get(metric, "")
        for value in re.findall(r"\d+\.\d+", formatted):
            fragments.append(tex_number(value))
    return fragments


def audit_spec(tex: str, final_tables_dir: Path, spec: TableSpec) -> AuditRow:
    block = find_table_block(tex, spec.label)
    if not block:
        return AuditRow(spec.label, "missing", "table block not found")
    csv_path = final_tables_dir / spec.csv_name
    rows = read_csv(csv_path)
    if not rows:
        return AuditRow(spec.label, "missing", f"{csv_path} missing or empty")
    missing: list[str] = []
    checked = 0
    for row in rows:
        model = row.get("model", "row")
        for fragment in expected_fragments(row, spec.metrics):
            checked += 1
            if fragment not in block:
                missing.append(f"{model}: {fragment}")
    if checked == 0:
        return AuditRow(spec.label, "incomplete", f"no numeric fragments extracted from {spec.csv_name}")
    return AuditRow(
        spec.label,
        ready(not missing),
        f"{checked} numeric fragments match {spec.csv_name}" if not missing else "; ".join(missing[:10]),
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
        "# Manuscript Table Values Audit",
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
    tex = args.tex.read_text(encoding="utf-8", errors="replace")
    rows = [audit_spec(tex, args.final_tables_dir, spec) for spec in TABLE_SPECS]
    write_outputs(args.final_tables_dir, args.output_prefix, rows)
    return 0 if all(row.status == "ready" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
