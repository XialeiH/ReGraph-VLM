#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class AuditRow:
    item: str
    status: str
    evidence: str


REQUIRED_LABELS = [
    "tab:split_accounting",
    "tab:session_order_pair_qc",
    "tab:implementation_details",
    "tab:cross_subject_main",
    "tab:sota_baselines",
    "tab:graph_only",
    "tab:adjacency_ablation",
    "tab:roi_token_controls",
    "tab:adjacency_perturbation",
    "tab:edge_bias_followup",
    "tab:single_ref_matched",
    "tab:single_ref_retrained",
    "tab:heldout",
    "tab:hardneg",
    "tab:lowshot",
    "tab:external_visual_roi_smoke",
    "tab:gate_confound",
    "tab:matched_deletion",
    "tab:fold_difficulty",
    "tab:top_gated_roi_names",
]

REQUIRED_RESULT_FILES = {
    "table_allfold_final.csv": 24,
    "table_hard_negative_allfold.csv": 24,
    "table_heldout_image.csv": 24,
    "table_phase2_sota_graph_baselines.csv": 100,
    "table_adjacency_ablation.csv": 1,
    "table_roi_token_controls.csv": 1,
    "table_adjacency_perturbation.csv": 1,
    "table_edge_bias_followup.csv": 1,
    "session_order_pair_qc.csv": 1,
    "fold_difficulty_qc.csv": 8,
    "single_ref_matched_all_runs.csv": 72,
    "single_ref_matched_allseed_all_runs.csv": 72,
    "publication_paired_stats.csv": 1,
}

REQUIRED_STATS = [
    ("main_allfold", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP"),
    ("main_allfold", "Gated ReGraph/BNT+CLIP - Flat ReGraph+CLIP"),
    ("hard_negative", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP"),
    ("heldout_real_vs_random_available_raw", "Gated ReGraph/BNT+CLIP - Gated random embedding"),
    ("component_baselines", "Gated ReGraph/BNT+CLIP - MindLink-style subject-adversarial ROI-MLP"),
    ("single_ref_eval_existing", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP"),
    ("single_ref_eval_existing", "Gated ReGraph/BNT+CLIP - No-adj gated ROI Transformer+CLIP"),
    ("single_ref_retrained", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP"),
    ("single_ref_retrained", "Gated ReGraph/BNT+CLIP - No-adj gated ROI Transformer+CLIP"),
    ("single_ref_retrained", "No-adj gated ROI Transformer+CLIP - ROI-MLP+CLIP"),
]

DEANON_PATTERNS = [
    r"Xialei",
    r"xialei",
    r"xialei\.huang",
    r"NYU Shanghai",
]

OVERCLAIM_PATTERNS = [
    r"adjacency improves",
    r"fixed adjacency improves",
    r"graph adjacency is the source",
    r"adjacency is the source",
    r"driven by explicit fixed adjacency",
    r"explicit fixed adjacency improves",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit manuscript/result consistency for the AAAI-style ReGraph-VLM submission.")
    parser.add_argument("--tex", type=Path, default=Path("reports/neurips_report/may30.tex"))
    parser.add_argument("--final-tables-dir", type=Path, default=Path("preproc_v0/repetition_familiarity/results/final_tables"))
    parser.add_argument("--output-dir", type=Path, default=Path("preproc_v0/repetition_familiarity/results/final_tables"))
    parser.add_argument("--output-prefix", default="manuscript_publication_claims_audit")
    return parser.parse_args()


def status(ok: bool) -> str:
    return "ready" if ok else "incomplete"


def count_env(text: str, env: str) -> tuple[int, int]:
    begin = len(re.findall(rf"\\begin\{{{re.escape(env)}\}}", text))
    end = len(re.findall(rf"\\end\{{{re.escape(env)}\}}", text))
    return begin, end


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def audit_text(tex_path: Path) -> list[AuditRow]:
    if not tex_path.exists():
        return [AuditRow("manuscript exists", "missing", f"{tex_path} not found")]
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    rows: list[AuditRow] = [AuditRow("manuscript exists", "ready", f"{tex_path}: {len(text.splitlines())} lines")]

    rows.append(AuditRow("anonymous author block", status("Anonymous Author(s)" in text), "Anonymous Author(s) present" if "Anonymous Author(s)" in text else "anonymous author marker missing"))

    deanon_hits = []
    for pattern in DEANON_PATTERNS:
        deanon_hits.extend(re.findall(pattern, text))
    rows.append(AuditRow("deanonymizing strings", status(not deanon_hits), "none found" if not deanon_hits else ", ".join(sorted(set(deanon_hits)))))

    overclaim_hits = []
    lower = text.lower()
    for pattern in OVERCLAIM_PATTERNS:
        if re.search(pattern, lower):
            overclaim_hits.append(pattern)
    rows.append(AuditRow("fixed-adjacency overclaims", status(not overclaim_hits), "none found" if not overclaim_hits else ", ".join(overclaim_hits)))

    labels = re.findall(r"\\label\{([^}]+)\}", text)
    refs = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text)
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    missing_refs = sorted(set(refs) - set(labels))
    rows.append(AuditRow("duplicate labels", status(not duplicate_labels), "none" if not duplicate_labels else ", ".join(duplicate_labels)))
    rows.append(AuditRow("unresolved refs", status(not missing_refs), "none" if not missing_refs else ", ".join(missing_refs)))

    missing_required = sorted(set(REQUIRED_LABELS) - set(labels))
    rows.append(AuditRow("required publication labels", status(not missing_required), "all present" if not missing_required else ", ".join(missing_required)))

    figure_paths = re.findall(r"\\IfFileExists\{([^}]+)\}", text)
    missing_figures = sorted(path for path in figure_paths if not (tex_path.parent / path).exists())
    rows.append(
        AuditRow(
            "figure file availability",
            status(not missing_figures),
            f"{len(figure_paths)} checked, all present" if not missing_figures else ", ".join(missing_figures),
        )
    )

    for env in ["table", "figure", "equation"]:
        begin, end = count_env(text, env)
        rows.append(AuditRow(f"{env} environment balance", status(begin == end), f"begin={begin}, end={end}"))

    return rows


def audit_result_files(final_tables_dir: Path) -> list[AuditRow]:
    rows: list[AuditRow] = []
    for name, min_rows in REQUIRED_RESULT_FILES.items():
        path = final_tables_dir / name
        if not path.exists():
            rows.append(AuditRow(f"result file: {name}", "missing", f"{path} not found"))
            continue
        df = read_csv(path)
        rows.append(
            AuditRow(
                f"result file: {name}",
                status(len(df) >= min_rows),
                f"{len(df)} rows, expected at least {min_rows}",
            )
        )
    return rows


def audit_publication_stats(final_tables_dir: Path) -> list[AuditRow]:
    path = final_tables_dir / "publication_paired_stats.csv"
    if not path.exists():
        return [AuditRow("publication paired stats coverage", "missing", f"{path} not found")]
    df = read_csv(path)
    rows: list[AuditRow] = []
    for setting, comparison in REQUIRED_STATS:
        if df.empty or "setting" not in df.columns or "comparison" not in df.columns:
            rows.append(AuditRow(f"paired stats: {setting} / {comparison}", "incomplete", "missing expected columns"))
            continue
        sub = df[(df["setting"] == setting) & (df["comparison"] == comparison)]
        rows.append(
            AuditRow(
                f"paired stats: {setting} / {comparison}",
                status(len(sub) >= 3),
                f"{len(sub)} metric rows, expected at least 3",
            )
        )
    return rows


def write_outputs(output_dir: Path, output_prefix: str, rows: list[AuditRow]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{output_prefix}.csv"
    md_path = output_dir / f"{output_prefix}.md"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item", "status", "evidence"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"item": row.item, "status": row.status, "evidence": row.evidence})

    counts = pd.Series([row.status for row in rows]).value_counts().to_dict()
    lines = [
        "# Manuscript Publication Claims Audit",
        "",
        "This audit checks manuscript/result consistency for the publication-facing ReGraph-VLM story.",
        "",
        f"Status counts: {counts}",
        "",
        "| Item | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row.item} | {row.status} | {row.evidence} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    rows = [
        *audit_text(args.tex),
        *audit_result_files(args.final_tables_dir),
        *audit_publication_stats(args.final_tables_dir),
    ]
    write_outputs(args.output_dir, args.output_prefix, rows)


if __name__ == "__main__":
    main()
