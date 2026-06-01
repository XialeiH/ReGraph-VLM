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


CAPTION_REQUIREMENTS = {
    "tab:split_accounting": ("strict T=3", "pair counts"),
    "tab:session_order_pair_qc": ("Session/order QC", "same anchor trial"),
    "tab:implementation_details": ("Implementation and training details",),
    "tab:within_subject": ("available smoke folds and seeds",),
    "tab:cross_subject_main": ("8 held-out-subject folds and 3 random seeds", "mean", "std"),
    "tab:sota_baselines": ("Task-matched", "mean", "std", "not full image-reconstruction"),
    "tab:graph_only": ("8 held-out-subject folds and 3 random seeds", "mean", "std"),
    "tab:adjacency_ablation": ("main all-fold cross-subject", "mean", "std"),
    "tab:roi_token_controls": ("ROI-token", "mean", "std"),
    "tab:adjacency_perturbation": ("diagnostic", "available diagnostic runs"),
    "tab:edge_bias_followup": ("mean", "std", "$n$ gives"),
    "tab:single_ref_matched": ("mean", "std", "$n=24$"),
    "tab:single_ref_retrained": ("mean", "std", "$n=24$"),
    "tab:heldout": ("8 folds", "3 seeds", "mean", "std"),
    "tab:hardneg": ("mean", "std", "point summaries"),
    "tab:lowshot": ("mean", "std", "$n$"),
    "tab:external_visual_roi_smoke": ("mean", "std", "not full HCP-MMP"),
    "tab:gate_confound": ("before and after controlling",),
    "tab:matched_deletion": ("Drops are absolute",),
    "tab:fold_difficulty": ("Fold-level robustness diagnostic", "Raw pair AUROC"),
    "tab:top_gated_roi_names": ("Top gated ROIs", "HCP-MMP"),
}

PRIMARY_RESULT_TABLES = {
    "tab:cross_subject_main",
    "tab:sota_baselines",
    "tab:graph_only",
    "tab:adjacency_ablation",
    "tab:roi_token_controls",
    "tab:edge_bias_followup",
    "tab:single_ref_matched",
    "tab:single_ref_retrained",
    "tab:heldout",
    "tab:hardneg",
    "tab:lowshot",
    "tab:external_visual_roi_smoke",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit uncertainty and scope language in manuscript table captions.")
    parser.add_argument("--tex", type=Path, default=Path("reports/neurips_report/may30.tex"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument("--output-prefix", default="table_uncertainty_language_audit")
    return parser.parse_args()


def ready(ok: bool) -> str:
    return "ready" if ok else "incomplete"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def table_captions(text: str) -> dict[str, str]:
    captions: dict[str, str] = {}
    pattern = re.compile(r"\\begin\{table\}.*?\\caption\{(.*?)\}.*?\\label\{([^}]+)\}", re.S)
    for match in pattern.finditer(text):
        caption, label = match.groups()
        captions[label] = normalize(caption)
    return captions


def audit_caption_language(tex: Path) -> list[AuditRow]:
    if not tex.exists():
        return [AuditRow("manuscript exists", "missing", f"{tex} not found")]
    captions = table_captions(tex.read_text(encoding="utf-8", errors="replace"))
    rows = [
        AuditRow("manuscript exists", "ready", str(tex)),
        AuditRow("table captions parsed", ready(bool(captions)), f"{len(captions)} table captions"),
    ]

    missing_expected = sorted(set(CAPTION_REQUIREMENTS) - set(captions))
    rows.append(
        AuditRow(
            "expected table captions present",
            ready(not missing_expected),
            f"{len(CAPTION_REQUIREMENTS)} expected captions present" if not missing_expected else "; ".join(missing_expected),
        )
    )

    for label, fragments in CAPTION_REQUIREMENTS.items():
        caption = captions.get(label, "")
        missing = [fragment for fragment in fragments if fragment not in caption]
        rows.append(
            AuditRow(
                f"caption scope: {label}",
                ready(not missing),
                "required uncertainty/scope language present" if not missing else "; ".join(missing),
            )
        )

    primary_missing_uncertainty = []
    for label in sorted(PRIMARY_RESULT_TABLES):
        caption = captions.get(label, "")
        has_mean_std = "mean" in caption and "std" in caption
        has_point_summary = "point summaries" in caption
        has_diagnostic_scope = "diagnostic" in caption or "not full HCP-MMP" in caption
        if not (has_mean_std or has_point_summary or has_diagnostic_scope):
            primary_missing_uncertainty.append(label)
    rows.append(
        AuditRow(
            "primary result captions disclose uncertainty scope",
            ready(not primary_missing_uncertainty),
            f"{len(PRIMARY_RESULT_TABLES)} primary result captions covered"
            if not primary_missing_uncertainty
            else "; ".join(primary_missing_uncertainty),
        )
    )
    return rows


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
        "# Table Uncertainty Language Audit",
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
    rows = audit_caption_language(args.tex)
    write_outputs(args.output_dir, args.output_prefix, rows)
    return 0 if all(row.status == "ready" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
