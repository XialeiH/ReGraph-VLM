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


REQUIRED_CLAIMS = {
    "Main cross-subject result",
    "Dataset accounting and split construction",
    "Model scope and non-claims",
    "Explicit adjacency is not the source of the gain",
    "Session/order confound control",
    "Implementation reproducibility",
    "Statistical reporting",
    "Component baseline framing",
    "Semantic alignment control",
    "External validation limits",
    "tab:laion_external_pairwise",
    "Fold_07 robustness",
    "Double-blind code sharing",
}

REQUIRED_ARTIFACTS = {
    "DATASET_CARD.md",
    "MODEL_CARD.md",
    "REVIEWER_RESPONSE.md",
    "table_allfold_final.csv",
    "split_accounting.csv",
    "session_order_pair_qc.csv",
    "fold_difficulty_qc.csv",
    "table_adjacency_ablation.csv",
    "table_roi_token_controls.csv",
    "final_adjacency_ablation_tests.csv",
    "single_ref_matched_summary.csv",
    "single_ref_matched_allseed_summary.csv",
    "model_parameter_counts.csv",
    "publication_paired_stats.csv",
    "table_phase2_sota_graph_baselines.csv",
    "table_heldout_image.csv",
    "table_external_visual_roi_smoke.csv",
    "external_visual_roi_all4_summary.md",
    "laion_fmri_visual_roi_pairwise_tests.csv",
    "external_validation_consistency_audit.csv",
    "anonymous_bundle_manifest.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the reviewer-facing publication evidence manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables/publication_evidence_manifest.md"),
    )
    parser.add_argument(
        "--final-tables-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument("--external-summary-dir", type=Path, default=Path("external_validation/summary"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument("--output-prefix", default="publication_evidence_manifest_audit")
    return parser.parse_args()


def ready(ok: bool) -> str:
    return "ready" if ok else "incomplete"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def markdown_rows(text: str, section: str) -> list[str]:
    start = text.find(f"## {section}")
    if start < 0:
        return []
    rest = text[start:]
    next_section = rest.find("\n## ", 1)
    body = rest if next_section < 0 else rest[:next_section]
    return [line for line in body.splitlines() if line.startswith("| ") and not line.startswith("| ---")]


def resolve_artifact(token: str, final_tables_dir: Path, external_summary_dir: Path) -> Path | None:
    if token.startswith("tab:") or token.startswith("Sec."):
        return None
    if token.endswith((".csv", ".txt")) and "/" not in token:
        if token.startswith("laion_fmri_"):
            return external_summary_dir / token
        return final_tables_dir / token
    if token == "external_visual_roi_all4_summary.md":
        return external_summary_dir / token
    if token == "anonymous_bundle_manifest.csv":
        return final_tables_dir / token
    if token.endswith((".md", ".py", ".toml")):
        return Path(token)
    return None


def audit_manifest(manifest: Path, final_tables_dir: Path, external_summary_dir: Path) -> list[AuditRow]:
    text = read_text(manifest)
    if not text:
        return [AuditRow("evidence manifest exists", "missing", f"{manifest} not found or empty")]

    main_rows = markdown_rows(text, "Main Evidence Map")
    audit_rows = markdown_rows(text, "Audit Artifacts")
    missing_claims = sorted(claim for claim in REQUIRED_CLAIMS if claim not in text)
    missing_artifact_mentions = sorted(artifact for artifact in REQUIRED_ARTIFACTS if artifact not in text)
    referenced = sorted(set(re.findall(r"`([^`]+)`", text)))
    missing_files: list[str] = []
    checked_files = 0
    for token in referenced:
        path = resolve_artifact(token, final_tables_dir, external_summary_dir)
        if path is None:
            continue
        checked_files += 1
        if not path.exists():
            missing_files.append(token)

    return [
        AuditRow("evidence manifest exists", "ready", str(manifest)),
        AuditRow("main evidence map row count", ready(len(main_rows) >= 14), f"{len(main_rows)} rows"),
        AuditRow("audit artifact row count", ready(len(audit_rows) >= 13), f"{len(audit_rows)} rows"),
        AuditRow(
            "required reviewer claims covered",
            ready(not missing_claims),
            f"{len(REQUIRED_CLAIMS)} required claims present" if not missing_claims else "; ".join(missing_claims),
        ),
        AuditRow(
            "required evidence artifacts mentioned",
            ready(not missing_artifact_mentions),
            f"{len(REQUIRED_ARTIFACTS)} required artifacts mentioned"
            if not missing_artifact_mentions
            else "; ".join(missing_artifact_mentions),
        ),
        AuditRow(
            "referenced evidence artifacts exist",
            ready(not missing_files),
            f"{checked_files} referenced artifact paths exist" if not missing_files else "; ".join(missing_files[:8]),
        ),
        AuditRow(
            "double-blind warning present",
            ready("do not submit public GitHub metadata" in text),
            "public GitHub metadata warning present",
        ),
        AuditRow(
            "external-validation caveat present",
            ready("not full HCP-MMP external replications" in text),
            "external-validation limitation is explicit",
        ),
        AuditRow(
            "adjacency non-claim present",
            ready("no-adj and adjacency variants are statistically tied" in text),
            "adjacency limitation is explicit",
        ),
        AuditRow(
            "fold_07 caveat present",
            ready("unresolved robustness case" in text),
            "fold_07 limitation is explicit",
        ),
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
        "# Publication Evidence Manifest Audit",
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
    rows = audit_manifest(args.manifest, args.final_tables_dir, args.external_summary_dir)
    write_outputs(args.output_dir, args.output_prefix, rows)
    return 0 if all(row.status == "ready" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
