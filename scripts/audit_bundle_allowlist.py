#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuditRow:
    item: str
    status: str
    evidence: str


REQUIRED_SOURCE_PATHS = {
    ".github/workflows/publication-preflight.yml",
    "ANONYMIZATION.md",
    "DATASET_CARD.md",
    "Makefile",
    "MODEL_CARD.md",
    "README.md",
    "REPRODUCIBILITY.md",
    "REVIEWER_RESPONSE.md",
    "reports/neurips_report/BUILD.md",
    "reports/neurips_report/may30.tex",
    "scripts/make_anonymous_submission_bundle.py",
    "scripts/run_publication_preflight.py",
    "scripts/smoke_test_anonymous_bundle_archive.py",
    "scripts/verify_anonymous_bundle_manifest.py",
}

REQUIRED_AUDIT_SCRIPTS = {
    "scripts/audit_aaai_publication_readiness.py",
    "scripts/audit_bundle_allowlist.py",
    "scripts/audit_citation_integrity.py",
    "scripts/audit_ci_workflow.py",
    "scripts/audit_dataset_accounting.py",
    "scripts/audit_external_data_policy.py",
    "scripts/audit_figure_assets.py",
    "scripts/audit_manuscript_publication_claims.py",
    "scripts/audit_manuscript_stat_claims.py",
    "scripts/audit_manuscript_table_values.py",
    "scripts/audit_makefile_targets.py",
    "scripts/audit_package_metadata.py",
    "scripts/audit_publication_artifact_provenance.py",
    "scripts/audit_publication_docs.py",
    "scripts/audit_publication_evidence_manifest.py",
    "scripts/audit_result_artifact_schemas.py",
    "scripts/audit_result_value_ranges.py",
    "scripts/audit_reviewer_response_readiness.py",
    "scripts/audit_table_uncertainty_language.py",
}

REQUIRED_GENERATED_AUDITS = {
    "preproc_v0/repetition_familiarity/results/final_tables/aaai_publication_readiness_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/bundle_allowlist_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/citation_integrity_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/ci_workflow_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/dataset_accounting_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/external_data_policy_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/figure_asset_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/makefile_targets_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/manuscript_publication_claims_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/manuscript_stat_claims_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/manuscript_table_values_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/package_metadata_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/publication_artifact_provenance_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/publication_docs_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/publication_evidence_manifest_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/result_artifact_schema_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/result_value_range_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/reviewer_response_readiness_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/table_uncertainty_language_audit.csv",
}

FORBIDDEN_PREFIXES = (
    ".git/",
    "dist/",
    "wandb/",
    "runs/",
    "slurm_logs/",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit anonymous bundle allowlist coverage and freshness.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument("--output-prefix", default="bundle_allowlist_audit")
    return parser.parse_args()


def ready(ok: bool) -> str:
    return "ready" if ok else "incomplete"


def run_git(root: Path, args: list[str]) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        return []
    return [line for line in completed.stdout.splitlines() if line.strip()]


def git_available(root: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def import_bundle_module(root: Path):
    bundle_path = root / "scripts/make_anonymous_submission_bundle.py"
    spec = importlib.util.spec_from_file_location("make_anonymous_submission_bundle", bundle_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {bundle_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def missing_paths(required: set[str], observed: set[str]) -> list[str]:
    return sorted(required - observed)


def audit_allowlist(root: Path) -> list[AuditRow]:
    try:
        bundle = import_bundle_module(root)
        exact_paths = set(bundle.EXACT_PATHS)
        publication_paths = set(bundle.PUBLICATION_ARTIFACT_PATHS)
        figure_paths = set(bundle.FIGURE_PATHS)
        manifest_path = str(bundle.BUNDLE_MANIFEST_RELATIVE_PATH)
        import_status = AuditRow("bundle module imports", "ready", "loaded make_anonymous_submission_bundle.py")
    except Exception as exc:
        return [AuditRow("bundle module imports", "incomplete", str(exc))]

    has_git_index = git_available(root)
    cached_paths = set(run_git(root, ["ls-files", "--cached"])) if has_git_index else set(exact_paths)
    rows = [
        import_status,
        AuditRow("exact allowlist nonempty", ready(bool(exact_paths)), f"{len(exact_paths)} exact paths"),
        AuditRow(
            "publication artifact allowlist nonempty",
            ready(bool(publication_paths)),
            f"{len(publication_paths)} publication artifact paths",
        ),
        AuditRow("figure allowlist nonempty", ready(bool(figure_paths)), f"{len(figure_paths)} figure paths"),
        AuditRow(
            "manifest path allowlisted",
            ready(manifest_path in publication_paths and manifest_path in exact_paths),
            manifest_path,
        ),
    ]

    for item, required in [
        ("required source paths allowlisted", REQUIRED_SOURCE_PATHS),
        ("required audit scripts allowlisted", REQUIRED_AUDIT_SCRIPTS),
        ("required generated audits allowlisted", REQUIRED_GENERATED_AUDITS),
    ]:
        missing = missing_paths(required, exact_paths)
        rows.append(
            AuditRow(
                item,
                ready(not missing),
                f"{len(required)} required paths present" if not missing else "; ".join(missing[:8]),
            )
        )

    missing_publication = missing_paths(publication_paths, exact_paths)
    missing_figures = missing_paths(figure_paths, exact_paths)
    rows.append(
        AuditRow(
            "publication artifacts included by exact allowlist",
            ready(not missing_publication),
            f"{len(publication_paths)} publication artifacts included" if not missing_publication else "; ".join(missing_publication[:8]),
        )
    )
    rows.append(
        AuditRow(
            "figures included by exact allowlist",
            ready(not missing_figures),
            f"{len(figure_paths)} figure artifacts included" if not missing_figures else "; ".join(missing_figures[:8]),
        )
    )

    missing_from_filesystem = sorted(path for path in exact_paths if not (root / path).exists())
    rows.append(
        AuditRow(
            "allowlisted paths exist in working tree",
            ready(not missing_from_filesystem),
            "all exact paths exist" if not missing_from_filesystem else "; ".join(missing_from_filesystem[:8]),
        )
    )

    unstaged_or_untracked = sorted(exact_paths - cached_paths)
    rows.append(
        AuditRow(
            "allowlisted paths tracked or staged",
            ready(not unstaged_or_untracked),
            (
                "all exact paths are in the Git index"
                if has_git_index and not unstaged_or_untracked
                else "Git index unavailable; extracted bundle paths verified by file presence"
                if not has_git_index
                else "; ".join(unstaged_or_untracked[:8])
            ),
        )
    )

    forbidden = sorted(
        path
        for path in exact_paths
        if path.startswith(FORBIDDEN_PREFIXES) or "/.git/" in path or path.endswith(".tar.gz")
    )
    rows.append(
        AuditRow(
            "no forbidden bundle path classes",
            ready(not forbidden),
            "no Git metadata, archives, run logs, or dist paths allowlisted" if not forbidden else "; ".join(forbidden[:8]),
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
        "# Bundle Allowlist Audit",
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
    rows = audit_allowlist(args.root.resolve())
    write_outputs(args.output_dir, args.output_prefix, rows)
    return 0 if all(row.status == "ready" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
