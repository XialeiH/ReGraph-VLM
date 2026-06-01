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


README_MODEL_LABELS = {
    "ROI-MLP+CLIP": "ROI-MLP+CLIP",
    "Flat ReGraph+CLIP": "Flat ReGraph+CLIP",
    "Gated ReGraph/BNT+CLIP": "Gated ReGraph+CLIP",
}

README_METRICS = ["AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "brain_R@5"]

PUBLICATION_DOC_PATHS = [
    Path("ANONYMIZATION.md"),
    Path("DATASET_CARD.md"),
    Path("Makefile"),
    Path("MODEL_CARD.md"),
    Path("README.md"),
    Path("REPRODUCIBILITY.md"),
    Path("REVIEWER_RESPONSE.md"),
    Path("pyproject.toml"),
    Path("reports/neurips_report/BUILD.md"),
    Path("reports/neurips_report/may30.tex"),
    Path("reports/neurips_report/regraph_vlm_report.tex"),
    Path("reports/neurips_report/external_validation_dataset_scan.md"),
]


def deanon_patterns() -> list[str]:
    literals = [
        "".join(("Xia", "lei", " Huang")),
        ".".join(("xia" + "lei", "huang")),
        " ".join(("NYU", "Shanghai")),
        "".join(("xh", "2906")),
    ]
    return [re.escape(value) for value in literals]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit README/BUILD publication documentation consistency.")
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--build-doc", type=Path, default=Path("reports/neurips_report/BUILD.md"))
    parser.add_argument("--workflow", type=Path, default=Path(".github/workflows/publication-preflight.yml"))
    parser.add_argument(
        "--allfold-table",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables/table_allfold_final.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument("--output-prefix", default="publication_docs_audit")
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


def format_value(mean: str, std: str) -> str:
    return f"{float(mean):.4f} +/- {float(std):.4f}"


def audit_readme_values(readme: str, table_rows: list[dict[str, str]]) -> AuditRow:
    if not table_rows:
        return AuditRow("README main-result table values", "missing", "table_allfold_final.csv missing or empty")
    missing: list[str] = []
    for row in table_rows:
        source_model = row["model"]
        readme_model = README_MODEL_LABELS.get(source_model)
        if not readme_model:
            continue
        if readme_model not in readme:
            missing.append(f"{readme_model}: model row")
            continue
        for metric in README_METRICS:
            expected = format_value(row[f"{metric}_mean"], row[f"{metric}_std"])
            if expected not in readme:
                missing.append(f"{readme_model} {metric}={expected}")
    return AuditRow(
        "README main-result table values",
        ready(not missing),
        "all values match table_allfold_final.csv" if not missing else "; ".join(missing[:8]),
    )


def audit_docs(readme_path: Path, build_path: Path, workflow_path: Path, allfold_path: Path) -> list[AuditRow]:
    readme = read_text(readme_path)
    build = read_text(build_path)
    workflow = read_text(workflow_path)
    makefile = read_text(Path("Makefile"))
    anonymization = read_text(Path("ANONYMIZATION.md"))
    dataset_card = read_text(Path("DATASET_CARD.md"))
    model_card = read_text(Path("MODEL_CARD.md"))
    reproducibility = read_text(Path("REPRODUCIBILITY.md"))
    reviewer_response = read_text(Path("REVIEWER_RESPONSE.md"))
    pyproject = read_text(Path("pyproject.toml"))
    bundle_smoke = read_text(Path("scripts/smoke_test_anonymous_bundle_archive.py"))
    bundle_allowlist_audit = read_text(Path("scripts/audit_bundle_allowlist.py"))
    citation_audit = read_text(Path("scripts/audit_citation_integrity.py"))
    makefile_audit = read_text(Path("scripts/audit_makefile_targets.py"))
    ci_audit = read_text(Path("scripts/audit_ci_workflow.py"))
    schema_audit = read_text(Path("scripts/audit_result_artifact_schemas.py"))
    value_range_audit = read_text(Path("scripts/audit_result_value_ranges.py"))
    table_uncertainty_audit = read_text(Path("scripts/audit_table_uncertainty_language.py"))
    evidence_manifest_audit = read_text(Path("scripts/audit_publication_evidence_manifest.py"))
    combined = readme + "\n" + build
    publication_doc_text = "\n".join(read_text(path) for path in PUBLICATION_DOC_PATHS)
    rows = [
        AuditRow("Makefile exists", "ready" if makefile else "missing", "Makefile"),
        AuditRow("DATASET_CARD exists", "ready" if dataset_card else "missing", "DATASET_CARD.md"),
        AuditRow("MODEL_CARD exists", "ready" if model_card else "missing", "MODEL_CARD.md"),
        AuditRow("README exists", "ready" if readme else "missing", str(readme_path)),
        AuditRow("REPRODUCIBILITY doc exists", "ready" if reproducibility else "missing", "REPRODUCIBILITY.md"),
        AuditRow("REVIEWER_RESPONSE exists", "ready" if reviewer_response else "missing", "REVIEWER_RESPONSE.md"),
        AuditRow("pyproject exists", "ready" if pyproject else "missing", "pyproject.toml"),
        AuditRow("BUILD doc exists", "ready" if build else "missing", str(build_path)),
        AuditRow("publication preflight workflow exists", "ready" if workflow else "missing", str(workflow_path)),
        AuditRow(
            "active report documented",
            ready("reports/neurips_report/may30.tex" in readme and "may30.tex" in build),
            "README and BUILD point to may30.tex",
        ),
        AuditRow(
            "one-command preflight documented",
            ready("python3 scripts/run_publication_preflight.py" in readme and "python3 scripts/run_publication_preflight.py" in build),
            "preflight command present in README and BUILD",
        ),
        AuditRow(
            "Makefile reviewer targets documented",
            ready(
                "make preflight" in readme
                and "make compile" in readme
                and "make bundle-check" in readme
                and "make bundle-verify" in readme
                and "make bundle-smoke" in readme
                and "make preflight" in build
                and "make compile" in build
                and "make bundle-check" in build
                and "make bundle-verify" in build
                and "make bundle-smoke" in build
                and "make parameter-counts" in reproducibility
            ),
            "README, BUILD, and REPRODUCIBILITY document reviewer-facing make targets",
        ),
        AuditRow(
            "Makefile reviewer targets implemented",
            ready(
                "preflight:" in makefile
                and "compile:" in makefile
                and "bundle-check:" in makefile
                and "bundle:" in makefile
                and "bundle-verify:" in makefile
                and "bundle-smoke:" in makefile
                and "manuscript-audit:" in makefile
                and "parameter-counts:" in makefile
            ),
            "Makefile implements preflight, compile, bundle, bundle-verify, bundle-smoke, manuscript-audit, and parameter-count targets",
        ),
        AuditRow(
            "Makefile target audit documented",
            ready(
                "Makefile target audit" in readme
                and "Makefile target audit" in build
                and "Makefile target audit" in reproducibility
                and "scripts/audit_makefile_targets.py" in readme
                and "scripts/audit_makefile_targets.py" in build
                and "scripts/audit_makefile_targets.py" in reproducibility
                and "makefile_targets_audit.csv" in readme
                and "makefile_targets_audit.csv" in build
                and "makefile_targets_audit.csv" in reproducibility
                and "reports/neurips_report/may30.tex" in makefile_audit
            ),
            "README, BUILD, and REPRODUCIBILITY document the Makefile target audit and generated artifact",
        ),
        AuditRow(
            "CI workflow audit documented",
            ready(
                "CI workflow audit" in readme
                and "CI workflow audit" in build
                and "CI workflow audit" in reproducibility
                and "scripts/audit_ci_workflow.py" in readme
                and "scripts/audit_ci_workflow.py" in build
                and "scripts/audit_ci_workflow.py" in reproducibility
                and "ci_workflow_audit.csv" in readme
                and "ci_workflow_audit.csv" in build
                and "ci_workflow_audit.csv" in reproducibility
                and "--compile --require-clean" in ci_audit
            ),
            "README, BUILD, and REPRODUCIBILITY document the CI workflow audit and generated artifact",
        ),
        AuditRow(
            "result artifact schema audit documented",
            ready(
                "result artifact schema audit" in readme
                and "result artifact schema audit" in build
                and "result artifact schema audit" in reproducibility
                and "scripts/audit_result_artifact_schemas.py" in readme
                and "scripts/audit_result_artifact_schemas.py" in build
                and "scripts/audit_result_artifact_schemas.py" in reproducibility
                and "result_artifact_schema_audit.csv" in readme
                and "result_artifact_schema_audit.csv" in build
                and "result_artifact_schema_audit.csv" in reproducibility
                and "publication_paired_stats.csv" in schema_audit
            ),
            "README, BUILD, and REPRODUCIBILITY document result artifact schema auditing",
        ),
        AuditRow(
            "result value-range audit documented",
            ready(
                "result value-range audit" in readme
                and "result value-range audit" in build
                and "result value-range audit" in reproducibility
                and "scripts/audit_result_value_ranges.py" in readme
                and "scripts/audit_result_value_ranges.py" in build
                and "scripts/audit_result_value_ranges.py" in reproducibility
                and "result_value_range_audit.csv" in readme
                and "result_value_range_audit.csv" in build
                and "result_value_range_audit.csv" in reproducibility
                and "paired_t_p" in value_range_audit
            ),
            "README, BUILD, and REPRODUCIBILITY document result value-range auditing",
        ),
        AuditRow(
            "table uncertainty-language audit documented",
            ready(
                "table uncertainty-language audit" in readme
                and "table uncertainty-language audit" in build
                and "table uncertainty-language audit" in reproducibility
                and "scripts/audit_table_uncertainty_language.py" in readme
                and "scripts/audit_table_uncertainty_language.py" in build
                and "scripts/audit_table_uncertainty_language.py" in reproducibility
                and "table_uncertainty_language_audit.csv" in readme
                and "table_uncertainty_language_audit.csv" in build
                and "table_uncertainty_language_audit.csv" in reproducibility
                and "primary result captions disclose uncertainty scope" in table_uncertainty_audit
            ),
            "README, BUILD, and REPRODUCIBILITY document table caption uncertainty-scope auditing",
        ),
        AuditRow(
            "reproducibility guide linked",
            ready("REPRODUCIBILITY.md" in readme and "REPRODUCIBILITY.md" in build),
            "README and BUILD link the reproducibility guide",
        ),
        AuditRow(
            "dependency and large-data policy documented",
            ready(
                "torch" in reproducibility
                and "nibabel" in reproducibility
                and "HPC" in reproducibility
                and "scratch" in reproducibility
                and "not full HCP-MMP 180-ROI" in reproducibility
                and "python3 -m pip install -e '.[dev]'" in reproducibility
                and "DATASET_CARD.md" in reproducibility
            ),
            "REPRODUCIBILITY.md covers model dependencies, neuroimaging dependencies, install extras, HPC scratch storage, external-validation limits, and the dataset card",
        ),
        AuditRow(
            "dataset card documents data accounting",
            ready(
                "split_accounting.csv" in dataset_card
                and "session_order_pair_qc.csv" in dataset_card
                and "fold_difficulty_qc.csv" in dataset_card
                and "table_external_visual_roi_smoke.csv" in dataset_card
                and "not full HCP-MMP 180-ROI external replications" in dataset_card
                and "remote HPC scratch storage" in dataset_card
                and "DATASET_CARD.md" in readme
                and "DATASET_CARD.md" in build
                and "DATASET_CARD.md" in reproducibility
            ),
            "DATASET_CARD.md documents split counts, session/order QC, fold difficulty, external-validation limits, and large-data policy",
        ),
        AuditRow(
            "model card documents model scope",
            ready(
                "fixed-order anatomical ROI-token" in model_card
                and "gated ROI-preserving readout" in model_card
                and "CLIP" in model_card
                and "Explicit fixed adjacency is not the source" in model_card
                and "not a clinical" in model_card
                and "task-matched component" in model_card
                and "MODEL_CARD.md" in readme
                and "MODEL_CARD.md" in build
                and "MODEL_CARD.md" in reproducibility
            ),
            "MODEL_CARD.md documents model scope, supported claims, non-claims, limitations, and reviewer-facing evidence",
        ),
        AuditRow(
            "reviewer response memo documents concerns",
            ready(
                "Central Framing" in reviewer_response
                and "Graph-Adjacency Novelty" in reviewer_response
                and "Session/Order Confounds" in reviewer_response
                and "Statistical Reporting" in reviewer_response
                and "External Validation" in reviewer_response
                and "MODEL_CARD.md" in reviewer_response
                and "DATASET_CARD.md" in reviewer_response
                and "REVIEWER_RESPONSE.md" in readme
                and "REVIEWER_RESPONSE.md" in build
                and "REVIEWER_RESPONSE.md" in reproducibility
            ),
            "REVIEWER_RESPONSE.md maps likely reviewer concerns to manuscript/result artifacts",
        ),
        AuditRow(
            "pyproject metadata aligned",
            ready(
                'name = "regraph-vlm"' in pyproject
                and "Fixed-order ROI-token brain graph" in pyproject
                and 'requires-python = ">=3.9"' in pyproject
                and "publication =" in pyproject
                and "legacy-graph =" in pyproject
                and '"models*"' in pyproject
            ),
            "pyproject names ReGraph-VLM, exposes dependency extras, and packages current models code",
        ),
        AuditRow(
            "package metadata audit documented",
            ready("package metadata" in readme and "package metadata" in build and "package metadata audit" in reproducibility),
            "README, BUILD, and REPRODUCIBILITY mention structural package metadata auditing",
        ),
        AuditRow(
            "external data policy audit documented",
            ready(
                "external data policy audit" in readme
                and "external data policy audit" in build
                and "external data policy audit" in reproducibility
                and "external data policy audit" in dataset_card
                and "scripts/external_data_policy.py" in reproducibility
                and "scripts/audit_external_data_policy.py" in build
            ),
            "README, BUILD, REPRODUCIBILITY, and DATASET_CARD mention HPC-only external-data policy auditing",
        ),
        AuditRow(
            "compile path documented",
            ready("python3 scripts/run_publication_preflight.py --compile" in readme and "python3 scripts/run_publication_preflight.py --compile" in build),
            "--compile command present in README and BUILD",
        ),
        AuditRow(
            "portable TeX note documented",
            ready("this machine currently" not in build and "If no local TeX compiler is available" in build),
            "BUILD uses a portable TeX availability note instead of local-machine state",
        ),
        AuditRow(
            "statistical-claims audit documented",
            ready("statistical claims" in readme and "statistical claims" in build),
            "README and BUILD mention statistical-claims verification",
        ),
        AuditRow(
            "artifact-provenance audit documented",
            ready("artifact provenance" in readme and "artifact provenance" in build),
            "README and BUILD mention artifact-provenance verification",
        ),
        AuditRow(
            "manuscript framing guardrails documented",
            ready("framing guardrails" in readme and "framing guardrails" in build and "fold_07" in readme and "fold_07" in build),
            "README and BUILD mention adjacency/component/external/fold_07/implementation guardrails",
        ),
        AuditRow(
            "reviewer-response audit documented",
            ready("reviewer-response readiness audit" in readme and "reviewer-response readiness audit" in build),
            "README and BUILD mention reviewer-response readiness audit",
        ),
        AuditRow(
            "publication evidence manifest documented",
            ready("Publication Evidence Manifest" in readme and "Publication Evidence Manifest" in build),
            "README and BUILD mention the publication evidence manifest",
        ),
        AuditRow(
            "publication evidence manifest audit documented",
            ready(
                "publication evidence manifest audit" in readme
                and "publication evidence manifest audit" in build
                and "publication evidence manifest audit" in reproducibility
                and "scripts/audit_publication_evidence_manifest.py" in readme
                and "scripts/audit_publication_evidence_manifest.py" in build
                and "scripts/audit_publication_evidence_manifest.py" in reproducibility
                and "publication_evidence_manifest_audit.csv" in readme
                and "publication_evidence_manifest_audit.csv" in build
                and "publication_evidence_manifest_audit.csv" in reproducibility
                and "required reviewer claims covered" in evidence_manifest_audit
            ),
            "README, BUILD, and REPRODUCIBILITY document reviewer evidence-manifest auditing",
        ),
        AuditRow(
            "anonymous submission bundle documented",
            ready(
                "Anonymous Submission Bundle" in readme
                and "Anonymous Submission Bundle" in build
                and "ANONYMIZATION.md" in readme
                and "ANONYMIZATION.md" in build
                and "SHA-256" in readme
                and "SHA-256" in build
                and "byte-stable" in readme
                and "byte-stable" in build
                and "byte-identical" in readme
                and "byte-identical" in build
                and "byte-identical" in reproducibility
                and "byte-identical" in anonymization
                and "byte level" in readme
                and "byte level" in build
                and "byte level" in reproducibility
                and "byte level" in anonymization
            ),
            "README and BUILD mention anonymous bundle instructions, checksum, byte-stable archive metadata, byte-identical smoke testing, and byte-level deanonymization scan",
        ),
        AuditRow(
            "anonymous bundle manifest documented",
            ready(
                "anonymous_bundle_manifest.csv" in readme
                and "anonymous_bundle_manifest.csv" in build
                and "anonymous_bundle_manifest.csv" in reproducibility
                and "anonymous_bundle_manifest.csv" in anonymization
                and "verify_anonymous_bundle_manifest.py" in readme
                and "verify_anonymous_bundle_manifest.py" in build
                and "verify_anonymous_bundle_manifest.py" in reproducibility
                and "verify_anonymous_bundle_manifest.py" in anonymization
                and "smoke_test_anonymous_bundle_archive.py" in readme
                and "smoke_test_anonymous_bundle_archive.py" in build
                and "smoke_test_anonymous_bundle_archive.py" in reproducibility
                and "smoke_test_anonymous_bundle_archive.py" in anonymization
                and "symlink/hardlink" in readme
                and "symlink/hardlink" in build
                and "symlink/hardlink" in reproducibility
                and "symlink/hardlink" in anonymization
                and "extracted anonymous bundle" in readme
                and "extracted anonymous bundle" in build
                and "extracted anonymous bundle" in reproducibility
                and "extracted anonymous bundle" in anonymization
                and "compile-required" in readme
                and "compile-required" in build
                and "compile-required" in reproducibility
                and "compile-required" in anonymization
                and "sidecar manifest" in readme
                and "sidecar manifest" in build
                and "sidecar manifest" in reproducibility
                and "sidecar paths" in anonymization
                and "anonymous sidecar manifest" in read_text(Path("scripts/run_publication_preflight.py"))
                and ".github/workflows/publication-preflight.yml" in read_text(Path("scripts/make_anonymous_submission_bundle.py"))
                and "repeat anonymous bundle build" in bundle_smoke
                and "byte-stable archive verified" in bundle_smoke
            ),
            "README, BUILD, REPRODUCIBILITY, and ANONYMIZATION document the per-file bundle manifest verifier, sidecar manifest support, byte-identical archive smoke test, and compile-capable extracted-bundle preflight",
        ),
        AuditRow(
            "bundle allowlist audit documented",
            ready(
                "bundle allowlist audit" in readme
                and "bundle allowlist audit" in build
                and "bundle allowlist audit" in reproducibility
                and "scripts/audit_bundle_allowlist.py" in readme
                and "scripts/audit_bundle_allowlist.py" in build
                and "scripts/audit_bundle_allowlist.py" in reproducibility
                and "bundle_allowlist_audit.csv" in readme
                and "bundle_allowlist_audit.csv" in build
                and "bundle_allowlist_audit.csv" in reproducibility
                and "allowlisted paths tracked or staged" in bundle_allowlist_audit
            ),
            "README, BUILD, and REPRODUCIBILITY document anonymous bundle allowlist freshness auditing",
        ),
        AuditRow(
            "citation integrity audit documented",
            ready(
                "citation integrity audit" in readme
                and "citation integrity audit" in build
                and "citation integrity audit" in reproducibility
                and "scripts/audit_citation_integrity.py" in readme
                and "scripts/audit_citation_integrity.py" in build
                and "scripts/audit_citation_integrity.py" in reproducibility
                and "citation_integrity_audit.csv" in readme
                and "citation_integrity_audit.csv" in build
                and "citation_integrity_audit.csv" in reproducibility
                and "all cited keys defined" in citation_audit
            ),
            "README, BUILD, and REPRODUCIBILITY document citation and bibliography integrity auditing",
        ),
        AuditRow(
            "CI runs publication preflight",
            ready("python scripts/run_publication_preflight.py" in workflow),
            "workflow runs preflight command",
        ),
        AuditRow(
            "CI requires TeX compilation",
            ready("--compile" in workflow and "latexmk" in workflow),
            "workflow installs TeX and runs compile-required preflight",
        ),
        AuditRow(
            "CI installs recommended TeX packages",
            ready("texlive-latex-recommended" in workflow and "texlive-latex-extra" in workflow),
            "workflow includes recommended and extra LaTeX package bundles",
        ),
        AuditRow(
            "CI verifies generated artifacts",
            ready("--require-clean" in workflow),
            "workflow delegates tracked/untracked artifact drift check to preflight",
        ),
        AuditRow(
            "CI catches untracked artifacts",
            ready("git diff --exit-code" not in workflow and "--require-clean" in workflow),
            "workflow uses preflight working-tree cleanliness instead of tracked-file diff only",
        ),
        AuditRow(
            "CI covers main and pull requests",
            ready("branches:" in workflow and "- main" in workflow and "pull_request:" in workflow),
            "workflow triggers on main pushes and pull requests",
        ),
        AuditRow("stale may23 references", ready("may23" not in combined), "none found" if "may23" not in combined else "may23 found"),
    ]

    deanon_hits: list[str] = []
    for pattern in deanon_patterns():
        deanon_hits.extend(re.findall(pattern, publication_doc_text))
    rows.append(
        AuditRow(
            "publication-facing docs deanonymizing strings",
            ready(not deanon_hits),
            "none found" if not deanon_hits else ", ".join(sorted(set(deanon_hits))),
        )
    )
    rows.append(audit_readme_values(readme, read_csv(allfold_path)))
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
        "# Publication Documentation Audit",
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
    rows = audit_docs(args.readme, args.build_doc, args.workflow, args.allfold_table)
    write_outputs(args.output_dir, args.output_prefix, rows)
    return 0 if all(row.status == "ready" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
