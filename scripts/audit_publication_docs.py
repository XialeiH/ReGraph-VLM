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
    pyproject = read_text(Path("pyproject.toml"))
    combined = readme + "\n" + build
    publication_doc_text = "\n".join(read_text(path) for path in PUBLICATION_DOC_PATHS)
    rows = [
        AuditRow("Makefile exists", "ready" if makefile else "missing", "Makefile"),
        AuditRow("DATASET_CARD exists", "ready" if dataset_card else "missing", "DATASET_CARD.md"),
        AuditRow("MODEL_CARD exists", "ready" if model_card else "missing", "MODEL_CARD.md"),
        AuditRow("README exists", "ready" if readme else "missing", str(readme_path)),
        AuditRow("REPRODUCIBILITY doc exists", "ready" if reproducibility else "missing", "REPRODUCIBILITY.md"),
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
            ),
            "README and BUILD mention anonymous bundle instructions, checksum, and byte-stable archive metadata",
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
            ),
            "README, BUILD, REPRODUCIBILITY, and ANONYMIZATION document the per-file bundle manifest verifier and archive smoke test",
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
