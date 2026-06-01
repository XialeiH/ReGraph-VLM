#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Step:
    name: str
    status: str
    evidence: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the publication preflight for reports/neurips_report/may30.tex.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tex", type=Path, default=Path("reports/neurips_report/may30.tex"))
    parser.add_argument(
        "--final-tables-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument("--external-summary-dir", type=Path, default=Path("external_validation/summary"))
    parser.add_argument("--compile", action="store_true", help="Compile the TeX file when a supported TeX tool is available.")
    parser.add_argument("--require-clean", action="store_true", help="Fail if preflight leaves tracked or untracked Git changes.")
    return parser.parse_args()


def run_command(root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def require_ok(name: str, completed: subprocess.CompletedProcess[str]) -> Step:
    if completed.returncode == 0:
        return Step(name, "ready", first_lines(completed.stdout))
    raise RuntimeError(f"{name} failed with exit code {completed.returncode}\n{completed.stdout}")


def first_lines(text: str, n: int = 3) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " | ".join(lines[:n]) if lines else "ok"


def audit_status(path: Path, expected_ready: int) -> Step:
    if not path.exists():
        return Step(path.name, "missing", f"{path} not found")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    non_ready = [row for row in rows if row.get("status") != "ready"]
    if non_ready:
        evidence = "; ".join(f"{row.get('item')}: {row.get('status')}" for row in non_ready[:5])
        return Step(path.name, "incomplete", evidence)
    status = "ready" if len(rows) >= expected_ready else "incomplete"
    evidence = f"{len(rows)}/{expected_ready} minimum checks ready"
    return Step(path.name, status, evidence)


def detect_tex_tool() -> str | None:
    for tool in ["latexmk", "pdflatex", "tectonic"]:
        if shutil.which(tool):
            return tool
    return None


def compile_tex(root: Path, tex: Path, tool: str) -> Step:
    tex_path = root / tex
    tex_dir = tex_path.parent
    tex_name = tex_path.name
    stem = tex_path.stem
    with tempfile.TemporaryDirectory(prefix="regraph_tex_build_") as tmp:
        build_dir = Path(tmp) / tex_dir.name
        shutil.copytree(tex_dir, build_dir)
        if tool == "latexmk":
            cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", tex_name]
            completed = run_command(build_dir, cmd)
        elif tool == "pdflatex":
            commands = [
                ["pdflatex", "-interaction=nonstopmode", tex_name],
                ["bibtex", stem],
                ["pdflatex", "-interaction=nonstopmode", tex_name],
                ["pdflatex", "-interaction=nonstopmode", tex_name],
            ]
            output = []
            code = 0
            for cmd in commands:
                completed = run_command(build_dir, cmd)
                output.append(completed.stdout)
                code = completed.returncode
                if code != 0:
                    break
            completed = subprocess.CompletedProcess(commands[-1], code, "\n".join(output), "")
        else:
            completed = run_command(build_dir, ["tectonic", tex_name])
        if completed.returncode == 0:
            built_pdf = build_dir / f"{stem}.pdf"
            if not built_pdf.exists():
                return Step("TeX compile", "incomplete", f"{tool} exited 0 but did not produce {built_pdf.name}")
            return Step("TeX compile", "ready", f"{tool} produced {tex_path.with_suffix('.pdf').name} in isolated build dir")
        return Step("TeX compile", "incomplete", first_lines(completed.stdout, n=8))


def audit_clean_worktree(root: Path) -> Step:
    completed = run_command(root, ["git", "status", "--porcelain"])
    if completed.returncode != 0:
        return Step("Git working tree clean", "incomplete", first_lines(completed.stdout, n=8))
    if completed.stdout.strip():
        return Step("Git working tree clean", "incomplete", first_lines(completed.stdout, n=8))
    return Step("Git working tree clean", "ready", "no tracked or untracked artifact drift")


def verify_model_parameter_counts(root: Path, final: Path) -> Step:
    if importlib.util.find_spec("torch") is None:
        return Step("model parameter-count code verification", "skipped", "torch not installed in this environment")
    return require_ok(
        "model parameter-count code verification",
        run_command(
            root,
            [
                sys.executable,
                "scripts/verify_model_parameter_counts.py",
                "--parameter-counts",
                str(final / "model_parameter_counts.csv"),
            ],
        ),
    )


def audit_bundle_manifest(path: Path) -> Step:
    if not path.exists():
        return Step("anonymous bundle manifest", "missing", f"{path} not found")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"path", "bytes", "sha256"}
    if not rows:
        return Step("anonymous bundle manifest", "incomplete", "manifest has no rows")
    if set(rows[0]) != required:
        return Step("anonymous bundle manifest", "incomplete", f"columns are {sorted(rows[0])}")

    paths = [row["path"] for row in rows]
    duplicate_paths = sorted({item for item in paths if paths.count(item) > 1})
    bad_rows: list[str] = []
    for row in rows:
        try:
            size = int(row["bytes"])
        except ValueError:
            bad_rows.append(f"{row['path']}: non-integer bytes")
            continue
        digest = row["sha256"]
        if size < 0:
            bad_rows.append(f"{row['path']}: negative bytes")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            bad_rows.append(f"{row['path']}: invalid sha256")

    if duplicate_paths:
        return Step("anonymous bundle manifest", "incomplete", f"duplicate paths: {', '.join(duplicate_paths[:5])}")
    if bad_rows:
        return Step("anonymous bundle manifest", "incomplete", "; ".join(bad_rows[:5]))
    if any(row_path.endswith("anonymous_bundle_manifest.csv") for row_path in paths):
        return Step("anonymous bundle manifest", "incomplete", "manifest should exclude its own row")

    status = "ready" if len(rows) >= 80 else "incomplete"
    return Step("anonymous bundle manifest", status, f"{len(rows)} files with source bytes and SHA-256 values")


def write_summary(rows: list[Step]) -> str:
    lines = [
        "# Publication Preflight",
        "",
        "| Step | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        evidence = row.evidence.replace("|", "/")
        lines.append(f"| {row.name} | {row.status} | {evidence} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    tex = args.tex
    final = args.final_tables_dir
    external = args.external_summary_dir
    bundle_manifest = final / "anonymous_bundle_manifest.csv"
    rows: list[Step] = []

    rows.append(
        require_ok(
            "materialize publication artifacts",
            run_command(
                root,
                [
                    sys.executable,
                    "scripts/materialize_publication_readiness_artifacts.py",
                    "--final-tables-dir",
                    str(final),
                    "--external-summary-dir",
                    str(external),
                    "--source-tex",
                    str(tex),
                ],
            ),
        )
    )
    rows.append(
        require_ok(
            "AAAI publication readiness audit",
            run_command(
                root,
                [
                    sys.executable,
                    "scripts/audit_aaai_publication_readiness.py",
                    "--final-tables-dir",
                    str(final),
                    "--external-summary-dir",
                    str(external),
                ],
            ),
        )
    )
    rows.append(audit_status(root / final / "aaai_publication_readiness_audit.csv", 33))

    rows.append(
        require_ok(
            "full manuscript/result audit",
            run_command(
                root,
                [
                    sys.executable,
                    "scripts/audit_manuscript_publication_claims.py",
                    "--tex",
                    str(tex),
                    "--final-tables-dir",
                    str(final),
                    "--output-dir",
                    str(final),
                ],
            ),
        )
    )
    rows.append(audit_status(root / final / "manuscript_publication_claims_audit.csv", 55))

    rows.append(
        require_ok(
            "publication docs audit",
            run_command(
                root,
                [
                    sys.executable,
                    "scripts/audit_publication_docs.py",
                    "--output-dir",
                    str(final),
                ],
            ),
        )
    )
    rows.append(audit_status(root / final / "publication_docs_audit.csv", 48))

    rows.append(
        require_ok(
            "citation integrity audit",
            run_command(
                root,
                [
                    sys.executable,
                    "scripts/audit_citation_integrity.py",
                    "--tex",
                    str(tex),
                    "--bib",
                    "reports/neurips_report/references.bib",
                    "--output-dir",
                    str(final),
                ],
            ),
        )
    )
    rows.append(audit_status(root / final / "citation_integrity_audit.csv", 11))

    rows.append(
        require_ok(
            "figure asset audit",
            run_command(
                root,
                [
                    sys.executable,
                    "scripts/audit_figure_assets.py",
                    "--tex",
                    str(tex),
                    "--output-dir",
                    str(final),
                ],
            ),
        )
    )
    rows.append(audit_status(root / final / "figure_asset_audit.csv", 9))

    rows.append(
        require_ok(
            "table uncertainty-language audit",
            run_command(
                root,
                [
                    sys.executable,
                    "scripts/audit_table_uncertainty_language.py",
                    "--tex",
                    str(tex),
                    "--output-dir",
                    str(final),
                ],
            ),
        )
    )
    rows.append(audit_status(root / final / "table_uncertainty_language_audit.csv", 25))

    rows.append(
        require_ok(
            "bundle allowlist audit",
            run_command(
                root,
                [
                    sys.executable,
                    "scripts/audit_bundle_allowlist.py",
                    "--output-dir",
                    str(final),
                ],
            ),
        )
    )
    rows.append(audit_status(root / final / "bundle_allowlist_audit.csv", 13))

    rows.append(
        require_ok(
            "Makefile target audit",
            run_command(
                root,
                [
                    sys.executable,
                    "scripts/audit_makefile_targets.py",
                    "--output-dir",
                    str(final),
                ],
            ),
        )
    )
    rows.append(audit_status(root / final / "makefile_targets_audit.csv", 9))

    rows.append(
        require_ok(
            "CI workflow audit",
            run_command(
                root,
                [
                    sys.executable,
                    "scripts/audit_ci_workflow.py",
                    "--output-dir",
                    str(final),
                ],
            ),
        )
    )
    rows.append(audit_status(root / final / "ci_workflow_audit.csv", 10))

    rows.append(
        require_ok(
            "external data policy audit",
            run_command(
                root,
                [
                    sys.executable,
                    "scripts/audit_external_data_policy.py",
                    "--output-dir",
                    str(final),
                ],
            ),
        )
    )
    rows.append(audit_status(root / final / "external_data_policy_audit.csv", 6))

    rows.append(
        require_ok(
            "package metadata audit",
            run_command(
                root,
                [
                    sys.executable,
                    "scripts/audit_package_metadata.py",
                    "--output-dir",
                    str(final),
                ],
            ),
        )
    )
    rows.append(audit_status(root / final / "package_metadata_audit.csv", 11))

    rows.append(
        require_ok(
            "reviewer-response readiness audit",
            run_command(
                root,
                [
                    sys.executable,
                    "scripts/audit_reviewer_response_readiness.py",
                    "--tex",
                    str(tex),
                    "--final-tables-dir",
                    str(final),
                    "--external-summary-dir",
                    str(external),
                    "--output-dir",
                    str(final),
                ],
            ),
        )
    )
    rows.append(audit_status(root / final / "reviewer_response_readiness_audit.csv", 11))

    rows.append(
        require_ok(
            "manuscript table-values audit",
            run_command(
                root,
                [
                    sys.executable,
                    "scripts/audit_manuscript_table_values.py",
                    "--tex",
                    str(tex),
                    "--final-tables-dir",
                    str(final),
                ],
            ),
        )
    )
    rows.append(audit_status(root / final / "manuscript_table_values_audit.csv", 24))
    rows.append(verify_model_parameter_counts(root, final))

    rows.append(
        require_ok(
            "result artifact schema audit",
            run_command(
                root,
                [
                    sys.executable,
                    "scripts/audit_result_artifact_schemas.py",
                    "--final-tables-dir",
                    str(final),
                    "--external-summary-dir",
                    str(external),
                ],
            ),
        )
    )
    rows.append(audit_status(root / final / "result_artifact_schema_audit.csv", 26))

    rows.append(
        require_ok(
            "result value-range audit",
            run_command(
                root,
                [
                    sys.executable,
                    "scripts/audit_result_value_ranges.py",
                    "--final-tables-dir",
                    str(final),
                    "--external-summary-dir",
                    str(external),
                ],
            ),
        )
    )
    rows.append(audit_status(root / final / "result_value_range_audit.csv", 26))

    rows.append(
        require_ok(
            "manuscript statistical-claims audit",
            run_command(
                root,
                [
                    sys.executable,
                    "scripts/audit_manuscript_stat_claims.py",
                    "--tex",
                    str(tex),
                    "--paired-stats",
                    str(final / "publication_paired_stats.csv"),
                    "--laion-pairwise",
                    str(external / "laion_fmri_visual_roi_pairwise_tests.csv"),
                    "--output-dir",
                    str(final),
                ],
            ),
        )
    )
    rows.append(audit_status(root / final / "manuscript_stat_claims_audit.csv", 22))

    rows.append(
        require_ok(
            "publication artifact provenance audit",
            run_command(
                root,
                [
                    sys.executable,
                    "scripts/audit_publication_artifact_provenance.py",
                    "--final-tables-dir",
                    str(final),
                    "--source-tex",
                    str(tex),
                ],
            ),
        )
    )
    rows.append(audit_status(root / final / "publication_artifact_provenance_audit.csv", 37))

    manuscript_only_dir = Path("/tmp/regraph_report_preflight")
    rows.append(
        require_ok(
            "manuscript-only audit",
            run_command(
                root,
                [
                    sys.executable,
                    "scripts/audit_manuscript_publication_claims.py",
                    "--tex",
                    str(tex),
                    "--manuscript-only",
                    "--output-dir",
                    str(manuscript_only_dir),
                ],
            ),
        )
    )
    rows.append(audit_status(manuscript_only_dir / "manuscript_publication_claims_audit.csv", 25))

    rows.append(
        require_ok(
            "publication evidence manifest",
            run_command(
                root,
                [
                    sys.executable,
                    "scripts/generate_publication_evidence_manifest.py",
                    "--tex",
                    str(tex),
                    "--final-tables-dir",
                    str(final),
                    "--external-summary-dir",
                    str(external),
                    "--output",
                    str(final / "publication_evidence_manifest.md"),
                ],
            ),
        )
    )
    rows.append(
        require_ok(
            "publication evidence manifest audit",
            run_command(
                root,
                [
                    sys.executable,
                    "scripts/audit_publication_evidence_manifest.py",
                    "--manifest",
                    str(final / "publication_evidence_manifest.md"),
                    "--final-tables-dir",
                    str(final),
                    "--external-summary-dir",
                    str(external),
                    "--output-dir",
                    str(final),
                ],
            ),
        )
    )
    rows.append(audit_status(root / final / "publication_evidence_manifest_audit.csv", 10))

    rows.append(
        require_ok(
            "anonymous submission bundle dry-run",
            run_command(
                root,
                [
                    sys.executable,
                    "scripts/make_anonymous_submission_bundle.py",
                    "--dry-run",
                    "--manifest-output",
                    str(bundle_manifest),
                ],
            ),
        )
    )
    rows.append(audit_bundle_manifest(root / bundle_manifest))
    rows.append(
        require_ok(
            "anonymous bundle manifest verification",
            run_command(
                root,
                [
                    sys.executable,
                    "scripts/verify_anonymous_bundle_manifest.py",
                    "--manifest",
                    str(bundle_manifest),
                    "--source",
                    "auto",
                ],
            ),
        )
    )
    with tempfile.TemporaryDirectory(prefix="regraph_bundle_sidecar_") as tmp:
        sidecar_manifest = Path(tmp) / "anonymous_bundle_manifest_sidecar.csv"
        rows.append(
            require_ok(
                "anonymous sidecar manifest dry-run",
                run_command(
                    root,
                    [
                        sys.executable,
                        "scripts/make_anonymous_submission_bundle.py",
                        "--dry-run",
                        "--manifest-output",
                        str(sidecar_manifest),
                    ],
                ),
            )
        )
        rows.append(
            require_ok(
                "anonymous sidecar manifest verification",
                run_command(
                    root,
                    [
                        sys.executable,
                        "scripts/verify_anonymous_bundle_manifest.py",
                        "--manifest",
                        str(sidecar_manifest),
                        "--source",
                        "auto",
                    ],
                ),
            )
        )
    rows.append(
        require_ok(
            "anonymous bundle archive smoke test",
            run_command(root, [sys.executable, "scripts/smoke_test_anonymous_bundle_archive.py"]),
        )
    )

    tex_tool = detect_tex_tool()
    if args.compile:
        if tex_tool is None:
            rows.append(Step("TeX compile", "missing", "no latexmk, pdflatex, or tectonic found"))
        else:
            rows.append(compile_tex(root, tex, tex_tool))
    else:
        rows.append(Step("TeX compile", "skipped", tex_tool or "no local TeX compiler found; pass --compile to require PDF build"))

    if args.require_clean:
        rows.append(audit_clean_worktree(root))

    summary = write_summary(rows)
    print(summary, end="")
    return 0 if all(row.status in {"ready", "skipped"} for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
