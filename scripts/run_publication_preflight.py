#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
    rows.append(audit_status(root / final / "aaai_publication_readiness_audit.csv", 32))

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
    rows.append(audit_status(root / final / "publication_artifact_provenance_audit.csv", 19))

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
    rows.append(audit_status(root / final / "manuscript_publication_claims_audit.csv", 53))

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
    rows.append(audit_status(root / final / "publication_docs_audit.csv", 17))

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
    rows.append(audit_status(root / final / "manuscript_table_values_audit.csv", 23))

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
    rows.append(audit_status(manuscript_only_dir / "manuscript_publication_claims_audit.csv", 23))

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
