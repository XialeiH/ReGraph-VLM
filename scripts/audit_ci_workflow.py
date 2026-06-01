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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the publication-preflight CI workflow.")
    parser.add_argument("--workflow", type=Path, default=Path(".github/workflows/publication-preflight.yml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument("--output-prefix", default="ci_workflow_audit")
    return parser.parse_args()


def ready(ok: bool) -> str:
    return "ready" if ok else "incomplete"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def has_all(text: str, fragments: list[str]) -> bool:
    normalized = normalize(text)
    return all(fragment in normalized for fragment in fragments)


def audit_workflow(workflow: Path) -> list[AuditRow]:
    if not workflow.exists():
        return [AuditRow("publication-preflight workflow exists", "missing", f"{workflow} not found")]
    text = workflow.read_text(encoding="utf-8")
    return [
        AuditRow(
            "workflow name",
            ready("name: publication-preflight" in text),
            "publication-preflight workflow name is present",
        ),
        AuditRow(
            "main push trigger",
            ready(has_all(text, ["push:", "branches:", "- main"])),
            "workflow runs on main pushes",
        ),
        AuditRow(
            "pull request trigger",
            ready("pull_request:" in text),
            "workflow runs on pull requests",
        ),
        AuditRow(
            "ubuntu runner",
            ready("runs-on: ubuntu-latest" in text),
            "publication preflight uses ubuntu-latest",
        ),
        AuditRow(
            "checkout and Python setup",
            ready("actions/checkout@v4" in text and "actions/setup-python@v5" in text),
            "workflow checks out source and configures Python",
        ),
        AuditRow(
            "Python version",
            ready('python-version: "3.10"' in text),
            "workflow pins Python 3.10 for audit execution",
        ),
        AuditRow(
            "TeX dependencies",
            ready(
                has_all(
                    text,
                    [
                        "latexmk",
                        "texlive-latex-base",
                        "texlive-latex-recommended",
                        "texlive-latex-extra",
                        "texlive-fonts-recommended",
                    ],
                )
            ),
            "workflow installs latexmk and required TeX package bundles",
        ),
        AuditRow(
            "audit dependencies",
            ready("python -m pip install pandas" in text),
            "workflow installs lightweight audit dependency set",
        ),
        AuditRow(
            "compile-required preflight",
            ready("python scripts/run_publication_preflight.py --compile --require-clean" in normalize(text)),
            "workflow runs compile-required publication preflight",
        ),
        AuditRow(
            "clean-worktree enforcement",
            ready("--require-clean" in text),
            "workflow fails if generated artifacts drift or new files appear",
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
        "# CI Workflow Audit",
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
    rows = audit_workflow(args.workflow)
    write_outputs(args.output_dir, args.output_prefix, rows)
    return 0 if all(row.status == "ready" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
