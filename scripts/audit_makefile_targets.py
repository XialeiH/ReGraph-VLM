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


REQUIRED_TARGETS = {
    "preflight": ["python3 scripts/run_publication_preflight.py"],
    "compile": ["python3 scripts/run_publication_preflight.py --compile"],
    "bundle-check": ["python3 scripts/make_anonymous_submission_bundle.py --dry-run"],
    "bundle": [
        "python3 scripts/make_anonymous_submission_bundle.py",
        "--manifest-output preproc_v0/repetition_familiarity/results/final_tables/anonymous_bundle_manifest.csv",
    ],
    "bundle-verify": ["python3 scripts/verify_anonymous_bundle_manifest.py"],
    "bundle-smoke": ["python3 scripts/smoke_test_anonymous_bundle_archive.py"],
    "manuscript-audit": [
        "python3 scripts/audit_manuscript_publication_claims.py",
        "--tex reports/neurips_report/may30.tex",
        "--manuscript-only",
        "--output-dir /tmp/regraph_report_preflight",
    ],
    "parameter-counts": ["python3 scripts/verify_model_parameter_counts.py"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit reviewer-facing Makefile publication targets.")
    parser.add_argument("--makefile", type=Path, default=Path("Makefile"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument("--output-prefix", default="makefile_targets_audit")
    return parser.parse_args()


def ready(ok: bool) -> str:
    return "ready" if ok else "incomplete"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\\\n", " ")).strip()


def target_blocks(text: str) -> dict[str, str]:
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+):(?:\s|$)", line)
        if match and not line.startswith("\t"):
            current = match.group(1)
            blocks.setdefault(current, [])
            continue
        if current and (line.startswith("\t") or not line.strip()):
            blocks[current].append(line)
    return {target: normalize("\n".join(lines)) for target, lines in blocks.items()}


def phony_targets(text: str) -> set[str]:
    targets: set[str] = set()
    for line in text.splitlines():
        if line.startswith(".PHONY:"):
            targets.update(line.split(":", 1)[1].split())
    return targets


def audit_makefile(makefile: Path) -> list[AuditRow]:
    if not makefile.exists():
        return [AuditRow("Makefile exists", "missing", f"{makefile} not found")]
    text = makefile.read_text(encoding="utf-8")
    blocks = target_blocks(text)
    phony = phony_targets(text)
    rows = [
        AuditRow(
            ".PHONY reviewer targets",
            ready(all(target in phony for target in REQUIRED_TARGETS)),
            "all reviewer-facing targets are declared phony"
            if all(target in phony for target in REQUIRED_TARGETS)
            else "missing: " + ", ".join(sorted(set(REQUIRED_TARGETS) - phony)),
        )
    ]
    for target, fragments in REQUIRED_TARGETS.items():
        command = blocks.get(target, "")
        missing = [fragment for fragment in fragments if fragment not in command]
        rows.append(
            AuditRow(
                f"make {target}",
                ready(not missing and bool(command)),
                "canonical command present" if not missing and command else "missing: " + "; ".join(missing or ["target body"]),
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
        "# Makefile Targets Audit",
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
    rows = audit_makefile(args.makefile)
    write_outputs(args.output_dir, args.output_prefix, rows)
    return 0 if all(row.status == "ready" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
