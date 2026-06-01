#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuditRow:
    item: str
    status: str
    evidence: str


REQUIRED_CITATION_KEYS = {
    "allen2022nsd",
    "glasser2016mmp",
    "li2021braingnn",
    "kan2022bnt",
    "radford2021clip",
    "oord2018infonce",
    "grill2001fmri",
    "grill2006repetition",
    "henson2003repetition",
    "krekelberg2006adaptation",
    "kriegeskorte2008rsa",
    "hasson2004isc",
    "haxby2011hyperalignment",
    "nastase2019isc",
    "malach1995loc",
    "kanwisher1997ffa",
    "grill2014visual",
    "scotti2023mindeye",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit manuscript citation and bibliography integrity.")
    parser.add_argument("--tex", type=Path, default=Path("reports/neurips_report/may30.tex"))
    parser.add_argument("--bib", type=Path, default=Path("reports/neurips_report/references.bib"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument("--output-prefix", default="citation_integrity_audit")
    return parser.parse_args()


def ready(ok: bool) -> str:
    return "ready" if ok else "incomplete"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def citation_commands(text: str) -> list[str]:
    return re.findall(r"\\cite[a-zA-Z*]*(?:\[[^\]]*\])*\{([^}]*)\}", text)


def citation_keys(text: str) -> list[str]:
    keys: list[str] = []
    for command in citation_commands(text):
        keys.extend(key.strip() for key in command.split(",") if key.strip())
    return keys


def bibliography_keys(text: str) -> list[str]:
    return re.findall(r"@\w+\s*\{\s*([^,\s]+)", text)


def audit_citations(tex: Path, bib: Path) -> list[AuditRow]:
    tex_text = read_text(tex)
    bib_text = read_text(bib)
    cited = citation_keys(tex_text)
    cite_commands = citation_commands(tex_text)
    bib_keys = bibliography_keys(bib_text)
    cited_set = set(cited)
    bib_set = set(bib_keys)
    duplicate_bib_keys = sorted(key for key, count in Counter(bib_keys).items() if count > 1)
    missing_bib = sorted(cited_set - bib_set)
    unused_bib = sorted(bib_set - cited_set)
    missing_required_cites = sorted(REQUIRED_CITATION_KEYS - cited_set)
    missing_required_bib = sorted(REQUIRED_CITATION_KEYS - bib_set)
    empty_commands = [command for command in cite_commands if not command.strip()]

    return [
        AuditRow("manuscript exists", "ready" if tex.exists() else "missing", str(tex)),
        AuditRow("bibliography exists", "ready" if bib.exists() else "missing", str(bib)),
        AuditRow("citation commands parsed", ready(bool(cite_commands)), f"{len(cite_commands)} citation commands"),
        AuditRow("unique cited keys parsed", ready(bool(cited_set)), f"{len(cited_set)} unique cited keys"),
        AuditRow("bibliography keys parsed", ready(bool(bib_set)), f"{len(bib_set)} bibliography keys"),
        AuditRow(
            "all cited keys defined",
            ready(not missing_bib),
            f"{len(cited_set)} cited keys defined" if not missing_bib else "; ".join(missing_bib),
        ),
        AuditRow(
            "no duplicate bibliography keys",
            ready(not duplicate_bib_keys),
            "no duplicates" if not duplicate_bib_keys else "; ".join(duplicate_bib_keys),
        ),
        AuditRow(
            "required publication citations present",
            ready(not missing_required_cites),
            f"{len(REQUIRED_CITATION_KEYS)} required keys cited" if not missing_required_cites else "; ".join(missing_required_cites),
        ),
        AuditRow(
            "required publication citations defined",
            ready(not missing_required_bib),
            f"{len(REQUIRED_CITATION_KEYS)} required keys defined" if not missing_required_bib else "; ".join(missing_required_bib),
        ),
        AuditRow(
            "no empty citation commands",
            ready(not empty_commands),
            "none found" if not empty_commands else f"{len(empty_commands)} empty citation commands",
        ),
        AuditRow(
            "unused bibliography entries bounded",
            ready(len(unused_bib) <= 2),
            f"{len(unused_bib)} unused entries" if unused_bib else "all bibliography entries cited",
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
        "# Citation Integrity Audit",
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
    rows = audit_citations(args.tex, args.bib)
    write_outputs(args.output_dir, args.output_prefix, rows)
    return 0 if all(row.status == "ready" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
