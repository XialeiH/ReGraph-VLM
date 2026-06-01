#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


FIGURE_EXISTS_RE = re.compile(r"\\IfFileExists\{([^{}]+)\}")
INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}")
ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class AuditRow:
    item: str
    status: str
    evidence: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit publication manuscript figure asset availability.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tex", type=Path, default=Path("reports/neurips_report/may30.tex"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument("--output-prefix", default="figure_asset_audit")
    return parser.parse_args()


def ready(ok: bool) -> str:
    return "ready" if ok else "incomplete"


def run_git(root: Path, args: list[str]) -> set[str]:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        return set()
    return {line for line in completed.stdout.splitlines() if line.strip()}


def git_available(root: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def import_bundle_paths(root: Path) -> set[str]:
    bundle_path = root / "scripts/make_anonymous_submission_bundle.py"
    spec = importlib.util.spec_from_file_location("make_anonymous_submission_bundle", bundle_path)
    if spec is None or spec.loader is None:
        return set()
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return set(module.EXACT_PATHS)


def relative_bundle_path(root: Path, tex: Path, figure_path: str) -> tuple[Path, str]:
    raw_path = Path(figure_path)
    if raw_path.is_absolute() or ".." in raw_path.parts:
        return raw_path, figure_path
    absolute = (tex.parent / raw_path).resolve()
    try:
        return absolute, absolute.relative_to(root).as_posix()
    except ValueError:
        return absolute, figure_path


def audit_figure_assets(root: Path, tex_relative: Path) -> list[AuditRow]:
    tex = (root / tex_relative).resolve()
    if not tex.exists():
        return [AuditRow("manuscript exists", "missing", f"{tex_relative} not found")]

    text = tex.read_text(encoding="utf-8", errors="replace")
    file_exists_paths = FIGURE_EXISTS_RE.findall(text)
    include_paths = INCLUDEGRAPHICS_RE.findall(text)
    unique_paths = sorted(set(file_exists_paths + include_paths))
    rows = [
        AuditRow("manuscript exists", "ready", str(tex_relative)),
        AuditRow(
            "figure dependencies parsed",
            ready(bool(unique_paths)),
            f"{len(file_exists_paths)} IfFileExists guards, {len(include_paths)} includegraphics calls, {len(unique_paths)} unique assets",
        ),
    ]

    unguarded = sorted(set(include_paths) - set(file_exists_paths))
    rows.append(
        AuditRow(
            "includegraphics calls guarded",
            ready(not unguarded),
            "all includegraphics assets are guarded by IfFileExists" if not unguarded else "; ".join(unguarded),
        )
    )

    unsafe = sorted(path for path in unique_paths if Path(path).is_absolute() or ".." in Path(path).parts)
    rows.append(
        AuditRow(
            "figure paths are portable",
            ready(not unsafe),
            "all figure paths are relative to the manuscript directory" if not unsafe else "; ".join(unsafe),
        )
    )

    has_git_index = git_available(root)
    cached_paths = run_git(root, ["ls-files", "--cached"]) if has_git_index else set()
    bundle_paths = import_bundle_paths(root)
    for figure_path in unique_paths:
        absolute, bundle_path = relative_bundle_path(root, tex, figure_path)
        problems = []
        if not absolute.exists():
            problems.append("missing")
            size = 0
        else:
            size = absolute.stat().st_size
            if size <= 0:
                problems.append("empty")
        if absolute.suffix.lower() not in ALLOWED_SUFFIXES:
            problems.append(f"unsupported suffix {absolute.suffix or '<none>'}")
        if bundle_paths and bundle_path not in bundle_paths:
            problems.append("not bundle-allowlisted")
        if has_git_index and bundle_path not in cached_paths:
            problems.append("not tracked or staged")
        rows.append(
            AuditRow(
                f"figure asset {figure_path}",
                ready(not problems),
                f"{bundle_path}; {size} bytes; bundle-allowlisted"
                if not problems
                else f"{bundle_path}; {', '.join(problems)}",
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
        "# Figure Asset Audit",
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
    root = args.root.resolve()
    rows = audit_figure_assets(root, args.tex)
    write_outputs(args.output_dir, args.output_prefix, rows)
    return 0 if all(row.status == "ready" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
