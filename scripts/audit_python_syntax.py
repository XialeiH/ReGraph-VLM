#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import py_compile
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuditRow:
    item: str
    status: str
    evidence: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit syntax validity for Python files included in the anonymous bundle.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument("--output-prefix", default="python_syntax_audit")
    return parser.parse_args()


def ready(ok: bool) -> str:
    return "ready" if ok else "incomplete"


def import_bundle_module(root: Path):
    bundle_path = root / "scripts/make_anonymous_submission_bundle.py"
    spec = importlib.util.spec_from_file_location("make_anonymous_submission_bundle", bundle_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {bundle_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compile_path(source_path: Path, cache_dir: Path) -> str | None:
    compiled_path = cache_dir / source_path.with_suffix(".pyc").name
    try:
        py_compile.compile(str(source_path), cfile=str(compiled_path), doraise=True)
    except py_compile.PyCompileError as exc:
        return str(exc).replace("\n", " ")
    return None


def audit_python_syntax(root: Path) -> list[AuditRow]:
    try:
        bundle = import_bundle_module(root)
    except Exception as exc:
        return [AuditRow("bundle module imports", "incomplete", str(exc))]

    python_paths = sorted(path for path in bundle.EXACT_PATHS if path.endswith(".py"))
    rows = [
        AuditRow("bundle module imports", "ready", "loaded make_anonymous_submission_bundle.py"),
        AuditRow("bundled Python files discovered", ready(bool(python_paths)), f"{len(python_paths)} Python files"),
    ]

    missing = [path for path in python_paths if not (root / path).exists()]
    rows.append(
        AuditRow(
            "bundled Python files exist",
            ready(not missing),
            "all bundled Python files exist" if not missing else "; ".join(missing[:8]),
        )
    )

    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="regraph_pycompile_") as tmp:
        cache_dir = Path(tmp)
        for path in python_paths:
            source_path = root / path
            if not source_path.exists():
                continue
            error = compile_path(source_path, cache_dir)
            if error:
                failures.append(f"{path}: {error}")
    rows.append(
        AuditRow(
            "bundled Python syntax",
            ready(not failures),
            f"{len(python_paths)} files compiled with py_compile" if not failures else "; ".join(failures[:5]),
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
        "# Python Syntax Audit",
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
    rows = audit_python_syntax(args.root.resolve())
    write_outputs(args.output_dir, args.output_prefix, rows)
    return 0 if all(row.status == "ready" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
