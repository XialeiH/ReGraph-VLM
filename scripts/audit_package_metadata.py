#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditRow:
    item: str
    status: str
    evidence: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit pyproject.toml metadata for the ReGraph-VLM publication package.")
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument("--output-prefix", default="package_metadata_audit")
    return parser.parse_args()


def ready(ok: bool) -> str:
    return "ready" if ok else "incomplete"


def load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib  # type: ignore[attr-defined]
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            from pip._vendor import tomli as tomllib  # type: ignore[no-redef]
    with path.open("rb") as handle:
        return tomllib.load(handle)


def contains_requirement(requirements: list[str], package: str) -> bool:
    normalized = package.lower().replace("_", "-")
    for requirement in requirements:
        head = requirement.split(";")[0].split("[")[0]
        name = head.split(">=")[0].split("==")[0].split("<")[0].strip().lower().replace("_", "-")
        if name == normalized:
            return True
    return False


def audit_metadata(path: Path) -> list[AuditRow]:
    if not path.exists():
        return [AuditRow("pyproject exists", "missing", str(path))]
    try:
        data = load_toml(path)
    except Exception as exc:
        return [AuditRow("pyproject parses", "incomplete", repr(exc))]

    project = data.get("project", {})
    dependencies = list(project.get("dependencies", []))
    optional = project.get("optional-dependencies", {})
    setuptools = data.get("tool", {}).get("setuptools", {})
    package_dir = setuptools.get("package-dir", {})
    find = setuptools.get("packages", {}).get("find", {})

    core_deps = ["numpy", "pandas", "scipy", "scikit-learn", "torch", "tqdm"]
    neuro_deps = ["h5py", "matplotlib", "nibabel", "nilearn", "Pillow", "requests"]
    legacy_deps = ["networkx", "PyYAML", "torch-geometric"]
    rows = [
        AuditRow("pyproject parses", "ready", str(path)),
        AuditRow(
            "project identity",
            ready(
                project.get("name") == "regraph-vlm"
                and project.get("version") == "0.1.0"
                and project.get("readme") == "README.md"
                and project.get("requires-python") == ">=3.9"
                and "Fixed-order ROI-token brain graph" in str(project.get("description", ""))
            ),
            f"name={project.get('name')}; requires-python={project.get('requires-python')}",
        ),
        AuditRow(
            "core dependencies",
            ready(all(contains_requirement(dependencies, dep) for dep in core_deps)),
            "core deps present: " + ", ".join(core_deps),
        ),
        AuditRow(
            "publication extra",
            ready("publication" in optional and contains_requirement(list(optional.get("publication", [])), "pandas")),
            "publication extra includes pandas",
        ),
        AuditRow(
            "neuroimaging extra",
            ready("neuro" in optional and all(contains_requirement(list(optional.get("neuro", [])), dep) for dep in neuro_deps)),
            "neuro extra present: " + ", ".join(neuro_deps),
        ),
        AuditRow(
            "CLIP extra",
            ready("clip" in optional and contains_requirement(list(optional.get("clip", [])), "open_clip_torch")),
            "clip extra includes open_clip_torch",
        ),
        AuditRow(
            "legacy graph extra",
            ready("legacy-graph" in optional and all(contains_requirement(list(optional.get("legacy-graph", [])), dep) for dep in legacy_deps)),
            "legacy-graph extra preserves bridgegen dependencies",
        ),
        AuditRow(
            "dev extra",
            ready(
                "dev" in optional
                and contains_requirement(list(optional.get("dev", [])), "pytest")
                and all(contains_requirement(list(optional.get("dev", [])), dep) for dep in legacy_deps)
            ),
            "dev extra includes pytest and legacy graph deps",
        ),
        AuditRow(
            "package directories",
            ready(package_dir.get("bridgegen") == "src/bridgegen" and package_dir.get("models") == "models"),
            f"package-dir={package_dir}",
        ),
        AuditRow(
            "package discovery",
            ready("src" in find.get("where", []) and "." in find.get("where", []) and "bridgegen*" in find.get("include", []) and "models*" in find.get("include", [])),
            f"find={find}",
        ),
        AuditRow(
            "model package files",
            ready(Path("models/__init__.py").exists() and Path("models/regraph_vlm.py").exists() and Path("models/bnt_encoder.py").exists()),
            "models package exposes ReGraph-VLM source files",
        ),
    ]
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
        "# Package Metadata Audit",
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
    rows = audit_metadata(args.pyproject)
    write_outputs(args.output_dir, args.output_prefix, rows)
    return 0 if all(row.status == "ready" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
