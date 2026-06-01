#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


MANIFEST_RELATIVE_PATH = "preproc_v0/repetition_familiarity/results/final_tables/anonymous_bundle_manifest.csv"
INNER_PREFLIGHT_ENV = "REGRAPH_BUNDLE_SMOKE_INNER"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build, extract, and verify the anonymous submission bundle.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def run_command(cwd: Path, args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, env=env)


def require_ok(description: str, completed: subprocess.CompletedProcess[str]) -> str:
    if completed.returncode != 0:
        raise RuntimeError(f"{description} failed with exit code {completed.returncode}\n{completed.stdout}")
    return completed.stdout


def first_lines(text: str, n: int = 3) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " | ".join(lines[:n]) if lines else "ok"


def safe_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    for member in members:
        path = Path(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"unsafe archive path: {member.name}")
        if member.name == ".git" or "/.git/" in member.name:
            raise RuntimeError(f"archive contains Git metadata: {member.name}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"archive contains link entry: {member.name}")
        if not member.isfile():
            raise RuntimeError(f"archive contains non-regular file entry: {member.name}")
    return members


def detect_prefix(members: list[tarfile.TarInfo]) -> str:
    prefixes = {Path(member.name).parts[0] for member in members if Path(member.name).parts}
    if len(prefixes) != 1:
        raise RuntimeError(f"expected one archive top-level directory, found {sorted(prefixes)}")
    return next(iter(prefixes))


def archive_relative_paths(members: list[tarfile.TarInfo], prefix: str) -> set[str]:
    relative_paths: set[str] = set()
    for member in members:
        parts = Path(member.name).parts
        if not parts or parts[0] != prefix:
            raise RuntimeError(f"archive member outside expected prefix {prefix}: {member.name}")
        relative_paths.add(Path(*parts[1:]).as_posix())
    return relative_paths


def read_manifest_paths(extracted_root: Path) -> set[str]:
    manifest_path = extracted_root / MANIFEST_RELATIVE_PATH
    if not manifest_path.exists():
        raise RuntimeError(f"extracted archive is missing {MANIFEST_RELATIVE_PATH}")
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    paths = {row["path"] for row in rows}
    if MANIFEST_RELATIVE_PATH in paths:
        raise RuntimeError("manifest should exclude its own row")
    return paths


def validate_archive_manifest_coverage(extracted_root: Path, members: list[tarfile.TarInfo], prefix: str) -> str:
    archive_paths = archive_relative_paths(members, prefix)
    manifest_paths = read_manifest_paths(extracted_root)
    expected_archive_paths = set(manifest_paths)
    expected_archive_paths.add(MANIFEST_RELATIVE_PATH)

    extra_paths = sorted(archive_paths - expected_archive_paths)
    missing_paths = sorted(expected_archive_paths - archive_paths)
    if extra_paths or missing_paths:
        evidence = []
        if extra_paths:
            evidence.append(f"extra archive paths: {extra_paths[:5]}")
        if missing_paths:
            evidence.append(f"missing archive paths: {missing_paths[:5]}")
        raise RuntimeError("; ".join(evidence))
    return f"{len(manifest_paths)} manifest rows plus manifest file match {len(archive_paths)} archive files"


def run_extracted_preflight(extracted_root: Path) -> str:
    if os.environ.get(INNER_PREFLIGHT_ENV) == "1":
        return "skipped extracted preflight inside recursive smoke-test guard"
    env = dict(os.environ)
    env[INNER_PREFLIGHT_ENV] = "1"
    command = [sys.executable, "scripts/run_publication_preflight.py"]
    tex_tool = next((tool for tool in ("latexmk", "pdflatex", "tectonic") if shutil.which(tool)), None)
    if tex_tool is not None:
        command.append("--compile")
    output = require_ok(
        "extracted anonymous bundle publication preflight",
        run_command(extracted_root, command, env=env),
    )
    mode = f"compile-required via {tex_tool}" if tex_tool is not None else "non-compiling; no TeX tool found"
    return f"extracted bundle preflight OK ({mode}): {first_lines(output)}"


def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    with tempfile.TemporaryDirectory(prefix="regraph_bundle_smoke_") as tmp:
        tmp_path = Path(tmp)
        archive_path = tmp_path / "regraph_vlm_anonymous_submission.tar.gz"
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        build_output = require_ok(
            "anonymous bundle build",
            run_command(
                root,
                [
                    sys.executable,
                    "scripts/make_anonymous_submission_bundle.py",
                    "--output",
                    str(archive_path),
                    "--manifest-output",
                    MANIFEST_RELATIVE_PATH,
                ],
            ),
        )

        with tarfile.open(archive_path, "r:gz") as archive:
            members = safe_members(archive)
            prefix = detect_prefix(members)
            archive.extractall(extract_dir)

        extracted_root = extract_dir / prefix
        coverage_evidence = validate_archive_manifest_coverage(extracted_root, members, prefix)
        verify_output = require_ok(
            "extracted anonymous bundle manifest verification",
            run_command(
                extracted_root,
                [
                    sys.executable,
                    "scripts/verify_anonymous_bundle_manifest.py",
                    "--source",
                    "filesystem",
                ],
            ),
        )
        extracted_preflight_evidence = run_extracted_preflight(extracted_root)

        print(
            f"Anonymous bundle archive smoke test OK: {len(members)} archive members, "
            f"prefix={prefix}, archive={archive_path.name}, {coverage_evidence}"
        )
        print(build_output.strip())
        print(verify_output.strip())
        print(extracted_preflight_evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
