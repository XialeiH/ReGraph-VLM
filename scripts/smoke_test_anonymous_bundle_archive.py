#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build, extract, and verify the anonymous submission bundle.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def run_command(cwd: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def require_ok(description: str, completed: subprocess.CompletedProcess[str]) -> str:
    if completed.returncode != 0:
        raise RuntimeError(f"{description} failed with exit code {completed.returncode}\n{completed.stdout}")
    return completed.stdout


def safe_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    for member in members:
        path = Path(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"unsafe archive path: {member.name}")
        if member.name == ".git" or "/.git/" in member.name:
            raise RuntimeError(f"archive contains Git metadata: {member.name}")
    return members


def detect_prefix(members: list[tarfile.TarInfo]) -> str:
    prefixes = {Path(member.name).parts[0] for member in members if Path(member.name).parts}
    if len(prefixes) != 1:
        raise RuntimeError(f"expected one archive top-level directory, found {sorted(prefixes)}")
    return next(iter(prefixes))


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
                    "preproc_v0/repetition_familiarity/results/final_tables/anonymous_bundle_manifest.csv",
                ],
            ),
        )

        with tarfile.open(archive_path, "r:gz") as archive:
            members = safe_members(archive)
            prefix = detect_prefix(members)
            archive.extractall(extract_dir)

        extracted_root = extract_dir / prefix
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

        print(
            f"Anonymous bundle archive smoke test OK: {len(members)} archive members, "
            f"prefix={prefix}, archive={archive_path.name}"
        )
        print(build_output.strip())
        print(verify_output.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
