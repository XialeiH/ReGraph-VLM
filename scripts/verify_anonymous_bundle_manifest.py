#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestRow:
    path: str
    bytes: int
    sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify anonymous bundle files against the per-file manifest.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables/anonymous_bundle_manifest.csv"),
    )
    parser.add_argument(
        "--source",
        choices=["auto", "git", "filesystem"],
        default="auto",
        help="Use Git index/HEAD contents when available, or filesystem contents for extracted bundles.",
    )
    return parser.parse_args()


def git_available(root: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def load_git_blob(root: Path, path: str) -> bytes | None:
    for ref in (":", "HEAD:"):
        completed = subprocess.run(
            ["git", "-C", str(root), "show", f"{ref}{path}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode == 0:
            return completed.stdout
    return None


def load_file(root: Path, path: str, source: str) -> bytes:
    if source == "git":
        data = load_git_blob(root, path)
        if data is None:
            raise FileNotFoundError(f"{path} not found in Git index or HEAD")
        return data

    filesystem_path = root / path
    if not filesystem_path.exists():
        raise FileNotFoundError(f"{path} not found on filesystem")
    return filesystem_path.read_bytes()


def read_manifest(path: Path) -> list[ManifestRow]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("manifest has no rows")
    expected_columns = {"path", "bytes", "sha256"}
    if set(rows[0]) != expected_columns:
        raise ValueError(f"manifest columns are {sorted(rows[0])}, expected {sorted(expected_columns)}")

    seen: set[str] = set()
    parsed: list[ManifestRow] = []
    for row in rows:
        path = row["path"]
        if path in seen:
            raise ValueError(f"duplicate manifest path: {path}")
        seen.add(path)
        if path.endswith("anonymous_bundle_manifest.csv"):
            raise ValueError("manifest should exclude its own row")
        try:
            byte_count = int(row["bytes"])
        except ValueError as exc:
            raise ValueError(f"{path}: non-integer bytes") from exc
        digest = row["sha256"]
        if byte_count < 0:
            raise ValueError(f"{path}: negative byte count")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"{path}: invalid sha256")
        parsed.append(ManifestRow(path=path, bytes=byte_count, sha256=digest))
    return parsed


def resolve_source(root: Path, requested: str) -> str:
    if requested == "auto":
        return "git" if git_available(root) else "filesystem"
    return requested


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    source = resolve_source(root, args.source)
    rows = read_manifest(manifest_path)

    failures: list[str] = []
    total_bytes = 0
    for row in rows:
        try:
            data = load_file(root, row.path, source)
        except FileNotFoundError as exc:
            failures.append(str(exc))
            continue
        total_bytes += len(data)
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != row.bytes:
            failures.append(f"{row.path}: bytes {len(data)} != manifest {row.bytes}")
        if digest != row.sha256:
            failures.append(f"{row.path}: sha256 {digest} != manifest {row.sha256}")

    if failures:
        print("Anonymous bundle manifest verification failed:")
        for failure in failures[:20]:
            print(f"- {failure}")
        if len(failures) > 20:
            print(f"- ... {len(failures) - 20} more")
        return 1

    print(
        f"Verified anonymous bundle manifest: {len(rows)} files, {total_bytes} bytes, "
        f"source={source}, manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
