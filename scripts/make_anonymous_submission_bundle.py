#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import tarfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path


EXACT_PATHS = {
    ".gitignore",
    "ANONYMIZATION.md",
    "README.md",
    "models/bnt_encoder.py",
    "models/regraph_vlm.py",
    "reports/neurips_report/BUILD.md",
    "reports/neurips_report/external_validation_dataset_scan.md",
    "reports/neurips_report/may30.tex",
    "reports/neurips_report/neurips_2025.sty",
    "reports/neurips_report/references.bib",
    "scripts/audit_aaai_publication_readiness.py",
    "scripts/audit_manuscript_publication_claims.py",
    "scripts/audit_manuscript_stat_claims.py",
    "scripts/audit_manuscript_table_values.py",
    "scripts/audit_publication_artifact_provenance.py",
    "scripts/audit_publication_docs.py",
    "scripts/audit_reviewer_response_readiness.py",
    "scripts/generate_publication_evidence_manifest.py",
    "scripts/make_anonymous_submission_bundle.py",
    "scripts/make_publication_stat_tests.py",
    "scripts/materialize_publication_readiness_artifacts.py",
    "scripts/run_publication_preflight.py",
    "scripts/run_regraph_vlm_fold.py",
    "scripts/update_may30_publication_tables.py",
}

PREFIXES = (
    "external_validation/summary/",
    "preproc_v0/repetition_familiarity/results/final_tables/",
    "reports/neurips_report/figures/",
)

FORBIDDEN_TEXT = (
    "".join(("Xia", "lei")),
    "".join(("xia", "lei")),
    "".join(("xh", "2906")),
    " ".join(("NYU", "Shanghai")),
    "".join(("Xia", "lei", "H")),
    "github.com/" + "".join(("Xia", "lei", "H")),
    "/" + "Users/",
    "/gpfsnyu/" + "scratch/",
    "/scratch/" + "".join(("xh", "2906")),
)

TEXT_SUFFIXES = {
    ".bib",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sty",
    ".tex",
    ".txt",
    ".yml",
    ".yaml",
}


@dataclass(frozen=True)
class BundleFile:
    path: str
    data: bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an anonymous, Git-history-free submission bundle.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("dist/regraph_vlm_anonymous_submission.tar.gz"))
    parser.add_argument("--prefix", default="regraph_vlm_anonymous/")
    parser.add_argument("--dry-run", action="store_true", help="Validate the bundle contents without writing the archive.")
    return parser.parse_args()


def run_git(root: Path, args: list[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"git {' '.join(args)} failed")
    return completed.stdout


def git_available(root: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def tracked_paths(root: Path) -> list[str]:
    if git_available(root):
        return [line for line in run_git(root, ["ls-files"]).splitlines() if include_path(line)]
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and include_path(path.relative_to(root).as_posix())
    )


def include_path(path: str) -> bool:
    if path in EXACT_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in PREFIXES)


def load_blob(root: Path, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), "show", f"HEAD:{path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode == 0:
        return completed.stdout
    filesystem_path = root / path
    if filesystem_path.exists():
        return filesystem_path.read_bytes()
    raise RuntimeError(completed.stderr.decode("utf-8", errors="replace").strip() or f"cannot read {path}")


def bundle_files(root: Path) -> list[BundleFile]:
    paths = tracked_paths(root)
    if not paths:
        raise RuntimeError("anonymous bundle allowlist matched no tracked files")
    return [BundleFile(path=path, data=load_blob(root, path)) for path in paths]


def scan_deanonymizing_text(files: list[BundleFile]) -> list[str]:
    hits: list[str] = []
    for item in files:
        if Path(item.path).suffix not in TEXT_SUFFIXES:
            continue
        text = item.data.decode("utf-8", errors="replace")
        for needle in FORBIDDEN_TEXT:
            if needle in text:
                hits.append(f"{item.path}: {needle}")
    return hits


def write_archive(output: Path, prefix: str, files: list[BundleFile]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized_prefix = prefix if prefix.endswith("/") else f"{prefix}/"
    with tarfile.open(output, "w:gz") as archive:
        for item in files:
            info = tarfile.TarInfo(name=f"{normalized_prefix}{item.path}")
            info.size = len(item.data)
            info.mode = 0o644
            archive.addfile(info, BytesIO(item.data))


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    files = bundle_files(root)
    hits = scan_deanonymizing_text(files)
    if hits:
        print("Anonymous bundle check failed; deanonymizing strings found:")
        for hit in hits[:40]:
            print(f"- {hit}")
        if len(hits) > 40:
            print(f"- ... {len(hits) - 40} more")
        return 1

    total_bytes = sum(len(item.data) for item in files)
    if args.dry_run:
        print(f"Anonymous bundle dry run OK: {len(files)} files, {total_bytes} bytes, no deanonymizing strings.")
        return 0

    write_archive(root / args.output, args.prefix, files)
    print(f"Wrote anonymous submission bundle: {args.output} ({len(files)} files, {total_bytes} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
