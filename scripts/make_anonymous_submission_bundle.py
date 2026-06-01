#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import subprocess
import tarfile
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path


BUNDLE_MANIFEST_RELATIVE_PATH = "preproc_v0/repetition_familiarity/results/final_tables/anonymous_bundle_manifest.csv"

PUBLICATION_ARTIFACT_PATHS = {
    "external_validation/summary/external_visual_roi_all4_summary.md",
    "external_validation/summary/laion_fmri_visual_roi_all_runs.csv",
    "external_validation/summary/laion_fmri_visual_roi_latex.txt",
    "external_validation/summary/laion_fmri_visual_roi_pairwise_tests.csv",
    "external_validation/summary/laion_fmri_visual_roi_summary.csv",
    "external_validation/summary/laion_fmri_visual_roi_summary.md",
    "preproc_v0/repetition_familiarity/results/final_tables/aaai_publication_readiness_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/aaai_publication_readiness_audit.md",
    BUNDLE_MANIFEST_RELATIVE_PATH,
    "preproc_v0/repetition_familiarity/results/final_tables/aaai_roi_token_story_summary.md",
    "preproc_v0/repetition_familiarity/results/final_tables/ci_workflow_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/ci_workflow_audit.md",
    "preproc_v0/repetition_familiarity/results/final_tables/final_adjacency_ablation_tests.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/final_adjacency_ablation_tests.md",
    "preproc_v0/repetition_familiarity/results/final_tables/external_data_policy_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/external_data_policy_audit.md",
    "preproc_v0/repetition_familiarity/results/final_tables/fold_difficulty_qc.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/makefile_targets_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/makefile_targets_audit.md",
    "preproc_v0/repetition_familiarity/results/final_tables/manuscript_publication_claims_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/manuscript_publication_claims_audit.md",
    "preproc_v0/repetition_familiarity/results/final_tables/manuscript_stat_claims_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/manuscript_stat_claims_audit.md",
    "preproc_v0/repetition_familiarity/results/final_tables/manuscript_table_values_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/manuscript_table_values_audit.md",
    "preproc_v0/repetition_familiarity/results/final_tables/model_parameter_counts.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/package_metadata_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/package_metadata_audit.md",
    "preproc_v0/repetition_familiarity/results/final_tables/publication_artifact_provenance_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/publication_artifact_provenance_audit.md",
    "preproc_v0/repetition_familiarity/results/final_tables/publication_docs_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/publication_docs_audit.md",
    "preproc_v0/repetition_familiarity/results/final_tables/publication_evidence_manifest.md",
    "preproc_v0/repetition_familiarity/results/final_tables/publication_paired_stats.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/reviewer_response_readiness_audit.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/reviewer_response_readiness_audit.md",
    "preproc_v0/repetition_familiarity/results/final_tables/session_order_pair_qc.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/single_ref_matched_allseed_latex.txt",
    "preproc_v0/repetition_familiarity/results/final_tables/single_ref_matched_allseed_pairwise_tests.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/single_ref_matched_allseed_summary.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/single_ref_matched_latex.txt",
    "preproc_v0/repetition_familiarity/results/final_tables/single_ref_matched_summary.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/split_accounting.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/table_adjacency_ablation.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/table_adjacency_ablation_latex.txt",
    "preproc_v0/repetition_familiarity/results/final_tables/table_adjacency_perturbation.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/table_allfold_final.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/table_edge_bias_followup.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/table_external_visual_roi_smoke.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/table_gate_confound.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/table_graph_only.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/table_hard_negative_allfold.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/table_heldout_image.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/table_lowshot_calibration.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/table_matched_deletion.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/table_phase2_sota_graph_baselines.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/table_roi_token_controls.csv",
    "preproc_v0/repetition_familiarity/results/final_tables/table_within_subject.csv",
}

FIGURE_PATHS = {
    "reports/neurips_report/figures/gate_interpretability.pdf",
    "reports/neurips_report/figures/model_overview.pdf",
    "reports/neurips_report/figures/model_overview.png",
    "reports/neurips_report/figures/natural_scene_two_stimuli_repeat_maps_left_lateral_large.png",
    "reports/neurips_report/figures/neuroscience_summary.pdf",
    "reports/neurips_report/figures/neuroscience_summary.png",
    "reports/neurips_report/figures/per_subject_performance.pdf",
    "reports/neurips_report/figures/results_tradeoff.pdf",
    "reports/neurips_report/figures/results_tradeoff.png",
}

EXACT_PATHS = {
    ".github/workflows/publication-preflight.yml",
    ".gitignore",
    "ANONYMIZATION.md",
    "DATASET_CARD.md",
    "Makefile",
    "MODEL_CARD.md",
    "README.md",
    "REPRODUCIBILITY.md",
    "REVIEWER_RESPONSE.md",
    "pyproject.toml",
    "models/__init__.py",
    "models/bnt_encoder.py",
    "models/regraph_vlm.py",
    "reports/neurips_report/BUILD.md",
    "reports/neurips_report/external_validation_dataset_scan.md",
    "reports/neurips_report/may30.tex",
    "reports/neurips_report/neurips_2025.sty",
    "reports/neurips_report/references.bib",
    "scripts/audit_aaai_publication_readiness.py",
    "scripts/audit_ci_workflow.py",
    "scripts/audit_external_data_policy.py",
    "scripts/audit_manuscript_publication_claims.py",
    "scripts/audit_manuscript_stat_claims.py",
    "scripts/audit_manuscript_table_values.py",
    "scripts/audit_makefile_targets.py",
    "scripts/audit_package_metadata.py",
    "scripts/audit_publication_artifact_provenance.py",
    "scripts/audit_publication_docs.py",
    "scripts/audit_reviewer_response_readiness.py",
    "scripts/external_data_policy.py",
    "scripts/generate_publication_evidence_manifest.py",
    "scripts/make_anonymous_submission_bundle.py",
    "scripts/make_publication_stat_tests.py",
    "scripts/materialize_publication_readiness_artifacts.py",
    "scripts/run_publication_preflight.py",
    "scripts/run_regraph_vlm_fold.py",
    "scripts/smoke_test_anonymous_bundle_archive.py",
    "scripts/update_may30_publication_tables.py",
    "scripts/verify_anonymous_bundle_manifest.py",
    "scripts/verify_model_parameter_counts.py",
    *FIGURE_PATHS,
    *PUBLICATION_ARTIFACT_PATHS,
}

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
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help=(
            "Write a CSV manifest with per-file source bytes and SHA-256 values. "
            "External paths are allowed; the in-bundle manifest file is refreshed when present."
        ),
    )
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
    return path in EXACT_PATHS


def load_blob(root: Path, path: str) -> bytes:
    for ref in (":", "HEAD:"):
        completed = subprocess.run(
            ["git", "-C", str(root), "show", f"{ref}{path}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode == 0:
            return completed.stdout
    filesystem_path = root / path
    if filesystem_path.exists():
        return filesystem_path.read_bytes()
    raise RuntimeError(f"cannot read {path} from Git index, HEAD, or filesystem")


def bundle_files(root: Path) -> list[BundleFile]:
    paths = tracked_paths(root)
    if not paths:
        raise RuntimeError("anonymous bundle allowlist matched no tracked files")
    return [BundleFile(path=path, data=load_blob(root, path)) for path in paths]


def scan_deanonymizing_bytes(files: list[BundleFile]) -> list[str]:
    hits: list[str] = []
    forbidden = [(needle, needle.encode("utf-8")) for needle in FORBIDDEN_TEXT]
    for item in files:
        for needle, encoded in forbidden:
            if encoded in item.data:
                hits.append(f"{item.path}: {needle}")
    return hits


def add_files_to_archive(archive: tarfile.TarFile, prefix: str, files: list[BundleFile]) -> None:
    normalized_prefix = prefix if prefix.endswith("/") else f"{prefix}/"
    for item in files:
        info = tarfile.TarInfo(name=f"{normalized_prefix}{item.path}")
        info.size = len(item.data)
        info.mode = 0o644
        info.mtime = 0
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        archive.addfile(info, BytesIO(item.data))


def archive_bytes(prefix: str, files: list[BundleFile]) -> bytes:
    buffer = BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as gzip_file:
        with tarfile.open(fileobj=gzip_file, mode="w") as archive:
            add_files_to_archive(archive, prefix, files)
    return buffer.getvalue()


def archive_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_archive(output: Path, data: bytes) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)


def repo_relative_path(root: Path, path: Path) -> str | None:
    full_path = path if path.is_absolute() else root / path
    try:
        return full_path.resolve().relative_to(root).as_posix()
    except ValueError:
        return None


def resolve_manifest_bundle_path(root: Path, output: Path, files: list[BundleFile]) -> str | None:
    file_paths = {item.path for item in files}
    output_path = repo_relative_path(root, output)
    if output_path in file_paths:
        return output_path
    if BUNDLE_MANIFEST_RELATIVE_PATH in file_paths:
        return BUNDLE_MANIFEST_RELATIVE_PATH
    return None


def build_manifest(manifest_bundle_path: str | None, files: list[BundleFile]) -> tuple[bytes, int]:
    rows = [
        {
            "path": item.path,
            "bytes": len(item.data),
            "sha256": hashlib.sha256(item.data).hexdigest(),
        }
        for item in sorted(files, key=lambda value: value.path)
        if item.path != manifest_bundle_path
    ]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["path", "bytes", "sha256"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8"), len(rows)


def replace_bundle_file(files: list[BundleFile], path: str, data: bytes) -> list[BundleFile]:
    return [BundleFile(path=item.path, data=data) if item.path == path else item for item in files]


def write_manifest(root: Path, output: Path, files: list[BundleFile]) -> tuple[int, list[BundleFile]]:
    output_path = output if output.is_absolute() else root / output
    manifest_path = resolve_manifest_bundle_path(root, output, files)
    manifest_data, row_count = build_manifest(manifest_path, files)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(manifest_data)
    if manifest_path is None:
        return row_count, files
    return row_count, replace_bundle_file(files, manifest_path, manifest_data)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    files = bundle_files(root)
    hits = scan_deanonymizing_bytes(files)
    if hits:
        print("Anonymous bundle check failed; deanonymizing byte strings found:")
        for hit in hits[:40]:
            print(f"- {hit}")
        if len(hits) > 40:
            print(f"- ... {len(hits) - 40} more")
        return 1

    manifest_rows = None
    if args.manifest_output is not None:
        manifest_rows, files = write_manifest(root, args.manifest_output, files)

    total_bytes = sum(len(item.data) for item in files)
    archive_data = archive_bytes(args.prefix, files)
    digest = archive_sha256(archive_data)
    manifest_suffix = (
        f", manifest={args.manifest_output.as_posix()} ({manifest_rows} rows)"
        if args.manifest_output is not None
        else ""
    )
    if args.dry_run:
        print(
            f"Anonymous bundle dry run OK: {len(files)} files, {total_bytes} source bytes, "
            f"{len(archive_data)} archive bytes, sha256={digest}, no deanonymizing byte strings{manifest_suffix}."
        )
        return 0

    write_archive(root / args.output, archive_data)
    print(
        f"Wrote anonymous submission bundle: {args.output} "
        f"({len(files)} files, {total_bytes} source bytes, {len(archive_data)} archive bytes, sha256={digest}{manifest_suffix})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
