#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuditRow:
    item: str
    status: str
    evidence: str


PROTECTED_PYTHON_SCRIPTS = [
    Path("scripts/prepare_external_validation_hpc.py"),
    Path("scripts/probe_laion_fmri_public_s3.py"),
    Path("scripts/analyze_laion_fmri_trial_metadata.py"),
    Path("scripts/export_laion_fmri_visual_roi_scalar4.py"),
]

PROTECTED_DOWNLOAD_SCRIPTS = [
    Path("scripts/run_laion_fmri_download_login.sh"),
]

EXTERNAL_DOWNLOAD_SCRIPTS = [*PROTECTED_PYTHON_SCRIPTS, *PROTECTED_DOWNLOAD_SCRIPTS]

DOC_PATHS = [
    Path("README.md"),
    Path("REPRODUCIBILITY.md"),
    Path("DATASET_CARD.md"),
    Path("reports/neurips_report/BUILD.md"),
    Path("reports/neurips_report/external_validation_dataset_scan.md"),
]

ALLOWED_LIGHTWEIGHT_EXTENSIONS = {".csv", ".md", ".txt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the external fMRI data storage policy.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument("--output-prefix", default="external_data_policy_audit")
    return parser.parse_args()


def ready(ok: bool) -> str:
    return "ready" if ok else "incomplete"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def in_git_checkout() -> bool:
    completed = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def tracked_files() -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        return []
    return completed.stdout.splitlines()


def audit_policy_utility() -> AuditRow:
    path = Path("scripts/external_data_policy.py")
    text = read_text(path)
    required = [
        "allowed_hpc_prefixes",
        "is_hpc_scratch_path",
        "enforce_hpc_external_path",
        "remote HPC scratch",
    ]
    missing = [fragment for fragment in required if fragment not in text]
    return AuditRow(
        "external data policy utility",
        ready(path.exists() and not missing),
        "shared HPC scratch path guard exists" if path.exists() and not missing else "missing: " + ", ".join(missing),
    )


def audit_protected_python_scripts() -> AuditRow:
    missing_scripts = [path.as_posix() for path in PROTECTED_PYTHON_SCRIPTS if not path.exists()]
    if missing_scripts and in_git_checkout():
        return AuditRow("protected external Python scripts", "incomplete", "missing: " + ", ".join(missing_scripts))

    missing_guards: list[str] = []
    checked = 0
    for path in PROTECTED_PYTHON_SCRIPTS:
        if not path.exists():
            continue
        checked += 1
        text = read_text(path)
        if "from external_data_policy import enforce_hpc_external_path" not in text:
            missing_guards.append(f"{path}: import")
        if "enforce_hpc_external_path(" not in text:
            missing_guards.append(f"{path}: call")
    if missing_guards:
        return AuditRow("protected external Python scripts", "incomplete", "; ".join(missing_guards))
    evidence = f"{checked} scripts enforce HPC scratch paths"
    if checked < len(PROTECTED_PYTHON_SCRIPTS):
        evidence = f"{checked} packaged scripts checked; full checkout enforces {len(PROTECTED_PYTHON_SCRIPTS)} protected scripts"
    return AuditRow("protected external Python scripts", "ready", evidence)


def audit_download_shell_scripts() -> AuditRow:
    missing = [path.as_posix() for path in PROTECTED_DOWNLOAD_SCRIPTS if not path.exists()]
    if missing and in_git_checkout():
        return AuditRow("external download shell scripts", "incomplete", "missing: " + ", ".join(missing))

    problems: list[str] = []
    checked = 0
    for path in PROTECTED_DOWNLOAD_SCRIPTS:
        if not path.exists():
            continue
        checked += 1
        text = read_text(path)
        hpc_cd = "cd " + (Path("/gpfsnyu") / "scratch").as_posix() + "/"
        if hpc_cd not in text:
            problems.append(f"{path}: missing HPC scratch cd")
        if "--root external_validation/" not in text:
            problems.append(f"{path}: root should stay relative to HPC checkout")
    if problems:
        return AuditRow("external download shell scripts", "incomplete", "; ".join(problems))
    evidence = f"{checked} shell scripts start from HPC scratch"
    if checked < len(PROTECTED_DOWNLOAD_SCRIPTS):
        evidence = f"{checked} packaged shell scripts checked; full checkout enforces login-node download wrapper"
    return AuditRow("external download shell scripts", "ready", evidence)


def audit_docs_policy() -> AuditRow:
    combined = "\n".join(read_text(path) for path in DOC_PATHS)
    required = [
        "remote HPC scratch",
        "not onto a local laptop checkout",
        "Large fMRI datasets should be downloaded and processed directly",
        "external_validation/summary",
    ]
    missing = [fragment for fragment in required if fragment not in combined]
    return AuditRow(
        "external data policy documentation",
        ready(not missing),
        "README, REPRODUCIBILITY, DATASET_CARD, BUILD, and dataset scan document HPC-only large-data handling"
        if not missing
        else "missing: " + ", ".join(missing),
    )


def audit_tracked_external_files() -> AuditRow:
    files = tracked_files()
    external_files = [path for path in files if path.startswith("external_validation/")]
    heavy = [
        path
        for path in external_files
        if not path.startswith("external_validation/summary/")
        or Path(path).suffix not in ALLOWED_LIGHTWEIGHT_EXTENSIONS
    ]
    return AuditRow(
        "tracked external validation files",
        ready(not heavy),
        f"{len(external_files)} tracked external-validation files; all are lightweight summaries"
        if not heavy
        else "unexpected tracked files: " + ", ".join(heavy[:8]),
    )


def audit_anonymous_bundle_policy() -> AuditRow:
    bundle_text = read_text(Path("scripts/make_anonymous_submission_bundle.py"))
    included_download_scripts = [path.as_posix() for path in EXTERNAL_DOWNLOAD_SCRIPTS if f'"{path.as_posix()}"' in bundle_text]
    required = [
        '"scripts/audit_external_data_policy.py"',
        '"scripts/external_data_policy.py"',
        '"preproc_v0/repetition_familiarity/results/final_tables/external_data_policy_audit.csv"',
        '"preproc_v0/repetition_familiarity/results/final_tables/external_data_policy_audit.md"',
    ]
    missing_required = [fragment.strip('"') for fragment in required if fragment not in bundle_text]
    ok = not included_download_scripts and not missing_required
    evidence = "anonymous bundle includes policy audit/guard but excludes external download scripts"
    if not ok:
        evidence = f"included_download_scripts={included_download_scripts or 'none'}; missing_required={missing_required or 'none'}"
    return AuditRow("anonymous bundle external-data scope", ready(ok), evidence)


def audit_rows() -> list[AuditRow]:
    return [
        audit_policy_utility(),
        audit_protected_python_scripts(),
        audit_download_shell_scripts(),
        audit_docs_policy(),
        audit_tracked_external_files(),
        audit_anonymous_bundle_policy(),
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
        "# External Data Policy Audit",
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
    rows = audit_rows()
    write_outputs(args.output_dir, args.output_prefix, rows)
    return 0 if all(row.status == "ready" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
