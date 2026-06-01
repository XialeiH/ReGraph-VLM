#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from external_data_policy import enforce_hpc_external_path


DATASETS = [
    {
        "id": "cneuromod_things",
        "priority": 1,
        "name": "CNeuroMod-THINGS",
        "metadata_repo": "https://github.com/courtois-neuromod/cneuromod-things.git",
        "official_page": "https://www.nature.com/articles/s41597-026-06591-y",
        "fit": "Closest external validation target: same-image repeats, shared images across subjects, naturalistic object images.",
        "validation_target": "Strict T=3 cross-subject same-image retrieval and gate-transfer analysis.",
        "blocker": "Requires DataLad/git-annex for selective file download from CONP; not installed in the base Shanghai environment.",
        "full_download": "Do not download all data. Pull only behavioral events, GLMsingle or fMRIPrep derivatives, and the minimum subject/session subset needed for a smoke fold.",
    },
    {
        "id": "bold5000",
        "priority": 2,
        "name": "BOLD5000",
        "metadata_repo": "https://github.com/BOLD5000-dataset/BOLD5000.git",
        "official_page": "https://bold5000-dataset.github.io/website/",
        "fit": "Good external image-retrieval benchmark with four subjects and natural images; several thousand images overlap with NSD.",
        "validation_target": "External brain-image retrieval and NSD-to-BOLD5000 transfer; not a strict repeated-image T=3 replication.",
        "blocker": "Full release is large; use OpenNeuro/KiltHub selective access or official precomputed responses if available.",
        "full_download": "Avoid full raw download. Start from processed GLM/percent-signal-change products or one subject/session subset.",
    },
    {
        "id": "nod",
        "priority": 3,
        "name": "Natural Object Dataset (NOD)",
        "metadata_repo": "https://github.com/BNUCNL/NOD-fmri.git",
        "official_page": "https://www.nature.com/articles/s41597-023-02471-x",
        "fit": "Large-scale naturalistic-image fMRI with many subjects; strong external generalization stress test.",
        "validation_target": "Cross-subject ROI-token retrieval after building an atlas/surface ROI cache; likely category/image retrieval rather than repetition retrieval.",
        "blocker": "Very large dataset; preprocessing/ROI extraction cost is substantial.",
        "full_download": "Only download a small subject/image subset first; expand only if the smoke pipeline works.",
    },
    {
        "id": "things_fmri",
        "priority": 4,
        "name": "THINGS-fMRI",
        "metadata_repo": "https://github.com/ViCCo-Group/THINGS-data.git",
        "official_page": "https://things-initiative.org/",
        "fit": "Object-image fMRI with rich semantic annotations; useful for concept/semantic validation.",
        "validation_target": "Semantic brain-image retrieval and concept-level transfer.",
        "blocker": "Images are generally not repeated in the same way as NSD/CNeuroMod-THINGS; exact OpenNeuro dataset IDs should be verified from the THINGS docs before download.",
        "full_download": "Use THINGS-data documentation to pull only fMRI derivatives and stimulus metadata.",
    },
]


def default_hpc_root() -> Path:
    user = os.environ.get("USER", "$USER")
    return Path("/gpfsnyu") / "scratch" / user / "ReGraph-VLM"


def run(cmd: list[str], cwd: Path | None = None) -> dict[str, object]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def clone_metadata(dataset: dict[str, str], out_dir: Path) -> dict[str, object]:
    repo = dataset.get("metadata_repo")
    if not repo:
        return {"status": "skip", "reason": "no metadata_repo"}
    target = out_dir / dataset["id"] / "metadata_repo"
    if target.exists():
        return {"status": "exists", "path": str(target)}
    target.parent.mkdir(parents=True, exist_ok=True)
    result = run(["git", "clone", "--depth", "1", repo, str(target)])
    result["path"] = str(target)
    return result


def write_markdown(out_dir: Path, probe_results: dict[str, object]) -> None:
    lines = [
        "# External Validation Dataset Scan",
        "",
        "All paths in this note are intended for Shanghai HPC scratch, not local storage.",
        "",
        "## Ranking",
        "",
    ]
    for ds in DATASETS:
        lines.extend(
            [
                f"### {ds['priority']}. {ds['name']}",
                "",
                f"- ID: `{ds['id']}`",
                f"- Official page: {ds['official_page']}",
                f"- Metadata/source repo: {ds.get('metadata_repo', 'n/a')}",
                f"- Fit: {ds['fit']}",
                f"- Validation target: {ds['validation_target']}",
                f"- Blocker: {ds['blocker']}",
                f"- Download policy: {ds['full_download']}",
                "",
            ]
        )
    lines.extend(["## Probe Results", ""])
    for key, result in probe_results.items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(result, indent=2))
        lines.append("```")
        lines.append("")
    lines.extend(
        [
            "## Next Commands",
            "",
            "Install missing tools in the project venv before any full CNeuroMod/NOD download:",
            "",
            "```bash",
            "cd $REGRAPH_VLM_HPC_ROOT",
            "source scripts/shanghai_env.sh",
            "python -m pip install h5py datalad datalad-installer",
            "# git-annex is still required for DataLad `get`; install it with an HPC-supported package manager or datalad-installer.",
            "```",
            "",
            "For the first real validation job, use CNeuroMod-THINGS and pull only metadata/events plus one small derivative subset before attempting ROI extraction.",
            "",
        ]
    )
    (out_dir / "external_validation_dataset_scan.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare external fMRI validation sources on Shanghai HPC.")
    parser.add_argument("--root", type=Path, default=default_hpc_root())
    parser.add_argument("--clone-metadata", action="store_true")
    args = parser.parse_args()
    enforce_hpc_external_path(args.root, "external validation project root")

    out_dir = args.root / "external_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_results: dict[str, object] = {}
    for ds in DATASETS:
        repo = ds.get("metadata_repo")
        if repo:
            probe_results[f"{ds['id']}_git_ls_remote"] = run(["git", "ls-remote", "--heads", repo])
        if args.clone_metadata:
            probe_results[f"{ds['id']}_clone_metadata"] = clone_metadata(ds, out_dir)

    manifest = {
        "root": str(out_dir),
        "policy": "Store external validation data directly on Shanghai HPC scratch. Do not download large datasets locally.",
        "datasets": DATASETS,
        "probe_results": probe_results,
    }
    (out_dir / "external_validation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_markdown(out_dir, probe_results)
    print(json.dumps({"out_dir": str(out_dir), "datasets": [d["id"] for d in DATASETS]}, indent=2))


if __name__ == "__main__":
    main()
