#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import torch

from external_data_policy import enforce_hpc_external_path
from export_laion_fmri_visual_roi_scalar4 import (
    FEATURE_NAMES,
    beta_key,
    local_name_from_key,
    read_trial_rows,
    scalar4,
    select_labels,
)


BASE_URL = "https://laion-fmri.s3.amazonaws.com"
DEFAULT_SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
APARC_NAMES = [
    "bankssts",
    "caudalanteriorcingulate",
    "caudalmiddlefrontal",
    "cuneus",
    "entorhinal",
    "fusiform",
    "inferiorparietal",
    "inferiortemporal",
    "isthmuscingulate",
    "lateraloccipital",
    "lateralorbitofrontal",
    "lingual",
    "medialorbitofrontal",
    "middletemporal",
    "parahippocampal",
    "paracentral",
    "parsopercularis",
    "parsorbitalis",
    "parstriangularis",
    "pericalcarine",
    "postcentral",
    "posteriorcingulate",
    "precentral",
    "precuneus",
    "rostralanteriorcingulate",
    "rostralmiddlefrontal",
    "superiorfrontal",
    "superiorparietal",
    "superiortemporal",
    "supramarginal",
    "frontalpole",
    "temporalpole",
    "transversetemporal",
    "insula",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LAION-fMRI FreeSurfer aparc whole-cortex scalar4 tensors.")
    parser.add_argument("--root", type=Path, default=Path("external_validation/laion_fmri_freesurfer_aparc"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("external_validation/laion_fmri_probe/trial_metadata/tsv"))
    parser.add_argument("--source-download-dir", type=Path, default=Path("external_validation/laion_fmri_matched_public_roi/downloads"))
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--max-session", type=int, default=30)
    parser.add_argument("--min-repeats", type=int, default=3)
    parser.add_argument("--max-labels", type=int, default=0)
    parser.add_argument("--top-quantile", type=float, default=0.90)
    parser.add_argument("--selection-manifest", type=Path, default=None)
    parser.add_argument("--download-only", action="store_true")
    return parser.parse_args()


def s3_download(key: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    urllib.request.urlretrieve(f"{BASE_URL}/{urllib.parse.quote(key)}", tmp)
    tmp.replace(path)


def fs_key(subject: str) -> str:
    return f"derivatives/freesurfer/{subject}/mri/aparc+aseg.mgz"


def aparc_nodes() -> list[tuple[int, str]]:
    nodes: list[tuple[int, str]] = []
    for hemi, offset in [("lh", 1000), ("rh", 2000)]:
        for idx, name in enumerate(APARC_NAMES, start=1):
            nodes.append((offset + idx, f"{hemi}.{name}"))
    return nodes


def manifest_selection(path: Path) -> tuple[list[str], dict[str, dict[str, list[dict[str, object]]]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    labels = [str(label) for label in manifest["labels"]]
    selected = {
        str(subject): {
            str(label): [dict(row) for row in rows]
            for label, rows in by_label.items()
        }
        for subject, by_label in manifest["selected"].items()
    }
    return labels, selected


def choose_labels(args: argparse.Namespace) -> tuple[list[str], dict[str, dict[str, list[dict[str, object]]]]]:
    if args.selection_manifest:
        return manifest_selection(args.selection_manifest)
    rows_by_subject = {subject: read_trial_rows(args.metadata_dir, subject, args.max_session) for subject in args.subjects}
    return select_labels(rows_by_subject, args.subjects, args.min_repeats, args.max_labels)


def beta_path_for(args: argparse.Namespace, subject: str, session: str) -> Path:
    key = beta_key(subject, session)
    source = args.source_download_dir / subject / local_name_from_key(key)
    if source.exists() and source.stat().st_size > 0:
        return source
    dest = args.root / "downloads" / subject / local_name_from_key(key)
    s3_download(key, dest)
    return dest


def prepare_downloads(args: argparse.Namespace, labels: list[str], selected: dict[str, dict[str, list[dict[str, object]]]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for subject in args.subjects:
        sessions = sorted({str(row["session"]) for label in labels for row in selected[subject][label]})
        fs_path = args.root / "downloads" / subject / "aparc+aseg.mgz"
        s3_download(fs_key(subject), fs_path)
        for session in sessions:
            beta_path_for(args, subject, session)
        out[subject] = {"n_beta_maps": len(sessions), "sessions": sessions, "fs_aparc": str(fs_path)}
        print({"downloaded": subject, **out[subject]}, flush=True)
    return out


def resampled_aparc(subject: str, beta_img: nib.spatialimages.SpatialImage, args: argparse.Namespace) -> np.ndarray:
    fs_path = args.root / "downloads" / subject / "aparc+aseg.mgz"
    s3_download(fs_key(subject), fs_path)
    aparc_img = nib.load(str(fs_path))
    resampled = resample_from_to(aparc_img, (beta_img.shape[:3], beta_img.affine), order=0)
    return np.rint(np.asanyarray(resampled.dataobj)).astype(np.int32)


def export_subject(
    subject: str,
    labels: list[str],
    selected: dict[str, list[dict[str, object]]],
    args: argparse.Namespace,
) -> dict[str, object]:
    out_dir = args.root / "freesurfer_aparc_scalar4_laion"
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = sorted({str(row["session"]) for label in labels for row in selected[label]})
    first_beta = nib.load(str(beta_path_for(args, subject, sessions[0])))
    aparc = resampled_aparc(subject, first_beta, args)
    nodes = aparc_nodes()
    roi_indices = [np.flatnonzero((aparc == label_id).reshape(-1)) for label_id, _ in nodes]

    n_trials = len(labels) * args.min_repeats
    x = np.zeros((n_trials, 180, len(FEATURE_NAMES)), dtype=np.float32)
    image_labels: list[str] = []
    repetitions: list[int] = []
    source_rows: list[dict[str, object]] = []
    position_by_key: dict[tuple[str, int], int] = {}
    row_idx = 0
    for label in labels:
        for rep, row in enumerate(selected[label], start=1):
            image_labels.append(label)
            repetitions.append(rep)
            position_by_key[(str(row["session"]), int(row["beta_index"]))] = row_idx
            source_rows.append({"row_index": row_idx, "label": label, "repetition": rep, **row})
            row_idx += 1

    for session in sessions:
        img = nib.load(str(beta_path_for(args, subject, session)))
        selected_indices = sorted(
            int(row["beta_index"])
            for label in labels
            for row in selected[label]
            if str(row["session"]) == session
        )
        if not selected_indices:
            continue
        volume = np.asanyarray(img.dataobj, dtype=np.float32)[..., selected_indices]
        if volume.ndim == 3:
            volume = volume[..., None]
        flat = volume.reshape(-1, len(selected_indices))
        out_positions = [position_by_key[(session, beta_idx)] for beta_idx in selected_indices]
        for node_idx, mask_idx in enumerate(roi_indices):
            if mask_idx.size == 0:
                continue
            values = flat[mask_idx, :].T
            x[out_positions, node_idx, :] = scalar4(values, args.top_quantile)

    node_labels = [name for _, name in nodes] + [f"pad_{idx:03d}" for idx in range(len(nodes) + 1, 181)]
    voxel_counts = np.zeros(180, dtype=np.int32)
    voxel_counts[: len(nodes)] = np.asarray([idx.size for idx in roi_indices], dtype=np.int32)
    payload = {
        "x": torch.from_numpy(x),
        "subject": subject,
        "image_label": image_labels,
        "repetition": torch.tensor(repetitions, dtype=torch.int16),
        "feature_names": FEATURE_NAMES,
        "node_set_name": "LAION-fMRI FreeSurfer aparc cortical parcels padded to 180 nodes",
        "node_labels": node_labels,
        "source_roi_labels": [name for _, name in nodes],
        "voxel_counts": torch.from_numpy(voxel_counts),
        "source_rows": source_rows,
    }
    torch.save(payload, out_dir / f"{subject}_laion_visual_roi_scalar4.pt")
    with (out_dir / f"{subject}_laion_visual_roi_scalar4_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)
    qc = {
        "subject": subject,
        "n_trials": int(n_trials),
        "n_images": int(len(labels)),
        "n_sessions": int(len(sessions)),
        "feature_shape": list(x.shape),
        "n_rois": len(nodes),
        "nan_count": int(np.isnan(x).sum()),
        "inf_count": int(np.isinf(x).sum()),
        "min_roi_voxels": int(voxel_counts[: len(nodes)].min()),
        "max_roi_voxels": int(voxel_counts[: len(nodes)].max()),
    }
    (out_dir / f"{subject}_laion_visual_roi_scalar4_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(qc, flush=True)
    return qc


def main() -> None:
    args = parse_args()
    enforce_hpc_external_path(args.root, "LAION-fMRI FreeSurfer export root")
    enforce_hpc_external_path(args.metadata_dir, "LAION-fMRI trial metadata directory")
    args.root.mkdir(parents=True, exist_ok=True)
    labels, selected = choose_labels(args)
    if not labels:
        raise RuntimeError("No LAION labels satisfy the requested constraints.")
    if args.download_only:
        downloads = prepare_downloads(args, labels, selected)
        out_dir = args.root / "freesurfer_aparc_scalar4_laion"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "laion_download_manifest.json").write_text(
            json.dumps({"subjects": args.subjects, "labels": labels, "selected": selected, "downloads": downloads}, indent=2),
            encoding="utf-8",
        )
        return
    qcs = [export_subject(subject, labels, selected[subject], args) for subject in args.subjects]
    out_dir = args.root / "freesurfer_aparc_scalar4_laion"
    (out_dir / "laion_freesurfer_aparc_scalar4_manifest.json").write_text(
        json.dumps({"subjects": args.subjects, "n_labels": len(labels), "n_rois": len(aparc_nodes()), "qcs": qcs}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
