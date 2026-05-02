#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd


SELECTED_ROIS = [
    {"roi_name": "V1", "label_values": [1, 2]},
    {"roi_name": "V2", "label_values": [3, 4]},
    {"roi_name": "V3", "label_values": [5, 6]},
    {"roi_name": "hV4", "label_values": [7]},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one NSD subject into trial-level ROI feature artifacts.")
    parser.add_argument("--subject", type=int, required=True, help="Subject number, for example 1 for subj01.")
    parser.add_argument("--root", type=Path, default=Path("."), help="v0_shared_unit root on HPC.")
    parser.add_argument("--output-root", type=Path, required=True, help="Stage 3B output root.")
    parser.add_argument("--manifest", type=Path, default=Path("preproc_v0/shared1000_manifest.csv"))
    parser.add_argument("--image-universe", type=Path, required=True, help="CSV containing the trial-level image universe.")
    return parser.parse_args()


def load_roi_masks(subject_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    roi_masks: dict[str, np.ndarray] = {}
    roi_dims: dict[str, int] = {}
    for spec in SELECTED_ROIS:
        merged = None
        for hemi in ["lh", "rh"]:
            arr = np.asanyarray(nib.load(str(subject_dir / f"{hemi}.prf-visualrois.nii.gz")).dataobj)
            hemi_mask = np.isin(arr, spec["label_values"])
            merged = hemi_mask if merged is None else (merged | hemi_mask)
        mask = np.transpose(merged.astype(bool), (2, 1, 0))
        roi_masks[spec["roi_name"]] = mask
        roi_dims[spec["roi_name"]] = int(mask.sum())
    return roi_masks, roi_dims


def main() -> None:
    args = parse_args()
    subject = f"subj{args.subject:02d}"
    root = args.root
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    image_universe = pd.read_csv(args.image_universe)
    selected_ids = set(image_universe["nsdId"].astype(int).tolist())

    manifest = pd.read_csv(args.manifest)
    subject_rows = manifest[
        (manifest["subject"] == subject)
        & (manifest["usable"] == True)
        & (manifest["nsdId"].isin(selected_ids))
    ].copy()
    if subject_rows.empty:
        raise ValueError(f"No usable trial-level rows found for {subject}")

    subject_rows["session"] = subject_rows["session"].astype(int)
    subject_rows["trial_in_session"] = subject_rows["trial_in_session"].astype(int)
    subject_rows["rep_index_for_subject"] = subject_rows["rep_index_for_subject"].astype(int)
    subject_rows = subject_rows.sort_values(["nsdId", "rep_index_for_subject", "session", "trial_in_session"]).reset_index(drop=True)

    subject_dir = root / f"data/nsddata/ppdata/{subject}/func1pt8mm/roi"
    beta_dir = root / f"data/nsddata_betas/ppdata/{subject}/func1pt8mm/betas_fithrf_GLMdenoise_RR"

    roi_masks, roi_dims = load_roi_masks(subject_dir)
    roi_names = [spec["roi_name"] for spec in SELECTED_ROIS]
    offsets = [0]
    for roi_name in roi_names:
        offsets.append(offsets[-1] + roi_dims[roi_name])

    concatenated = np.empty((len(subject_rows), offsets[-1]), dtype=np.float32)
    roi_features = {
        roi_name: np.empty((len(subject_rows), roi_dims[roi_name]), dtype=np.float32)
        for roi_name in roi_names
    }

    handles: dict[int, h5py.File] = {}
    metadata_rows: list[dict[str, object]] = []
    try:
        for sample_index, row in enumerate(subject_rows.itertuples(index=False)):
            session = int(row.session)
            if session not in handles:
                handles[session] = h5py.File(beta_dir / f"betas_session{session:02d}.hdf5", "r")
            trial_idx = int(row.trial_in_session) - 1
            beta = handles[session]["betas"][trial_idx].astype(np.float32)

            parts = []
            for roi_name in roi_names:
                vec = beta[roi_masks[roi_name]]
                roi_features[roi_name][sample_index] = vec
                parts.append(vec)
            concatenated[sample_index] = np.concatenate(parts, axis=0)

            metadata_rows.append(
                {
                    "sample_index": sample_index,
                    "subject": subject,
                    "nsdId": int(row.nsdId),
                    "session": int(row.session),
                    "trial_in_session": int(row.trial_in_session),
                    "rep_index_for_subject": int(row.rep_index_for_subject),
                    "beta_index_in_session": int(row.beta_index_in_session),
                    "global_trial_index": int(row.global_trial_index),
                    "beta_path": str(row.beta_path),
                    "usable": True,
                }
            )
    finally:
        for handle in handles.values():
            handle.close()

    feature_path = output_root / f"{subject}_triallevel_shared907_roi_features.npz"
    np.savez(
        feature_path,
        roi_names=np.array(roi_names, dtype="<U16"),
        roi_dims=np.array([roi_dims[roi_name] for roi_name in roi_names], dtype=np.int32),
        concatenated=concatenated,
        concatenated_offsets=np.array(offsets, dtype=np.int32),
        nsd_ids=subject_rows["nsdId"].astype(np.int32).to_numpy(),
        sessions=subject_rows["session"].astype(np.int16).to_numpy(),
        trials=subject_rows["trial_in_session"].astype(np.int16).to_numpy(),
        rep_indices=subject_rows["rep_index_for_subject"].astype(np.int16).to_numpy(),
        **roi_features,
    )

    metadata_path = output_root / f"{subject}_triallevel_shared907_metadata.csv"
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)

    qc_payload = {
        "subject": subject,
        "n_trial_samples": int(len(subject_rows)),
        "n_unique_images": int(subject_rows["nsdId"].nunique()),
        "mean_trials_per_image": float(subject_rows.groupby("nsdId").size().mean()),
        "min_trials_per_image": int(subject_rows.groupby("nsdId").size().min()),
        "max_trials_per_image": int(subject_rows.groupby("nsdId").size().max()),
        "concatenated_dim": int(concatenated.shape[1]),
        "nan_count": int(np.isnan(concatenated).sum()),
        "inf_count": int(np.isinf(concatenated).sum()),
        "feature_path": str(feature_path),
        "metadata_path": str(metadata_path),
    }
    (output_root / f"{subject}_triallevel_shared907_qc.json").write_text(json.dumps(qc_payload, indent=2), encoding="utf-8")
    print(json.dumps(qc_payload))


if __name__ == "__main__":
    main()
