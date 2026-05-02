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
    parser = argparse.ArgumentParser(description="Export one NSD subject into averaged-beta and ROI feature artifacts.")
    parser.add_argument("--subject", type=int, required=True, help="Subject number, for example 1 for subj01.")
    parser.add_argument("--root", type=Path, default=Path("."), help="v0_shared_unit root on HPC.")
    parser.add_argument("--output-root", type=Path, default=Path("preproc_v0"), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subject = f"subj{args.subject:02d}"
    root = args.root
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(output_root / "shared1000_manifest.csv")
    summary = pd.read_csv(output_root / "shared1000_summary.csv")
    roi_summary = pd.read_csv(output_root / "roi_voxel_count_summary.csv")

    subject_manifest = manifest[(manifest["subject"] == subject) & (manifest["usable"] == True)].copy()
    if subject_manifest.empty:
        raise ValueError(f"No usable manifest rows found for {subject}")

    subject_manifest["session"] = subject_manifest["session"].astype(int)
    subject_manifest["trial_in_session"] = subject_manifest["trial_in_session"].astype(int)
    subject_manifest["rep_index_for_subject"] = subject_manifest["rep_index_for_subject"].astype(int)
    subject_manifest = subject_manifest.sort_values(["nsdId", "rep_index_for_subject"])

    subject_dir = root / f"data/nsddata/ppdata/{subject}/func1pt8mm/roi"
    beta_dir = root / f"data/nsddata_betas/ppdata/{subject}/func1pt8mm/betas_fithrf_GLMdenoise_RR"

    roi_masks: dict[str, np.ndarray] = {}
    for spec in SELECTED_ROIS:
        merged = None
        for hemi in ["lh", "rh"]:
            arr = np.asanyarray(nib.load(str(subject_dir / f"{hemi}.prf-visualrois.nii.gz")).dataobj)
            hemi_mask = np.isin(arr, spec["label_values"])
            merged = hemi_mask if merged is None else (merged | hemi_mask)
        roi_masks[spec["roi_name"]] = np.transpose(merged.astype(bool), (2, 1, 0))

    subject_groups = list(subject_manifest.groupby("nsdId"))
    nsd_ids = np.array([int(nsd_id) for nsd_id, _ in subject_groups], dtype=np.int32)
    roi_dims = {name: int(mask.sum()) for name, mask in roi_masks.items()}
    union_mask = np.logical_or.reduce(list(roi_masks.values()))

    averaged_union = np.empty((len(nsd_ids), int(union_mask.sum())), dtype=np.float32)
    roi_features = {
        name: np.empty((len(nsd_ids), dim), dtype=np.float32)
        for name, dim in roi_dims.items()
    }
    concatenated = np.empty((len(nsd_ids), int(sum(roi_dims.values()))), dtype=np.float32)
    reps_used = np.empty(len(nsd_ids), dtype=np.int16)

    handles: dict[int, h5py.File] = {}
    metadata_rows: list[dict[str, object]] = []

    for idx, (nsd_id, rows) in enumerate(subject_groups):
        rows = rows.sort_values("rep_index_for_subject")
        rep_volumes = []
        sessions_used: list[int] = []
        trials_used: list[int] = []
        beta_files_used: list[str] = []
        rep_indices_used: list[int] = []

        for row in rows.itertuples(index=False):
            session = int(row.session)
            if session not in handles:
                handles[session] = h5py.File(beta_dir / f"betas_session{session:02d}.hdf5", "r")
            trial_idx = int(row.trial_in_session) - 1
            rep_volumes.append(handles[session]["betas"][trial_idx].astype(np.float32))
            sessions_used.append(session)
            trials_used.append(int(row.trial_in_session))
            beta_files_used.append(str(row.beta_path))
            rep_indices_used.append(int(row.rep_index_for_subject))

        averaged = np.mean(np.stack(rep_volumes, axis=0), axis=0, dtype=np.float32)
        averaged_union[idx] = averaged[union_mask]

        parts = []
        for spec in SELECTED_ROIS:
            roi_name = spec["roi_name"]
            vec = averaged[roi_masks[roi_name]]
            roi_features[roi_name][idx] = vec
            parts.append(vec)
        concatenated[idx] = np.concatenate(parts, axis=0)
        reps_used[idx] = len(rep_volumes)

        metadata_rows.append(
            {
                "nsdId": int(nsd_id),
                "subject": subject,
                "n_repetitions_used": int(len(rep_volumes)),
                "all_rep_indices": json.dumps(rep_indices_used),
                "all_sessions": json.dumps(sessions_used),
                "all_trials": json.dumps(trials_used),
                "beta_files_used": json.dumps(beta_files_used),
                "is_complete_average": bool(len(rep_volumes) == 3),
                "usable": True,
            }
        )

    for handle in handles.values():
        handle.close()

    np.savez(
        output_root / f"{subject}_shared1000_avgbetas.npz",
        nsd_ids=nsd_ids,
        averaged_betas_union=averaged_union,
        union_mask_hdf5_order=union_mask,
        volume_shape_hdf5_order=np.array([83, 104, 81], dtype=np.int32),
        n_repetitions_used=reps_used,
    )

    offsets = [0]
    for spec in SELECTED_ROIS:
        offsets.append(offsets[-1] + roi_dims[spec["roi_name"]])

    np.savez(
        output_root / f"{subject}_shared1000_roi_features.npz",
        nsd_ids=nsd_ids,
        roi_names=np.array([spec["roi_name"] for spec in SELECTED_ROIS], dtype="<U16"),
        roi_dims=np.array([roi_dims[spec["roi_name"]] for spec in SELECTED_ROIS], dtype=np.int32),
        concatenated=concatenated,
        concatenated_offsets=np.array(offsets, dtype=np.int32),
        **roi_features,
    )

    pd.DataFrame(metadata_rows).to_csv(output_root / f"{subject}_shared1000_metadata.csv", index=False)

    summary_row = summary[summary["subject"] == subject].iloc[0]
    roi_lines = []
    for spec in SELECTED_ROIS:
        subset = roi_summary[roi_summary["roi_name"] == spec["roi_name"]]
        roi_lines.append(
            f"- `{spec['roi_name']}`: voxel count range across hemispheres/subjects = "
            f"{int(subset['n_voxels'].min())} to {int(subset['n_voxels'].max())}"
        )

    qc_lines = [
        "# Preprocessing QC Note",
        "",
        f"- Pilot subject: `{subject}`",
        "- Version 0 scope: shared1000 manifest -> stable retinotopic ROI parsing -> repeated-image averaging for one subject.",
        f"- Shared images found for subject: `{int(summary_row['n_shared_images_found'])}`",
        f"- Total repetitions for subject: `{int(summary_row['n_total_repetitions'])}`",
        f"- Mean repetitions per image in manifest: `{summary_row['mean_reps_per_image']:.2f}`",
        f"- Usable manifest rows for subject: `{int(summary_row['n_usable_rows'])}`",
        f"- Averaged shared images exported: `{len(nsd_ids)}`",
        f"- Averaged beta union dimension: `{averaged_union.shape[1]}`",
        f"- Concatenated feature dimension: `{concatenated.shape[1]}`",
        f"- NaN count in concatenated features: `{int(np.isnan(concatenated).sum())}`",
        f"- Inf count in concatenated features: `{int(np.isinf(concatenated).sum())}`",
        "",
        "## ROI Choice",
        "",
        "- Version 0 keeps only `V1`, `V2`, `V3`, and `hV4` from `prf-visualrois`.",
        "- This small retinotopic set is used because it is consistently available across subjects and gives a stable shared-unit starting point.",
        "- Whole-brain is intentionally excluded in the first pass to avoid unstable voxel support and unnecessary preprocessing complexity before shared-unit validation.",
        "",
        "## ROI Voxel Count Ranges",
        "",
        *roi_lines,
        "",
        "## Subject-Level Outputs",
        "",
        f"- Averaged betas: `{output_root / f'{subject}_shared1000_avgbetas.npz'}`",
        f"- ROI features: `{output_root / f'{subject}_shared1000_roi_features.npz'}`",
        f"- Metadata: `{output_root / f'{subject}_shared1000_metadata.csv'}`",
        "",
        "## Sanity Checks",
        "",
        f"- Repetition count range for averaged images: `{int(reps_used.min())}` to `{int(reps_used.max())}`",
        f"- Mean repetitions used: `{float(reps_used.mean()):.2f}`",
        f"- All-zero images per ROI: `{json.dumps({name: int(np.all(roi_features[name] == 0, axis=1).sum()) for name in roi_features})}`",
        "- No split-aware normalization or PCA was applied in this step.",
        "",
        "## Next-Step Feasibility",
        "",
        "- The manifest and ROI definitions are now fixed enough to scale the averaging/export step to all subjects.",
        "- The next implementation step should extend the same pipeline to all subjects, then generate split-aware normalization and PCA artifacts.",
        "",
    ]
    (output_root / f"{subject}_preprocessing_qc_note.md").write_text("\n".join(qc_lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "subject": subject,
                "n_images_exported": int(len(nsd_ids)),
                "mean_repetitions_used": float(reps_used.mean()),
                "maxrss_hint": "see Slurm sstat for runtime memory",
            }
        )
    )


if __name__ == "__main__":
    main()
