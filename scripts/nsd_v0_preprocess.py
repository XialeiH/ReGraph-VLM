#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.io import loadmat


TRIALS_PER_SESSION = 750
SELECTED_ROIS = [
    {"roi_name": "V1", "label_values": [1, 2]},
    {"roi_name": "V2", "label_values": [3, 4]},
    {"roi_name": "V3", "label_values": [5, 6]},
    {"roi_name": "hV4", "label_values": [7]},
]
HEMISPHERES = ["lh", "rh"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Version 0 NSD preprocessing artifacts.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subject", type=int, default=1, help="Pilot subject for averaged beta export.")
    return parser.parse_args()


def beta_prefix(subject: int) -> Path:
    return Path(
        f"nsddata_betas/ppdata/subj{subject:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR"
    )


def roi_prefix(subject: int) -> Path:
    return Path(f"nsddata/ppdata/subj{subject:02d}/func1pt8mm/roi")


def load_design(data_root: Path) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    expdesign_path = data_root / "nsddata/experiments/nsd/nsd_expdesign.mat"
    stim_info_path = data_root / "nsddata/experiments/nsd/nsd_stim_info_merged.csv"
    mat = loadmat(expdesign_path, squeeze_me=True, struct_as_record=False)
    stim_info = pd.read_csv(stim_info_path)
    return mat, stim_info


def available_beta_files(data_root: Path, subject: int) -> dict[int, Path]:
    prefix = data_root / beta_prefix(subject)
    files = {}
    for path in sorted(prefix.glob("betas_session*.hdf5")):
        session = int(path.stem.replace("betas_session", ""))
        files[session] = path
    return files


def expected_nsd_id_one_based(
    subject: int,
    global_trial_index: int,
    masterordering: np.ndarray,
    subjectim: np.ndarray,
) -> int:
    subjectim_index = int(masterordering[global_trial_index - 1])
    return int(subjectim[subject - 1, subjectim_index - 1])


def build_manifest(
    data_root: Path,
    output_root: Path,
    mat: dict[str, np.ndarray],
    stim_info: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared_df = stim_info.loc[stim_info["shared1000"] == True].copy()
    shared_df = shared_df.sort_values("nsdId").reset_index(drop=True)
    shared_df["shared_group_id"] = np.arange(1, len(shared_df) + 1, dtype=np.int32)

    sharedix_zero_based = np.sort(mat["sharedix"].astype(np.int64) - 1)
    csv_shared_zero_based = np.sort(shared_df["nsdId"].to_numpy(dtype=np.int64))
    if not np.array_equal(sharedix_zero_based, csv_shared_zero_based):
        raise ValueError("shared1000 in stim_info does not match sharedix in expdesign")

    subjectim = mat["subjectim"].astype(np.int64)
    masterordering = mat["masterordering"].astype(np.int64)

    all_rows: list[dict[str, object]] = []
    subject_rows_for_summary: dict[int, list[dict[str, object]]] = defaultdict(list)

    for subject in range(1, 9):
        rep_cols = [f"subject{subject}_rep0", f"subject{subject}_rep1", f"subject{subject}_rep2"]
        available_sessions = available_beta_files(data_root, subject)
        available_session_ids = set(available_sessions)
        for shared_record in shared_df.itertuples(index=False):
            for rep_index, rep_col in enumerate(rep_cols):
                global_trial_index = int(getattr(shared_record, rep_col))
                stim_info_available = bool(getattr(shared_record, f"subject{subject}") == 1 and global_trial_index > 0)
                session = ((global_trial_index - 1) // TRIALS_PER_SESSION + 1) if global_trial_index > 0 else pd.NA
                trial_in_session = ((global_trial_index - 1) % TRIALS_PER_SESSION + 1) if global_trial_index > 0 else pd.NA
                beta_path = (
                    data_root
                    / beta_prefix(subject)
                    / f"betas_session{int(session):02d}.hdf5"
                    if session is not pd.NA
                    else None
                )
                beta_exists = bool(session in available_session_ids) if session is not pd.NA else False
                mapping_matches = False
                indexing_in_range = False
                if global_trial_index > 0 and global_trial_index <= len(masterordering):
                    indexing_in_range = True
                    expected = expected_nsd_id_one_based(subject, global_trial_index, masterordering, subjectim)
                    mapping_matches = expected == int(shared_record.nsdId) + 1
                usable = bool(
                    stim_info_available
                    and indexing_in_range
                    and mapping_matches
                    and beta_exists
                    and trial_in_session is not pd.NA
                )
                row = {
                    "nsdId": int(shared_record.nsdId),
                    "subject": f"subj{subject:02d}",
                    "session": int(session) if session is not pd.NA else pd.NA,
                    "trial_in_session": int(trial_in_session) if trial_in_session is not pd.NA else pd.NA,
                    "beta_index_in_session": int(trial_in_session) if trial_in_session is not pd.NA else pd.NA,
                    "is_shared": True,
                    "shared_group_id": int(shared_record.shared_group_id),
                    "rep_index_for_subject": rep_index,
                    "global_trial_index": global_trial_index if global_trial_index > 0 else pd.NA,
                    "beta_path": str(beta_path) if beta_path is not None else "",
                    "stim_info_available": stim_info_available,
                    "mapping_matches_design": mapping_matches,
                    "beta_file_exists": beta_exists,
                    "usable": usable,
                }
                all_rows.append(row)
                subject_rows_for_summary[subject].append(row)

    manifest_df = pd.DataFrame(all_rows).sort_values(
        ["subject", "shared_group_id", "rep_index_for_subject"]
    )
    manifest_path = output_root / "shared1000_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    summary_rows: list[dict[str, object]] = []
    for subject in range(1, 9):
        subject_df = pd.DataFrame(subject_rows_for_summary[subject])
        reps_per_image = (
            subject_df.groupby("nsdId")["global_trial_index"]
            .apply(lambda s: int(s.notna().sum()))
            .reset_index(drop=True)
        )
        summary_rows.append(
            {
                "subject": f"subj{subject:02d}",
                "n_shared_images_found": int(subject_df["nsdId"].nunique()),
                "n_total_repetitions": int(subject_df["global_trial_index"].notna().sum()),
                "mean_reps_per_image": float(reps_per_image.mean()),
                "min_reps_per_image": int(reps_per_image.min()),
                "max_reps_per_image": int(reps_per_image.max()),
                "n_usable_rows": int(subject_df["usable"].sum()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root / "shared1000_summary.csv", index=False)
    return manifest_df, summary_df


def build_roi_artifacts(data_root: Path, output_root: Path) -> tuple[pd.DataFrame, dict]:
    rows: list[dict[str, object]] = []
    subject_file_inventory: dict[str, list[str]] = {}
    for subject in range(1, 9):
        subject_dir = data_root / roi_prefix(subject)
        subject_key = f"subj{subject:02d}"
        subject_file_inventory[subject_key] = sorted(path.name for path in subject_dir.glob("*.nii.gz"))
        for roi_spec in SELECTED_ROIS:
            for hemisphere in HEMISPHERES:
                mask_name = f"{hemisphere}.prf-visualrois.nii.gz"
                mask_path = subject_dir / mask_name
                exists = mask_path.exists()
                n_voxels = 0
                if exists:
                    arr = np.asanyarray(nib.load(str(mask_path)).dataobj)
                    n_voxels = int(np.isin(arr, roi_spec["label_values"]).sum())
                rows.append(
                    {
                        "subject": subject_key,
                        "roi_name": roi_spec["roi_name"],
                        "hemisphere": hemisphere,
                        "mask_file": str(mask_path),
                        "n_voxels": n_voxels,
                        "exists": bool(exists),
                        "usable": bool(exists and n_voxels > 0),
                    }
                )
    roi_summary_df = pd.DataFrame(rows).sort_values(["subject", "roi_name", "hemisphere"])
    roi_summary_df.to_csv(output_root / "roi_voxel_count_summary.csv", index=False)

    roi_selection = {
        "version": "v0_shared_unit",
        "selection_name": "small_stable_retinotopic_visual_set",
        "selection_rationale": (
            "Version 0 uses only retinotopic prf-visualrois-derived V1/V2/V3/hV4 because these masks "
            "exist consistently across subjects and are more stable than whole-brain or higher-level ROI unions "
            "for the first shared-unit pass."
        ),
        "source_mask_family": "prf-visualrois",
        "hemisphere_policy": (
            "Voxel counts are tracked separately for left and right hemispheres, then merged within each ROI "
            "name for model-ready ROI-separated and concatenated features."
        ),
        "feature_storage": {
            "roi_separated": True,
            "concatenated": True,
            "concatenated_order": [spec["roi_name"] for spec in SELECTED_ROIS],
        },
        "selected_rois": [
            {
                "roi_name": spec["roi_name"],
                "label_values": spec["label_values"],
                "mask_files": [f"lh.prf-visualrois.nii.gz", f"rh.prf-visualrois.nii.gz"],
                "merge_hemispheres": True,
            }
            for spec in SELECTED_ROIS
        ],
        "excluded_for_v0": [
            "whole-brain",
            "streams",
            "floc-bodies",
            "floc-faces",
            "floc-places",
            "floc-words",
        ],
        "subject_file_inventory": subject_file_inventory,
    }
    with (output_root / "roi_selection_v0.json").open("w", encoding="utf-8") as handle:
        json.dump(roi_selection, handle, indent=2)
    return roi_summary_df, roi_selection


def merged_roi_masks_hdf5_order(data_root: Path, subject: int) -> dict[str, np.ndarray]:
    subject_dir = data_root / roi_prefix(subject)
    masks: dict[str, np.ndarray] = {}
    for roi_spec in SELECTED_ROIS:
        merged_mask = None
        for hemisphere in HEMISPHERES:
            arr = np.asanyarray(nib.load(str(subject_dir / f"{hemisphere}.prf-visualrois.nii.gz")).dataobj)
            hemi_mask = np.isin(arr, roi_spec["label_values"])
            merged_mask = hemi_mask if merged_mask is None else (merged_mask | hemi_mask)
        masks[roi_spec["roi_name"]] = np.transpose(merged_mask.astype(bool), (2, 1, 0))
    return masks


def export_subject_features(
    data_root: Path,
    output_root: Path,
    manifest_df: pd.DataFrame,
    subject: int,
) -> dict[str, object]:
    subject_key = f"subj{subject:02d}"
    subject_manifest = manifest_df.loc[(manifest_df["subject"] == subject_key) & (manifest_df["usable"] == True)].copy()
    subject_manifest["global_trial_index"] = subject_manifest["global_trial_index"].astype(int)
    subject_manifest["session"] = subject_manifest["session"].astype(int)
    subject_manifest["trial_in_session"] = subject_manifest["trial_in_session"].astype(int)
    grouped = list(subject_manifest.groupby("nsdId"))

    if len(grouped) != 1000:
        raise ValueError(f"Expected 1000 shared images for {subject_key}, found {len(grouped)} usable groups")

    beta_files = available_beta_files(data_root, subject)
    hdf5_handles = {session: h5py.File(path, "r") for session, path in beta_files.items()}
    roi_masks = merged_roi_masks_hdf5_order(data_root, subject)
    roi_dims = {roi_name: int(mask.sum()) for roi_name, mask in roi_masks.items()}
    total_concat_dim = int(sum(roi_dims.values()))
    union_mask = np.logical_or.reduce(list(roi_masks.values()))
    union_dim = int(union_mask.sum())

    averaged_betas_union = np.empty((len(grouped), union_dim), dtype=np.float32)
    roi_features = {
        roi_name: np.empty((len(grouped), dim), dtype=np.float32)
        for roi_name, dim in roi_dims.items()
    }
    concatenated = np.empty((len(grouped), total_concat_dim), dtype=np.float32)

    metadata_rows: list[dict[str, object]] = []
    n_repetitions_used = np.empty(len(grouped), dtype=np.int16)
    nsd_ids = np.empty(len(grouped), dtype=np.int32)

    for image_idx, (nsd_id, rows_df) in enumerate(grouped):
        rows_df = rows_df.sort_values("rep_index_for_subject")
        rep_volumes = []
        sessions_used: list[int] = []
        trials_used: list[int] = []
        beta_paths_used: list[str] = []
        rep_indices_used: list[int] = []
        for row in rows_df.itertuples(index=False):
            session = int(row.session)
            trial_idx = int(row.trial_in_session) - 1
            rep_volumes.append(hdf5_handles[session]["betas"][trial_idx].astype(np.float32))
            sessions_used.append(session)
            trials_used.append(int(row.trial_in_session))
            beta_paths_used.append(str(row.beta_path))
            rep_indices_used.append(int(row.rep_index_for_subject))
        averaged_volume = np.mean(np.stack(rep_volumes, axis=0), axis=0, dtype=np.float32)
        averaged_betas_union[image_idx] = averaged_volume[union_mask]
        concat_parts = []
        for roi_name in [spec["roi_name"] for spec in SELECTED_ROIS]:
            roi_vector = averaged_volume[roi_masks[roi_name]]
            roi_features[roi_name][image_idx] = roi_vector
            concat_parts.append(roi_vector)
        concatenated[image_idx] = np.concatenate(concat_parts, axis=0)
        nsd_ids[image_idx] = int(nsd_id)
        n_repetitions_used[image_idx] = len(rep_volumes)
        metadata_rows.append(
            {
                "nsdId": int(nsd_id),
                "subject": subject_key,
                "n_repetitions_used": int(len(rep_volumes)),
                "all_rep_indices": json.dumps(rep_indices_used),
                "all_sessions": json.dumps(sessions_used),
                "all_trials": json.dumps(trials_used),
                "beta_files_used": json.dumps(beta_paths_used),
                "is_complete_average": bool(len(rep_volumes) == 3),
                "usable": True,
            }
        )

    for handle in hdf5_handles.values():
        handle.close()

    avgbeta_path = output_root / f"{subject_key}_shared1000_avgbetas.npz"
    np.savez(
        avgbeta_path,
        nsd_ids=nsd_ids,
        averaged_betas_union=averaged_betas_union,
        union_mask_hdf5_order=union_mask,
        volume_shape_hdf5_order=np.array([83, 104, 81], dtype=np.int32),
        n_repetitions_used=n_repetitions_used,
    )

    offsets = [0]
    for roi_name in [spec["roi_name"] for spec in SELECTED_ROIS]:
        offsets.append(offsets[-1] + roi_dims[roi_name])
    roi_feature_path = output_root / f"{subject_key}_shared1000_roi_features.npz"
    np.savez(
        roi_feature_path,
        nsd_ids=nsd_ids,
        roi_names=np.array([spec["roi_name"] for spec in SELECTED_ROIS], dtype="<U16"),
        roi_dims=np.array([roi_dims[spec["roi_name"]] for spec in SELECTED_ROIS], dtype=np.int32),
        concatenated=concatenated,
        concatenated_offsets=np.array(offsets, dtype=np.int32),
        **roi_features,
    )

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = output_root / f"{subject_key}_shared1000_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    qc = {
        "subject": subject_key,
        "n_shared_images": int(len(nsd_ids)),
        "mean_repetitions": float(n_repetitions_used.mean()),
        "min_repetitions": int(n_repetitions_used.min()),
        "max_repetitions": int(n_repetitions_used.max()),
        "nan_count": int(np.isnan(concatenated).sum()),
        "inf_count": int(np.isinf(concatenated).sum()),
        "averaged_beta_union_dim": int(averaged_betas_union.shape[1]),
        "concatenated_dim": int(concatenated.shape[1]),
        "all_zero_images_per_roi": {
            roi_name: int(np.all(roi_features[roi_name] == 0, axis=1).sum())
            for roi_name in roi_features
        },
        "roi_dims": roi_dims,
        "output_files": {
            "avgbetas": str(avgbeta_path),
            "roi_features": str(roi_feature_path),
            "metadata": str(metadata_path),
        },
    }
    return qc


def write_qc_note(
    output_root: Path,
    manifest_summary: pd.DataFrame,
    roi_summary: pd.DataFrame,
    subject_qc: dict[str, object],
) -> None:
    roi_lines = []
    for roi_name in [spec["roi_name"] for spec in SELECTED_ROIS]:
        subset = roi_summary.loc[roi_summary["roi_name"] == roi_name]
        roi_lines.append(
            f"- `{roi_name}`: voxel count range across hemispheres/subjects = "
            f"{int(subset['n_voxels'].min())} to {int(subset['n_voxels'].max())}"
        )

    summary_row = manifest_summary.loc[manifest_summary["subject"] == subject_qc["subject"]].iloc[0]
    qc_path = output_root / "preprocessing_qc_note.md"
    qc_path.write_text(
        "\n".join(
            [
                "# Preprocessing QC Note",
                "",
                f"- Pilot subject: `{subject_qc['subject']}`",
                "- Version 0 scope: shared1000 manifest -> stable retinotopic ROI parsing -> repeated-image averaging for one subject.",
                f"- Shared images found for pilot subject: `{int(summary_row['n_shared_images_found'])}`",
                f"- Total repetitions for pilot subject: `{int(summary_row['n_total_repetitions'])}`",
                f"- Mean repetitions per image: `{summary_row['mean_reps_per_image']:.2f}`",
                f"- Usable manifest rows for pilot subject: `{int(summary_row['n_usable_rows'])}`",
                f"- Averaged shared images exported: `{subject_qc['n_shared_images']}`",
                f"- Averaged beta union dimension: `{subject_qc['averaged_beta_union_dim']}`",
                f"- Concatenated feature dimension: `{subject_qc['concatenated_dim']}`",
                f"- NaN count in concatenated features: `{subject_qc['nan_count']}`",
                f"- Inf count in concatenated features: `{subject_qc['inf_count']}`",
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
                f"- Averaged betas: `{subject_qc['output_files']['avgbetas']}`",
                f"- ROI features: `{subject_qc['output_files']['roi_features']}`",
                f"- Metadata: `{subject_qc['output_files']['metadata']}`",
                "",
                "## Sanity Checks",
                "",
                f"- Repetition count range for averaged images: `{subject_qc['min_repetitions']}` to `{subject_qc['max_repetitions']}`",
                f"- Mean repetitions used: `{subject_qc['mean_repetitions']:.2f}`",
                f"- All-zero images per ROI: `{json.dumps(subject_qc['all_zero_images_per_roi'])}`",
                "- No split-aware normalization or PCA was applied in this step.",
                "",
                "## Next-Step Feasibility",
                "",
                "- The manifest and ROI definitions are now fixed enough to scale the averaging/export step to all subjects.",
                "- The next implementation step should extend the same pipeline to all subjects, then generate split-aware normalization and PCA artifacts.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    mat, stim_info = load_design(args.data_root)
    manifest_df, manifest_summary = build_manifest(args.data_root, args.output_root, mat, stim_info)
    roi_summary, _ = build_roi_artifacts(args.data_root, args.output_root)
    subject_qc = export_subject_features(args.data_root, args.output_root, manifest_df, args.subject)
    write_qc_note(args.output_root, manifest_summary, roi_summary, subject_qc)


if __name__ == "__main__":
    main()
