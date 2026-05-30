#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


FEATURE_NAMES = ["mean_beta", "std_beta", "q90_beta", "positive_fraction"]
PREFERRED_VISUAL_ROIS = [
    "V1",
    "V2",
    "V3",
    "hV4",
    "VO1",
    "VO2",
    "LO1",
    "LO2",
    "TO1",
    "TO2",
    "V3A",
    "V3B",
    "lFFA",
    "rFFA",
    "lOFA",
    "rOFA",
    "lEBA",
    "rEBA",
    "lPPA",
    "rPPA",
    "lOPA",
    "rOPA",
    "lRSC",
    "rRSC",
    "lLOC",
    "rLOC",
    "FFA",
    "OFA",
    "EBA",
    "PPA",
    "OPA",
    "RSC",
    "LOC",
]
IMAGE_COLUMN_CANDIDATES = [
    "image_name",
    "image_file",
    "image_filename",
    "filename",
    "file_name",
    "stim_file",
    "stimulus",
    "stimulus_file",
    "things_id",
    "uniqueID",
    "image_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export THINGS-fMRI table-format visual-ROI scalar4 features.")
    parser.add_argument(
        "--betas-csv-dir",
        type=Path,
        default=Path(
            "/gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/"
            "things_fmri/figshare_table/betas_csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/"
            "things_fmri/visual_roi_scalar4_smoke"
        ),
    )
    parser.add_argument("--subjects", nargs="+", default=["01", "02", "03"])
    parser.add_argument("--image-column", default="auto")
    parser.add_argument("--roi-columns", nargs="*", default=None)
    parser.add_argument("--max-shared-images", type=int, default=1000)
    parser.add_argument("--top-quantile", type=float, default=0.90)
    return parser.parse_args()


def subject_prefix(subject: str) -> str:
    subject = str(subject)
    return subject if subject.startswith("sub-") else f"sub-{subject}"


def infer_image_column(stim: pd.DataFrame) -> str:
    for col in IMAGE_COLUMN_CANDIDATES:
        if col in stim.columns:
            return col
    object_cols = [col for col in stim.columns if stim[col].dtype == object]
    if object_cols:
        return max(object_cols, key=lambda col: stim[col].astype(str).nunique())
    raise ValueError(f"Could not infer image column from columns: {list(stim.columns)}")


def is_binary_roi_column(series: pd.Series) -> bool:
    vals = pd.to_numeric(series, errors="coerce").dropna().unique()
    if len(vals) == 0 or len(vals) > 2:
        return False
    return set(vals).issubset({0, 1, 0.0, 1.0})


def infer_roi_columns(vox: pd.DataFrame, requested: list[str] | None) -> list[str]:
    if requested:
        missing = [col for col in requested if col not in vox.columns]
        if missing:
            raise ValueError(f"Requested ROI columns not found in VoxelMetadata: {missing}")
        return requested
    preferred = [col for col in PREFERRED_VISUAL_ROIS if col in vox.columns and is_binary_roi_column(vox[col])]
    if preferred:
        return preferred
    excluded = {
        "voxel_id",
        "x",
        "y",
        "z",
        "i",
        "j",
        "k",
        "nc_testset",
        "nc_singletrial",
        "splithalf_uncorrected",
        "splithalf_corrected",
        "prf-eccentricity",
        "prf-polarangle",
        "prf-size",
        "prf-rsquared",
    }
    return [col for col in vox.columns if col not in excluded and is_binary_roi_column(vox[col])]


def load_subject_metadata(betas_csv_dir: Path, subject: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix = subject_prefix(subject)
    stim = pd.read_csv(betas_csv_dir / f"{prefix}_StimulusMetadata.csv")
    vox = pd.read_csv(betas_csv_dir / f"{prefix}_VoxelMetadata.csv")
    return stim, vox


def response_hdf_path(betas_csv_dir: Path, subject: str) -> Path:
    prefix = subject_prefix(subject)
    return betas_csv_dir / f"{prefix}_ResponseData.h5"


def response_key_and_columns(hdf_path: Path) -> tuple[str, list[str]]:
    with pd.HDFStore(hdf_path, mode="r") as store:
        keys = store.keys()
        if not keys:
            raise ValueError(f"No HDF keys found in {hdf_path}")
        key = keys[0]
    empty = pd.read_hdf(hdf_path, key=key, start=0, stop=0)
    columns = [col for col in empty.columns if col != "voxel_id"]
    return key, columns


def normalize_trial_columns(response_columns: list[object], source_indices: np.ndarray) -> list[object]:
    available = set(response_columns)
    out: list[object] = []
    for idx in source_indices.tolist():
        if idx in available:
            out.append(idx)
        elif str(idx) in available:
            out.append(str(idx))
        else:
            raise KeyError(f"Trial column {idx} not found in ResponseData columns.")
    return out


def response_columns_for_roi(response_columns: list[str], vox: pd.DataFrame, roi: str) -> list[str]:
    mask = pd.to_numeric(vox[roi], errors="coerce").fillna(0).to_numpy() > 0
    if mask.sum() == 0:
        return []
    if "voxel_id" in vox.columns:
        ids = vox.loc[mask, "voxel_id"].astype(str).tolist()
        direct = [col for col in ids if col in response_columns]
        if direct:
            return direct
        prefixed = [f"voxel_{col}" for col in ids if f"voxel_{col}" in response_columns]
        if prefixed:
            return prefixed
    if len(response_columns) != len(vox):
        raise ValueError(f"Response column count {len(response_columns)} does not match voxel metadata rows {len(vox)}")
    return [response_columns[i] for i in np.flatnonzero(mask)]


def load_response_columns(hdf_path: Path, key: str, columns: list[str]) -> pd.DataFrame:
    try:
        return pd.read_hdf(hdf_path, key=key, columns=columns)
    except TypeError:
        responses = pd.read_hdf(hdf_path, key=key)
        if "voxel_id" in responses.columns:
            responses = responses.drop(columns="voxel_id")
    return responses[columns]


def response_row_indices_for_roi(vox: pd.DataFrame, roi: str) -> np.ndarray:
    mask = pd.to_numeric(vox[roi], errors="coerce").fillna(0).to_numpy() > 0
    return np.flatnonzero(mask)


def load_response_full(hdf_path: Path, key: str) -> pd.DataFrame:
    responses = pd.read_hdf(hdf_path, key=key)
    if "voxel_id" in responses.columns:
        responses = responses.drop(columns="voxel_id")
    return responses


def scalar4(matrix: np.ndarray, top_quantile: float) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], len(FEATURE_NAMES)), dtype=np.float32)
    out = np.zeros((matrix.shape[0], len(FEATURE_NAMES)), dtype=np.float32)
    out[:, 0] = matrix.mean(axis=1, dtype=np.float64).astype(np.float32)
    out[:, 1] = matrix.std(axis=1, dtype=np.float64).astype(np.float32)
    out[:, 2] = np.quantile(matrix, top_quantile, axis=1).astype(np.float32)
    out[:, 3] = (matrix > 0).mean(axis=1, dtype=np.float64).astype(np.float32)
    return out


def shared_single_images(stim_by_subject: dict[str, pd.DataFrame], image_column: str, limit: int) -> list[str]:
    sets = []
    for stim in stim_by_subject.values():
        labels = stim[image_column].astype(str)
        counts = labels.value_counts()
        sets.append(set(counts[counts == 1].index.astype(str)))
    shared = sorted(set.intersection(*sets))
    return shared[:limit] if limit > 0 else shared


def export_subject(
    subject: str,
    stim: pd.DataFrame,
    vox: pd.DataFrame,
    hdf_path: Path,
    shared_images: list[str],
    image_column: str,
    roi_columns: list[str],
    out_dir: Path,
    top_quantile: float,
) -> dict[str, object]:
    start = time.time()
    selected = stim[stim[image_column].astype(str).isin(shared_images)].copy()
    selected["_image_label"] = selected[image_column].astype(str)
    selected = selected.sort_values("_image_label").reset_index(drop=False).rename(columns={"index": "source_index"})
    source_indices = selected["source_index"].to_numpy(dtype=np.int64)
    x = np.zeros((len(selected), 180, len(FEATURE_NAMES)), dtype=np.float32)
    voxel_counts = np.zeros(180, dtype=np.int32)
    used_rois = []
    key, response_columns = response_key_and_columns(hdf_path)
    row_oriented = len(response_columns) == len(stim)
    if row_oriented:
        trial_cols = normalize_trial_columns(response_columns, source_indices)
        responses = load_response_full(hdf_path, key)
        for node_idx, roi in enumerate(roi_columns[:180]):
            row_idx = response_row_indices_for_roi(vox, roi)
            used_rois.append(roi)
            voxel_counts[node_idx] = len(row_idx)
            values = responses.iloc[row_idx][trial_cols].T.to_numpy(dtype=np.float32, copy=True)
            x[:, node_idx, :] = scalar4(values, top_quantile)
        n_response_values_loaded = int(responses.shape[0] * responses.shape[1])
    else:
        roi_to_cols = {roi: response_columns_for_roi(response_columns, vox, roi) for roi in roi_columns[:180]}
        selected_response_cols = sorted({col for cols in roi_to_cols.values() for col in cols})
        responses = load_response_columns(hdf_path, key, selected_response_cols)
        for node_idx, roi in enumerate(roi_columns[:180]):
            cols = roi_to_cols[roi]
            used_rois.append(roi)
            voxel_counts[node_idx] = len(cols)
            values = responses.iloc[source_indices][cols].to_numpy(dtype=np.float32, copy=True)
            x[:, node_idx, :] = scalar4(values, top_quantile)
        n_response_values_loaded = int(responses.shape[0] * responses.shape[1])

    node_labels = used_rois + [f"pad_{i:03d}" for i in range(len(used_rois) + 1, 181)]
    prefix = subject_prefix(subject)
    pt = {
        "x": torch.from_numpy(x),
        "subject": prefix,
        "image_label": selected["_image_label"].tolist(),
        "repetition": torch.ones(len(selected), dtype=torch.int16),
        "feature_names": FEATURE_NAMES,
        "node_set_name": "THINGS-fMRI table-format visual ROI tokens padded to 180 nodes",
        "node_labels": node_labels,
        "source_roi_labels": used_rois,
        "voxel_counts": torch.from_numpy(voxel_counts),
        "source_indices": torch.from_numpy(source_indices.astype(np.int32)),
        "image_column": image_column,
    }
    out_path = out_dir / f"{prefix}_things_fmri_visual_roi_scalar4.pt"
    torch.save(pt, out_path)
    selected.to_csv(out_dir / f"{prefix}_things_fmri_visual_roi_scalar4_metadata.csv", index=False)
    qc = {
        "subject": prefix,
        "n_trials": int(len(selected)),
        "n_images": int(selected["_image_label"].nunique()),
        "feature_shape": list(x.shape),
        "nan_count": int(np.isnan(x).sum()),
        "inf_count": int(np.isinf(x).sum()),
        "image_column": image_column,
        "hdf_key": key,
        "row_oriented_hdf": bool(row_oriented),
        "n_response_values_loaded": n_response_values_loaded,
        "source_roi_labels": used_rois,
        "voxel_counts": {roi: int(voxel_counts[i]) for i, roi in enumerate(used_rois)},
        "output_path": str(out_path),
        "elapsed_seconds": float(time.time() - start),
    }
    (out_dir / f"{prefix}_things_fmri_visual_roi_scalar4_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc), flush=True)
    return qc


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {subject: load_subject_metadata(args.betas_csv_dir, subject) for subject in args.subjects}
    image_column = args.image_column
    if image_column == "auto":
        inferred = [infer_image_column(stim) for stim, _ in metadata.values()]
        if len(set(inferred)) != 1:
            raise ValueError(f"Image-column inference differed by subject: {dict(zip(args.subjects, inferred))}")
        image_column = inferred[0]
    roi_columns = infer_roi_columns(next(iter(metadata.values()))[1], args.roi_columns)
    if not roi_columns:
        raise ValueError("No ROI columns found in VoxelMetadata.")
    shared_images = shared_single_images({s: v[0] for s, v in metadata.items()}, image_column, args.max_shared_images)
    if not shared_images:
        raise RuntimeError("No shared single-presentation images found across requested THINGS-fMRI subjects.")

    qcs = []
    for subject, (stim, vox) in metadata.items():
        qcs.append(export_subject(subject, stim, vox, response_hdf_path(args.betas_csv_dir, subject), shared_images, image_column, roi_columns, args.out_dir, args.top_quantile))
    manifest = {
        "subjects": [subject_prefix(s) for s in args.subjects],
        "image_column": image_column,
        "roi_columns": roi_columns,
        "n_shared_single_images_used": len(shared_images),
        "max_shared_images": args.max_shared_images,
        "shared_image_labels": shared_images,
        "qcs": qcs,
    }
    (args.out_dir / "things_fmri_visual_roi_scalar4_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.out_dir)


if __name__ == "__main__":
    main()
