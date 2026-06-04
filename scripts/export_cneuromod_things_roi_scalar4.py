#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nilearn.image import resample_to_img


FEATURE_NAMES = ["mean_beta", "std_beta", "q90_beta", "positive_fraction"]
SUB_RE = re.compile(r"sub-(\d+)")
SES_RE = re.compile(r"ses-(\d+)")
RUN_RE = re.compile(r"run-(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CNeuroMod-THINGS HCP-MMP ROI scalar4 features.")
    parser.add_argument(
        "--things-root",
        type=Path,
        default=Path("/gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/cneuromod_things/metadata_repo/THINGS"),
    )
    parser.add_argument(
        "--conp-base-url",
        default="https://sftp.conp.ca/users/cneuromod/cneuromod.all",
        help="Public CONP HTTP mirror used when local GLMsingle/smriprep files are absent.",
    )
    parser.add_argument(
        "--conp-host-header",
        default=None,
        help="Optional Host header for compute nodes that can reach CONP by IP but cannot resolve DNS.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("/gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/cneuromod_things/conp_public_downloads"),
    )
    parser.add_argument(
        "--atlas",
        type=Path,
        default=Path(
            "/gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/atlases/"
            "hcp_mmp1_fsl/MNI_Glasser_HCP_v1.0.nii.gz"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/cneuromod_things/roi_scalar4_smoke"),
    )
    parser.add_argument("--subjects", nargs="+", default=["sub-01", "sub-02"])
    parser.add_argument("--space", default="MNI152NLin2009cAsym")
    parser.add_argument(
        "--atlas-to-beta-space",
        choices=["none", "mni2009c_to_t1w"],
        default="none",
        help="Use mni2009c_to_t1w for public CNeuroMod GLMsingle T1w trial betas.",
    )
    parser.add_argument("--max-shared-images", type=int, default=200)
    parser.add_argument("--top-quantile", type=float, default=0.90)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def path_int(pattern: re.Pattern[str], path: Path) -> int:
    match = pattern.search(str(path))
    if not match:
        raise ValueError(f"Could not parse {pattern.pattern} from {path}")
    return int(match.group(1))


def false_mask(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.lower()
    return ~(values.isin({"true", "1", "1.0", "yes"}))


def load_subject_events(events_root: Path, subject: str) -> pd.DataFrame:
    rows = []
    for path in sorted((events_root / subject).glob("ses-*/func/*_events.tsv")):
        df = pd.read_csv(path, sep="\t")
        df["source_file"] = str(path)
        df["subject"] = subject
        df["session_num"] = path_int(SES_RE, path)
        df["run_num"] = path_int(RUN_RE, path)
        df["trial_in_run"] = np.arange(len(df), dtype=np.int32)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No event files found for {subject} under {events_root}")
    df = pd.concat(rows, ignore_index=True)
    if "ran" in df.columns:
        df = df[df["ran"].fillna(0).astype(float) == 1.0]
    if "not_for_memory" in df.columns:
        df = df[false_mask(df["not_for_memory"])]
    df = df[df["things_image_nr"].notna()].copy()
    df["image_key"] = df["things_image_nr"].astype("Int64").astype(str)
    df["repetition"] = df["repetition"].astype("Int64")
    return df.reset_index(drop=True)


def strict_t3_images(events_by_subject: dict[str, pd.DataFrame], limit: int) -> list[str]:
    strict_sets = []
    for df in events_by_subject.values():
        by_image = df.groupby("image_key")
        counts = by_image.size()
        reps = by_image["repetition"].nunique(dropna=True)
        strict_sets.append(set(counts[(counts == 3) & (reps == 3)].index))
    shared = sorted(set.intersection(*strict_sets), key=lambda x: int(x))
    return shared[:limit] if limit > 0 else shared


def h5_path(things_root: Path, subject: str, space: str) -> Path:
    return (
        things_root
        / "glmsingle"
        / subject
        / "glmsingle"
        / "output"
        / f"{subject}_task-things_space-{space}_model-fitHrfGLMdenoiseRR_stat-trialBetas_desc-zscore_statseries.h5"
    )


def conp_h5_url(base_url: str, subject: str, space: str) -> str:
    return (
        f"{base_url}/things/glmsingle/{subject}/glmsingle/output/"
        f"{subject}_task-things_space-{space}_model-fitHrfGLMdenoiseRR_stat-trialBetas_desc-zscore_statseries.h5"
    )


def conp_transform_url(base_url: str, subject: str) -> str:
    return (
        f"{base_url}/anat/smriprep/{subject}/anat/"
        f"{subject}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5"
    )


def download_with_wget(url: str, out_path: Path, host_header: str | None = None) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    cmd = ["wget", "-c", "-O", str(out_path)]
    if host_header is not None:
        cmd.extend(["--no-check-certificate", "--header", f"Host: {host_header}"])
    cmd.append(url)
    print(f"[download] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    return out_path


def local_or_conp_h5(args: argparse.Namespace, subject: str) -> Path:
    local = h5_path(args.things_root, subject, args.space)
    if local.exists():
        return local
    out = (
        args.download_dir
        / "things"
        / "glmsingle"
        / subject
        / "glmsingle"
        / "output"
        / f"{subject}_task-things_space-{args.space}_model-fitHrfGLMdenoiseRR_stat-trialBetas_desc-zscore_statseries.h5"
    )
    return download_with_wget(conp_h5_url(args.conp_base_url, subject, args.space), out, args.conp_host_header)


def local_or_conp_transform(args: argparse.Namespace, subject: str) -> Path:
    local = (
        args.things_root.parent
        / "anat"
        / "smriprep"
        / subject
        / "anat"
        / f"{subject}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5"
    )
    if local.exists():
        return local
    out = (
        args.download_dir
        / "anat"
        / "smriprep"
        / subject
        / "anat"
        / f"{subject}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5"
    )
    return download_with_wget(conp_transform_url(args.conp_base_url, subject), out, args.conp_host_header)


def bilateral_label_values(atlas_data: np.ndarray) -> np.ndarray:
    labels = np.rint(atlas_data).astype(np.int32)
    out = np.zeros_like(labels, dtype=np.int16)
    left = (labels >= 1) & (labels <= 180)
    right = (labels >= 1001) & (labels <= 1180)
    out[left] = labels[left].astype(np.int16)
    out[right] = (labels[right] - 1000).astype(np.int16)
    return out


def mask_img_from_h5(h5: h5py.File) -> nib.Nifti1Image:
    mask_array = np.asarray(h5["mask_array"]).astype(bool)
    mask_affine = np.asarray(h5["mask_affine"])
    return nib.Nifti1Image(mask_array.astype(np.uint8), affine=mask_affine)


def warp_atlas_to_mask(
    atlas_path: Path,
    mask_img: nib.Nifti1Image,
    transform_path: Path,
    cache_path: Path,
) -> nib.Nifti1Image:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return nib.load(str(cache_path))
    ref_path = cache_path.with_name(cache_path.name.replace(".nii.gz", "_reference.nii.gz"))
    nib.save(mask_img, str(ref_path))
    import ants

    fixed = ants.image_read(str(ref_path))
    moving = ants.image_read(str(atlas_path))
    warped = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=[str(transform_path)],
        interpolator="genericLabel",
    )
    ants.image_write(warped, str(cache_path))
    return nib.load(str(cache_path))


def roi_indices_for_h5(
    h5: h5py.File,
    atlas_path: Path,
    atlas_to_beta_space: str,
    transform_path: Path | None,
    warped_atlas_cache: Path | None,
) -> tuple[list[np.ndarray], list[int]]:
    mask_img = mask_img_from_h5(h5)
    if atlas_to_beta_space == "mni2009c_to_t1w":
        if transform_path is None or warped_atlas_cache is None:
            raise ValueError("mni2009c_to_t1w requires a transform path and warped atlas cache path.")
        atlas_resampled = warp_atlas_to_mask(atlas_path, mask_img, transform_path, warped_atlas_cache)
    else:
        atlas_img = nib.load(str(atlas_path))
        atlas_resampled = resample_to_img(
            atlas_img,
            mask_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
    labels = bilateral_label_values(np.asanyarray(atlas_resampled.dataobj))
    mask_array = np.asarray(h5["mask_array"]).astype(bool)
    flat_labels = labels[mask_array]
    indices = [np.flatnonzero(flat_labels == roi_id).astype(np.int64) for roi_id in range(1, 181)]
    return indices, [int(idx.size) for idx in indices]


def scalar4(values: np.ndarray, roi_indices: list[np.ndarray], top_quantile: float) -> np.ndarray:
    out = np.zeros((len(roi_indices), len(FEATURE_NAMES)), dtype=np.float32)
    for roi_idx, voxel_idx in enumerate(roi_indices):
        if voxel_idx.size == 0:
            continue
        roi_values = values[voxel_idx]
        out[roi_idx, 0] = float(roi_values.mean(dtype=np.float64))
        out[roi_idx, 1] = float(roi_values.std(dtype=np.float64))
        out[roi_idx, 2] = float(np.quantile(roi_values, top_quantile))
        out[roi_idx, 3] = float((roi_values > 0).mean(dtype=np.float64))
    return out


def h5_betas(h5: h5py.File, session: int, run: int) -> h5py.Dataset:
    return h5[str(session)][str(run)]["betas"]


def export_subject(
    subject: str,
    df: pd.DataFrame,
    h5_file: Path,
    atlas: Path,
    atlas_to_beta_space: str,
    transform_path: Path | None,
    out_dir: Path,
    top_quantile: float,
    progress_every: int,
) -> dict[str, object]:
    start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not h5_file.exists():
        raise FileNotFoundError(f"Missing HDF5 content for {subject}: {h5_file}")

    with h5py.File(h5_file, "r") as h5:
        warped_cache = out_dir / "warped_atlases" / f"{subject}_hcp_mmp180_space-{atlas_to_beta_space}_in_h5mask.nii.gz"
        roi_indices, voxel_counts = roi_indices_for_h5(
            h5,
            atlas,
            atlas_to_beta_space=atlas_to_beta_space,
            transform_path=transform_path,
            warped_atlas_cache=warped_cache,
        )
        features = np.zeros((len(df), 180, len(FEATURE_NAMES)), dtype=np.float32)
        for i, row in enumerate(df.itertuples(index=False), start=0):
            betas = h5_betas(h5, int(row.session_num), int(row.run_num))
            values = np.asarray(betas[int(row.trial_in_run)], dtype=np.float32)
            features[i] = scalar4(values, roi_indices, top_quantile)
            done = i + 1
            if done == 1 or done == len(df) or done % progress_every == 0:
                elapsed = max(time.time() - start, 1e-6)
                print(
                    f"[progress] {subject} {done}/{len(df)} "
                    f"elapsed={elapsed / 60:.1f}m rate={done / elapsed:.2f}/s",
                    flush=True,
                )

    subject_int = int(SUB_RE.search(subject).group(1))
    pt = {
        "x": torch.from_numpy(features),
        "subject": torch.full((len(df),), subject_int, dtype=torch.int16),
        "image_id": torch.from_numpy(df["things_image_nr"].astype(np.int32).to_numpy()),
        "image_label": df["image_key"].astype(str).to_numpy(),
        "repetition": torch.from_numpy(df["repetition"].astype(np.int16).to_numpy()),
        "session": torch.from_numpy(df["session_num"].astype(np.int16).to_numpy()),
        "run": torch.from_numpy(df["run_num"].astype(np.int16).to_numpy()),
        "trial_in_run": torch.from_numpy(df["trial_in_run"].astype(np.int16).to_numpy()),
        "feature_names": FEATURE_NAMES,
        "node_set_name": "HCP-MMP1 volumetric MNI, bilateral 180-token",
        "node_labels": list(range(1, 181)),
        "voxel_counts": torch.tensor(voxel_counts, dtype=torch.int32),
        "source_h5": str(h5_file),
        "source_atlas": str(atlas),
        "atlas_to_beta_space": atlas_to_beta_space,
        "source_transform": str(transform_path) if transform_path is not None else None,
    }
    out_path = out_dir / f"{subject}_cneuromod_things_trial_scalar4.pt"
    torch.save(pt, out_path)

    meta_cols = [
        "subject",
        "things_image_nr",
        "image_path",
        "repetition",
        "session_num",
        "run_num",
        "trial_in_run",
        "source_file",
    ]
    df[meta_cols].to_csv(out_dir / f"{subject}_cneuromod_things_trial_scalar4_metadata.csv", index=False)
    qc = {
        "subject": subject,
        "n_trials": int(len(df)),
        "n_images": int(df["image_key"].nunique()),
        "feature_shape": list(features.shape),
        "min_roi_voxels": int(min(voxel_counts)),
        "max_roi_voxels": int(max(voxel_counts)),
        "zero_voxel_rois": int(sum(v == 0 for v in voxel_counts)),
        "nan_count": int(np.isnan(features).sum()),
        "inf_count": int(np.isinf(features).sum()),
        "output_path": str(out_path),
        "atlas_to_beta_space": atlas_to_beta_space,
        "source_transform": str(transform_path) if transform_path is not None else None,
        "elapsed_seconds": float(time.time() - start),
    }
    (out_dir / f"{subject}_cneuromod_things_trial_scalar4_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc), flush=True)
    return qc


def main() -> None:
    args = parse_args()
    events_root = args.things_root / "fmriprep" / "sourcedata" / "things"
    events_by_subject = {subject: load_subject_events(events_root, subject) for subject in args.subjects}
    shared_images = strict_t3_images(events_by_subject, args.max_shared_images)
    if not shared_images:
        raise RuntimeError("No shared strict-T3 images found for requested subjects.")

    qcs = []
    for subject, df in events_by_subject.items():
        sub = df[df["image_key"].isin(shared_images)].copy()
        sub = sub.sort_values(["image_key", "repetition", "session_num", "run_num", "trial_in_run"]).reset_index(drop=True)
        h5_file = local_or_conp_h5(args, subject)
        transform_path = local_or_conp_transform(args, subject) if args.atlas_to_beta_space == "mni2009c_to_t1w" else None
        qcs.append(
            export_subject(
                subject=subject,
                df=sub,
                h5_file=h5_file,
                atlas=args.atlas,
                atlas_to_beta_space=args.atlas_to_beta_space,
                transform_path=transform_path,
                out_dir=args.out_dir,
                top_quantile=args.top_quantile,
                progress_every=args.progress_every,
            )
        )

    manifest = {
        "subjects": args.subjects,
        "space": args.space,
        "atlas_to_beta_space": args.atlas_to_beta_space,
        "n_shared_strict_t3_images_used": len(shared_images),
        "max_shared_images": args.max_shared_images,
        "shared_image_ids": shared_images,
        "qcs": qcs,
    }
    (args.out_dir / "cneuromod_things_roi_scalar4_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.out_dir)


if __name__ == "__main__":
    main()
