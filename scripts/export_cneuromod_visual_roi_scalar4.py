#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


FEATURE_NAMES = ["mean_beta", "std_beta", "q90_beta", "positive_fraction"]
ROI_RE = re.compile(r"_roi-([^_]+)_")
SUB_RE = re.compile(r"sub-(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export public CNeuroMod visual-ROI scalar4 features.")
    parser.add_argument(
        "--glmsingle-root",
        type=Path,
        default=Path(
            "/gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/"
            "cneuromod_things/metadata_repo/THINGS/glmsingle"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/cneuromod_things/visual_roi_scalar4_smoke"),
    )
    parser.add_argument("--subjects", nargs="+", default=["sub-01", "sub-02"])
    parser.add_argument("--max-shared-images", type=int, default=500)
    parser.add_argument("--top-quantile", type=float, default=0.90)
    return parser.parse_args()


def roi_name(path: Path) -> str:
    match = ROI_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse ROI name from {path}")
    return match.group(1)


def subject_int(subject: str) -> int:
    match = SUB_RE.match(subject)
    if not match:
        raise ValueError(f"Invalid subject label: {subject}")
    return int(match.group(1))


def load_subject_arrays(root: Path, subject: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    desc = root / subject / "descriptive"
    labels_path = desc / f"{subject}_task-things_desc-perTrial_labels.npy"
    labels = np.load(labels_path, allow_pickle=True).astype(str)
    arrays = {}
    for path in sorted(desc.glob("*_contrast-*_stat-betas_desc-perTrial_statseries.npy")):
        arr = np.load(path, mmap_mode="r")
        if arr.shape[0] != len(labels):
            raise ValueError(f"Length mismatch for {path}: {arr.shape[0]} vs labels {len(labels)}")
        arrays[roi_name(path)] = arr
    if not arrays:
        raise FileNotFoundError(f"No visual ROI beta arrays found for {subject} under {desc}")
    return labels, arrays


def shared_strict_t3_labels(labels_by_subject: dict[str, np.ndarray], limit: int) -> list[str]:
    strict_sets = []
    for labels in labels_by_subject.values():
        counts = pd.Series(labels).value_counts()
        strict_sets.append(set(counts[counts == 3].index.astype(str)))
    shared = sorted(set.intersection(*strict_sets))
    return shared[:limit] if limit > 0 else shared


def scalar4(matrix: np.ndarray, top_quantile: float) -> np.ndarray:
    out = np.zeros((matrix.shape[0], len(FEATURE_NAMES)), dtype=np.float32)
    out[:, 0] = matrix.mean(axis=1, dtype=np.float64).astype(np.float32)
    out[:, 1] = matrix.std(axis=1, dtype=np.float64).astype(np.float32)
    out[:, 2] = np.quantile(matrix, top_quantile, axis=1).astype(np.float32)
    out[:, 3] = (matrix > 0).mean(axis=1, dtype=np.float64).astype(np.float32)
    return out


def export_subject(
    subject: str,
    labels: np.ndarray,
    arrays: dict[str, np.ndarray],
    roi_order: list[str],
    shared_labels: list[str],
    out_dir: Path,
    top_quantile: float,
) -> dict[str, object]:
    start = time.time()
    keep = np.isin(labels, np.asarray(shared_labels))
    kept_labels = labels[keep]
    order = np.lexsort((np.arange(len(kept_labels)), kept_labels))
    kept_labels = kept_labels[order]
    source_indices = np.flatnonzero(keep)[order]

    x = np.zeros((len(kept_labels), 180, len(FEATURE_NAMES)), dtype=np.float32)
    voxel_counts = np.zeros(180, dtype=np.int32)
    for node_idx, roi in enumerate(roi_order):
        roi_matrix = np.asarray(arrays[roi][source_indices], dtype=np.float32)
        x[:, node_idx, :] = scalar4(roi_matrix, top_quantile)
        voxel_counts[node_idx] = int(roi_matrix.shape[1])

    occurrence = pd.Series(kept_labels).groupby(kept_labels).cumcount().to_numpy(dtype=np.int16) + 1
    pt = {
        "x": torch.from_numpy(x),
        "subject": torch.full((len(kept_labels),), subject_int(subject), dtype=torch.int16),
        "image_label": kept_labels.tolist(),
        "repetition": torch.from_numpy(occurrence),
        "feature_names": FEATURE_NAMES,
        "node_set_name": "CNeuroMod public visual fLoc ROI tokens padded to 180 nodes",
        "node_labels": roi_order + [f"pad_{i:03d}" for i in range(len(roi_order) + 1, 181)],
        "voxel_counts": torch.from_numpy(voxel_counts),
        "source_indices": torch.from_numpy(source_indices.astype(np.int32)),
    }
    out_path = out_dir / f"{subject}_cneuromod_visual_roi_scalar4.pt"
    torch.save(pt, out_path)

    pd.DataFrame(
        {
            "subject": subject,
            "image_label": kept_labels,
            "repetition": occurrence,
            "source_index": source_indices,
        }
    ).to_csv(out_dir / f"{subject}_cneuromod_visual_roi_scalar4_metadata.csv", index=False)

    qc = {
        "subject": subject,
        "n_trials": int(len(kept_labels)),
        "n_images": int(pd.Series(kept_labels).nunique()),
        "roi_order": roi_order,
        "feature_shape": list(x.shape),
        "nan_count": int(np.isnan(x).sum()),
        "inf_count": int(np.isinf(x).sum()),
        "voxel_counts": {roi: int(voxel_counts[i]) for i, roi in enumerate(roi_order)},
        "output_path": str(out_path),
        "elapsed_seconds": float(time.time() - start),
    }
    (out_dir / f"{subject}_cneuromod_visual_roi_scalar4_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc), flush=True)
    return qc


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels_by_subject = {}
    arrays_by_subject = {}
    for subject in args.subjects:
        labels, arrays = load_subject_arrays(args.glmsingle_root, subject)
        labels_by_subject[subject] = labels
        arrays_by_subject[subject] = arrays

    common_rois = sorted(set.intersection(*(set(arrays) for arrays in arrays_by_subject.values())))
    shared_labels = shared_strict_t3_labels(labels_by_subject, args.max_shared_images)
    if not common_rois:
        raise RuntimeError("No common public visual ROI arrays across requested subjects.")
    if not shared_labels:
        raise RuntimeError("No shared image labels with exactly three trials across requested subjects.")

    qcs = []
    for subject in args.subjects:
        qcs.append(
            export_subject(
                subject=subject,
                labels=labels_by_subject[subject],
                arrays=arrays_by_subject[subject],
                roi_order=common_rois,
                shared_labels=shared_labels,
                out_dir=args.out_dir,
                top_quantile=args.top_quantile,
            )
        )

    manifest = {
        "subjects": args.subjects,
        "common_rois": common_rois,
        "n_shared_strict_t3_images_used": len(shared_labels),
        "max_shared_images": args.max_shared_images,
        "shared_image_labels": shared_labels,
        "qcs": qcs,
    }
    (args.out_dir / "cneuromod_visual_roi_scalar4_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.out_dir)


if __name__ == "__main__":
    main()
