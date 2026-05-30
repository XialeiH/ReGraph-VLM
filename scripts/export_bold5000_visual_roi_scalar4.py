#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch


FEATURE_NAMES = ["mean_beta", "std_beta", "q90_beta", "positive_fraction"]
CANONICAL_ROIS = [
    "LHEarlyVis",
    "LHLOC",
    "LHOPA",
    "LHPPA",
    "LHRSC",
    "RHEarlyVis",
    "RHLOC",
    "RHOPA",
    "RHPPA",
    "RHRSC",
]
ROI_ALIASES = {
    "LHLOC": ["LHLOC", "LHLO"],
    "RHLOC": ["RHLOC", "RHLO"],
    "RHRSC": ["RHRSC", "RHRRSC"],
}
ZIP_RE = re.compile(r"(CSI\d)_.*_allses_(.*)\.npy$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export BOLD5000 public visual-ROI scalar4 features.")
    parser.add_argument(
        "--openneuro-root",
        type=Path,
        default=Path("/gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/bold5000/openneuro_ds001499"),
    )
    parser.add_argument(
        "--roi-zip",
        type=Path,
        default=Path(
            "/gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/bold5000/"
            "release2_smoke/BOLD5000_GLMsingle_ROI_betas.zip"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/bold5000/visual_roi_scalar4_smoke"),
    )
    parser.add_argument("--subjects", nargs="+", default=["CSI1", "CSI2", "CSI3"])
    parser.add_argument("--max-shared-images", type=int, default=1000)
    parser.add_argument("--top-quantile", type=float, default=0.90)
    return parser.parse_args()


def sorted_event_files(root: Path, subject: str) -> list[Path]:
    files = list(root.glob(f"sub-{subject}/ses-*/func/sub-{subject}_ses-*_task-5000scenes_run-*_events.tsv"))

    def key(path: Path) -> tuple[int, int]:
        match = re.search(r"_ses-(\d+)_task-5000scenes_run-(\d+)_events", path.name)
        if not match:
            raise ValueError(f"Cannot parse session/run from {path}")
        return int(match.group(1)), int(match.group(2))

    return sorted(files, key=key)


def load_events(root: Path, subject: str) -> pd.DataFrame:
    rows = []
    for path in sorted_event_files(root, subject):
        rows.append(pd.read_csv(path, sep="\t"))
    if not rows:
        raise FileNotFoundError(f"No BOLD5000 event files found for {subject}")
    events = pd.concat(rows, ignore_index=True)
    events["image_label"] = events["stim_file"].astype(str)
    return events


def zip_index(roi_zip: Path) -> dict[tuple[str, str], str]:
    out = {}
    with zipfile.ZipFile(roi_zip) as zf:
        for name in zf.namelist():
            if not name.startswith("BOLD5000_GLMsingle_ROI_betas/py/") or not name.endswith(".npy"):
                continue
            match = ZIP_RE.match(Path(name).name)
            if match:
                out[(match.group(1), match.group(2))] = name
    return out


def resolve_roi_name(index: dict[tuple[str, str], str], subject: str, canonical_roi: str) -> str:
    for roi in ROI_ALIASES.get(canonical_roi, [canonical_roi]):
        if (subject, roi) in index:
            return roi
    raise FileNotFoundError(f"No ROI array found for {subject} {canonical_roi}")


def load_roi_array(roi_zip: Path, member: str) -> np.ndarray:
    with zipfile.ZipFile(roi_zip) as zf:
        return np.load(io.BytesIO(zf.read(member))).astype(np.float32, copy=False)


def scalar4(matrix: np.ndarray, top_quantile: float) -> np.ndarray:
    out = np.zeros((matrix.shape[0], len(FEATURE_NAMES)), dtype=np.float32)
    out[:, 0] = matrix.mean(axis=1, dtype=np.float64).astype(np.float32)
    out[:, 1] = matrix.std(axis=1, dtype=np.float64).astype(np.float32)
    out[:, 2] = np.quantile(matrix, top_quantile, axis=1).astype(np.float32)
    out[:, 3] = (matrix > 0).mean(axis=1, dtype=np.float64).astype(np.float32)
    return out


def shared_single_images(events_by_subject: dict[str, pd.DataFrame], limit: int) -> list[str]:
    single_sets = []
    for events in events_by_subject.values():
        counts = events["image_label"].value_counts()
        single_sets.append(set(counts[counts == 1].index.astype(str)))
    shared = sorted(set.intersection(*single_sets))
    return shared[:limit] if limit > 0 else shared


def export_subject(
    subject: str,
    events: pd.DataFrame,
    shared_images: list[str],
    roi_zip: Path,
    index: dict[tuple[str, str], str],
    out_dir: Path,
    top_quantile: float,
) -> dict[str, object]:
    start = time.time()
    selected = events[events["image_label"].isin(shared_images)].copy()
    selected = selected.sort_values("image_label").reset_index(drop=False).rename(columns={"index": "source_index"})
    x = np.zeros((len(selected), 180, len(FEATURE_NAMES)), dtype=np.float32)
    voxel_counts = np.zeros(180, dtype=np.int32)
    source_indices = selected["source_index"].to_numpy(dtype=np.int64)

    resolved_rois = []
    for node_idx, roi in enumerate(CANONICAL_ROIS):
        source_roi = resolve_roi_name(index, subject, roi)
        resolved_rois.append(source_roi)
        arr = load_roi_array(roi_zip, index[(subject, source_roi)])
        if arr.shape[0] != len(events):
            raise ValueError(f"Length mismatch for {subject} {source_roi}: {arr.shape[0]} vs {len(events)}")
        x[:, node_idx, :] = scalar4(arr[source_indices], top_quantile)
        voxel_counts[node_idx] = int(arr.shape[1])

    pt = {
        "x": torch.from_numpy(x),
        "subject": subject,
        "image_label": selected["image_label"].astype(str).tolist(),
        "repetition": torch.ones(len(selected), dtype=torch.int16),
        "feature_names": FEATURE_NAMES,
        "node_set_name": "BOLD5000 public visual ROI tokens padded to 180 nodes",
        "node_labels": CANONICAL_ROIS + [f"pad_{i:03d}" for i in range(len(CANONICAL_ROIS) + 1, 181)],
        "source_roi_labels": resolved_rois,
        "voxel_counts": torch.from_numpy(voxel_counts),
        "source_indices": torch.from_numpy(source_indices.astype(np.int32)),
    }
    out_path = out_dir / f"{subject}_bold5000_visual_roi_scalar4.pt"
    torch.save(pt, out_path)
    selected[["image_label", "ImgType", "Sess", "Run", "Trial", "source_index"]].to_csv(
        out_dir / f"{subject}_bold5000_visual_roi_scalar4_metadata.csv", index=False
    )
    qc = {
        "subject": subject,
        "n_trials": int(len(selected)),
        "n_images": int(selected["image_label"].nunique()),
        "feature_shape": list(x.shape),
        "nan_count": int(np.isnan(x).sum()),
        "inf_count": int(np.isinf(x).sum()),
        "canonical_rois": CANONICAL_ROIS,
        "source_roi_labels": resolved_rois,
        "voxel_counts": {roi: int(voxel_counts[i]) for i, roi in enumerate(CANONICAL_ROIS)},
        "output_path": str(out_path),
        "elapsed_seconds": float(time.time() - start),
    }
    (out_dir / f"{subject}_bold5000_visual_roi_scalar4_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc), flush=True)
    return qc


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    events_by_subject = {subject: load_events(args.openneuro_root, subject) for subject in args.subjects}
    shared_images = shared_single_images(events_by_subject, args.max_shared_images)
    if not shared_images:
        raise RuntimeError("No shared single-presentation images found across requested BOLD5000 subjects.")

    index = zip_index(args.roi_zip)
    qcs = [
        export_subject(subject, events_by_subject[subject], shared_images, args.roi_zip, index, args.out_dir, args.top_quantile)
        for subject in args.subjects
    ]
    manifest = {
        "subjects": args.subjects,
        "canonical_rois": CANONICAL_ROIS,
        "n_shared_single_images_used": len(shared_images),
        "max_shared_images": args.max_shared_images,
        "shared_image_labels": shared_images,
        "qcs": qcs,
    }
    (args.out_dir / "bold5000_visual_roi_scalar4_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.out_dir)


if __name__ == "__main__":
    main()
