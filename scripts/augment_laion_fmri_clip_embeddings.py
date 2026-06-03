#!/usr/bin/env python3
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from external_data_policy import enforce_hpc_external_path


BASE_URL = "https://laion-fmri.s3.amazonaws.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach public LAION-fMRI CLIP embeddings to exported ROI tensors.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--stimulus-dir", type=Path, default=Path("external_validation/laion_fmri/stimuli"))
    parser.add_argument("--subjects", nargs="+", default=["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"])
    parser.add_argument("--file-template", default="{subject}_laion_visual_roi_scalar4.pt")
    return parser.parse_args()


def download(key: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    urllib.request.urlretrieve(f"{BASE_URL}/{key}", tmp)
    tmp.replace(path)


def decode_id(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_clip_map(stimulus_dir: Path) -> tuple[dict[str, np.ndarray], int]:
    metadata_path = stimulus_dir / "task-images_metadata.csv"
    embedding_path = stimulus_dir / "task-images_desc-CLIP_embeddings.h5"
    download("stimuli/task-images_metadata.csv", metadata_path)
    download("stimuli/task-images_desc-CLIP_embeddings.h5", embedding_path)
    metadata = pd.read_csv(metadata_path)
    with h5py.File(embedding_path, "r") as handle:
        embeddings = np.asarray(handle["embedding"], dtype=np.float32)
        image_ids = [decode_id(value) for value in handle["image_ids"][:]]
        valid = np.asarray(handle["valid"], dtype=bool) if "valid" in handle else np.ones(len(image_ids), dtype=bool)
    if "image_name" not in metadata.columns:
        raise ValueError(f"Missing image_name column in {metadata_path}")
    names = metadata["image_name"].astype(str).tolist()
    if len(names) != embeddings.shape[0] or names != image_ids:
        # Fall back to the H5 ids if the CSV order ever changes.
        names = image_ids
    clip = {
        name: embeddings[idx]
        for idx, name in enumerate(names)
        if bool(valid[idx])
    }
    return clip, int(embeddings.shape[1])


def main() -> None:
    args = parse_args()
    enforce_hpc_external_path(args.data_dir, "LAION-fMRI ROI tensor directory")
    enforce_hpc_external_path(args.stimulus_dir, "LAION-fMRI stimulus embedding directory")
    clip_map, clip_dim = load_clip_map(args.stimulus_dir)
    rows = []
    for subject in args.subjects:
        path = args.data_dir / args.file_template.format(subject=subject)
        payload = torch.load(path, map_location="cpu")
        labels = [str(label) for label in payload["image_label"]]
        missing = sorted({label for label in labels if label not in clip_map})
        if missing:
            raise KeyError(f"{subject}: {len(missing)} labels missing CLIP embeddings, e.g. {missing[:5]}")
        clip = np.stack([clip_map[label] for label in labels], axis=0).astype(np.float32)
        payload["clip"] = torch.from_numpy(clip)
        payload["clip_dim"] = clip_dim
        payload["clip_source"] = "LAION-fMRI public stimuli/task-images_desc-CLIP_embeddings.h5"
        torch.save(payload, path)
        rows.append({"subject": subject, "n_trials": len(labels), "clip_dim": clip_dim, "missing": 0})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(args.data_dir / "laion_clip_augmentation_qc.csv", index=False)


if __name__ == "__main__":
    main()
