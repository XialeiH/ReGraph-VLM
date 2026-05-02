#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import OrderedDict
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np


FEATURE_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "q05",
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
    "q95",
    "positive_fraction",
    "negative_fraction",
    "abs_mean",
    "rms",
    "top10_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export cached atlas ROI 16-stat node features for one NSD subject.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--node-set", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-open-sessions", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class SessionCache:
    def __init__(self, beta_dir: Path, max_open: int):
        self.beta_dir = beta_dir
        self.max_open = max_open
        self.handles: OrderedDict[int, h5py.File] = OrderedDict()

    def get(self, session: int) -> h5py.File:
        if session in self.handles:
            handle = self.handles.pop(session)
            self.handles[session] = handle
            return handle
        handle = h5py.File(
            self.beta_dir / f"betas_session{session:02d}.hdf5",
            "r",
            rdcc_nbytes=8 * 1024 * 1024,
            rdcc_nslots=1009,
        )
        self.handles[session] = handle
        while len(self.handles) > self.max_open:
            _, old = self.handles.popitem(last=False)
            old.close()
        return handle

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def load_label_indices(root: Path, subject: str, node_set: dict[str, object]) -> tuple[list[np.ndarray], list[int], tuple[int, ...]]:
    roi_family = str(node_set["roi_family"])
    mask_path = root / f"data/nsddata/ppdata/{subject}/func1pt8mm/roi/{roi_family}.nii.gz"
    arr = np.asanyarray(nib.load(str(mask_path)).dataobj).astype(np.int32)
    beta_probe = (
        root
        / f"data/nsddata_betas/ppdata/{subject}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session01.hdf5"
    )
    with h5py.File(beta_probe, "r", rdcc_nbytes=1024 * 1024) as handle:
        beta_shape = tuple(int(v) for v in handle["betas"].shape[1:])
    if tuple(arr.shape) != beta_shape:
        transposed = np.transpose(arr, (2, 1, 0))
        if tuple(transposed.shape) != beta_shape:
            raise ValueError(f"ROI mask shape {arr.shape} cannot align to beta shape {beta_shape}")
        arr = transposed
    flat = arr.reshape(-1)
    labels = [int(row["label_id"]) for row in node_set["node_labels"]]
    indices = [np.flatnonzero(flat == label).astype(np.int64) for label in labels]
    return indices, [int(idx.size) for idx in indices], beta_shape


def roi_stats16(volume_flat: np.ndarray, node_indices: list[np.ndarray]) -> np.ndarray:
    out = np.zeros((len(node_indices), len(FEATURE_NAMES)), dtype=np.float32)
    for node_idx, flat_idx in enumerate(node_indices):
        v = volume_flat[flat_idx].astype(np.float32, copy=False)
        if v.size == 0:
            continue
        q05, q10, q25, q50, q75, q90, q95 = np.quantile(v, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
        top_k = max(1, int(np.ceil(0.10 * v.size)))
        top10 = np.partition(v, v.size - top_k)[-top_k:]
        out[node_idx] = np.asarray(
            [
                float(v.mean(dtype=np.float64)),
                float(v.std(dtype=np.float64)),
                float(v.min()),
                float(v.max()),
                float(q05),
                float(q10),
                float(q25),
                float(q50),
                float(q75),
                float(q90),
                float(q95),
                float((v > 0).mean(dtype=np.float64)),
                float((v < 0).mean(dtype=np.float64)),
                float(np.abs(v).mean(dtype=np.float64)),
                float(np.sqrt(np.square(v, dtype=np.float32).mean(dtype=np.float64))),
                float(top10.mean(dtype=np.float64)),
            ],
            dtype=np.float32,
        )
    return out


def log_progress(subject: str, done: int, total: int, start_time: float) -> None:
    elapsed = max(time.time() - start_time, 1e-6)
    rate = done / elapsed
    eta = (total - done) / rate if rate > 0 else float("inf")
    print(
        f"[progress] {subject} {done}/{total} ({done / max(total, 1):.1%}) "
        f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m rate={rate:.3f}/s",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    node_set = json.loads(args.node_set.read_text(encoding="utf-8"))
    subject = args.subject
    rows = read_csv(args.root / "preproc_v0" / f"{subject}_shared1000_metadata.csv")
    node_indices, voxel_counts, beta_shape = load_label_indices(args.root, subject, node_set)
    beta_dir = args.root / f"data/nsddata_betas/ppdata/{subject}/func1pt8mm/betas_fithrf_GLMdenoise_RR"
    session_cache = SessionCache(beta_dir, args.max_open_sessions)
    features = np.zeros((len(rows), int(node_set["n_nodes"]), len(FEATURE_NAMES)), dtype=np.float32)
    nsd_ids = np.zeros(len(rows), dtype=np.int32)
    n_repetitions = np.zeros(len(rows), dtype=np.int16)
    qc_rows: list[dict[str, object]] = []
    start_time = time.time()
    print(f"[start] subject={subject} samples={len(rows)} nodes={node_set['n_nodes']} features=16", flush=True)
    try:
        for sample_index, row in enumerate(rows):
            nsd_ids[sample_index] = int(row["nsdId"])
            sessions = [int(v) for v in json.loads(row["all_sessions"])]
            trials = [int(v) for v in json.loads(row["all_trials"])]
            n_repetitions[sample_index] = len(sessions)
            rep_sum = None
            for session, trial in zip(sessions, trials):
                volume = session_cache.get(session)["betas"][trial - 1].astype(np.float32, copy=False)
                rep_sum = np.asarray(volume, dtype=np.float32).copy() if rep_sum is None else rep_sum + volume
            if rep_sum is not None:
                averaged = (rep_sum / float(len(sessions))).reshape(-1)
                features[sample_index] = roi_stats16(averaged, node_indices)
                status = "ok"
            else:
                status = "no_repetitions"
            qc_rows.append(
                {
                    "sample_index": sample_index,
                    "subject": subject,
                    "nsdId": int(row["nsdId"]),
                    "n_repetitions_used": int(len(sessions)),
                    "status": status,
                }
            )
            done = sample_index + 1
            if done == 1 or done == len(rows) or done % args.progress_every == 0:
                log_progress(subject, done, len(rows), start_time)
    finally:
        session_cache.close()

    npz_path = args.output_dir / f"{subject}_hcp_mmp1_roi_stats16_features.npz"
    print(f"[write] subject={subject} writing={npz_path}", flush=True)
    np.savez_compressed(
        npz_path,
        features=features,
        nsd_ids=nsd_ids,
        sample_indices=np.arange(len(rows), dtype=np.int32),
        n_repetitions_used=n_repetitions,
        voxel_counts=np.asarray(voxel_counts, dtype=np.int32),
        beta_shape=np.asarray(beta_shape, dtype=np.int32),
        feature_names=np.asarray(FEATURE_NAMES),
        node_set_name=np.asarray(str(node_set["node_set_name"])),
        subject=np.asarray(subject),
    )
    meta_path = args.output_dir / f"{subject}_hcp_mmp1_roi_stats16_metadata.csv"
    write_csv(meta_path, qc_rows, ["sample_index", "subject", "nsdId", "n_repetitions_used", "status"])
    qc = {
        "subject": subject,
        "node_set_name": node_set["node_set_name"],
        "feature_set": "roi_stats16",
        "n_nodes": int(node_set["n_nodes"]),
        "n_samples": int(features.shape[0]),
        "feature_shape": list(features.shape),
        "feature_nan_count": int(np.isnan(features).sum()),
        "feature_inf_count": int(np.isinf(features).sum()),
        "min_roi_voxels": int(min(voxel_counts)),
        "max_roi_voxels": int(max(voxel_counts)),
        "elapsed_seconds": float(time.time() - start_time),
        "status": "ok",
    }
    (args.output_dir / f"{subject}_hcp_mmp1_roi_stats16_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc), flush=True)


if __name__ == "__main__":
    main()
