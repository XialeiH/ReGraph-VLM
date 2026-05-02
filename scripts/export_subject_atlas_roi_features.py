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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export cached atlas ROI scalar features for one NSD subject."
    )
    parser.add_argument("--root", type=Path, required=True, help="v0_shared_unit root on HPC.")
    parser.add_argument("--subject", type=str, required=True, help="Subject id, e.g. subj01.")
    parser.add_argument("--node-set", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-quantile", type=float, default=0.90)
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
        path = self.beta_dir / f"betas_session{session:02d}.hdf5"
        handle = h5py.File(path, "r", rdcc_nbytes=8 * 1024 * 1024, rdcc_nslots=1009)
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

    beta_probe_path = (
        root
        / f"data/nsddata_betas/ppdata/{subject}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session01.hdf5"
    )
    with h5py.File(beta_probe_path, "r", rdcc_nbytes=1024 * 1024) as handle:
        beta_shape = tuple(int(v) for v in handle["betas"].shape[1:])

    if tuple(arr.shape) != beta_shape:
        transposed = np.transpose(arr, (2, 1, 0))
        if tuple(transposed.shape) != beta_shape:
            raise ValueError(f"ROI mask shape {arr.shape} cannot align to beta shape {beta_shape}")
        arr = transposed

    labels = [int(row["label_id"]) for row in node_set["node_labels"]]
    flat = arr.reshape(-1)
    indices: list[np.ndarray] = []
    counts: list[int] = []
    for label in labels:
        idx = np.flatnonzero(flat == label).astype(np.int64)
        indices.append(idx)
        counts.append(int(idx.size))
    return indices, counts, beta_shape


def roi_scalar_features(volume_flat: np.ndarray, node_indices: list[np.ndarray], top_quantile: float) -> np.ndarray:
    out = np.zeros((len(node_indices), 4), dtype=np.float32)
    for node_idx, flat_idx in enumerate(node_indices):
        values = volume_flat[flat_idx]
        if values.size == 0:
            continue
        out[node_idx, 0] = float(values.mean(dtype=np.float64))
        out[node_idx, 1] = float(values.std(dtype=np.float64))
        out[node_idx, 2] = float(np.quantile(values, top_quantile))
        out[node_idx, 3] = float((values > 0).mean(dtype=np.float64))
    return out


def log_progress(prefix: str, done: int, total: int, start_time: float) -> None:
    elapsed = max(time.time() - start_time, 1e-6)
    rate = done / elapsed
    remaining = (total - done) / rate if rate > 0 else float("inf")
    print(
        f"[progress] {prefix} {done}/{total} "
        f"({done / max(total, 1):.1%}) elapsed={elapsed / 60:.1f}m "
        f"eta={remaining / 60:.1f}m rate={rate:.3f}/s",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    node_set = json.loads(args.node_set.read_text(encoding="utf-8"))
    subject = args.subject

    metadata_path = args.root / "preproc_v0" / f"{subject}_shared1000_metadata.csv"
    rows = read_csv(metadata_path)
    node_indices, voxel_counts, beta_shape = load_label_indices(args.root, subject, node_set)
    beta_dir = args.root / f"data/nsddata_betas/ppdata/{subject}/func1pt8mm/betas_fithrf_GLMdenoise_RR"
    session_cache = SessionCache(beta_dir, args.max_open_sessions)

    features = np.zeros((len(rows), int(node_set["n_nodes"]), 4), dtype=np.float32)
    nsd_ids = np.zeros(len(rows), dtype=np.int32)
    n_repetitions = np.zeros(len(rows), dtype=np.int16)
    qc_rows: list[dict[str, object]] = []
    start_time = time.time()
    print(
        f"[start] subject={subject} samples={len(rows)} nodes={node_set['n_nodes']} "
        f"top_quantile={args.top_quantile}",
        flush=True,
    )

    try:
        for sample_index, row in enumerate(rows):
            nsd_ids[sample_index] = int(row["nsdId"])
            sessions = [int(v) for v in json.loads(row["all_sessions"])]
            trials = [int(v) for v in json.loads(row["all_trials"])]
            n_repetitions[sample_index] = len(sessions)
            if not sessions:
                qc_rows.append(
                    {
                        "sample_index": sample_index,
                        "subject": subject,
                        "nsdId": int(row["nsdId"]),
                        "n_repetitions_used": 0,
                        "status": "no_repetitions",
                    }
                )
                continue
            rep_sum: np.ndarray | None = None
            for session, trial in zip(sessions, trials):
                handle = session_cache.get(session)
                volume = handle["betas"][trial - 1].astype(np.float32, copy=False)
                if rep_sum is None:
                    rep_sum = np.asarray(volume, dtype=np.float32).copy()
                else:
                    rep_sum += volume
            assert rep_sum is not None
            averaged = (rep_sum / float(len(sessions))).reshape(-1)
            features[sample_index] = roi_scalar_features(averaged, node_indices, args.top_quantile)
            qc_rows.append(
                {
                    "sample_index": sample_index,
                    "subject": subject,
                    "nsdId": int(row["nsdId"]),
                    "n_repetitions_used": int(len(sessions)),
                    "status": "ok",
                }
            )
            done = sample_index + 1
            if done == 1 or done == len(rows) or done % args.progress_every == 0:
                log_progress(subject, done, len(rows), start_time)
    finally:
        session_cache.close()

    npz_path = args.output_dir / f"{subject}_hcp_mmp1_roi_summary_features.npz"
    print(f"[write] subject={subject} writing={npz_path}", flush=True)
    np.savez_compressed(
        npz_path,
        features=features,
        nsd_ids=nsd_ids,
        sample_indices=np.arange(len(rows), dtype=np.int32),
        n_repetitions_used=n_repetitions,
        voxel_counts=np.asarray(voxel_counts, dtype=np.int32),
        beta_shape=np.asarray(beta_shape, dtype=np.int32),
        node_set_name=np.asarray(str(node_set["node_set_name"])),
        subject=np.asarray(subject),
    )
    meta_path = args.output_dir / f"{subject}_hcp_mmp1_roi_summary_metadata.csv"
    print(f"[write] subject={subject} writing={meta_path}", flush=True)
    write_csv(
        meta_path,
        qc_rows,
        ["sample_index", "subject", "nsdId", "n_repetitions_used", "status"],
    )
    qc = {
        "subject": subject,
        "node_set_name": node_set["node_set_name"],
        "n_nodes": int(node_set["n_nodes"]),
        "n_samples": int(features.shape[0]),
        "feature_shape": list(features.shape),
        "feature_nan_count": int(np.isnan(features).sum()),
        "feature_inf_count": int(np.isinf(features).sum()),
        "min_roi_voxels": int(min(voxel_counts)),
        "max_roi_voxels": int(max(voxel_counts)),
        "output_npz": str(npz_path),
        "status": "ok",
        "elapsed_seconds": float(time.time() - start_time),
    }
    print(f"[write] subject={subject} writing_qc", flush=True)
    (args.output_dir / f"{subject}_hcp_mmp1_roi_summary_qc.json").write_text(
        json.dumps(qc, indent=2), encoding="utf-8"
    )
    print(json.dumps(qc))


if __name__ == "__main__":
    main()
