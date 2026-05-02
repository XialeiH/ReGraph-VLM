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
    parser = argparse.ArgumentParser(description="Cache averaged voxel patterns for each HCP-MMP ROI and subject.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--node-set", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--max-open-sessions", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
        handle = h5py.File(self.beta_dir / f"betas_session{session:02d}.hdf5", "r", rdcc_nbytes=16 * 1024 * 1024)
        self.handles[session] = handle
        while len(self.handles) > self.max_open:
            _, old = self.handles.popitem(last=False)
            old.close()
        return handle

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def load_roi_indices(root: Path, subject: str, node_set: dict[str, object]) -> tuple[list[np.ndarray], list[int], tuple[int, ...]]:
    roi_family = str(node_set["roi_family"])
    mask_path = root / f"data/nsddata/ppdata/{subject}/func1pt8mm/roi/{roi_family}.nii.gz"
    arr = np.asanyarray(nib.load(str(mask_path)).dataobj).astype(np.int32)
    beta_probe = root / f"data/nsddata_betas/ppdata/{subject}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session01.hdf5"
    with h5py.File(beta_probe, "r") as handle:
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
    subject = args.subject
    node_set = json.loads(args.node_set.read_text(encoding="utf-8"))
    rows = read_csv(args.root / "preproc_v0" / f"{subject}_shared1000_metadata.csv")
    roi_indices, voxel_counts, beta_shape = load_roi_indices(args.root, subject, node_set)
    beta_dir = args.root / f"data/nsddata_betas/ppdata/{subject}/func1pt8mm/betas_fithrf_GLMdenoise_RR"
    out_path = args.output_dir / f"{subject}_hcp_mmp1_roi_voxels.h5"
    tmp_path = args.output_dir / f"{subject}_hcp_mmp1_roi_voxels.h5.tmp"
    store_dtype = np.float16 if args.dtype == "float16" else np.float32
    start = time.time()
    print(f"[start] subject={subject} samples={len(rows)} nodes={len(roi_indices)} dtype={args.dtype}", flush=True)

    session_cache = SessionCache(beta_dir, args.max_open_sessions)
    nsd_ids = np.asarray([int(row["nsdId"]) for row in rows], dtype=np.int32)
    sample_indices = np.arange(len(rows), dtype=np.int32)
    n_repetitions = np.zeros(len(rows), dtype=np.int16)
    try:
        with h5py.File(tmp_path, "w") as out:
            out.attrs["subject"] = subject
            out.attrs["node_set_name"] = str(node_set["node_set_name"])
            out.attrs["storage_dtype"] = args.dtype
            out.create_dataset("nsd_ids", data=nsd_ids)
            out.create_dataset("sample_indices", data=sample_indices)
            out.create_dataset("voxel_counts", data=np.asarray(voxel_counts, dtype=np.int32))
            out.create_dataset("beta_shape", data=np.asarray(beta_shape, dtype=np.int32))
            roi_group = out.create_group("roi")
            datasets = []
            for node_idx, flat_idx in enumerate(roi_indices):
                ds = roi_group.create_dataset(
                    f"node_{node_idx:03d}",
                    shape=(len(rows), int(flat_idx.size)),
                    dtype=store_dtype,
                    chunks=(1, int(flat_idx.size)),
                )
                datasets.append(ds)
            for sample_index, row in enumerate(rows):
                sessions = [int(v) for v in json.loads(row["all_sessions"])]
                trials = [int(v) for v in json.loads(row["all_trials"])]
                n_repetitions[sample_index] = len(sessions)
                rep_sum = None
                for session, trial in zip(sessions, trials):
                    vol = session_cache.get(session)["betas"][trial - 1].astype(np.float32, copy=False)
                    rep_sum = np.asarray(vol, dtype=np.float32).copy() if rep_sum is None else rep_sum + vol
                if rep_sum is None:
                    averaged_flat = np.zeros(int(np.prod(beta_shape)), dtype=np.float32)
                else:
                    averaged_flat = (rep_sum / float(len(sessions))).reshape(-1)
                for node_idx, flat_idx in enumerate(roi_indices):
                    datasets[node_idx][sample_index, :] = averaged_flat[flat_idx].astype(store_dtype, copy=False)
                done = sample_index + 1
                if done == 1 or done == len(rows) or done % args.progress_every == 0:
                    log_progress(subject, done, len(rows), start)
            out.create_dataset("n_repetitions_used", data=n_repetitions)
        tmp_path.replace(out_path)
    finally:
        session_cache.close()
    qc = {
        "subject": subject,
        "node_set_name": node_set["node_set_name"],
        "n_samples": len(rows),
        "n_nodes": len(roi_indices),
        "storage_dtype": args.dtype,
        "min_roi_voxels": int(min(voxel_counts)),
        "max_roi_voxels": int(max(voxel_counts)),
        "output_h5": str(out_path),
        "elapsed_seconds": float(time.time() - start),
        "status": "ok",
    }
    (args.output_dir / f"{subject}_hcp_mmp1_roi_voxels_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc), flush=True)


if __name__ == "__main__":
    main()
