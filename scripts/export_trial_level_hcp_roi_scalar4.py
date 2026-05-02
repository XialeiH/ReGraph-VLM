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
import pandas as pd
import torch


SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
FEATURE_NAMES = ["mean_beta", "std_beta", "q90_beta", "positive_fraction"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trial-level HCP-MMP ROI scalar4 features.")
    parser.add_argument("--root", type=Path, required=True, help="v0_shared_unit root.")
    parser.add_argument("--subject", type=str, required=True, help="subj01..subj08 or all.")
    parser.add_argument("--inventory", type=Path, default=Path("preproc_v0/repetition_familiarity/repetition_inventory.csv"))
    parser.add_argument("--node-set", type=Path, default=Path("preproc_v0/roi_graph_atlas_v1/roi_node_set_v1.json"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/trial_roi_features_scalar4"),
    )
    parser.add_argument("--top-quantile", type=float, default=0.90)
    parser.add_argument("--max-open-sessions", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


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


def scalar4(volume_flat: np.ndarray, node_indices: list[np.ndarray], top_quantile: float) -> np.ndarray:
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


def subject_to_int(subject: str) -> int:
    return int(subject.replace("subj", ""))


def log_progress(subject: str, done: int, total: int, start_time: float) -> None:
    elapsed = max(time.time() - start_time, 1e-6)
    rate = done / elapsed
    eta = (total - done) / rate if rate > 0 else float("inf")
    print(
        f"[progress] {subject} {done}/{total} ({done / max(total, 1):.1%}) "
        f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m rate={rate:.3f}/s",
        flush=True,
    )


def export_subject(args: argparse.Namespace, subject: str) -> dict[str, object]:
    root = args.root.resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    node_set = json.loads((root / args.node_set).read_text(encoding="utf-8"))
    inventory = pd.read_csv(root / args.inventory)
    rows = inventory[(inventory["subject"] == subject) & (inventory["has_beta"] == True) & (inventory["has_roi_mask"] == True)].copy()
    if rows.empty:
        raise ValueError(f"No usable inventory rows for {subject}")
    rows = rows.sort_values(["nsdId", "repeat_index", "session_index", "trial_index"]).reset_index(drop=True)

    node_indices, voxel_counts, beta_shape = load_label_indices(root, subject, node_set)
    n_trials = len(rows)
    n_nodes = int(node_set["n_nodes"])
    features = np.zeros((n_trials, n_nodes, len(FEATURE_NAMES)), dtype=np.float32)

    beta_dir = root / f"data/nsddata_betas/ppdata/{subject}/func1pt8mm/betas_fithrf_GLMdenoise_RR"
    cache = SessionCache(beta_dir, args.max_open_sessions)
    start = time.time()
    print(f"[start] subject={subject} n_trials={n_trials} n_nodes={n_nodes}", flush=True)
    try:
        for idx, row in rows.iterrows():
            session = int(row["session_index"])
            trial = int(row["trial_index"])
            volume = cache.get(session)["betas"][trial - 1].astype(np.float32, copy=False)
            features[idx] = scalar4(volume.reshape(-1), node_indices, args.top_quantile)
            done = idx + 1
            if done == 1 or done == n_trials or done % args.progress_every == 0:
                log_progress(subject, done, n_trials, start)
    finally:
        cache.close()

    valid = np.ones(n_trials, dtype=np.bool_)
    subject_values = np.full(n_trials, subject_to_int(subject), dtype=np.int16)
    data = {
        "x": torch.from_numpy(features),
        "subject": torch.from_numpy(subject_values),
        "nsdId": torch.from_numpy(rows["nsdId"].astype(np.int32).to_numpy()),
        "repeat_index": torch.from_numpy(rows["repeat_index"].astype(np.int16).to_numpy()),
        "raw_rep_index_for_subject": torch.from_numpy(rows["raw_rep_index_for_subject"].astype(np.int16).to_numpy()),
        "session_index": torch.from_numpy(rows["session_index"].astype(np.int16).to_numpy()),
        "trial_index": torch.from_numpy(rows["trial_index"].astype(np.int16).to_numpy()),
        "trial_in_session": torch.from_numpy(rows["trial_index"].astype(np.int16).to_numpy()),
        "valid": torch.from_numpy(valid),
        "feature_names": FEATURE_NAMES,
        "node_set_name": str(node_set["node_set_name"]),
        "node_labels": [int(row["label_id"]) for row in node_set["node_labels"]],
        "voxel_counts": torch.tensor(voxel_counts, dtype=torch.int32),
        "beta_shape": torch.tensor(beta_shape, dtype=torch.int32),
    }
    out_path = output_dir / f"{subject}_trial_scalar4.pt"
    torch.save(data, out_path)

    qc = {
        "subject": subject,
        "n_trials": int(n_trials),
        "shape": str(list(features.shape)),
        "nan_count": int(np.isnan(features).sum()),
        "inf_count": int(np.isinf(features).sum()),
        "min": float(np.min(features)),
        "max": float(np.max(features)),
        "mean": float(np.mean(features, dtype=np.float64)),
        "std": float(np.std(features, dtype=np.float64)),
        "n_t2_rows": int(rows["usable_T2"].sum()),
        "n_t3_rows": int(rows["usable_T3"].sum()),
        "n_unique_images": int(rows["nsdId"].nunique()),
        "n_nodes": int(n_nodes),
        "node_feature_dim": int(len(FEATURE_NAMES)),
        "output_path": str(out_path),
        "status": "ok",
        "elapsed_seconds": float(time.time() - start),
    }
    qc_path = output_dir / f"{subject}_trial_scalar4_qc.json"
    qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc), flush=True)
    return qc


def summarize(output_dir: Path) -> Path:
    rows: list[dict[str, object]] = []
    for subject in SUBJECTS:
        path = output_dir / f"{subject}_trial_scalar4_qc.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            rows.append({"subject": subject, "status": "missing"})
    fields = [
        "subject",
        "n_trials",
        "shape",
        "nan_count",
        "inf_count",
        "min",
        "max",
        "mean",
        "std",
        "n_t2_rows",
        "n_t3_rows",
        "n_unique_images",
        "n_nodes",
        "node_feature_dim",
        "output_path",
        "status",
        "elapsed_seconds",
    ]
    out_path = output_dir / "trial_roi_features_scalar4_qc.csv"
    write_csv(out_path, rows, fields)
    return out_path


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize_only:
        print(summarize(output_dir))
        return

    subjects = SUBJECTS if args.subject == "all" else [args.subject]
    for subject in subjects:
        export_subject(args, subject)
    print(summarize(output_dir))


if __name__ == "__main__":
    main()
