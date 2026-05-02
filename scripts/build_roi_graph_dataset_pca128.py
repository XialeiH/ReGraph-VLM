#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch


PCA_DIM = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HCP-MMP ROI graph dataset with training-only per-ROI PCA128 node features.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--fold-name", type=str, required=True)
    parser.add_argument("--node-set", type=Path, required=True)
    parser.add_argument("--voxel-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pca-dim", type=int, default=PCA_DIM)
    parser.add_argument("--oversample", type=int, default=16)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def make_edges(adjacency: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    src, dst = np.nonzero(np.isfinite(adjacency))
    return (
        torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long),
        torch.tensor(adjacency[src, dst, None], dtype=torch.float32),
    )


class VoxelCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.handles: dict[str, h5py.File] = {}

    def get_handle(self, subject: str) -> h5py.File:
        if subject not in self.handles:
            self.handles[subject] = h5py.File(self.cache_dir / f"{subject}_hcp_mmp1_roi_voxels.h5", "r")
        return self.handles[subject]

    def get_roi(self, row: dict[str, str], node_idx: int) -> np.ndarray:
        subject = row["subject"]
        sample_index = int(row["sample_index"])
        handle = self.get_handle(subject)
        actual = int(handle["nsd_ids"][sample_index])
        expected = int(row["nsdId"])
        if actual != expected:
            raise ValueError(f"{subject} sample_index {sample_index} nsdId mismatch: {actual} != {expected}")
        return handle["roi"][f"node_{node_idx:03d}"][sample_index, :].astype(np.float32, copy=False)

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def stack_roi(rows: list[dict[str, str]], cache: VoxelCache, node_idx: int) -> np.ndarray:
    return np.stack([cache.get_roi(row, node_idx) for row in rows], axis=0).astype(np.float32, copy=False)


def randomized_pca_fit(x: np.ndarray, dim: int, oversample: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int]:
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    xc = x - mean[None, :]
    n_samples, n_features = xc.shape
    k_eff = int(min(dim, n_samples - 1, n_features))
    if k_eff <= 0:
        return mean, np.zeros((0, n_features), dtype=np.float32), 0
    q = int(min(k_eff + oversample, n_samples, n_features))
    omega = rng.standard_normal((n_features, q), dtype=np.float32)
    y = xc @ omega
    q_mat, _ = np.linalg.qr(y, mode="reduced")
    b = q_mat.T @ xc
    _, _, vt = np.linalg.svd(b, full_matrices=False)
    components = vt[:k_eff].astype(np.float32, copy=False)
    return mean, components, k_eff


def transform_pca(x: np.ndarray, mean: np.ndarray, components: np.ndarray, out_dim: int) -> np.ndarray:
    out = np.zeros((x.shape[0], out_dim), dtype=np.float32)
    if components.shape[0] > 0:
        out[:, : components.shape[0]] = (x - mean[None, :]) @ components.T
    return out


def rows_to_graphs(
    rows: list[dict[str, str]],
    split: str,
    x_all: np.ndarray,
    class_map: dict[int, int],
    adjacency: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    edge_index, edge_attr = make_edges(adjacency)
    graphs: list[dict[str, object]] = []
    meta: list[dict[str, object]] = []
    for idx, row in enumerate(rows):
        y = class_map[int(row["nsdId"])]
        graphs.append(
            {
                "x": torch.tensor(x_all[idx], dtype=torch.float32),
                "edge_index": edge_index.clone(),
                "edge_attr": edge_attr.clone(),
                "y": torch.tensor(y, dtype=torch.long),
                "subject": row["subject"],
                "nsdId": int(row["nsdId"]),
                "split": split,
                "source_feature_path": row["source_feature_path"],
                "source_sample_index": int(row["sample_index"]),
            }
        )
        meta.append(
            {
                "split": split,
                "row_index": idx,
                "subject": row["subject"],
                "nsdId": int(row["nsdId"]),
                "label": y,
                "source_feature_path": row["source_feature_path"],
                "source_sample_index": int(row["sample_index"]),
            }
        )
    return graphs, meta


def main() -> None:
    args = parse_args()
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    node_set = json.loads(args.node_set.read_text(encoding="utf-8"))
    n_nodes = int(node_set["n_nodes"])
    train_rows_all = read_csv(args.fold_root / f"{args.fold_name}_train_features_index.csv")
    test_rows = read_csv(args.fold_root / f"{args.fold_name}_test_features_index.csv")
    train_subjects = sorted({row["subject"] for row in train_rows_all})
    val_subject = train_subjects[-1]
    train_rows = [row for row in train_rows_all if row["subject"] != val_subject]
    val_rows = [row for row in train_rows_all if row["subject"] == val_subject]
    image_ids = sorted({int(row["nsdId"]) for row in train_rows_all + test_rows})
    class_map = {nsd_id: idx for idx, nsd_id in enumerate(image_ids)}
    print(
        f"[start] fold={args.fold_name} train={len(train_rows)} val={len(val_rows)} "
        f"test={len(test_rows)} nodes={n_nodes} pca_dim={args.pca_dim}",
        flush=True,
    )
    train_x = np.zeros((len(train_rows), n_nodes, args.pca_dim), dtype=np.float32)
    val_x = np.zeros((len(val_rows), n_nodes, args.pca_dim), dtype=np.float32)
    test_x = np.zeros((len(test_rows), n_nodes, args.pca_dim), dtype=np.float32)
    pca_rows = []
    rng = np.random.default_rng(args.seed)
    cache = VoxelCache(args.voxel_cache_dir)
    try:
        for node_idx in range(n_nodes):
            t0 = time.time()
            x_train_raw = stack_roi(train_rows, cache, node_idx)
            mean, components, k_eff = randomized_pca_fit(x_train_raw, args.pca_dim, args.oversample, rng)
            train_x[:, node_idx, :] = transform_pca(x_train_raw, mean, components, args.pca_dim)
            val_x[:, node_idx, :] = transform_pca(stack_roi(val_rows, cache, node_idx), mean, components, args.pca_dim)
            test_x[:, node_idx, :] = transform_pca(stack_roi(test_rows, cache, node_idx), mean, components, args.pca_dim)
            pca_rows.append({"node_index": node_idx, "n_voxels": int(mean.shape[0]), "k_eff": k_eff})
            if node_idx == 0 or (node_idx + 1) % 10 == 0 or node_idx + 1 == n_nodes:
                print(f"[progress] roi={node_idx + 1}/{n_nodes} k_eff={k_eff} elapsed_roi={time.time() - t0:.1f}s", flush=True)
    finally:
        cache.close()

    roi_signal = train_x[:, :, 0]
    corr = np.corrcoef(roi_signal, rowvar=False).astype(np.float32)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    adjacency = np.abs(corr)
    np.fill_diagonal(adjacency, 1.0)
    train_graphs, train_meta = rows_to_graphs(train_rows, "train", train_x, class_map, adjacency)
    val_graphs, val_meta = rows_to_graphs(val_rows, "val", val_x, class_map, adjacency)
    test_graphs, test_meta = rows_to_graphs(test_rows, "test", test_x, class_map, adjacency)
    print("[write] saving graph tensors", flush=True)
    torch.save(train_graphs, args.output_dir / "train_graphs.pt")
    torch.save(val_graphs, args.output_dir / "val_graphs.pt")
    torch.save(test_graphs, args.output_dir / "test_graphs.pt")
    np.save(args.output_dir / "adjacency.npy", adjacency)
    with (args.output_dir / "graph_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        all_meta = train_meta + val_meta + test_meta
        writer = csv.DictWriter(handle, fieldnames=list(all_meta[0].keys()))
        writer.writeheader()
        writer.writerows(all_meta)
    with (args.output_dir / "pca_node_diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node_index", "n_voxels", "k_eff"])
        writer.writeheader()
        writer.writerows(pca_rows)
    all_features = np.concatenate([train_x, val_x, test_x], axis=0)
    qc = {
        "fold": args.fold_name,
        "node_set_name": node_set["node_set_name"],
        "feature_set": "roi_pca128",
        "n_nodes": n_nodes,
        "node_feature_dim": args.pca_dim,
        "train_graphs": len(train_graphs),
        "val_graphs": len(val_graphs),
        "test_graphs": len(test_graphs),
        "val_subject": val_subject,
        "test_subject": test_rows[0]["subject"],
        "num_classes": len(class_map),
        "feature_nan_count": int(np.isnan(all_features).sum()),
        "feature_inf_count": int(np.isinf(all_features).sum()),
        "adjacency_nan_count": int(np.isnan(adjacency).sum()),
        "adjacency_inf_count": int(np.isinf(adjacency).sum()),
        "adjacency_density": float(np.count_nonzero(adjacency) / adjacency.size),
        "elapsed_seconds": float(time.time() - start),
        "status": "ok",
    }
    (args.output_dir / "graph_dataset_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc), flush=True)


if __name__ == "__main__":
    main()
