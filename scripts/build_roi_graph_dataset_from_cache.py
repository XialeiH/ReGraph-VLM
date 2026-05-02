#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build atlas ROI graph datasets from cached subject ROI features.")
    parser.add_argument("--root", type=Path, required=True, help="v0_shared_unit root on HPC.")
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--fold-name", type=str, required=True)
    parser.add_argument("--node-set", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-strategy", choices=["fixed_last"], default="fixed_last")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def make_edges(adjacency: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    src, dst = np.nonzero(np.isfinite(adjacency))
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_attr = torch.tensor(adjacency[src, dst, None], dtype=torch.float32)
    return edge_index, edge_attr


class FeatureCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self._features: dict[str, np.ndarray] = {}
        self._nsd_ids: dict[str, np.ndarray] = {}

    def load_subject(self, subject: str) -> tuple[np.ndarray, np.ndarray]:
        if subject not in self._features:
            path = self.cache_dir / f"{subject}_hcp_mmp1_roi_summary_features.npz"
            with np.load(path, allow_pickle=False) as npz:
                self._features[subject] = npz["features"].astype(np.float32, copy=False)
                self._nsd_ids[subject] = npz["nsd_ids"].astype(np.int32, copy=False)
        return self._features[subject], self._nsd_ids[subject]

    def get(self, row: dict[str, str]) -> np.ndarray:
        subject = row["subject"]
        sample_index = int(row["sample_index"])
        features, nsd_ids = self.load_subject(subject)
        if sample_index < 0 or sample_index >= features.shape[0]:
            raise IndexError(f"{subject} sample_index {sample_index} outside cache with {features.shape[0]} rows")
        expected_nsd_id = int(row["nsdId"])
        actual_nsd_id = int(nsd_ids[sample_index])
        if actual_nsd_id != expected_nsd_id:
            raise ValueError(f"{subject} sample_index {sample_index} nsdId mismatch: {actual_nsd_id} != {expected_nsd_id}")
        return features[sample_index]


def build_graphs(
    rows: list[dict[str, str]],
    split_name: str,
    cache: FeatureCache,
    class_map: dict[int, int],
    adjacency: np.ndarray,
    progress_every: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], np.ndarray]:
    edge_index, edge_attr = make_edges(adjacency)
    graphs: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    features: list[np.ndarray] = []
    start_time = time.time()
    print(f"[start] split={split_name} rows={len(rows)}", flush=True)
    for row_index, row in enumerate(rows):
        x_np = cache.get(row)
        y = class_map[int(row["nsdId"])]
        graphs.append(
            {
                "x": torch.tensor(x_np, dtype=torch.float32),
                "edge_index": edge_index.clone(),
                "edge_attr": edge_attr.clone(),
                "y": torch.tensor(y, dtype=torch.long),
                "subject": row["subject"],
                "nsdId": int(row["nsdId"]),
                "split": split_name,
                "source_feature_path": row["source_feature_path"],
                "source_sample_index": int(row["sample_index"]),
            }
        )
        metadata.append(
            {
                "split": split_name,
                "row_index": row_index,
                "subject": row["subject"],
                "nsdId": int(row["nsdId"]),
                "label": y,
                "source_feature_path": row["source_feature_path"],
                "source_sample_index": int(row["sample_index"]),
            }
        )
        features.append(x_np)
        done = row_index + 1
        if done == 1 or done == len(rows) or done % progress_every == 0:
            elapsed = max(time.time() - start_time, 1e-6)
            rate = done / elapsed
            eta = (len(rows) - done) / rate if rate > 0 else float("inf")
            print(
                f"[progress] split={split_name} {done}/{len(rows)} "
                f"({done / max(len(rows), 1):.1%}) elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    return graphs, metadata, np.stack(features, axis=0)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    node_set = json.loads(args.node_set.read_text(encoding="utf-8"))
    train_rows_all = read_csv(args.fold_root / f"{args.fold_name}_train_features_index.csv")
    test_rows = read_csv(args.fold_root / f"{args.fold_name}_test_features_index.csv")

    train_subjects = sorted({row["subject"] for row in train_rows_all})
    val_subject = train_subjects[-1]
    train_rows = [row for row in train_rows_all if row["subject"] != val_subject]
    val_rows = [row for row in train_rows_all if row["subject"] == val_subject]
    image_ids = sorted({int(row["nsdId"]) for row in train_rows_all + test_rows})
    class_map = {nsd_id: idx for idx, nsd_id in enumerate(image_ids)}

    cache = FeatureCache(args.cache_dir)
    identity = np.eye(int(node_set["n_nodes"]), dtype=np.float32)
    print(
        f"[start] fold={args.fold_name} train={len(train_rows)} val={len(val_rows)} "
        f"test={len(test_rows)} classes={len(class_map)} nodes={node_set['n_nodes']}",
        flush=True,
    )
    train_graphs, train_meta, train_features = build_graphs(
        train_rows, "train", cache, class_map, identity, args.progress_every
    )

    print("[stage] computing training-only ROI correlation adjacency", flush=True)
    roi_signal = train_features[:, :, 0]
    corr = np.corrcoef(roi_signal, rowvar=False).astype(np.float32)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    adjacency = np.abs(corr)
    np.fill_diagonal(adjacency, 1.0)

    train_edge_index, train_edge_attr = make_edges(adjacency)
    for graph in train_graphs:
        graph["edge_index"] = train_edge_index.clone()
        graph["edge_attr"] = train_edge_attr.clone()

    val_graphs, val_meta, val_features = build_graphs(
        val_rows, "val", cache, class_map, adjacency, args.progress_every
    )
    test_graphs, test_meta, test_features = build_graphs(
        test_rows, "test", cache, class_map, adjacency, args.progress_every
    )

    print("[write] saving graph tensors", flush=True)
    torch.save(train_graphs, args.output_dir / "train_graphs.pt")
    torch.save(val_graphs, args.output_dir / "val_graphs.pt")
    torch.save(test_graphs, args.output_dir / "test_graphs.pt")
    np.save(args.output_dir / "adjacency.npy", adjacency)

    print("[write] saving metadata", flush=True)
    all_meta = train_meta + val_meta + test_meta
    with (args.output_dir / "graph_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_meta[0].keys()))
        writer.writeheader()
        writer.writerows(all_meta)

    all_features = np.concatenate([train_features, val_features, test_features], axis=0)
    qc = {
        "fold": args.fold_name,
        "node_set_name": node_set["node_set_name"],
        "n_nodes": int(node_set["n_nodes"]),
        "node_feature_dim": 4,
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
        "status": "ok",
    }
    print("[write] saving qc", flush=True)
    (args.output_dir / "graph_dataset_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc))


if __name__ == "__main__":
    main()
