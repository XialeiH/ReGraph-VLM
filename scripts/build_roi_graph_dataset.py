#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build averaged-response atlas ROI graph datasets for one fold.")
    parser.add_argument("--root", type=Path, required=True, help="v0_shared_unit root on HPC.")
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--fold-name", type=str, required=True)
    parser.add_argument("--node-set", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-strategy", choices=["fixed_last"], default="fixed_last")
    parser.add_argument("--top-quantile", type=float, default=0.90)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_roi_indices(root: Path, subject: str, node_set: dict[str, object]) -> tuple[list[np.ndarray], list[int]]:
    roi_family = str(node_set["roi_family"])
    mask_path = root / f"data/nsddata/ppdata/{subject}/func1pt8mm/roi/{roi_family}.nii.gz"
    arr = np.asanyarray(nib.load(str(mask_path)).dataobj).astype(np.int32)
    labels = [int(row["label_id"]) for row in node_set["node_labels"]]
    masks: list[np.ndarray] = []
    counts: list[int] = []
    for label in labels:
        mask = arr == label
        if mask.shape != (83, 104, 81):
            mask = np.transpose(mask, (2, 1, 0))
        flat_idx = np.flatnonzero(mask.reshape(-1))
        masks.append(flat_idx)
        counts.append(int(flat_idx.size))
    return masks, counts


def load_metadata_row(index_row: dict[str, str], cache: dict[Path, list[dict[str, str]]]) -> dict[str, str]:
    feature_path = Path(index_row["source_feature_path"])
    metadata_path = feature_path.with_name(feature_path.name.replace("_roi_features.npz", "_metadata.csv"))
    if metadata_path not in cache:
        cache[metadata_path] = read_csv(metadata_path)
    return cache[metadata_path][int(index_row["sample_index"])]


def open_beta(handles: dict[tuple[str, int], h5py.File], root: Path, subject: str, session: int) -> h5py.File:
    key = (subject, session)
    if key not in handles:
        path = root / f"data/nsddata_betas/ppdata/{subject}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session{session:02d}.hdf5"
        handles[key] = h5py.File(path, "r")
    return handles[key]


def graph_features_for_row(
    row: dict[str, str],
    root: Path,
    node_indices: list[np.ndarray],
    metadata_cache: dict[Path, list[dict[str, str]]],
    beta_handles: dict[tuple[str, int], h5py.File],
    top_quantile: float,
) -> np.ndarray:
    subject = row["subject"]
    meta = load_metadata_row(row, metadata_cache)
    sessions = json.loads(meta["all_sessions"])
    trials = json.loads(meta["all_trials"])
    volumes = []
    for session, trial in zip(sessions, trials):
        handle = open_beta(beta_handles, root, subject, int(session))
        volumes.append(handle["betas"][int(trial) - 1].astype(np.float32))
    averaged = np.mean(np.stack(volumes, axis=0), axis=0, dtype=np.float32).reshape(-1)

    features = np.zeros((len(node_indices), 4), dtype=np.float32)
    for node_idx, flat_idx in enumerate(node_indices):
        values = averaged[flat_idx]
        if values.size == 0:
            continue
        features[node_idx, 0] = float(values.mean())
        features[node_idx, 1] = float(values.std())
        features[node_idx, 2] = float(np.quantile(values, top_quantile))
        features[node_idx, 3] = float((values > 0).mean())
    return features


def make_edges(adjacency: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    src, dst = np.nonzero(np.isfinite(adjacency))
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_attr = torch.tensor(adjacency[src, dst, None], dtype=torch.float32)
    return edge_index, edge_attr


def build_graphs(
    rows: list[dict[str, str]],
    split_name: str,
    root: Path,
    node_set: dict[str, object],
    class_map: dict[int, int],
    adjacency: np.ndarray,
    top_quantile: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], np.ndarray]:
    metadata_cache: dict[Path, list[dict[str, str]]] = {}
    beta_handles: dict[tuple[str, int], h5py.File] = {}
    roi_cache: dict[str, tuple[list[np.ndarray], list[int]]] = {}
    edge_index, edge_attr = make_edges(adjacency)
    graphs: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    node_feature_stack: list[np.ndarray] = []
    try:
        for sample_idx, row in enumerate(rows):
            subject = row["subject"]
            if subject not in roi_cache:
                roi_cache[subject] = load_roi_indices(root, subject, node_set)
            node_indices, voxel_counts = roi_cache[subject]
            x_np = graph_features_for_row(
                row=row,
                root=root,
                node_indices=node_indices,
                metadata_cache=metadata_cache,
                beta_handles=beta_handles,
                top_quantile=top_quantile,
            )
            y = class_map[int(row["nsdId"])]
            graphs.append(
                {
                    "x": torch.tensor(x_np, dtype=torch.float32),
                    "edge_index": edge_index.clone(),
                    "edge_attr": edge_attr.clone(),
                    "y": torch.tensor(y, dtype=torch.long),
                    "subject": subject,
                    "nsdId": int(row["nsdId"]),
                    "split": split_name,
                    "source_feature_path": row["source_feature_path"],
                    "source_sample_index": int(row["sample_index"]),
                }
            )
            node_feature_stack.append(x_np)
            metadata_rows.append(
                {
                    "split": split_name,
                    "row_index": sample_idx,
                    "subject": subject,
                    "nsdId": int(row["nsdId"]),
                    "label": y,
                    "source_feature_path": row["source_feature_path"],
                    "source_sample_index": int(row["sample_index"]),
                    "min_roi_voxels": min(voxel_counts),
                    "max_roi_voxels": max(voxel_counts),
                }
            )
    finally:
        for handle in beta_handles.values():
            handle.close()
    return graphs, metadata_rows, np.stack(node_feature_stack, axis=0)


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

    identity = np.eye(int(node_set["n_nodes"]), dtype=np.float32)
    train_graphs_tmp, train_meta, train_features = build_graphs(
        train_rows, "train", args.root, node_set, class_map, identity, args.top_quantile
    )
    roi_signal = train_features[:, :, 0]
    corr = np.corrcoef(roi_signal, rowvar=False).astype(np.float32)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    adjacency = np.abs(corr)
    np.fill_diagonal(adjacency, 1.0)

    train_edge_index, train_edge_attr = make_edges(adjacency)
    for graph in train_graphs_tmp:
        graph["edge_index"] = train_edge_index.clone()
        graph["edge_attr"] = train_edge_attr.clone()

    val_graphs, val_meta, val_features = build_graphs(
        val_rows, "val", args.root, node_set, class_map, adjacency, args.top_quantile
    )
    test_graphs, test_meta, test_features = build_graphs(
        test_rows, "test", args.root, node_set, class_map, adjacency, args.top_quantile
    )

    torch.save(train_graphs_tmp, args.output_dir / "train_graphs.pt")
    torch.save(val_graphs, args.output_dir / "val_graphs.pt")
    torch.save(test_graphs, args.output_dir / "test_graphs.pt")
    np.save(args.output_dir / "adjacency.npy", adjacency)

    with (args.output_dir / "graph_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list((train_meta + val_meta + test_meta)[0].keys()))
        writer.writeheader()
        writer.writerows(train_meta + val_meta + test_meta)

    all_features = np.concatenate([train_features, val_features, test_features], axis=0)
    qc = {
        "fold": args.fold_name,
        "node_set_name": node_set["node_set_name"],
        "n_nodes": int(node_set["n_nodes"]),
        "node_feature_dim": 4,
        "train_graphs": len(train_graphs_tmp),
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
    (args.output_dir / "graph_dataset_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc))


if __name__ == "__main__":
    main()
