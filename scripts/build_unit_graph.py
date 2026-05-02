#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from run_prototype_fold import PrototypeModel, read_index_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fold-specific static unit graph for Stage 3A.")
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--prototype-root", type=Path, required=True)
    parser.add_argument("--fold-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--coactivation-topk", type=int, default=3)
    parser.add_argument("--graph-topk", type=int, default=8)
    parser.add_argument("--min-pair-count", type=int, default=12)
    return parser.parse_args()


def read_simple_yaml(path: Path) -> dict[str, object]:
    payload: dict[str, object] = {}
    current_list_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if line.startswith("  - ") and current_list_key is not None:
            payload.setdefault(current_list_key, []).append(line[4:])
            continue
        current_list_key = None
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            payload[key] = []
            current_list_key = key
        else:
            payload[key] = value
    return payload


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def yaml_dump_simple(path: Path, payload: dict[str, object]) -> None:
    lines: list[str] = []
    for key, value in payload.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_canonical_validation_subject(fold_dir: Path) -> dict[str, object]:
    metrics = json.loads((fold_dir / "prototype_metrics.json").read_text(encoding="utf-8"))
    selected = set(str(subject) for subject in metrics["selected_validation_subjects"])
    rows: list[dict[str, object]] = []
    with (fold_dir / "prototype_valsweep_summary.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["validation_subject"] not in selected:
                continue
            rows.append(
                {
                    "validation_subject": row["validation_subject"],
                    "best_val_top1": float(row["best_val_top1"]),
                    "best_val_top5": float(row["best_val_top5"]),
                    "top1_acc": float(row["top1_acc"]),
                    "top5_acc": float(row["top5_acc"]),
                    "best_seed": int(row["best_seed"]),
                }
            )
    if not rows:
        raise ValueError(f"No selected validation subjects found in {fold_dir}")
    rows.sort(
        key=lambda row: (
            float(row["best_val_top5"]),
            float(row["best_val_top1"]),
            float(row["top5_acc"]),
            float(row["top1_acc"]),
        ),
        reverse=True,
    )
    return rows[0]


def load_canonical_model(subrun_dir: Path, device: torch.device) -> tuple[PrototypeModel, dict[str, object], np.ndarray]:
    metrics = json.loads((subrun_dir / "prototype_metrics.json").read_text(encoding="utf-8"))
    config = read_simple_yaml(subrun_dir / "prototype_run_config.yaml")
    best_seed = int(metrics["best_seed"])
    model = PrototypeModel(
        input_dim=int(metrics["input_dim"]),
        hidden_dim=int(metrics["hidden_dim"]),
        num_classes=int(metrics["num_classes"]),
        num_prototypes=int(metrics["num_prototypes"]),
        tau=float(config["tau"]),
        dropout=float(config["dropout"]),
    ).to(device)
    state_dict = torch.load(subrun_dir / f"prototype_seed_{best_seed}_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    proto_embed = model.prototypes.detach().cpu().numpy().astype(np.float32)
    return model, metrics, proto_embed


def forward_assignments(model: PrototypeModel, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            outputs.append(model(xb)["assignments"].cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def labels_from_index(index_rows: list[dict[str, str]], class_map: dict[int, int]) -> np.ndarray:
    return np.array([class_map[int(row["nsdId"])] for row in index_rows], dtype=np.int64)


def build_adjacency(assignments: np.ndarray, coactivation_topk: int, graph_topk: int, min_pair_count: int) -> tuple[np.ndarray, list[dict[str, object]]]:
    num_units = assignments.shape[1]
    top_units = np.argsort(-assignments, axis=1)[:, :coactivation_topk]
    pair_counts: dict[tuple[int, int], int] = {}
    unit_support = np.zeros(num_units, dtype=np.int64)

    for row in top_units:
        units = sorted(int(v) for v in row.tolist())
        for unit in units:
            unit_support[unit] += 1
        for i in range(len(units)):
            for j in range(i + 1, len(units)):
                key = (units[i], units[j])
                pair_counts[key] = pair_counts.get(key, 0) + 1

    n_samples = max(assignments.shape[0], 1)
    weighted = np.zeros((num_units, num_units), dtype=np.float32)
    edge_rows: list[dict[str, object]] = []
    for (unit_i, unit_j), count in pair_counts.items():
        if count < min_pair_count:
            continue
        weight = count / n_samples
        weighted[unit_i, unit_j] = weight
        weighted[unit_j, unit_i] = weight
        edge_rows.append(
            {
                "unit_i": unit_i,
                "unit_j": unit_j,
                "pair_count": int(count),
                "pair_fraction": float(weight),
                "unit_i_support": float(unit_support[unit_i] / n_samples),
                "unit_j_support": float(unit_support[unit_j] / n_samples),
            }
        )

    pruned = np.zeros_like(weighted)
    for unit in range(num_units):
        row = weighted[unit].copy()
        row[unit] = 0.0
        if graph_topk > 0 and np.count_nonzero(row) > graph_topk:
            keep_idx = np.argpartition(-row, kth=graph_topk - 1)[:graph_topk]
            pruned[unit, keep_idx] = row[keep_idx]
        else:
            pruned[unit] = row

    pruned = np.maximum(pruned, pruned.T)
    np.fill_diagonal(pruned, 1.0)
    degree = pruned.sum(axis=1)
    inv_sqrt = np.where(degree > 0, degree ** -0.5, 0.0).astype(np.float32)
    normalized = (inv_sqrt[:, None] * pruned) * inv_sqrt[None, :]
    return normalized.astype(np.float32), edge_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    fold_dir = args.prototype_root / args.fold_name
    canonical = choose_canonical_validation_subject(fold_dir)
    canonical_subject = str(canonical["validation_subject"])
    subrun_dir = fold_dir / "valsweep_runs" / canonical_subject
    model, sub_metrics, prototype_embeddings = load_canonical_model(subrun_dir, device=device)

    train_index = read_index_csv(args.fold_root / f"{args.fold_name}_train_features_index.csv")
    test_index = read_index_csv(args.fold_root / f"{args.fold_name}_test_features_index.csv")
    train_x = np.load(args.fold_root / f"{args.fold_name}_train_pca512.npy").astype(np.float32)
    test_x = np.load(args.fold_root / f"{args.fold_name}_test_pca512.npy").astype(np.float32)

    image_ids = sorted({int(row["nsdId"]) for row in train_index + test_index})
    class_map = {nsd_id: idx for idx, nsd_id in enumerate(image_ids)}
    train_labels = labels_from_index(train_index, class_map)
    test_labels = labels_from_index(test_index, class_map)

    val_subjects = [str(subject) for subject in sub_metrics["validation_subjects"]]
    val_set = set(val_subjects)
    fit_mask = np.array([row["subject"] not in val_set for row in train_index], dtype=bool)
    val_mask = ~fit_mask

    fit_x = train_x[fit_mask]
    fit_y = train_labels[fit_mask]
    val_x = train_x[val_mask]
    val_y = train_labels[val_mask]
    fit_assignments = forward_assignments(model=model, x=fit_x, device=device, batch_size=args.batch_size)
    val_assignments = forward_assignments(model=model, x=val_x, device=device, batch_size=args.batch_size)
    test_assignments = forward_assignments(model=model, x=test_x, device=device, batch_size=args.batch_size)

    adjacency, edge_rows = build_adjacency(
        assignments=fit_assignments,
        coactivation_topk=args.coactivation_topk,
        graph_topk=args.graph_topk,
        min_pair_count=args.min_pair_count,
    )

    np.savez_compressed(
        args.output_dir / f"{args.fold_name}_light_interaction_inputs.npz",
        fit_assignments=fit_assignments.astype(np.float32),
        val_assignments=val_assignments.astype(np.float32),
        test_assignments=test_assignments.astype(np.float32),
        fit_labels=fit_y.astype(np.int64),
        val_labels=val_y.astype(np.int64),
        test_labels=test_labels.astype(np.int64),
        prototype_embeddings=prototype_embeddings.astype(np.float32),
    )
    np.savez_compressed(
        args.output_dir / f"{args.fold_name}_unit_graph.npz",
        adjacency=adjacency.astype(np.float32),
    )

    edge_path = args.output_dir / f"{args.fold_name}_unit_graph_edges.csv"
    write_csv(
        edge_path,
        edge_rows,
        ["unit_i", "unit_j", "pair_count", "pair_fraction", "unit_i_support", "unit_j_support"],
    )

    metadata = {
        "fold": args.fold_name,
        "held_out_subject": test_index[0]["subject"],
        "canonical_validation_subject": canonical_subject,
        "canonical_best_seed": int(canonical["best_seed"]),
        "canonical_best_val_top5": float(canonical["best_val_top5"]),
        "canonical_best_val_top1": float(canonical["best_val_top1"]),
        "num_units": int(prototype_embeddings.shape[0]),
        "prototype_dim": int(prototype_embeddings.shape[1]),
        "fit_samples": int(fit_assignments.shape[0]),
        "val_samples": int(val_assignments.shape[0]),
        "test_samples": int(test_assignments.shape[0]),
        "coactivation_topk": int(args.coactivation_topk),
        "graph_topk": int(args.graph_topk),
        "min_pair_count": int(args.min_pair_count),
        "nonzero_edges_undirected": int(len(edge_rows)),
        "graph_density_with_self_loops": float(np.count_nonzero(adjacency) / adjacency.size),
    }
    (args.output_dir / f"{args.fold_name}_unit_graph_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    yaml_dump_simple(
        args.output_dir / f"{args.fold_name}_unit_graph_config.yaml",
        {
            "fold": args.fold_name,
            "canonical_validation_subject": canonical_subject,
            "device": str(device),
            "batch_size": args.batch_size,
            "coactivation_topk": args.coactivation_topk,
            "graph_topk": args.graph_topk,
            "min_pair_count": args.min_pair_count,
            "fit_samples": int(fit_assignments.shape[0]),
            "val_samples": int(val_assignments.shape[0]),
            "test_samples": int(test_assignments.shape[0]),
        },
    )


if __name__ == "__main__":
    main()
