#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROI_ORDER = ["V1", "V2", "V3", "hV4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate train-only normalization and shared PCA artifacts for one LOSO fold.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--fold-name", type=str, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-view", type=str, required=True)
    parser.add_argument("--pca-dim", type=int, default=512)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_split(path: Path, fold_name: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for fold in payload["folds"]:
        if fold["fold_name"] == fold_name:
            return fold
    raise ValueError(f"Fold {fold_name} not found in {path}")


def canonical_roi_dims(manifest_rows: list[dict[str, str]]) -> dict[str, int]:
    dims = {roi: 0 for roi in ROI_ORDER}
    feature_paths = sorted({row["feature_path"] for row in manifest_rows})
    for feature_path in feature_paths:
        npz = np.load(feature_path)
        roi_names = [str(name) for name in npz["roi_names"].tolist()]
        roi_dims = [int(value) for value in npz["roi_dims"].tolist()]
        dim_map = dict(zip(roi_names, roi_dims))
        for roi in ROI_ORDER:
            dims[roi] = max(dims[roi], dim_map[roi])
    return dims


def canonical_offsets(roi_dims: dict[str, int]) -> dict[str, tuple[int, int]]:
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for roi in ROI_ORDER:
        offsets[roi] = (cursor, cursor + roi_dims[roi])
        cursor += roi_dims[roi]
    return offsets


def build_matrix(rows: list[dict[str, str]], roi_dims: dict[str, int]) -> tuple[np.ndarray, list[dict[str, object]]]:
    offsets = canonical_offsets(roi_dims)
    total_dim = sum(roi_dims.values())
    feature_cache: dict[str, dict[str, object]] = {}
    matrix = np.zeros((len(rows), total_dim), dtype=np.float32)
    index_rows: list[dict[str, object]] = []

    for row_idx, row in enumerate(rows):
        feature_path = row["feature_path"]
        sample_index = int(row["sample_index"])
        if feature_path not in feature_cache:
            loaded = np.load(feature_path)
            feature_cache[feature_path] = {
                "concatenated": np.asarray(loaded["concatenated"], dtype=np.float32),
                "roi_names": [str(name) for name in loaded["roi_names"].tolist()],
                "roi_dims": [int(value) for value in loaded["roi_dims"].tolist()],
            }
        npz = feature_cache[feature_path]
        source_offset = 0
        for roi_name, roi_dim in zip(npz["roi_names"], npz["roi_dims"]):
            roi_values = npz["concatenated"][sample_index, source_offset : source_offset + roi_dim]
            left, _ = offsets[roi_name]
            matrix[row_idx, left : left + roi_dim] = roi_values
            source_offset += roi_dim
        index_rows.append(
            {
                "subject": row["subject"],
                "nsdId": int(row["nsdId"]),
                "source_feature_path": feature_path,
                "sample_index": sample_index,
            }
        )
    return matrix, index_rows


def fit_pca(train_matrix: np.ndarray, pca_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_matrix.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std == 0] = 1.0

    train_norm = (train_matrix - mean) / std
    pca_mean = train_norm.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = train_norm - pca_mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:pca_dim].astype(np.float32)
    explained_variance = ((singular_values[:pca_dim] ** 2) / max(centered.shape[0] - 1, 1)).astype(np.float32)
    full_explained = ((singular_values**2) / max(centered.shape[0] - 1, 1)).astype(np.float64)
    explained_ratio = (full_explained[:pca_dim] / full_explained.sum()).astype(np.float32)
    return mean, std, pca_mean, components, explained_variance, explained_ratio


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_csv_rows(args.manifest)
    split = load_split(args.splits, args.fold_name)
    train_subjects = set(split["train_subjects"])
    test_subjects = set(split["test_subjects"])

    train_rows = [row for row in manifest_rows if row["subject"] in train_subjects]
    test_rows = [row for row in manifest_rows if row["subject"] in test_subjects]

    roi_dims = canonical_roi_dims(manifest_rows)
    train_matrix, train_index_rows = build_matrix(train_rows, roi_dims)
    test_matrix, test_index_rows = build_matrix(test_rows, roi_dims)

    effective_pca_dim = min(args.pca_dim, train_matrix.shape[0], train_matrix.shape[1])
    mean, std, pca_mean, components, explained_variance, explained_ratio = fit_pca(train_matrix, effective_pca_dim)

    train_norm = ((train_matrix - mean) / std).astype(np.float32)
    test_norm = ((test_matrix - mean) / std).astype(np.float32)
    train_pca = ((train_norm - pca_mean) @ components.T).astype(np.float32)
    test_pca = ((test_norm - pca_mean) @ components.T).astype(np.float32)

    fold_prefix = args.output_root / args.fold_name
    np.savez(
        fold_prefix.with_name(f"{args.fold_name}_train_norm_stats.npz"),
        dataset_view=np.array(args.dataset_view),
        fold_name=np.array(args.fold_name),
        roi_order=np.array(ROI_ORDER),
        roi_max_dims=np.array([roi_dims[roi] for roi in ROI_ORDER], dtype=np.int32),
        mean=mean,
        std=std,
    )
    np.savez(
        fold_prefix.with_name(f"{args.fold_name}_pca512.npz"),
        dataset_view=np.array(args.dataset_view),
        fold_name=np.array(args.fold_name),
        pca_dim=np.array(effective_pca_dim, dtype=np.int32),
        roi_order=np.array(ROI_ORDER),
        roi_max_dims=np.array([roi_dims[roi] for roi in ROI_ORDER], dtype=np.int32),
        normalization_mean=mean,
        normalization_std=std,
        pca_mean=pca_mean,
        components=components,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_ratio,
    )
    np.save(fold_prefix.with_name(f"{args.fold_name}_train_normalized.npy"), train_norm)
    np.save(fold_prefix.with_name(f"{args.fold_name}_test_normalized.npy"), test_norm)
    np.save(fold_prefix.with_name(f"{args.fold_name}_train_pca512.npy"), train_pca)
    np.save(fold_prefix.with_name(f"{args.fold_name}_test_pca512.npy"), test_pca)

    train_norm_path = str(fold_prefix.with_name(f"{args.fold_name}_train_normalized.npy").resolve())
    test_norm_path = str(fold_prefix.with_name(f"{args.fold_name}_test_normalized.npy").resolve())
    train_pca_path = str(fold_prefix.with_name(f"{args.fold_name}_train_pca512.npy").resolve())
    test_pca_path = str(fold_prefix.with_name(f"{args.fold_name}_test_pca512.npy").resolve())

    for row_idx, row in enumerate(train_index_rows):
        row["normalized_feature_path"] = train_norm_path
        row["normalized_row_index"] = row_idx
        row["pca_feature_path"] = train_pca_path
        row["pca_row_index"] = row_idx
    for row_idx, row in enumerate(test_index_rows):
        row["normalized_feature_path"] = test_norm_path
        row["normalized_row_index"] = row_idx
        row["pca_feature_path"] = test_pca_path
        row["pca_row_index"] = row_idx

    for path, rows in [
        (fold_prefix.with_name(f"{args.fold_name}_train_features_index.csv"), train_index_rows),
        (fold_prefix.with_name(f"{args.fold_name}_test_features_index.csv"), test_index_rows),
    ]:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "subject",
                    "nsdId",
                    "source_feature_path",
                    "sample_index",
                    "normalized_feature_path",
                    "normalized_row_index",
                    "pca_feature_path",
                    "pca_row_index",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    held_out_subject = sorted(test_subjects)[0]
    diagnostics = {
        "dataset_view": args.dataset_view,
        "fold_name": args.fold_name,
        "held_out_subject": held_out_subject,
        "train_subjects": sorted(train_subjects),
        "test_subjects": sorted(test_subjects),
        "n_train_samples": int(train_matrix.shape[0]),
        "n_test_samples": int(test_matrix.shape[0]),
        "canonical_input_dim": int(train_matrix.shape[1]),
        "roi_max_dims": {roi: int(roi_dims[roi]) for roi in ROI_ORDER},
        "pca_dim_requested": int(args.pca_dim),
        "pca_dim_effective": int(effective_pca_dim),
        "explained_variance_ratio_sum": float(explained_ratio.sum()),
        "train_nan_count": int(np.isnan(train_pca).sum()),
        "test_nan_count": int(np.isnan(test_pca).sum()),
        "train_inf_count": int(np.isinf(train_pca).sum()),
        "test_inf_count": int(np.isinf(test_pca).sum()),
    }
    (fold_prefix.with_name(f"{args.fold_name}_pca_diagnostics.json")).write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    note_lines = [
        "# Normalization and PCA Note",
        "",
        f"- Primary dataset view: `{args.dataset_view}`.",
        "- Normalization is training-only feature-wise z-score on a canonical ROI-block-padded concatenated feature space.",
        f"- Canonical ROI blocks are formed by padding each ROI to the maximum ROI dimension observed across subjects in `{args.dataset_view}`.",
        "- PCA is fit only on normalized training-subject samples and the same shared basis is used to project both train and held-out test subject samples.",
        f"- This run generated `{args.fold_name}` with held-out subject `{held_out_subject}` and effective PCA dimension `{effective_pca_dim}`.",
        f"- Diagnostics summary: explained variance ratio sum = `{explained_ratio.sum():.4f}`, train NaN = `{int(np.isnan(train_pca).sum())}`, test NaN = `{int(np.isnan(test_pca).sum())}`.",
        "",
    ]
    (args.output_root / "normalization_pca_note.md").write_text("\n".join(note_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
