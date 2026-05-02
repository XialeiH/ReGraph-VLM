#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from run_prototype_fold import PrototypeModel, read_index_csv  # noqa: E402


ROI_ORDER = ["V1", "V2", "V3", "hV4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2B ROI attribution and montage index generation.")
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--prototype-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--montage-top-n", type=int, default=12)
    parser.add_argument("--representative-units", type=int, default=10)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_selected_models_for_fold(fold_dir: Path, device: torch.device) -> tuple[list[PrototypeModel], dict[str, object]]:
    metrics = json.loads((fold_dir / "prototype_metrics.json").read_text(encoding="utf-8"))
    selected_validation_subjects = [str(subject) for subject in metrics["selected_validation_subjects"]]
    models: list[PrototypeModel] = []
    model_meta: dict[str, object] = {
        "fold": metrics["fold"],
        "held_out_subject": metrics["held_out_subject"],
        "selected_validation_subjects": selected_validation_subjects,
        "num_prototypes": int(metrics["num_prototypes"]),
    }

    for validation_subject in selected_validation_subjects:
        subrun_dir = fold_dir / "valsweep_runs" / validation_subject
        sub_metrics = json.loads((subrun_dir / "prototype_metrics.json").read_text(encoding="utf-8"))
        config = read_simple_yaml(subrun_dir / "prototype_run_config.yaml")
        best_seed = int(sub_metrics["best_seed"])
        model = PrototypeModel(
            input_dim=int(sub_metrics["input_dim"]),
            hidden_dim=int(sub_metrics["hidden_dim"]),
            num_classes=int(sub_metrics["num_classes"]),
            num_prototypes=int(sub_metrics["num_prototypes"]),
            tau=float(config["tau"]),
            dropout=float(config["dropout"]),
        ).to(device)
        state_dict = torch.load(subrun_dir / f"prototype_seed_{best_seed}_model.pt", map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    return models, model_meta


def forward_assignments(models: list[PrototypeModel], x_pca: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x_pca.shape[0], batch_size):
            xb = torch.from_numpy(x_pca[start : start + batch_size]).to(device)
            assignment_stack = []
            for model in models:
                assignment_stack.append(model(xb)["assignments"].cpu().numpy().astype(np.float32))
            outputs.append(np.mean(np.stack(assignment_stack, axis=0), axis=0).astype(np.float32))
    return np.concatenate(outputs, axis=0)


def load_fold_data(fold_root: Path, fold_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, tuple[int, int]], list[dict[str, object]]]:
    train_norm = np.load(fold_root / f"{fold_name}_train_normalized.npy").astype(np.float32)
    test_norm = np.load(fold_root / f"{fold_name}_test_normalized.npy").astype(np.float32)
    train_pca = np.load(fold_root / f"{fold_name}_train_pca512.npy").astype(np.float32)
    test_pca = np.load(fold_root / f"{fold_name}_test_pca512.npy").astype(np.float32)
    pca = np.load(fold_root / f"{fold_name}_pca512.npz")
    pca_mean = np.asarray(pca["pca_mean"], dtype=np.float32)
    components = np.asarray(pca["components"], dtype=np.float32)
    roi_max_dims = [int(v) for v in pca["roi_max_dims"].tolist()]
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for roi_name, roi_dim in zip(ROI_ORDER, roi_max_dims):
        offsets[roi_name] = (cursor, cursor + roi_dim)
        cursor += roi_dim

    rows: list[dict[str, object]] = []
    for split, index_rows in [
        ("train", read_index_csv(fold_root / f"{fold_name}_train_features_index.csv")),
        ("test", read_index_csv(fold_root / f"{fold_name}_test_features_index.csv")),
    ]:
        for row in index_rows:
            rows.append(
                {
                    "split": split,
                    "subject": row["subject"],
                    "nsdId": int(row["nsdId"]),
                }
            )

    x_norm = np.concatenate([train_norm, test_norm], axis=0)
    x_pca = np.concatenate([train_pca, test_pca], axis=0)
    return x_norm, x_pca, pca_mean, components, offsets, rows


def project_from_normalized(x_norm: np.ndarray, pca_mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return ((x_norm - pca_mean) @ components.T).astype(np.float32)


def choose_representative_units(link_rows: list[dict[str, object]], max_units: int) -> list[tuple[str, int]]:
    by_roi: dict[str, list[dict[str, object]]] = {}
    for row in link_rows:
        if str(row["dominant_roi"]) == "mixed":
            continue
        by_roi.setdefault(str(row["dominant_roi"]), []).append(row)
    selected: list[tuple[str, int]] = []
    for roi_name in ROI_ORDER:
        candidates = sorted(
            by_roi.get(roi_name, []),
            key=lambda r: (float(r["top1_fraction"]), float(r["concentration_score"]), float(r["mean_activation"])),
            reverse=True,
        )
        for row in candidates[:2]:
            selected.append((str(row["fold"]), int(row["unit_id"])))
    if len(selected) < max_units:
        all_rows = sorted(
            link_rows,
            key=lambda r: (float(r["top1_fraction"]), float(r["concentration_score"]), float(r["mean_activation"])),
            reverse=True,
        )
        seen = set(selected)
        for row in all_rows:
            key = (str(row["fold"]), int(row["unit_id"]))
            if key in seen:
                continue
            selected.append(key)
            seen.add(key)
            if len(selected) >= max_units:
                break
    return selected[:max_units]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    ablation_long_rows: list[dict[str, object]] = []
    ablation_summary_rows: list[dict[str, object]] = []
    drop_rows: list[dict[str, object]] = []
    recruitment_link_rows: list[dict[str, object]] = []
    full_assignments_by_fold: dict[str, np.ndarray] = {}
    full_rows_by_fold: dict[str, list[dict[str, object]]] = {}

    fold_dirs = sorted(path for path in args.prototype_root.iterdir() if path.is_dir() and path.name.startswith("fold_"))
    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        models, meta = load_selected_models_for_fold(fold_dir, device=device)
        x_norm, x_pca, pca_mean, components, offsets, sample_rows = load_fold_data(args.fold_root, fold_name)
        full_assign = forward_assignments(models=models, x_pca=x_pca, device=device, batch_size=args.batch_size)
        full_assignments_by_fold[fold_name] = full_assign
        full_rows_by_fold[fold_name] = sample_rows

        roi_only_assignments: dict[str, np.ndarray] = {}
        roi_drop_assignments: dict[str, np.ndarray] = {}
        for roi_name in ROI_ORDER:
            left, right = offsets[roi_name]
            roi_only_norm = np.zeros_like(x_norm)
            roi_only_norm[:, left:right] = x_norm[:, left:right]
            roi_drop_norm = x_norm.copy()
            roi_drop_norm[:, left:right] = 0.0
            roi_only_assignments[roi_name] = forward_assignments(
                models=models,
                x_pca=project_from_normalized(roi_only_norm, pca_mean, components),
                device=device,
                batch_size=args.batch_size,
            )
            roi_drop_assignments[roi_name] = forward_assignments(
                models=models,
                x_pca=project_from_normalized(roi_drop_norm, pca_mean, components),
                device=device,
                batch_size=args.batch_size,
            )

        unit_ratios: dict[int, dict[str, float]] = {}
        mean_activation = full_assign.mean(axis=0)
        top_order = np.argsort(-full_assign, axis=1)
        top1 = top_order[:, 0]
        top3 = top_order[:, :3]
        for unit_id in range(full_assign.shape[1]):
            ratios: dict[str, float] = {}
            full_mean = float(mean_activation[unit_id])
            for roi_name in ROI_ORDER:
                roi_only_mean = float(roi_only_assignments[roi_name][:, unit_id].mean())
                retained_ratio = roi_only_mean / max(full_mean, 1e-12)
                ratios[roi_name] = retained_ratio
            unit_ratios[unit_id] = ratios

            dominant_roi = max(ratios, key=ratios.get)
            total_ratio = sum(max(v, 0.0) for v in ratios.values())
            concentration = ratios[dominant_roi] / max(total_ratio, 1e-12)
            if concentration < 0.45:
                dominant_label = "mixed"
            else:
                dominant_label = dominant_roi

            recruitment_link_rows.append(
                {
                    "fold": fold_name,
                    "held_out_subject": meta["held_out_subject"],
                    "unit_id": unit_id,
                    "dominant_roi": dominant_label,
                    "concentration_score": float(concentration),
                    "mean_activation": full_mean,
                    "top1_count": int(np.sum(top1 == unit_id)),
                    "top1_fraction": float(np.mean(top1 == unit_id)),
                    "top3_count": int(np.sum(top3 == unit_id)),
                    "top3_fraction": float(np.mean(top3 == unit_id)),
                }
            )

            summary_row: dict[str, object] = {
                "fold": fold_name,
                "held_out_subject": meta["held_out_subject"],
                "unit_id": unit_id,
                "dominant_roi": dominant_label,
                "concentration_score": float(concentration),
                "mean_full_activation": full_mean,
            }
            for roi_name in ROI_ORDER:
                roi_only_mean = float(roi_only_assignments[roi_name][:, unit_id].mean())
                retained_ratio = ratios[roi_name]
                summary_row[f"mean_roi_only_activation_{roi_name}"] = roi_only_mean
                summary_row[f"retained_ratio_{roi_name}"] = retained_ratio
                ablation_long_rows.append(
                    {
                        "fold": fold_name,
                        "held_out_subject": meta["held_out_subject"],
                        "unit_id": unit_id,
                        "roi_name": roi_name,
                        "mean_full_activation": full_mean,
                        "mean_roi_only_activation": roi_only_mean,
                        "retained_ratio": retained_ratio,
                        "dominant_or_not": int(roi_name == dominant_roi and dominant_label != "mixed"),
                        "dominant_label": dominant_label,
                        "concentration_score": float(concentration),
                    }
                )
                roi_drop_mean = float(roi_drop_assignments[roi_name][:, unit_id].mean())
                retained_after_drop = roi_drop_mean / max(full_mean, 1e-12)
                drop_rows.append(
                    {
                        "fold": fold_name,
                        "held_out_subject": meta["held_out_subject"],
                        "unit_id": unit_id,
                        "roi_name": roi_name,
                        "mean_full_activation": full_mean,
                        "mean_roi_drop_activation": roi_drop_mean,
                        "retained_ratio_after_drop": retained_after_drop,
                        "drop_ratio": 1.0 - retained_after_drop,
                        "necessary_or_not": int((1.0 - retained_after_drop) == max(1.0 - (float(roi_drop_assignments[r][:, unit_id].mean()) / max(full_mean, 1e-12)) for r in ROI_ORDER)),
                    }
                )
            ablation_summary_rows.append(summary_row)

    write_csv(
        args.output_dir / "unit_roi_ablation_foldwise.csv",
        ablation_long_rows,
        [
            "fold",
            "held_out_subject",
            "unit_id",
            "roi_name",
            "mean_full_activation",
            "mean_roi_only_activation",
            "retained_ratio",
            "dominant_or_not",
            "dominant_label",
            "concentration_score",
        ],
    )
    write_csv(
        args.output_dir / "unit_roi_ablation_summary.csv",
        ablation_summary_rows,
        [
            "fold",
            "held_out_subject",
            "unit_id",
            "dominant_roi",
            "concentration_score",
            "mean_full_activation",
            "mean_roi_only_activation_V1",
            "retained_ratio_V1",
            "mean_roi_only_activation_V2",
            "retained_ratio_V2",
            "mean_roi_only_activation_V3",
            "retained_ratio_V3",
            "mean_roi_only_activation_hV4",
            "retained_ratio_hV4",
        ],
    )
    write_csv(
        args.output_dir / "unit_roi_drop_summary.csv",
        drop_rows,
        [
            "fold",
            "held_out_subject",
            "unit_id",
            "roi_name",
            "mean_full_activation",
            "mean_roi_drop_activation",
            "retained_ratio_after_drop",
            "drop_ratio",
            "necessary_or_not",
        ],
    )
    write_csv(
        args.output_dir / "unit_roi_recruitment_link.csv",
        recruitment_link_rows,
        [
            "fold",
            "held_out_subject",
            "unit_id",
            "dominant_roi",
            "concentration_score",
            "mean_activation",
            "top1_count",
            "top1_fraction",
            "top3_count",
            "top3_fraction",
        ],
    )

    selected_units = choose_representative_units(recruitment_link_rows, max_units=args.representative_units)
    montage_rows: list[dict[str, object]] = []
    link_lookup = {(str(row["fold"]), int(row["unit_id"])): row for row in recruitment_link_rows}
    for fold_name, unit_id in selected_units:
        assignments = full_assignments_by_fold[fold_name][:, unit_id]
        sample_rows = full_rows_by_fold[fold_name]
        top_idx = np.argsort(-assignments)[: args.montage_top_n]
        meta = link_lookup[(fold_name, unit_id)]
        for rank, idx in enumerate(top_idx.tolist(), start=1):
            sample = sample_rows[idx]
            montage_rows.append(
                {
                    "fold": fold_name,
                    "unit_id": unit_id,
                    "rank": rank,
                    "nsdId": int(sample["nsdId"]),
                    "subject": str(sample["subject"]),
                    "split": str(sample["split"]),
                    "activation": float(assignments[idx]),
                    "roi_dominance_label": str(meta["dominant_roi"]),
                    "concentration_score": float(meta["concentration_score"]),
                }
            )
    write_csv(
        args.output_dir / "unit_montage_index.csv",
        montage_rows,
        ["fold", "unit_id", "rank", "nsdId", "subject", "split", "activation", "roi_dominance_label", "concentration_score"],
    )


if __name__ == "__main__":
    main()
