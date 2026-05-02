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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze shared-unit recruitment patterns from saved prototype models.")
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--prototype-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--coactivation-topk", type=int, default=3)
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


def forward_assignments(models: list[PrototypeModel], x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            assignment_stack = []
            for model in models:
                assignment_stack.append(model(xb)["assignments"].cpu().numpy().astype(np.float32))
            outputs.append(np.mean(np.stack(assignment_stack, axis=0), axis=0).astype(np.float32))
    return np.concatenate(outputs, axis=0)


def build_full_fold_matrix(fold_root: Path, fold_name: str) -> tuple[np.ndarray, list[dict[str, object]]]:
    train_index = read_index_csv(fold_root / f"{fold_name}_train_features_index.csv")
    test_index = read_index_csv(fold_root / f"{fold_name}_test_features_index.csv")
    train_x = np.load(fold_root / f"{fold_name}_train_pca512.npy").astype(np.float32)
    test_x = np.load(fold_root / f"{fold_name}_test_pca512.npy").astype(np.float32)

    rows: list[dict[str, object]] = []
    for split, index_rows in [("train", train_index), ("test", test_index)]:
        for row in index_rows:
            rows.append(
                {
                    "split": split,
                    "subject": row["subject"],
                    "nsdId": int(row["nsdId"]),
                }
            )
    x = np.concatenate([train_x, test_x], axis=0)
    return x, rows


def jaccard_overlap(a: np.ndarray, b: np.ndarray) -> float:
    sa = set(int(v) for v in a.tolist())
    sb = set(int(v) for v in b.tolist())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    recruitment_rows: list[dict[str, object]] = []
    coactivation_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []

    fold_dirs = sorted(path for path in args.prototype_root.iterdir() if path.is_dir() and path.name.startswith("fold_"))
    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        models, meta = load_selected_models_for_fold(fold_dir, device=device)
        full_x, full_rows = build_full_fold_matrix(args.fold_root, fold_name)
        assignments = forward_assignments(models=models, x=full_x, device=device, batch_size=args.batch_size)

        num_prototypes = int(meta["num_prototypes"])
        top_order = np.argsort(-assignments, axis=1)
        top1 = top_order[:, 0]
        top3 = top_order[:, :3]
        top5 = top_order[:, :5]
        entropy = -(assignments * np.log(np.clip(assignments, 1e-12, None))).sum(axis=1)
        max_activation = assignments.max(axis=1)
        active_ge_uniform = (assignments >= (1.0 / num_prototypes)).sum(axis=1)
        active_ge_005 = (assignments >= 0.05).sum(axis=1)
        top3_mass = np.take_along_axis(assignments, top3, axis=1).sum(axis=1)
        top5_mass = np.take_along_axis(assignments, top5, axis=1).sum(axis=1)

        overlap_top1 = []
        overlap_top3 = []
        overlap_top5 = []
        by_image: dict[int, list[tuple[str, int]]] = {}
        for idx, row in enumerate(full_rows):
            by_image.setdefault(int(row["nsdId"]), []).append((str(row["subject"]), idx))

        for nsd_id, samples in by_image.items():
            samples = sorted(samples)
            for i in range(len(samples)):
                for j in range(i + 1, len(samples)):
                    subject_a, idx_a = samples[i]
                    subject_b, idx_b = samples[j]
                    top1_match = int(top1[idx_a] == top1[idx_b])
                    top3_j = jaccard_overlap(top3[idx_a], top3[idx_b])
                    top5_j = jaccard_overlap(top5[idx_a], top5[idx_b])
                    overlap_top1.append(top1_match)
                    overlap_top3.append(top3_j)
                    overlap_top5.append(top5_j)
                    overlap_rows.append(
                        {
                            "fold": fold_name,
                            "held_out_subject": meta["held_out_subject"],
                            "nsdId": int(nsd_id),
                            "subject_a": subject_a,
                            "subject_b": subject_b,
                            "top1_match": top1_match,
                            "top3_jaccard": float(top3_j),
                            "top5_jaccard": float(top5_j),
                        }
                    )

        recruitment_rows.append(
            {
                "fold": fold_name,
                "held_out_subject": meta["held_out_subject"],
                "selected_validation_subjects": ";".join(meta["selected_validation_subjects"]),
                "n_samples": int(assignments.shape[0]),
                "mean_entropy": float(np.mean(entropy)),
                "std_entropy": float(np.std(entropy)),
                "mean_max_activation": float(np.mean(max_activation)),
                "mean_active_units_ge_uniform": float(np.mean(active_ge_uniform)),
                "mean_active_units_ge_0p05": float(np.mean(active_ge_005)),
                "mean_top3_mass": float(np.mean(top3_mass)),
                "mean_top5_mass": float(np.mean(top5_mass)),
                "same_image_top1_match_rate": float(np.mean(overlap_top1)) if overlap_top1 else 0.0,
                "same_image_top3_jaccard_mean": float(np.mean(overlap_top3)) if overlap_top3 else 0.0,
                "same_image_top5_jaccard_mean": float(np.mean(overlap_top5)) if overlap_top5 else 0.0,
            }
        )

        unit_support = np.zeros(num_prototypes, dtype=np.int64)
        pair_counts: dict[tuple[int, int], int] = {}
        for row in top_order[:, : args.coactivation_topk]:
            present = sorted(int(v) for v in row.tolist())
            for unit in present:
                unit_support[unit] += 1
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    key = (present[i], present[j])
                    pair_counts[key] = pair_counts.get(key, 0) + 1

        n_samples = assignments.shape[0]
        for (unit_i, unit_j), count in sorted(pair_counts.items()):
            support_i = unit_support[unit_i] / n_samples
            support_j = unit_support[unit_j] / n_samples
            pair_fraction = count / n_samples
            denom = max(support_i * support_j, 1e-12)
            lift = pair_fraction / denom
            coactivation_rows.append(
                {
                    "fold": fold_name,
                    "held_out_subject": meta["held_out_subject"],
                    "unit_i": unit_i,
                    "unit_j": unit_j,
                    "topk": args.coactivation_topk,
                    "pair_count": int(count),
                    "pair_fraction": float(pair_fraction),
                    "unit_i_support": float(support_i),
                    "unit_j_support": float(support_j),
                    "lift": float(lift),
                    "pmi": float(math.log(max(pair_fraction, 1e-12) / denom)),
                }
            )

    write_csv(
        args.output_dir / "unit_recruitment_summary.csv",
        recruitment_rows,
        [
            "fold",
            "held_out_subject",
            "selected_validation_subjects",
            "n_samples",
            "mean_entropy",
            "std_entropy",
            "mean_max_activation",
            "mean_active_units_ge_uniform",
            "mean_active_units_ge_0p05",
            "mean_top3_mass",
            "mean_top5_mass",
            "same_image_top1_match_rate",
            "same_image_top3_jaccard_mean",
            "same_image_top5_jaccard_mean",
        ],
    )
    write_csv(
        args.output_dir / "unit_coactivation_summary.csv",
        coactivation_rows,
        [
            "fold",
            "held_out_subject",
            "unit_i",
            "unit_j",
            "topk",
            "pair_count",
            "pair_fraction",
            "unit_i_support",
            "unit_j_support",
            "lift",
            "pmi",
        ],
    )
    write_csv(
        args.output_dir / "topk_unit_overlap_across_subjects.csv",
        overlap_rows,
        ["fold", "held_out_subject", "nsdId", "subject_a", "subject_b", "top1_match", "top3_jaccard", "top5_jaccard"],
    )


if __name__ == "__main__":
    main()
