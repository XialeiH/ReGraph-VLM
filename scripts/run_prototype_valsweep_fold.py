#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prototype model with validation-subject sweep and aggregate logits.")
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--fold-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inner-script", type=Path, default=None)
    parser.add_argument("--num-prototypes", type=int, default=64)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--align-weight", type=float, default=0.20)
    parser.add_argument("--balance-weight", type=float, default=0.05)
    parser.add_argument("--orth-weight", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--images-per-batch", type=int, default=32)
    parser.add_argument("--subjects-per-image", type=int, default=4)
    parser.add_argument("--lr-encoder", type=float, default=1e-3)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-prototypes", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seeds", type=str, default="11,22,33")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--aggregation-space",
        choices=["probabilities_mean", "logits_mean"],
        default="probabilities_mean",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["best_val_top1", "best_val_top5"],
        default="best_val_top5",
    )
    parser.add_argument("--top-k-validation-subjects", type=int, default=0)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def read_index_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    topk = np.argpartition(-logits, kth=min(k - 1, logits.shape[1] - 1), axis=1)[:, :k]
    return float((topk == labels[:, None]).any(axis=1).mean())


def row_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def mean_assignment_stats(stats_list: list[dict[str, float | bool]]) -> dict[str, float | bool]:
    keys = [key for key in stats_list[0].keys() if key != "collapse_flag"]
    output: dict[str, float | bool] = {}
    for key in keys:
        output[key] = float(np.mean([float(stats[key]) for stats in stats_list]))
    output["collapse_flag"] = any(bool(stats["collapse_flag"]) for stats in stats_list)
    return output


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inner_script = args.inner_script or (Path(__file__).resolve().parent / "run_prototype_fold.py")

    train_index = read_index_csv(args.fold_root / f"{args.fold_name}_train_features_index.csv")
    test_index = read_index_csv(args.fold_root / f"{args.fold_name}_test_features_index.csv")
    train_subjects = sorted({row["subject"] for row in train_index})
    held_out_subject = test_index[0]["subject"]

    sweep_root = args.output_dir / "valsweep_runs"
    sweep_root.mkdir(parents=True, exist_ok=True)

    per_subject_rows: list[dict[str, object]] = []
    all_learning_rows: list[dict[str, object]] = []
    all_seed_rows: list[dict[str, object]] = []
    logits_list: list[np.ndarray] = []
    assignments_list: list[np.ndarray] = []
    train_assignment_stats_list: list[dict[str, float | bool]] = []
    test_assignment_stats_list: list[dict[str, float | bool]] = []
    labels: np.ndarray | None = None
    sample_rows: list[dict[str, str]] | None = None

    for val_subject in train_subjects:
        subject_out = sweep_root / val_subject
        subject_out.mkdir(parents=True, exist_ok=True)
        metrics_path = subject_out / "prototype_metrics.json"
        if not (args.reuse_existing and metrics_path.exists()):
            cmd = [
                sys.executable,
                str(inner_script),
                "--fold-root",
                str(args.fold_root),
                "--fold-name",
                args.fold_name,
                "--output-dir",
                str(subject_out),
                "--num-prototypes",
                str(args.num_prototypes),
                "--input-dim",
                str(args.input_dim),
                "--hidden-dim",
                str(args.hidden_dim),
                "--tau",
                str(args.tau),
                "--align-weight",
                str(args.align_weight),
                "--balance-weight",
                str(args.balance_weight),
                "--orth-weight",
                str(args.orth_weight),
                "--epochs",
                str(args.epochs),
                "--warmup-epochs",
                str(args.warmup_epochs),
                "--patience",
                str(args.patience),
                "--images-per-batch",
                str(args.images_per_batch),
                "--subjects-per-image",
                str(args.subjects_per_image),
                "--lr-encoder",
                str(args.lr_encoder),
                "--lr-head",
                str(args.lr_head),
                "--lr-prototypes",
                str(args.lr_prototypes),
                "--weight-decay",
                str(args.weight_decay),
                "--dropout",
                str(args.dropout),
                "--grad-clip",
                str(args.grad_clip),
                "--seeds",
                args.seeds,
                "--device",
                args.device,
                "--val-subjects",
                val_subject,
            ]
            subprocess.run(
                cmd,
                check=True,
            )

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assign_stats = json.loads((subject_out / "prototype_assignment_stats.json").read_text(encoding="utf-8"))
        per_subject_rows.append(
            {
                "fold": args.fold_name,
                "validation_subject": metrics["validation_subjects"][0],
                "best_val_top1": float(metrics["best_val_top1"]),
                "best_val_top5": float(metrics["best_val_top5"]),
                "top1_acc": float(metrics["top1_acc"]),
                "top5_acc": float(metrics["top5_acc"]),
                "n_train_samples": int(metrics["n_train_samples"]),
                "n_val_samples": int(metrics["n_val_samples"]),
                "best_seed": int(metrics["best_seed"]),
            }
        )
        train_assignment_stats_list.append(assign_stats["train"])
        test_assignment_stats_list.append(assign_stats["test"])

        with (subject_out / "prototype_learning_curve.csv").open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["validation_subject"] = metrics["validation_subjects"][0]
                all_learning_rows.append(row)

        with (subject_out / "prototype_seed_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["validation_subject"] = metrics["validation_subjects"][0]
                all_seed_rows.append(row)

        logits = np.load(subject_out / "prototype_logits.npy")
        assignments = np.load(subject_out / "prototype_assignments.npy")
        logits_list.append(logits)
        assignments_list.append(assignments)

        with (subject_out / "prototype_predictions.csv").open("r", encoding="utf-8", newline="") as handle:
            predictions = list(csv.DictReader(handle))
        current_labels = np.array([int(row["true_label"]) for row in predictions], dtype=np.int64)
        if labels is None:
            labels = current_labels
            sample_rows = predictions
        else:
            if not np.array_equal(labels, current_labels):
                raise ValueError(f"Inconsistent labels across validation subjects for {args.fold_name}")

    assert labels is not None
    assert sample_rows is not None

    ranked_rows = sorted(
        per_subject_rows,
        key=lambda row: (
            float(row[args.selection_metric]),
            float(row["best_val_top5"]),
            float(row["best_val_top1"]),
        ),
        reverse=True,
    )
    if args.top_k_validation_subjects > 0:
        selected_rows = ranked_rows[: min(args.top_k_validation_subjects, len(ranked_rows))]
    else:
        selected_rows = ranked_rows
    selected_subjects = [str(row["validation_subject"]) for row in selected_rows]
    selected_mask = [str(row["validation_subject"]) in set(selected_subjects) for row in per_subject_rows]
    selected_logits = [logits for logits, keep in zip(logits_list, selected_mask) if keep]
    selected_assignments = [assignments for assignments, keep in zip(assignments_list, selected_mask) if keep]
    selected_train_stats = [stats for stats, keep in zip(train_assignment_stats_list, selected_mask) if keep]
    selected_test_stats = [stats for stats, keep in zip(test_assignment_stats_list, selected_mask) if keep]

    if args.aggregation_space == "probabilities_mean":
        ensemble_logits = np.mean([row_softmax(logits) for logits in selected_logits], axis=0).astype(np.float32)
    else:
        ensemble_logits = np.mean(np.stack(selected_logits, axis=0), axis=0).astype(np.float32)
    ensemble_assignments = np.mean(np.stack(selected_assignments, axis=0), axis=0).astype(np.float32)
    top1_preds = ensemble_logits.argmax(axis=1)
    top5_idx = np.argpartition(-ensemble_logits, kth=min(4, ensemble_logits.shape[1] - 1), axis=1)[:, : min(5, ensemble_logits.shape[1])]

    top1_acc = topk_accuracy(ensemble_logits, labels, 1)
    top5_acc = topk_accuracy(ensemble_logits, labels, min(5, ensemble_logits.shape[1]))
    per_subject_top1 = [float(row["top1_acc"]) for row in per_subject_rows]
    per_subject_top5 = [float(row["top5_acc"]) for row in per_subject_rows]

    metrics = {
        "baseline": "prototype",
        "selection_mode": "validation_subject_sweep",
        "aggregation_space": args.aggregation_space,
        "selection_metric": args.selection_metric,
        "fold": args.fold_name,
        "held_out_subject": held_out_subject,
        "validation_subjects_swept": [row["validation_subject"] for row in per_subject_rows],
        "selected_validation_subjects": selected_subjects,
        "n_validation_subjects": len(per_subject_rows),
        "n_selected_validation_subjects": len(selected_subjects),
        "n_train_samples_per_run": int(per_subject_rows[0]["n_train_samples"]),
        "n_val_samples_per_run": int(per_subject_rows[0]["n_val_samples"]),
        "n_test_samples": int(labels.shape[0]),
        "input_dim": int(args.input_dim),
        "hidden_dim": int(args.hidden_dim),
        "num_prototypes": int(args.num_prototypes),
        "num_classes": int(ensemble_logits.shape[1]),
        "chance_level": float(1.0 / ensemble_logits.shape[1]),
        "top1_acc": float(top1_acc),
        "top5_acc": float(top5_acc),
        "per_val_subject_top1_mean": float(np.mean(per_subject_top1)),
        "per_val_subject_top1_std": float(np.std(per_subject_top1)),
        "per_val_subject_top5_mean": float(np.mean(per_subject_top5)),
        "per_val_subject_top5_std": float(np.std(per_subject_top5)),
        "n_seeds": len([token for token in args.seeds.split(",") if token.strip()]),
        "seeds": [int(token) for token in args.seeds.split(",") if token.strip()],
    }
    (args.output_dir / "prototype_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    assignment_stats = {
        "train": mean_assignment_stats(selected_train_stats),
        "test": mean_assignment_stats(selected_test_stats),
    }
    (args.output_dir / "prototype_assignment_stats.json").write_text(json.dumps(assignment_stats, indent=2), encoding="utf-8")

    yaml_dump_simple(
        args.output_dir / "prototype_run_config.yaml",
        {
            "fold": args.fold_name,
            "selection_mode": "validation_subject_sweep",
            "aggregation_space": args.aggregation_space,
            "selection_metric": args.selection_metric,
            "validation_subjects_swept": [row["validation_subject"] for row in per_subject_rows],
            "selected_validation_subjects": selected_subjects,
            "top_k_validation_subjects": args.top_k_validation_subjects,
            "num_prototypes": args.num_prototypes,
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "tau": args.tau,
            "align_weight": args.align_weight,
            "balance_weight": args.balance_weight,
            "orth_weight": args.orth_weight,
            "epochs": args.epochs,
            "warmup_epochs": args.warmup_epochs,
            "patience": args.patience,
            "images_per_batch": args.images_per_batch,
            "subjects_per_image": args.subjects_per_image,
            "lr_encoder": args.lr_encoder,
            "lr_head": args.lr_head,
            "lr_prototypes": args.lr_prototypes,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "grad_clip": args.grad_clip,
            "seeds": [int(token) for token in args.seeds.split(",") if token.strip()],
            "device": args.device,
        },
    )

    with (args.output_dir / "prototype_valsweep_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold",
                "validation_subject",
                "best_val_top1",
                "best_val_top5",
                "top1_acc",
                "top5_acc",
                "n_train_samples",
                "n_val_samples",
                "best_seed",
            ],
        )
        writer.writeheader()
        writer.writerows(per_subject_rows)

    with (args.output_dir / "prototype_learning_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["validation_subject", "seed", "epoch", "train_loss", "train_top1", "val_loss", "val_top1", "val_top5"],
        )
        writer.writeheader()
        writer.writerows(all_learning_rows)

    with (args.output_dir / "prototype_seed_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["validation_subject", "seed", "best_epoch", "best_val_top1", "best_val_top5", "test_top1", "test_top5"],
        )
        writer.writeheader()
        writer.writerows(all_seed_rows)

    with (args.output_dir / "prototype_subject_breakdown.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject", "top1_acc", "top5_acc", "n_test_samples", "baseline"])
        writer.writeheader()
        writer.writerow(
            {
                "subject": held_out_subject,
                "top1_acc": f"{top1_acc:.6f}",
                "top5_acc": f"{top5_acc:.6f}",
                "n_test_samples": int(labels.shape[0]),
                "baseline": "prototype",
            }
        )

    with (args.output_dir / "prototype_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["subject", "nsdId", "true_label", "pred_label", "top1_correct", "top5_correct", "top1_prototype"],
        )
        writer.writeheader()
        top1_prototypes = ensemble_assignments.argmax(axis=1)
        for idx, row in enumerate(sample_rows):
            writer.writerow(
                {
                    "subject": row["subject"],
                    "nsdId": row["nsdId"],
                    "true_label": int(labels[idx]),
                    "pred_label": int(top1_preds[idx]),
                    "top1_correct": int(top1_preds[idx] == labels[idx]),
                    "top5_correct": int(labels[idx] in top5_idx[idx]),
                    "top1_prototype": int(top1_prototypes[idx]),
                }
            )

    np.save(args.output_dir / "prototype_logits.npy", ensemble_logits)
    np.save(args.output_dir / "prototype_assignments.npy", ensemble_assignments)


if __name__ == "__main__":
    main()
