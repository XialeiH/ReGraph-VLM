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
    parser = argparse.ArgumentParser(description="Run B4 with validation-subject sweep and aggregate logits.")
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--fold-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inner-script", type=Path, default=None)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seeds", type=str, default="101,102,103")
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inner_script = args.inner_script or (Path(__file__).resolve().parent / "run_npy_baseline_fold.py")

    train_index = read_index_csv(args.fold_root / f"{args.fold_name}_train_features_index.csv")
    test_index = read_index_csv(args.fold_root / f"{args.fold_name}_test_features_index.csv")
    input_dim = int(np.load(args.fold_root / f"{args.fold_name}_test_pca512.npy", mmap_mode="r").shape[1])
    train_subjects = sorted({row["subject"] for row in train_index})
    held_out_subject = test_index[0]["subject"]

    sweep_root = args.output_dir / "valsweep_runs"
    sweep_root.mkdir(parents=True, exist_ok=True)

    per_subject_rows: list[dict[str, object]] = []
    all_learning_rows: list[dict[str, object]] = []
    all_seed_rows: list[dict[str, object]] = []
    logits_list: list[np.ndarray] = []
    hidden_list: list[np.ndarray] = []
    labels: np.ndarray | None = None
    image_ids: list[str] | None = None

    for val_subject in train_subjects:
        subject_out = sweep_root / val_subject
        subject_out.mkdir(parents=True, exist_ok=True)
        metrics_path = subject_out / "b4_metrics.json"
        if not (args.reuse_existing and metrics_path.exists()):
            cmd = [
                sys.executable,
                str(inner_script),
                "--fold-root",
                str(args.fold_root),
                "--fold-name",
                args.fold_name,
                "--baseline",
                "b4",
                "--output-dir",
                str(subject_out),
                "--hidden-dim",
                str(args.hidden_dim),
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--weight-decay",
                str(args.weight_decay),
                "--dropout",
                str(args.dropout),
                "--seeds",
                args.seeds,
                "--val-subjects",
                val_subject,
            ]
            subprocess.run(cmd, check=True)

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        per_subject_rows.append(
            {
                "fold": args.fold_name,
                "validation_subject": val_subject,
                "top1_acc": float(metrics["top1_acc"]),
                "top5_acc": float(metrics["top5_acc"]),
                "n_train_samples": int(metrics["n_train_samples"]),
                "n_val_samples": int(metrics["n_val_samples"]),
                "best_seed": int(metrics["best_seed"]),
            }
        )

        with (subject_out / "b4_learning_curve.csv").open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["validation_subject"] = val_subject
                all_learning_rows.append(row)

        with (subject_out / "b4_seed_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["validation_subject"] = val_subject
                all_seed_rows.append(row)

        subject_logits = np.load(subject_out / "b4_logits.npy")
        logits_list.append(subject_logits)
        hidden_list.append(np.load(subject_out / "b4_hidden.npy"))

        with (subject_out / "b4_predictions.csv").open("r", encoding="utf-8", newline="") as handle:
            predictions = list(csv.DictReader(handle))
        current_labels = np.array([int(row["true_label"]) for row in predictions], dtype=np.int64)
        current_image_ids = [row["nsdId"] for row in predictions]
        if labels is None:
            labels = current_labels
            image_ids = current_image_ids
        else:
            if not np.array_equal(labels, current_labels):
                raise ValueError(f"Inconsistent labels across validation subjects for {args.fold_name}")
            if image_ids != current_image_ids:
                raise ValueError(f"Inconsistent sample ordering across validation subjects for {args.fold_name}")

    assert labels is not None
    assert image_ids is not None

    ensemble_logits = np.mean(np.stack(logits_list, axis=0), axis=0).astype(np.float32)
    ensemble_hidden = np.mean(np.stack(hidden_list, axis=0), axis=0).astype(np.float32)
    top1_preds = ensemble_logits.argmax(axis=1)
    top5_idx = np.argpartition(-ensemble_logits, kth=min(4, ensemble_logits.shape[1] - 1), axis=1)[:, : min(5, ensemble_logits.shape[1])]

    top1_acc = topk_accuracy(ensemble_logits, labels, 1)
    top5_acc = topk_accuracy(ensemble_logits, labels, min(5, ensemble_logits.shape[1]))
    per_subject_top1 = [float(row["top1_acc"]) for row in per_subject_rows]
    per_subject_top5 = [float(row["top5_acc"]) for row in per_subject_rows]

    metrics = {
        "baseline": "b4",
        "selection_mode": "validation_subject_sweep_logits_mean",
        "fold": args.fold_name,
        "held_out_subject": held_out_subject,
        "validation_subjects_swept": train_subjects,
        "n_validation_subjects": len(train_subjects),
        "n_train_samples_per_run": int(per_subject_rows[0]["n_train_samples"]),
        "n_val_samples_per_run": int(per_subject_rows[0]["n_val_samples"]),
        "n_test_samples": int(labels.shape[0]),
        "input_dim": input_dim,
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
    (args.output_dir / "b4_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    yaml_dump_simple(
        args.output_dir / "b4_run_config.yaml",
        {
            "baseline": "b4",
            "fold": args.fold_name,
            "selection_mode": "validation_subject_sweep_logits_mean",
            "validation_subjects_swept": train_subjects,
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "seeds": [int(token) for token in args.seeds.split(",") if token.strip()],
        },
    )

    with (args.output_dir / "b4_valsweep_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["fold", "validation_subject", "top1_acc", "top5_acc", "n_train_samples", "n_val_samples", "best_seed"],
        )
        writer.writeheader()
        writer.writerows(per_subject_rows)

    with (args.output_dir / "b4_learning_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["validation_subject", "seed", "epoch", "train_loss", "train_top1", "val_loss", "val_top1"],
        )
        writer.writeheader()
        writer.writerows(all_learning_rows)

    with (args.output_dir / "b4_seed_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["validation_subject", "seed", "best_epoch", "best_val_top1", "test_top1", "test_top5"],
        )
        writer.writeheader()
        writer.writerows(all_seed_rows)

    with (args.output_dir / "b4_subject_breakdown.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject", "top1_acc", "top5_acc", "n_test_samples", "baseline"])
        writer.writeheader()
        writer.writerow(
            {
                "subject": held_out_subject,
                "top1_acc": f"{top1_acc:.6f}",
                "top5_acc": f"{top5_acc:.6f}",
                "n_test_samples": int(labels.shape[0]),
                "baseline": "b4",
            }
        )

    with (args.output_dir / "b4_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject", "nsdId", "true_label", "pred_label", "top1_correct", "top5_correct"])
        writer.writeheader()
        for idx, row in enumerate(test_index):
            writer.writerow(
                {
                    "subject": row["subject"],
                    "nsdId": row["nsdId"],
                    "true_label": int(labels[idx]),
                    "pred_label": int(top1_preds[idx]),
                    "top1_correct": int(top1_preds[idx] == labels[idx]),
                    "top5_correct": int(labels[idx] in top5_idx[idx]),
                }
            )

    np.save(args.output_dir / "b4_logits.npy", ensemble_logits)
    np.save(args.output_dir / "b4_hidden.npy", ensemble_hidden)


if __name__ == "__main__":
    main()
