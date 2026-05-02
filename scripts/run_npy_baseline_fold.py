#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run B1 or B4 baseline on one all8_ge2_766 fold using PCA features.")
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--fold-name", type=str, required=True)
    parser.add_argument("--baseline", choices=["b1", "b4"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=str, default="101")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ridge-alphas", type=str, default="0.01,0.1,1.0,10.0")
    parser.add_argument("--val-subject-count", type=int, default=1)
    parser.add_argument("--val-subject-strategy", choices=["fixed_last", "rotating"], default="fixed_last")
    parser.add_argument("--val-subjects", type=str, default="")
    return parser.parse_args()


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted, dtype=np.float64).astype(np.float32)
    return exp / exp.sum(axis=1, keepdims=True)


def topk_accuracy(logits: np.ndarray, labels: np.ndarray, k: int) -> float:
    topk = np.argpartition(-logits, kth=min(k - 1, logits.shape[1] - 1), axis=1)[:, :k]
    return float((topk == labels[:, None]).any(axis=1).mean())


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


def init_mlp_params(rng: np.random.Generator, input_dim: int, num_classes: int, hidden_dim: int) -> dict[str, np.ndarray]:
    scale1 = np.sqrt(2.0 / input_dim)
    scale2 = np.sqrt(2.0 / hidden_dim)
    return {
        "W1": (rng.standard_normal((input_dim, hidden_dim), dtype=np.float32) * scale1).astype(np.float32),
        "b1": np.zeros(hidden_dim, dtype=np.float32),
        "W2": (rng.standard_normal((hidden_dim, num_classes), dtype=np.float32) * scale2).astype(np.float32),
        "b2": np.zeros(num_classes, dtype=np.float32),
    }


def mlp_forward(params: dict[str, np.ndarray], x: np.ndarray, dropout: float, rng: np.random.Generator | None, train: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    hidden_pre = x @ params["W1"] + params["b1"]
    hidden = relu(hidden_pre)
    if train and dropout > 0:
        assert rng is not None
        mask = (rng.random(hidden.shape, dtype=np.float32) >= dropout).astype(np.float32) / (1.0 - dropout)
    else:
        mask = np.ones_like(hidden, dtype=np.float32)
    hidden_drop = hidden * mask
    logits = hidden_drop @ params["W2"] + params["b2"]
    return logits, {"x": x, "hidden_pre": hidden_pre, "mask": mask, "hidden_drop": hidden_drop}


def mlp_loss_and_grads(
    params: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
    logits: np.ndarray,
    labels: np.ndarray,
    weight_decay: float,
) -> tuple[float, dict[str, np.ndarray]]:
    probs = softmax(logits)
    n = labels.shape[0]
    loss = float(-np.log(np.clip(probs[np.arange(n), labels], 1e-12, 1.0)).mean())
    grad_logits = probs
    grad_logits[np.arange(n), labels] -= 1.0
    grad_logits /= n

    grads: dict[str, np.ndarray] = {}
    grads["W2"] = cache["hidden_drop"].T @ grad_logits + weight_decay * params["W2"]
    grads["b2"] = grad_logits.sum(axis=0)
    grad_hidden = grad_logits @ params["W2"].T
    grad_hidden *= cache["mask"]
    grad_hidden[cache["hidden_pre"] <= 0] = 0.0
    grads["W1"] = cache["x"].T @ grad_hidden + weight_decay * params["W1"]
    grads["b1"] = grad_hidden.sum(axis=0)
    loss += 0.5 * weight_decay * float((params["W1"] ** 2).sum() + (params["W2"] ** 2).sum())
    return loss, grads


def adam_step(params: dict[str, np.ndarray], grads: dict[str, np.ndarray], opt_state: dict[str, object], lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
    opt_state["t"] += 1
    t = int(opt_state["t"])
    for name, grad in grads.items():
        m = opt_state["m"].setdefault(name, np.zeros_like(grad, dtype=np.float32))
        v = opt_state["v"].setdefault(name, np.zeros_like(grad, dtype=np.float32))
        m[:] = beta1 * m + (1.0 - beta1) * grad
        v[:] = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        params[name] -= lr * m_hat / (np.sqrt(v_hat) + eps)


def evaluate_mlp(params: dict[str, np.ndarray], x: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    logits, cache = mlp_forward(params, x, dropout=0.0, rng=None, train=False)
    probs = softmax(logits)
    loss = float(-np.log(np.clip(probs[np.arange(labels.shape[0]), labels], 1e-12, 1.0)).mean())
    return {
        "loss": loss,
        "top1": topk_accuracy(logits, labels, 1),
        "top5": topk_accuracy(logits, labels, min(5, logits.shape[1])),
        "logits": logits.astype(np.float32),
        "hidden": cache["hidden_drop"].astype(np.float32),
    }


def train_one_seed(
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    hidden_dim: int,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    params = init_mlp_params(rng, x_train.shape[1], int(y_train.max()) + 1, hidden_dim)
    best_params = {k: v.copy() for k, v in params.items()}
    best_val = -1.0
    best_epoch = -1
    bad_epochs = 0
    learning_curve: list[dict[str, object]] = []
    opt_state: dict[str, object] = {"m": {}, "v": {}, "t": 0}

    for epoch in range(1, epochs + 1):
        order = rng.permutation(x_train.shape[0])
        for start in range(0, x_train.shape[0], batch_size):
            batch_idx = order[start : start + batch_size]
            xb = x_train[batch_idx]
            yb = y_train[batch_idx]
            logits, cache = mlp_forward(params, xb, dropout=dropout, rng=rng, train=True)
            _, grads = mlp_loss_and_grads(params, cache, logits, yb, weight_decay)
            adam_step(params, grads, opt_state, lr)

        train_eval = evaluate_mlp(params, x_train, y_train)
        val_eval = evaluate_mlp(params, x_val, y_val)
        learning_curve.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_loss": float(train_eval["loss"]),
                "train_top1": float(train_eval["top1"]),
                "val_loss": float(val_eval["loss"]),
                "val_top1": float(val_eval["top1"]),
            }
        )

        if float(val_eval["top1"]) > best_val:
            best_val = float(val_eval["top1"])
            best_epoch = epoch
            best_params = {k: v.copy() for k, v in params.items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    test_eval = evaluate_mlp(best_params, x_test, y_test)
    val_eval = evaluate_mlp(best_params, x_val, y_val)
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_top1": float(best_val),
        "val_top1": float(val_eval["top1"]),
        "val_top5": float(val_eval["top5"]),
        "test_top1": float(test_eval["top1"]),
        "test_top5": float(test_eval["top5"]),
        "test_logits": test_eval["logits"],
        "test_hidden": test_eval["hidden"],
        "best_params": {k: v.copy() for k, v in best_params.items()},
        "learning_curve": learning_curve,
    }


def fit_ridge_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    ridge_alphas: list[float],
) -> dict[str, object]:
    num_classes = int(y_train.max()) + 1
    y_onehot = np.zeros((y_train.shape[0], num_classes), dtype=np.float32)
    y_onehot[np.arange(y_train.shape[0]), y_train] = 1.0

    x_train_aug = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=np.float32)], axis=1)
    x_val_aug = np.concatenate([x_val, np.ones((x_val.shape[0], 1), dtype=np.float32)], axis=1)
    x_test_aug = np.concatenate([x_test, np.ones((x_test.shape[0], 1), dtype=np.float32)], axis=1)

    gram = x_train_aug.T @ x_train_aug
    xty = x_train_aug.T @ y_onehot
    eye = np.eye(gram.shape[0], dtype=np.float32)

    best: dict[str, object] | None = None
    learning_curve: list[dict[str, object]] = []
    for idx, alpha in enumerate(ridge_alphas, start=1):
        weights = np.linalg.solve(gram + alpha * eye, xty).astype(np.float32)
        train_logits = x_train_aug @ weights
        val_logits = x_val_aug @ weights
        learning_curve.append(
            {
                "seed": idx,
                "epoch": idx,
                "train_loss": 0.0,
                "train_top1": float(topk_accuracy(train_logits, y_train, 1)),
                "val_loss": 0.0,
                "val_top1": float(topk_accuracy(val_logits, y_val, 1)),
            }
        )
        candidate = {
            "alpha": alpha,
            "weights": weights,
            "val_top1": float(topk_accuracy(val_logits, y_val, 1)),
            "val_top5": float(topk_accuracy(val_logits, y_val, min(5, val_logits.shape[1]))),
        }
        if best is None or float(candidate["val_top1"]) > float(best["val_top1"]):
            best = candidate

    assert best is not None
    test_logits = (x_test_aug @ best["weights"]).astype(np.float32)
    return {
        "seed": 0,
        "best_epoch": 1,
        "best_val_top1": float(best["val_top1"]),
        "val_top1": float(best["val_top1"]),
        "val_top5": float(best["val_top5"]),
        "test_top1": float(topk_accuracy(test_logits, y_test, 1)),
        "test_top5": float(topk_accuracy(test_logits, y_test, min(5, test_logits.shape[1]))),
        "test_logits": test_logits,
        "learning_curve": learning_curve,
        "best_alpha": float(best["alpha"]),
    }


def select_validation_subjects(
    train_subjects: list[str], fold_name: str, strategy: str, count: int, explicit_subjects: list[str]
) -> list[str]:
    if explicit_subjects:
        invalid = [subject for subject in explicit_subjects if subject not in train_subjects]
        if invalid:
            raise ValueError(f"Explicit validation subjects not in training pool: {invalid}")
        if len(set(explicit_subjects)) != len(explicit_subjects):
            raise ValueError("Explicit validation subjects must be unique")
        return explicit_subjects

    if count < 1:
        raise ValueError("val-subject-count must be at least 1")
    if count >= len(train_subjects):
        raise ValueError("val-subject-count must be smaller than the number of training subjects")

    if strategy == "fixed_last":
        return train_subjects[-count:]

    fold_num = int(fold_name.split("_")[-1])
    start = (fold_num - 1) % len(train_subjects)
    ordered = train_subjects[start:] + train_subjects[:start]
    return ordered[:count]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_index = read_index_csv(args.fold_root / f"{args.fold_name}_train_features_index.csv")
    test_index = read_index_csv(args.fold_root / f"{args.fold_name}_test_features_index.csv")
    train_features = np.load(args.fold_root / f"{args.fold_name}_train_pca512.npy").astype(np.float32)
    test_features = np.load(args.fold_root / f"{args.fold_name}_test_pca512.npy").astype(np.float32)

    image_ids = sorted({int(row["nsdId"]) for row in train_index + test_index})
    class_map = {nsd_id: idx for idx, nsd_id in enumerate(image_ids)}
    train_labels = np.array([class_map[int(row["nsdId"])] for row in train_index], dtype=np.int64)
    test_labels = np.array([class_map[int(row["nsdId"])] for row in test_index], dtype=np.int64)

    train_subjects = sorted({row["subject"] for row in train_index})
    explicit_val_subjects = [token.strip() for token in args.val_subjects.split(",") if token.strip()]
    val_subjects = select_validation_subjects(
        train_subjects=train_subjects,
        fold_name=args.fold_name,
        strategy=args.val_subject_strategy,
        count=args.val_subject_count,
        explicit_subjects=explicit_val_subjects,
    )
    val_subject_set = set(val_subjects)
    train_mask = np.array([row["subject"] not in val_subject_set for row in train_index], dtype=bool)
    val_mask = ~train_mask

    x_fit = train_features[train_mask]
    y_fit = train_labels[train_mask]
    x_val = train_features[val_mask]
    y_val = train_labels[val_mask]
    x_test = test_features
    y_test = test_labels

    results: list[dict[str, object]] = []
    learning_curve_rows: list[dict[str, object]] = []
    if args.baseline == "b1":
        ridge_alphas = [float(token) for token in args.ridge_alphas.split(",") if token.strip()]
        result = fit_ridge_classifier(x_fit, y_fit, x_val, y_val, x_test, y_test, ridge_alphas)
        results.append(result)
        learning_curve_rows.extend(result["learning_curve"])
        seeds = [0]
    else:
        seeds = [int(token) for token in args.seeds.split(",") if token.strip()]
        for seed in seeds:
            result = train_one_seed(
                seed=seed,
                x_train=x_fit,
                y_train=y_fit,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                dropout=args.dropout,
            )
            results.append(result)
            learning_curve_rows.extend(result["learning_curve"])

    best_result = max(results, key=lambda item: float(item["best_val_top1"]))
    logits = np.asarray(best_result["test_logits"], dtype=np.float32)
    top1_preds = logits.argmax(axis=1)
    top5 = np.argpartition(-logits, kth=min(4, logits.shape[1] - 1), axis=1)[:, : min(5, logits.shape[1])]

    metrics = {
        "baseline": args.baseline,
        "fold": args.fold_name,
        "held_out_subject": test_index[0]["subject"],
        "validation_subjects": val_subjects,
        "n_train_samples": int(x_fit.shape[0]),
        "n_val_samples": int(x_val.shape[0]),
        "n_test_samples": int(x_test.shape[0]),
        "input_dim": int(x_test.shape[1]),
        "num_classes": int(len(image_ids)),
        "chance_level": float(1.0 / len(image_ids)),
        "top1_acc": float(np.mean([float(item["test_top1"]) for item in results])),
        "top1_std": float(np.std([float(item["test_top1"]) for item in results])),
        "top5_acc": float(np.mean([float(item["test_top5"]) for item in results])),
        "top5_std": float(np.std([float(item["test_top5"]) for item in results])),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "best_seed": int(best_result["seed"]),
    }
    if args.baseline == "b1":
        metrics["best_alpha"] = float(best_result["best_alpha"])

    (args.output_dir / f"{args.baseline}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    yaml_dump_simple(
        args.output_dir / f"{args.baseline}_run_config.yaml",
        {
            "baseline": args.baseline,
            "fold": args.fold_name,
            "validation_subjects": val_subjects,
            "seeds": seeds,
            "hidden_dim": 0 if args.baseline == "b1" else args.hidden_dim,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": 0.0 if args.baseline == "b1" else args.dropout,
            "ridge_alphas": args.ridge_alphas,
            "val_subject_count": args.val_subject_count,
            "val_subject_strategy": args.val_subject_strategy,
            "val_subjects_arg": args.val_subjects,
        },
    )

    with (args.output_dir / f"{args.baseline}_learning_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "epoch", "train_loss", "train_top1", "val_loss", "val_top1"])
        writer.writeheader()
        writer.writerows(learning_curve_rows)

    with (args.output_dir / f"{args.baseline}_subject_breakdown.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject", "top1_acc", "top5_acc", "n_test_samples", "baseline"])
        writer.writeheader()
        writer.writerow(
            {
                "subject": test_index[0]["subject"],
                "top1_acc": f"{metrics['top1_acc']:.6f}",
                "top5_acc": f"{metrics['top5_acc']:.6f}",
                "n_test_samples": int(x_test.shape[0]),
                "baseline": args.baseline,
            }
        )

    with (args.output_dir / f"{args.baseline}_seed_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "best_epoch", "best_val_top1", "test_top1", "test_top5"])
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "seed": item["seed"],
                    "best_epoch": item["best_epoch"],
                    "best_val_top1": f"{float(item['best_val_top1']):.6f}",
                    "test_top1": f"{float(item['test_top1']):.6f}",
                    "test_top5": f"{float(item['test_top5']):.6f}",
                }
            )

    with (args.output_dir / f"{args.baseline}_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject", "nsdId", "true_label", "pred_label", "top1_correct", "top5_correct"])
        writer.writeheader()
        for idx, row in enumerate(test_index):
            writer.writerow(
                {
                    "subject": row["subject"],
                    "nsdId": row["nsdId"],
                    "true_label": int(test_labels[idx]),
                    "pred_label": int(top1_preds[idx]),
                    "top1_correct": int(top1_preds[idx] == test_labels[idx]),
                    "top5_correct": int(test_labels[idx] in top5[idx]),
                }
            )

    np.save(args.output_dir / f"{args.baseline}_logits.npy", logits.astype(np.float32))
    if args.baseline == "b4":
        np.save(args.output_dir / "b4_hidden.npy", np.asarray(best_result["test_hidden"], dtype=np.float32))
        np.savez(
            args.output_dir / "b4_best_params.npz",
            W1=np.asarray(best_result["best_params"]["W1"], dtype=np.float32),
            b1=np.asarray(best_result["best_params"]["b1"], dtype=np.float32),
            W2=np.asarray(best_result["best_params"]["W2"], dtype=np.float32),
            b2=np.asarray(best_result["best_params"]["b2"], dtype=np.float32),
        )


if __name__ == "__main__":
    main()
