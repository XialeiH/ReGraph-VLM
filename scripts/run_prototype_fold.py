#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Version 0 shared prototype model on one fold.")
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--fold-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--val-subject-count", type=int, default=2)
    parser.add_argument("--val-subject-strategy", choices=["fixed_last", "rotating"], default="rotating")
    parser.add_argument("--val-subjects", type=str, default="")
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


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    topk = torch.topk(logits, k=min(k, logits.shape[1]), dim=1).indices
    return float((topk == labels[:, None]).any(dim=1).float().mean().item())


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


class PrototypeModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_prototypes: int, tau: float, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, hidden_dim) * 0.02)
        self.head = nn.Sequential(
            nn.LayerNorm(num_prototypes),
            nn.Linear(num_prototypes, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.tau = tau

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)
        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        similarities = h_norm @ p_norm.T
        assignments = torch.softmax(similarities / self.tau, dim=-1)
        logits = self.head(assignments)
        return {
            "h": h,
            "h_norm": h_norm,
            "p_norm": p_norm,
            "assignments": assignments,
            "logits": logits,
        }


def build_image_to_indices(labels: np.ndarray) -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = {}
    for idx, label in enumerate(labels.tolist()):
        mapping.setdefault(int(label), []).append(int(idx))
    return mapping


def make_epoch_batches(
    image_to_indices: dict[int, list[int]],
    images_per_batch: int,
    subjects_per_image: int,
    rng: random.Random,
) -> list[list[int]]:
    image_ids = list(image_to_indices.keys())
    rng.shuffle(image_ids)
    batches: list[list[int]] = []
    for start in range(0, len(image_ids), images_per_batch):
        batch_images = image_ids[start : start + images_per_batch]
        batch_indices: list[int] = []
        for image_id in batch_images:
            candidates = list(image_to_indices[image_id])
            rng.shuffle(candidates)
            batch_indices.extend(candidates[: min(subjects_per_image, len(candidates))])
        rng.shuffle(batch_indices)
        if batch_indices:
            batches.append(batch_indices)
    return batches


def alignment_loss(assignments: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(assignments, dim=-1)
    unique_labels = labels.unique()
    losses = []
    for label in unique_labels:
        idx = torch.where(labels == label)[0]
        if idx.numel() < 2:
            continue
        group = normalized[idx]
        cosine = group @ group.T
        upper = torch.triu_indices(idx.numel(), idx.numel(), offset=1, device=group.device)
        pairwise = cosine[upper[0], upper[1]]
        losses.append(1.0 - pairwise.mean())
    if not losses:
        return assignments.new_tensor(0.0)
    return torch.stack(losses).mean()


def balance_loss(assignments: torch.Tensor) -> torch.Tensor:
    mean_assignment = assignments.mean(dim=0)
    uniform = torch.full_like(mean_assignment, 1.0 / mean_assignment.numel())
    return torch.sum(mean_assignment * (mean_assignment.clamp_min(1e-8).log() - uniform.log()))


def orthogonality_loss(p_norm: torch.Tensor) -> torch.Tensor:
    gram = p_norm @ p_norm.T
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return torch.mean((gram - eye) ** 2)


@dataclass
class EvalOutputs:
    loss: float
    top1: float
    top5: float
    logits: np.ndarray
    assignments: np.ndarray


def evaluate_model(
    model: PrototypeModel,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    ce_weight: float,
    align_weight: float,
    balance_weight: float,
    orth_weight: float,
) -> EvalOutputs:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_assignments: list[torch.Tensor] = []
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size]
            yb = y[start : start + batch_size]
            outputs = model(xb)
            ce = F.cross_entropy(outputs["logits"], yb)
            align = alignment_loss(outputs["assignments"], yb)
            balance = balance_loss(outputs["assignments"])
            orth = orthogonality_loss(outputs["p_norm"])
            loss = ce_weight * ce + align_weight * align + balance_weight * balance + orth_weight * orth
            total_loss += float(loss.item()) * xb.shape[0]
            total_count += xb.shape[0]
            all_logits.append(outputs["logits"].cpu())
            all_assignments.append(outputs["assignments"].cpu())
    logits = torch.cat(all_logits, dim=0)
    assignments = torch.cat(all_assignments, dim=0)
    return EvalOutputs(
        loss=total_loss / max(total_count, 1),
        top1=topk_accuracy(logits, y.cpu(), 1),
        top5=topk_accuracy(logits, y.cpu(), min(5, logits.shape[1])),
        logits=logits.numpy().astype(np.float32),
        assignments=assignments.numpy().astype(np.float32),
    )


def assignment_diagnostics(
    assignments: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    mean_assignment = assignments.mean(axis=0)
    entropy = float(-(mean_assignment * np.log(np.clip(mean_assignment, 1e-8, 1.0))).sum())
    effective_cells = float(np.exp(entropy))
    max_mass = float(mean_assignment.max())

    z_norm = assignments / np.clip(np.linalg.norm(assignments, axis=1, keepdims=True), 1e-8, None)
    same_cosines: list[float] = []
    diff_cosines: list[float] = []
    proto_agreement: list[float] = []
    unique_labels = np.unique(labels)
    rng = np.random.default_rng(0)
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        if idx.size >= 2:
            sims = z_norm[idx] @ z_norm[idx].T
            upper = np.triu_indices(idx.size, k=1)
            same_cosines.extend(sims[upper].tolist())
            top_proto = assignments[idx].argmax(axis=1)
            agreement = (top_proto[:, None] == top_proto[None, :])[upper]
            proto_agreement.append(float(np.mean(agreement)) if agreement.size > 0 else 0.0)
            other_labels = unique_labels[unique_labels != label]
            if other_labels.size > 0:
                sampled_other = int(rng.choice(other_labels))
                other_idx = np.where(labels == sampled_other)[0]
                pair_sims = z_norm[idx] @ z_norm[other_idx].T
                diff_cosines.extend(pair_sims.reshape(-1).tolist())
    same_mean = float(np.mean(same_cosines)) if same_cosines else 0.0
    diff_mean = float(np.mean(diff_cosines)) if diff_cosines else 0.0
    return {
        "effective_cells": effective_cells,
        "max_cell_mass": max_mass,
        "same_image_cosine": same_mean,
        "different_image_cosine": diff_mean,
        "same_minus_different": same_mean - diff_mean,
        "top1_prototype_agreement": float(np.mean(proto_agreement)) if proto_agreement else 0.0,
        "collapse_flag": bool(effective_cells < 16.0 or max_mass > 0.25),
    }


def cosine_lr_factor(epoch: int, epochs: int, warmup_epochs: int, min_factor: float = 0.01) -> float:
    if epoch <= warmup_epochs:
        return epoch / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_factor + (1.0 - min_factor) * cosine


def train_one_seed(
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    num_classes: int,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = PrototypeModel(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_prototypes=args.num_prototypes,
        tau=args.tau,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.lr_encoder, "weight_decay": args.weight_decay},
            {"params": model.head.parameters(), "lr": args.lr_head, "weight_decay": args.weight_decay},
            {"params": [model.prototypes], "lr": args.lr_prototypes, "weight_decay": args.weight_decay},
        ]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_lr_factor(step + 1, args.epochs, args.warmup_epochs),
    )

    x_train_t = torch.from_numpy(x_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)
    x_test_t = torch.from_numpy(x_test).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)

    image_to_indices = build_image_to_indices(y_train)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_top1 = -1.0
    best_val_top5 = -1.0
    best_epoch = -1
    bad_epochs = 0
    learning_rows: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_rng = random.Random(seed * 1000 + epoch)
        batches = make_epoch_batches(
            image_to_indices=image_to_indices,
            images_per_batch=args.images_per_batch,
            subjects_per_image=args.subjects_per_image,
            rng=epoch_rng,
        )
        total_train_loss = 0.0
        total_train_count = 0
        total_correct = 0.0

        for batch_indices in batches:
            xb = x_train_t[batch_indices]
            yb = y_train_t[batch_indices]
            outputs = model(xb)
            ce = F.cross_entropy(outputs["logits"], yb)
            align = alignment_loss(outputs["assignments"], yb)
            balance = balance_loss(outputs["assignments"])
            orth = orthogonality_loss(outputs["p_norm"])
            loss = ce + args.align_weight * align + args.balance_weight * balance + args.orth_weight * orth

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_train_loss += float(loss.item()) * xb.shape[0]
            total_train_count += xb.shape[0]
            total_correct += float((outputs["logits"].argmax(dim=1) == yb).float().sum().item())

        scheduler.step()

        val_outputs = evaluate_model(
            model=model,
            x=x_val_t,
            y=y_val_t,
            batch_size=args.images_per_batch * args.subjects_per_image,
            ce_weight=1.0,
            align_weight=args.align_weight,
            balance_weight=args.balance_weight,
            orth_weight=args.orth_weight,
        )
        train_top1 = total_correct / max(total_train_count, 1)
        train_loss = total_train_loss / max(total_train_count, 1)
        learning_rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_top1": train_top1,
                "val_loss": val_outputs.loss,
                "val_top1": val_outputs.top1,
                "val_top5": val_outputs.top5,
            }
        )

        improved = (val_outputs.top1 > best_val_top1) or (
            math.isclose(val_outputs.top1, best_val_top1) and val_outputs.top5 > best_val_top5
        )
        if improved:
            best_val_top1 = val_outputs.top1
            best_val_top5 = val_outputs.top5
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break

    model.load_state_dict(best_state)
    val_outputs = evaluate_model(
        model=model,
        x=x_val_t,
        y=y_val_t,
        batch_size=args.images_per_batch * args.subjects_per_image,
        ce_weight=1.0,
        align_weight=args.align_weight,
        balance_weight=args.balance_weight,
        orth_weight=args.orth_weight,
    )
    test_outputs = evaluate_model(
        model=model,
        x=x_test_t,
        y=y_test_t,
        batch_size=args.images_per_batch * args.subjects_per_image,
        ce_weight=1.0,
        align_weight=args.align_weight,
        balance_weight=args.balance_weight,
        orth_weight=args.orth_weight,
    )
    train_eval_outputs = evaluate_model(
        model=model,
        x=x_train_t,
        y=y_train_t,
        batch_size=args.images_per_batch * args.subjects_per_image,
        ce_weight=1.0,
        align_weight=args.align_weight,
        balance_weight=args.balance_weight,
        orth_weight=args.orth_weight,
    )

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_top1": best_val_top1,
        "best_val_top5": best_val_top5,
        "val_top1": val_outputs.top1,
        "val_top5": val_outputs.top5,
        "test_top1": test_outputs.top1,
        "test_top5": test_outputs.top5,
        "test_logits": test_outputs.logits,
        "test_assignments": test_outputs.assignments,
        "train_assignments": train_eval_outputs.assignments,
        "learning_rows": learning_rows,
        "state_dict": best_state,
    }


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
    fit_mask = np.array([row["subject"] not in val_subject_set for row in train_index], dtype=bool)
    val_mask = ~fit_mask

    x_fit = train_features[fit_mask]
    y_fit = train_labels[fit_mask]
    x_val = train_features[val_mask]
    y_val = train_labels[val_mask]
    x_test = test_features
    y_test = test_labels

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    results: list[dict[str, object]] = []
    learning_rows: list[dict[str, object]] = []
    seeds = [int(token) for token in args.seeds.split(",") if token.strip()]

    for seed in seeds:
        seed_result = train_one_seed(
            seed=seed,
            x_train=x_fit,
            y_train=y_fit,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            args=args,
            num_classes=len(image_ids),
            device=device,
        )
        results.append(seed_result)
        learning_rows.extend(seed_result["learning_rows"])
        torch.save(seed_result["state_dict"], args.output_dir / f"prototype_seed_{seed}_model.pt")

    best_result = max(results, key=lambda item: (float(item["best_val_top1"]), float(item["best_val_top5"])))
    logits = np.asarray(best_result["test_logits"], dtype=np.float32)
    assignments = np.asarray(best_result["test_assignments"], dtype=np.float32)
    top1_preds = logits.argmax(axis=1)
    top5_idx = np.argpartition(-logits, kth=min(4, logits.shape[1] - 1), axis=1)[:, : min(5, logits.shape[1])]

    train_assignment_stats = assignment_diagnostics(
        assignments=np.asarray(best_result["train_assignments"], dtype=np.float32),
        labels=y_fit,
    )
    test_assignment_stats = assignment_diagnostics(
        assignments=assignments,
        labels=y_test,
    )
    assignment_stats = {
        "train": train_assignment_stats,
        "test": test_assignment_stats,
    }
    (args.output_dir / "prototype_assignment_stats.json").write_text(json.dumps(assignment_stats, indent=2), encoding="utf-8")

    metrics = {
        "baseline": "prototype",
        "fold": args.fold_name,
        "held_out_subject": test_index[0]["subject"],
        "validation_subjects": val_subjects,
        "n_train_samples": int(x_fit.shape[0]),
        "n_val_samples": int(x_val.shape[0]),
        "n_test_samples": int(x_test.shape[0]),
        "input_dim": int(args.input_dim),
        "hidden_dim": int(args.hidden_dim),
        "num_prototypes": int(args.num_prototypes),
        "num_classes": int(len(image_ids)),
        "chance_level": float(1.0 / len(image_ids)),
        "top1_acc": float(np.mean([float(item["test_top1"]) for item in results])),
        "top1_std": float(np.std([float(item["test_top1"]) for item in results])),
        "top5_acc": float(np.mean([float(item["test_top5"]) for item in results])),
        "top5_std": float(np.std([float(item["test_top5"]) for item in results])),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "best_seed": int(best_result["seed"]),
        "best_val_top1": float(best_result["best_val_top1"]),
        "best_val_top5": float(best_result["best_val_top5"]),
    }
    (args.output_dir / "prototype_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    yaml_dump_simple(
        args.output_dir / "prototype_run_config.yaml",
        {
            "fold": args.fold_name,
            "validation_subjects": val_subjects,
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
            "seeds": seeds,
            "device": str(device),
            "val_subjects_arg": args.val_subjects,
        },
    )

    with (args.output_dir / "prototype_learning_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seed", "epoch", "train_loss", "train_top1", "val_loss", "val_top1", "val_top5"],
        )
        writer.writeheader()
        writer.writerows(learning_rows)

    with (args.output_dir / "prototype_seed_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seed", "best_epoch", "best_val_top1", "best_val_top5", "test_top1", "test_top5"],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "seed": item["seed"],
                    "best_epoch": item["best_epoch"],
                    "best_val_top1": f"{float(item['best_val_top1']):.6f}",
                    "best_val_top5": f"{float(item['best_val_top5']):.6f}",
                    "test_top1": f"{float(item['test_top1']):.6f}",
                    "test_top5": f"{float(item['test_top5']):.6f}",
                }
            )

    with (args.output_dir / "prototype_subject_breakdown.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject", "top1_acc", "top5_acc", "n_test_samples", "baseline"])
        writer.writeheader()
        writer.writerow(
            {
                "subject": test_index[0]["subject"],
                "top1_acc": f"{metrics['top1_acc']:.6f}",
                "top5_acc": f"{metrics['top5_acc']:.6f}",
                "n_test_samples": int(x_test.shape[0]),
                "baseline": "prototype",
            }
        )

    with (args.output_dir / "prototype_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["subject", "nsdId", "true_label", "pred_label", "top1_correct", "top5_correct", "top1_prototype"],
        )
        writer.writeheader()
        top1_prototypes = assignments.argmax(axis=1)
        for idx, row in enumerate(test_index):
            writer.writerow(
                {
                    "subject": row["subject"],
                    "nsdId": row["nsdId"],
                    "true_label": int(y_test[idx]),
                    "pred_label": int(top1_preds[idx]),
                    "top1_correct": int(top1_preds[idx] == y_test[idx]),
                    "top5_correct": int(y_test[idx] in top5_idx[idx]),
                    "top1_prototype": int(top1_prototypes[idx]),
                }
            )

    np.save(args.output_dir / "prototype_logits.npy", logits)
    np.save(args.output_dir / "prototype_assignments.npy", assignments)


if __name__ == "__main__":
    main()
