#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-trial repeat/familiarity graph classification baselines.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fold", required=True, choices=["fold_01", "fold_04"])
    parser.add_argument("--task", default="first_vs_repeated", choices=["first_vs_repeated", "repeat_index"])
    parser.add_argument("--model", required=True, choices=["roi_mlp", "gcn", "gat", "braingnn", "session_mlp"])
    parser.add_argument("--adjacency", default="topk20", choices=["topk20", "identity", "shuffled"])
    parser.add_argument("--control", default="none", choices=["none", "shuffle_within_subject"])
    parser.add_argument("--braingnn-root", type=Path, default=Path("external/BrainGNN_Pytorch"))
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/repeat_state_baselines"),
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pool-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


class TrialDataset(Dataset):
    def __init__(self, items: list[dict[str, object]], task: str):
        self.items = items
        self.task = task

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, object]:
        item = self.items[idx]
        label_key = "y_first_vs_repeated" if self.task == "first_vs_repeated" else "y_repeat_index"
        return {
            "x": item["x"].float(),
            "y": torch.tensor(int(item[label_key]), dtype=torch.long),
            "subject": int(item["subject"]),
            "session_index": int(item["session_index"]),
            "repeat_index": int(item["repeat_index"]),
            "nsdId": int(item["nsdId"]),
        }


def collate_trials(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "x": torch.stack([item["x"] for item in batch]),  # type: ignore[list-item]
        "y": torch.stack([item["y"] for item in batch]),  # type: ignore[list-item]
        "subject": torch.tensor([int(item["subject"]) for item in batch], dtype=torch.long),
        "session_index": torch.tensor([int(item["session_index"]) for item in batch], dtype=torch.float32),
        "repeat_index": torch.tensor([int(item["repeat_index"]) for item in batch], dtype=torch.long),
        "nsdId": torch.tensor([int(item["nsdId"]) for item in batch], dtype=torch.long),
    }


def shuffle_labels_within_subject(items: list[dict[str, object]], task: str, seed: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    key = "y_first_vs_repeated" if task == "first_vs_repeated" else "y_repeat_index"
    out = [dict(item) for item in items]
    by_subject: dict[int, list[int]] = {}
    for idx, item in enumerate(out):
        by_subject.setdefault(int(item["subject"]), []).append(idx)
    for indices in by_subject.values():
        labels = [int(out[idx][key]) for idx in indices]
        rng.shuffle(labels)
        for idx, label in zip(indices, labels):
            out[idx][key] = label
    return out


class RoiMLPClassifier(nn.Module):
    def __init__(self, n_nodes: int, in_dim: int, hidden_dim: int, n_classes: int, dropout: float):
        super().__init__()
        flat = n_nodes * in_dim
        self.net = nn.Sequential(
            nn.LayerNorm(flat),
            nn.Linear(flat, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, session_index: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(x.flatten(start_dim=1))


class DenseGCNClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, session_index: torch.Tensor | None = None) -> torch.Tensor:
        h = torch.einsum("ij,bjf->bif", adjacency, x)
        h = F.gelu(self.lin1(h))
        h = self.drop(h)
        h = torch.einsum("ij,bjf->bif", adjacency, h)
        h = self.norm(F.gelu(self.lin2(h)))
        return self.head(h.mean(dim=1))


class DenseGATClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.att_src = nn.Linear(hidden_dim, 1, bias=False)
        self.att_dst = nn.Linear(hidden_dim, 1, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, session_index: torch.Tensor | None = None) -> torch.Tensor:
        h = F.gelu(self.proj(x))
        scores = self.att_src(h) + self.att_dst(h).transpose(1, 2)
        scores = F.leaky_relu(scores, negative_slope=0.2)
        mask = adjacency > 0
        scores = scores.masked_fill(~mask[None, :, :], -1e9)
        attn = torch.softmax(scores, dim=-1)
        h = torch.bmm(attn, h)
        h = self.norm(F.gelu(self.out(self.drop(h))))
        return self.head(h.mean(dim=1))


class SessionMLPClassifier(nn.Module):
    def __init__(self, n_classes: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, n_classes))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, session_index: torch.Tensor | None = None) -> torch.Tensor:
        assert session_index is not None
        s = session_index.float().view(-1, 1)
        s = (s - 20.0) / 20.0
        return self.net(s)


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    adj = np.abs(adjacency.astype(np.float32)).copy()
    np.fill_diagonal(adj, 1.0)
    degree = adj.sum(axis=1)
    inv_sqrt = np.where(degree > 0, degree ** -0.5, 0.0).astype(np.float32)
    return (inv_sqrt[:, None] * adj) * inv_sqrt[None, :]


def load_adjacency(fold_dir: Path, mode: str, seed: int) -> np.ndarray:
    base = np.load(fold_dir / "adjacency.npy")
    if mode == "identity":
        return np.eye(base.shape[0], dtype=np.float32)
    if mode == "shuffled":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(base.shape[0])
        base = base[perm][:, perm]
    return normalize_adjacency(base)


def rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def auroc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata_average(scores)
    pos_rank_sum = float(ranks[labels == 1].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auroc_ovr(labels: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    vals = []
    for cls in range(n_classes):
        binary = (labels == cls).astype(np.int64)
        vals.append(auroc_binary(binary, probs[:, cls]))
    return float(np.nanmean(vals))


def average_precision_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float((precision * sorted_labels).sum() / n_pos)


def auprc_ovr(labels: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    vals = []
    for cls in range(n_classes):
        binary = (labels == cls).astype(np.int64)
        vals.append(average_precision_binary(binary, probs[:, cls]))
    return float(np.nanmean(vals))


def confusion_matrix(labels: np.ndarray, preds: np.ndarray, n_classes: int) -> list[list[int]]:
    mat = np.zeros((n_classes, n_classes), dtype=np.int64)
    for y, p in zip(labels.astype(np.int64), preds.astype(np.int64)):
        mat[int(y), int(p)] += 1
    return mat.tolist()


def balanced_accuracy(labels: np.ndarray, preds: np.ndarray, n_classes: int) -> float:
    mat = np.asarray(confusion_matrix(labels, preds, n_classes), dtype=np.float64)
    recalls = []
    for cls in range(n_classes):
        denom = mat[cls].sum()
        recalls.append(mat[cls, cls] / denom if denom > 0 else np.nan)
    return float(np.nanmean(recalls))


def macro_f1(labels: np.ndarray, preds: np.ndarray, n_classes: int) -> float:
    mat = np.asarray(confusion_matrix(labels, preds, n_classes), dtype=np.float64)
    f1s = []
    for cls in range(n_classes):
        tp = mat[cls, cls]
        fp = mat[:, cls].sum() - tp
        fn = mat[cls, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)
    return float(np.mean(f1s))


def best_binary_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float, float]:
    candidates = np.unique(probs)
    if len(candidates) > 5000:
        candidates = np.unique(np.quantile(probs, np.linspace(0.0, 1.0, 5000)))
    best = (float(candidates[0]), -1.0, -1.0)
    for threshold in candidates:
        preds = (probs >= threshold).astype(np.int64)
        bal = balanced_accuracy(labels, preds, 2)
        acc = float((labels == preds).mean())
        if bal > best[1] or (math.isclose(bal, best[1]) and acc > best[2]):
            best = (float(threshold), float(bal), float(acc))
    return best


def load_items(fold_dir: Path, split: str, task: str, control: str, seed: int) -> list[dict[str, object]]:
    items = torch.load(fold_dir / f"{split}_single_trials.pt", map_location="cpu", weights_only=False)
    if control == "shuffle_within_subject" and split == "train":
        items = shuffle_labels_within_subject(items, task, seed)
    return items


def dense_evaluate(model: nn.Module, loader: DataLoader, adjacency: torch.Tensor, device: torch.device, n_classes: int, threshold: float | None = None) -> dict[str, object]:
    model.eval()
    labels_parts: list[np.ndarray] = []
    probs_parts: list[np.ndarray] = []
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            session = batch["session_index"].to(device)
            logits = model(x, adjacency, session)
            loss = F.cross_entropy(logits, y)
            probs = torch.softmax(logits, dim=-1)
            labels_parts.append(y.cpu().numpy())
            probs_parts.append(probs.cpu().numpy())
            total_loss += float(loss.item()) * y.numel()
            total_n += y.numel()
    labels = np.concatenate(labels_parts)
    probs = np.concatenate(probs_parts)
    if n_classes == 2 and threshold is not None:
        preds = (probs[:, 1] >= threshold).astype(np.int64)
    else:
        preds = probs.argmax(axis=1)
    return {
        "loss": total_loss / max(total_n, 1),
        "labels": labels,
        "probs": probs,
        "preds": preds,
        "accuracy": float((labels == preds).mean()),
        "balanced_accuracy": balanced_accuracy(labels, preds, n_classes),
        "macro_f1": macro_f1(labels, preds, n_classes),
        "auroc": auroc_binary(labels, probs[:, 1]) if n_classes == 2 else auroc_ovr(labels, probs, n_classes),
        "auprc": average_precision_binary(labels, probs[:, 1]) if n_classes == 2 else auprc_ovr(labels, probs, n_classes),
    }


def make_dense_model(model_name: str, n_nodes: int, in_dim: int, hidden_dim: int, n_classes: int, dropout: float) -> nn.Module:
    if model_name == "roi_mlp":
        return RoiMLPClassifier(n_nodes, in_dim, hidden_dim, n_classes, dropout)
    if model_name == "gcn":
        return DenseGCNClassifier(in_dim, hidden_dim, n_classes, dropout)
    if model_name == "gat":
        return DenseGATClassifier(in_dim, hidden_dim, n_classes, dropout)
    if model_name == "session_mlp":
        return SessionMLPClassifier(n_classes)
    raise ValueError(f"Unsupported dense model: {model_name}")


def train_dense(args: argparse.Namespace, fold_dir: Path, out_dir: Path, device: torch.device) -> dict[str, object]:
    train_items = load_items(fold_dir, "train", args.task, args.control, args.seed)
    val_items = load_items(fold_dir, "val", args.task, args.control, args.seed)
    test_items = load_items(fold_dir, "test", args.task, args.control, args.seed)
    n_classes = 2 if args.task == "first_vs_repeated" else 3
    first_x = train_items[0]["x"]
    n_nodes, in_dim = int(first_x.shape[0]), int(first_x.shape[1])  # type: ignore[union-attr]
    adjacency = torch.from_numpy(load_adjacency(fold_dir, args.adjacency, args.seed)).to(device)

    model = make_dense_model(args.model, n_nodes, in_dim, args.hidden_dim, n_classes, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(TrialDataset(train_items, args.task), batch_size=args.batch_size, shuffle=True, collate_fn=collate_trials)
    eval_train_loader = DataLoader(TrialDataset(train_items, args.task), batch_size=args.batch_size, shuffle=False, collate_fn=collate_trials)
    val_loader = DataLoader(TrialDataset(val_items, args.task), batch_size=args.batch_size, shuffle=False, collate_fn=collate_trials)
    test_loader = DataLoader(TrialDataset(test_items, args.task), batch_size=args.batch_size, shuffle=False, collate_fn=collate_trials)

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = -1.0
    best_epoch = -1
    bad = 0
    curve = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_n = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            session = batch["session_index"].to(device)
            logits = model(x, adjacency, session)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_loss += float(loss.item()) * y.numel()
            train_n += y.numel()

        train_eval = dense_evaluate(model, eval_train_loader, adjacency, device, n_classes)
        val_eval = dense_evaluate(model, val_loader, adjacency, device, n_classes)
        curve.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(train_n, 1),
                "train_auroc": train_eval["auroc"],
                "train_auprc": train_eval["auprc"],
                "val_loss": val_eval["loss"],
                "val_auroc": val_eval["auroc"],
                "val_auprc": val_eval["auprc"],
                "val_balanced_accuracy": val_eval["balanced_accuracy"],
            }
        )
        score = float(val_eval["auroc"])
        if score > best_val:
            best_val = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= args.patience:
            break

    model.load_state_dict(best_state)
    val_for_threshold = dense_evaluate(model, val_loader, adjacency, device, n_classes)
    threshold = None
    if n_classes == 2:
        threshold, _bal, _acc = best_binary_threshold(val_for_threshold["labels"], val_for_threshold["probs"][:, 1])
    test = dense_evaluate(model, test_loader, adjacency, device, n_classes, threshold=threshold)
    metrics = {
        "model": args.model,
        "task": args.task,
        "fold": args.fold,
        "adjacency": args.adjacency,
        "control": args.control,
        "n_train": len(train_items),
        "n_val": len(val_items),
        "n_test": len(test_items),
        "n_classes": n_classes,
        "n_nodes": n_nodes,
        "node_feature_dim": in_dim,
        "pool_ratio": None,
        "best_epoch": best_epoch,
        "best_val_auroc": best_val,
        "val_auroc": val_for_threshold["auroc"],
        "val_auprc": val_for_threshold["auprc"],
        "val_balanced_accuracy": val_for_threshold["balanced_accuracy"],
        "test_loss": test["loss"],
        "auroc": test["auroc"],
        "auprc": test["auprc"],
        "balanced_accuracy": test["balanced_accuracy"],
        "macro_f1": test["macro_f1"],
        "accuracy": test["accuracy"],
        "confusion_matrix": confusion_matrix(test["labels"], test["preds"], n_classes),
        "status": "ok",
    }
    torch.save({"model_state": best_state, "args": vars(args), "metrics": metrics}, out_dir / "checkpoint.pt")
    np.savez_compressed(out_dir / "test_predictions.npz", labels=test["labels"], probs=test["probs"], preds=test["preds"])
    write_curve(out_dir / "learning_curve.csv", curve)
    return metrics


def write_curve(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "train_auroc", "train_auprc", "val_loss", "val_auroc", "val_auprc", "val_balanced_accuracy"],
        )
        writer.writeheader()
        writer.writerows(rows)


def train_braingnn(args: argparse.Namespace, fold_dir: Path, out_dir: Path, device: torch.device) -> dict[str, object]:
    from torch_geometric.data import Batch, Data

    braingnn_root = (args.root / args.braingnn_root).resolve()
    if str(braingnn_root) not in sys.path:
        sys.path.insert(0, str(braingnn_root))
    from net.braingnn import Network  # noqa: PLC0415

    train_items = load_items(fold_dir, "train", args.task, args.control, args.seed)
    val_items = load_items(fold_dir, "val", args.task, args.control, args.seed)
    test_items = load_items(fold_dir, "test", args.task, args.control, args.seed)
    n_classes = 2 if args.task == "first_vs_repeated" else 3
    first_x = train_items[0]["x"]
    n_nodes, in_dim = int(first_x.shape[0]), int(first_x.shape[1])  # type: ignore[union-attr]
    adjacency = load_adjacency(fold_dir, args.adjacency, args.seed)
    src, dst = np.nonzero(adjacency)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_attr = torch.tensor(adjacency[src, dst], dtype=torch.float32)
    pos = torch.eye(n_nodes, dtype=torch.float32)

    def to_batch(batch: dict[str, object]) -> tuple[Batch, torch.Tensor]:
        xs = batch["x"]
        ys = batch["y"]
        data_list = [Data(x=xs[i], edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=ys[i].view(1)) for i in range(xs.shape[0])]  # type: ignore[index,union-attr]
        return Batch.from_data_list(data_list).to(device), ys.to(device)  # type: ignore[union-attr]

    model = Network(indim=in_dim, ratio=args.pool_ratio, nclass=n_classes, R=n_nodes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=max(args.weight_decay, 5e-4))
    train_loader = DataLoader(TrialDataset(train_items, args.task), batch_size=args.batch_size, shuffle=True, collate_fn=collate_trials)
    eval_train_loader = DataLoader(TrialDataset(train_items, args.task), batch_size=args.batch_size, shuffle=False, collate_fn=collate_trials)
    val_loader = DataLoader(TrialDataset(val_items, args.task), batch_size=args.batch_size, shuffle=False, collate_fn=collate_trials)
    test_loader = DataLoader(TrialDataset(test_items, args.task), batch_size=args.batch_size, shuffle=False, collate_fn=collate_trials)

    def evaluate(loader: DataLoader, threshold: float | None = None) -> dict[str, object]:
        model.eval()
        labels_parts = []
        probs_parts = []
        total_loss = 0.0
        total_n = 0
        roi_sum = None
        roi_count = 0
        with torch.no_grad():
            for raw in loader:
                batch, y = to_batch(raw)
                out, _w1, _w2, s1, _s2 = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr, batch.pos)
                loss = F.nll_loss(out, y)
                probs = out.exp()
                labels_parts.append(y.cpu().numpy())
                probs_parts.append(probs.cpu().numpy())
                total_loss += float(loss.item()) * y.numel()
                total_n += y.numel()
                if s1.ndim == 2 and s1.shape[1] == n_nodes:
                    roi_sum = s1.detach().cpu().mean(dim=0) if roi_sum is None else roi_sum + s1.detach().cpu().mean(dim=0)
                    roi_count += 1
        labels = np.concatenate(labels_parts)
        probs = np.concatenate(probs_parts)
        preds = (probs[:, 1] >= threshold).astype(np.int64) if n_classes == 2 and threshold is not None else probs.argmax(axis=1)
        return {
            "loss": total_loss / max(total_n, 1),
            "labels": labels,
            "probs": probs,
            "preds": preds,
            "accuracy": float((labels == preds).mean()),
            "balanced_accuracy": balanced_accuracy(labels, preds, n_classes),
            "macro_f1": macro_f1(labels, preds, n_classes),
            "auroc": auroc_binary(labels, probs[:, 1]) if n_classes == 2 else auroc_ovr(labels, probs, n_classes),
            "auprc": average_precision_binary(labels, probs[:, 1]) if n_classes == 2 else auprc_ovr(labels, probs, n_classes),
            "roi_importance": (roi_sum / max(roi_count, 1)).numpy() if roi_sum is not None else np.zeros((n_nodes,), dtype=np.float32),
        }

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = -1.0
    best_epoch = -1
    bad = 0
    curve = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_n = 0
        for raw in train_loader:
            batch, y = to_batch(raw)
            out, _w1, _w2, _s1, _s2 = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr, batch.pos)
            loss = F.nll_loss(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_loss += float(loss.item()) * y.numel()
            train_n += y.numel()
        train_eval = evaluate(eval_train_loader)
        val_eval = evaluate(val_loader)
        curve.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(train_n, 1),
                "train_auroc": train_eval["auroc"],
                "train_auprc": train_eval["auprc"],
                "val_loss": val_eval["loss"],
                "val_auroc": val_eval["auroc"],
                "val_auprc": val_eval["auprc"],
                "val_balanced_accuracy": val_eval["balanced_accuracy"],
            }
        )
        score = float(val_eval["auroc"])
        if score > best_val:
            best_val = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= args.patience:
            break

    model.load_state_dict(best_state)
    val_for_threshold = evaluate(val_loader)
    threshold = None
    if n_classes == 2:
        threshold, _bal, _acc = best_binary_threshold(val_for_threshold["labels"], val_for_threshold["probs"][:, 1])
    test = evaluate(test_loader, threshold=threshold)
    metrics = {
        "model": args.model,
        "task": args.task,
        "fold": args.fold,
        "adjacency": args.adjacency,
        "control": args.control,
        "n_train": len(train_items),
        "n_val": len(val_items),
        "n_test": len(test_items),
        "n_classes": n_classes,
        "n_nodes": n_nodes,
        "node_feature_dim": in_dim,
        "pool_ratio": args.pool_ratio,
        "best_epoch": best_epoch,
        "best_val_auroc": best_val,
        "val_auroc": val_for_threshold["auroc"],
        "val_auprc": val_for_threshold["auprc"],
        "val_balanced_accuracy": val_for_threshold["balanced_accuracy"],
        "test_loss": test["loss"],
        "auroc": test["auroc"],
        "auprc": test["auprc"],
        "balanced_accuracy": test["balanced_accuracy"],
        "macro_f1": test["macro_f1"],
        "accuracy": test["accuracy"],
        "confusion_matrix": confusion_matrix(test["labels"], test["preds"], n_classes),
        "status": "ok",
    }
    torch.save({"model_state": best_state, "args": vars(args), "metrics": metrics}, out_dir / "checkpoint.pt")
    np.savez_compressed(out_dir / "test_predictions.npz", labels=test["labels"], probs=test["probs"], preds=test["preds"])
    write_curve(out_dir / "learning_curve.csv", curve)
    with (out_dir / "roi_importance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node_index", "importance"])
        writer.writeheader()
        for idx, value in enumerate(test["roi_importance"].tolist()):
            writer.writerow({"node_index": idx, "importance": float(value)})
    return metrics


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    fold_dir = args.root / args.dataset_root / args.fold
    variant = f"{args.model}_{args.adjacency}_{args.control}"
    if args.model == "braingnn":
        variant += f"_pool{args.pool_ratio:g}"
    out_dir = args.root / args.output_root / args.task / variant / args.fold
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    if args.model == "braingnn":
        metrics = train_braingnn(args, fold_dir, out_dir, device)
    else:
        metrics = train_dense(args, fold_dir, out_dir, device)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
