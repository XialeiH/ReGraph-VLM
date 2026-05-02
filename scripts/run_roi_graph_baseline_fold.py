#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple ROI graph sanity baselines.")
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--fold-name", type=str, required=True)
    parser.add_argument("--model", choices=["roi_mlp", "gcn", "gat"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    topk = torch.topk(logits, k=min(k, logits.shape[1]), dim=1).indices
    return float((topk == labels[:, None]).any(dim=1).float().mean().item())


def load_graph_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    graphs = torch.load(path, map_location="cpu", weights_only=False)
    x = np.stack([graph["x"].numpy().astype(np.float32) for graph in graphs], axis=0)
    y = np.asarray([int(graph["y"]) for graph in graphs], dtype=np.int64)
    return x, y


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    adj = adjacency.astype(np.float32).copy()
    np.fill_diagonal(adj, 1.0)
    degree = adj.sum(axis=1)
    inv_sqrt = np.where(degree > 0, degree ** -0.5, 0.0).astype(np.float32)
    return (inv_sqrt[:, None] * adj) * inv_sqrt[None, :]


class RoiMLP(nn.Module):
    def __init__(self, n_nodes: int, in_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(n_nodes * in_dim),
            nn.Linear(n_nodes * in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(start_dim=1))


class DenseGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = torch.einsum("ij,bjf->bif", adjacency, x)
        h = F.gelu(self.lin1(h))
        h = self.drop(h)
        h = torch.einsum("ij,bjf->bif", adjacency, h)
        h = self.norm(F.gelu(self.lin2(h)))
        return self.head(h.mean(dim=1))


class DenseGAT(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.att_src = nn.Linear(hidden_dim, 1, bias=False)
        self.att_dst = nn.Linear(hidden_dim, 1, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.proj(x))
        scores = self.att_src(h) + self.att_dst(h).transpose(1, 2)
        scores = F.leaky_relu(scores, negative_slope=0.2)
        mask = adjacency > 0
        scores = scores.masked_fill(~mask[None, :, :], -1e9)
        attn = torch.softmax(scores, dim=-1)
        h = torch.bmm(attn, h)
        h = self.norm(F.gelu(self.out(self.drop(h))))
        return self.head(h.mean(dim=1))


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, adjacency: torch.Tensor, batch_size: int) -> dict[str, object]:
    model.eval()
    losses: list[float] = []
    logits_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size]
            yb = y[start : start + batch_size]
            logits = model(xb, adjacency)
            losses.append(float(F.cross_entropy(logits, yb).item()) * xb.shape[0])
            logits_parts.append(logits.cpu())
    logits_all = torch.cat(logits_parts, dim=0)
    return {
        "loss": sum(losses) / max(x.shape[0], 1),
        "top1": topk_accuracy(logits_all, y.cpu(), 1),
        "top5": topk_accuracy(logits_all, y.cpu(), 5),
        "logits": logits_all.numpy().astype(np.float32),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_x, train_y = load_graph_split(args.graph_dir / "train_graphs.pt")
    val_x, val_y = load_graph_split(args.graph_dir / "val_graphs.pt")
    test_x, test_y = load_graph_split(args.graph_dir / "test_graphs.pt")
    adjacency = normalize_adjacency(np.load(args.graph_dir / "adjacency.npy"))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    train_x_t = torch.from_numpy(train_x).to(device)
    train_y_t = torch.from_numpy(train_y).to(device)
    val_x_t = torch.from_numpy(val_x).to(device)
    val_y_t = torch.from_numpy(val_y).to(device)
    test_x_t = torch.from_numpy(test_x).to(device)
    test_y_t = torch.from_numpy(test_y).to(device)
    adjacency_t = torch.from_numpy(adjacency).to(device)

    n_nodes, in_dim = train_x.shape[1], train_x.shape[2]
    num_classes = int(max(train_y.max(), val_y.max(), test_y.max()) + 1)
    if args.model == "roi_mlp":
        model: nn.Module = RoiMLP(n_nodes, in_dim, args.hidden_dim, num_classes, args.dropout)
    elif args.model == "gcn":
        model = DenseGCN(in_dim, args.hidden_dim, num_classes, args.dropout)
    else:
        model = DenseGAT(in_dim, args.hidden_dim, num_classes, args.dropout)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_top1 = -1.0
    best_val_top5 = -1.0
    best_epoch = -1
    bad_epochs = 0
    curve: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(train_x.shape[0])
        total_loss = 0.0
        total_correct = 0.0
        total_count = 0
        for start in range(0, train_x.shape[0], args.batch_size):
            idx = order[start : start + args.batch_size]
            xb = train_x_t[idx]
            yb = train_y_t[idx]
            logits = model(xb, adjacency_t)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item()) * xb.shape[0]
            total_correct += float((logits.argmax(dim=1) == yb).float().sum().item())
            total_count += xb.shape[0]
        val = evaluate(model, val_x_t, val_y_t, adjacency_t, args.batch_size)
        curve.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total_count, 1),
                "train_top1": total_correct / max(total_count, 1),
                "val_loss": val["loss"],
                "val_top1": val["top1"],
                "val_top5": val["top5"],
            }
        )
        improved = (val["top1"] > best_val_top1) or (np.isclose(val["top1"], best_val_top1) and val["top5"] > best_val_top5)
        if improved:
            best_val_top1 = float(val["top1"])
            best_val_top5 = float(val["top5"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break

    model.load_state_dict(best_state)
    test = evaluate(model, test_x_t, test_y_t, adjacency_t, args.batch_size)
    torch.save(best_state, args.output_dir / "checkpoint.pt")
    np.save(args.output_dir / "logits.npy", test["logits"])
    metrics = {
        "model": args.model,
        "fold": args.fold_name,
        "n_train": int(train_x.shape[0]),
        "n_val": int(val_x.shape[0]),
        "n_test": int(test_x.shape[0]),
        "n_nodes": int(n_nodes),
        "node_feature_dim": int(in_dim),
        "num_classes": int(num_classes),
        "chance_level": float(1.0 / num_classes),
        "top1": float(test["top1"]),
        "top5": float(test["top5"]),
        "best_val_top1": float(best_val_top1),
        "best_val_top5": float(best_val_top5),
        "best_epoch": int(best_epoch),
        "status": "ok",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.output_dir / "learning_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "train_top1", "val_loss", "val_top1", "val_top5"])
        writer.writeheader()
        writer.writerows(curve)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
