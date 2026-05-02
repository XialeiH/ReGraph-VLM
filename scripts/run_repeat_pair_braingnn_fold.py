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
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BrainGNN Siamese encoder for repeat-pair matching.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fold", required=True, choices=["fold_01", "fold_04"])
    parser.add_argument("--braingnn-root", type=Path, default=Path("external/BrainGNN_Pytorch"))
    parser.add_argument("--adjacency", default="topk20", choices=["topk20", "identity", "shuffled"])
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/repeat_pair_encoder_results"),
    )
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


class PairDataset(Dataset):
    def __init__(self, pairs: list[dict[str, object]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, object]:
        pair = self.pairs[idx]
        return {
            "x1": pair["x1"].float(),
            "x2": pair["x2"].float(),
            "same_image": torch.tensor(float(pair["same_image"]), dtype=torch.float32),
            "subject": int(pair["subject"]),
            "nsdId_1": int(pair["nsdId_1"]),
            "nsdId_2": int(pair["nsdId_2"]),
            "repeat_1": int(pair["repeat_1"]),
            "repeat_2": int(pair["repeat_2"]),
        }


def collate_pairs(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "x1": torch.stack([item["x1"] for item in batch]),  # type: ignore[list-item]
        "x2": torch.stack([item["x2"] for item in batch]),  # type: ignore[list-item]
        "same_image": torch.stack([item["same_image"] for item in batch]),  # type: ignore[list-item]
    }


def load_network(root: Path):
    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from net.braingnn import Network  # noqa: PLC0415

    return Network


class BrainGNNEncoder(nn.Module):
    def __init__(self, braingnn_root: Path, indim: int, ratio: float, n_nodes: int, embedding_dim: int):
        super().__init__()
        Network = load_network(braingnn_root)
        self.net = Network(indim=indim, ratio=ratio, nclass=2, R=n_nodes)
        self.proj = nn.Linear(512, embedding_dim)

    def forward(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.net.conv1(batch.x, batch.edge_index, batch.edge_attr, batch.pos)
        x, edge_index, edge_attr, batch_vec, perm, score1 = self.net.pool1(x, batch.edge_index, batch.edge_attr, batch.batch)
        pos = batch.pos[perm]
        x1 = torch.cat([gmp(x, batch_vec), gap(x, batch_vec)], dim=1)

        edge_attr = edge_attr.squeeze()
        edge_index, edge_attr = self.net.augment_adj(edge_index, edge_attr, x.size(0))

        x = self.net.conv2(x, edge_index, edge_attr, pos)
        x, edge_index, edge_attr, batch_vec, _perm, _score2 = self.net.pool2(x, edge_index, edge_attr, batch_vec)
        x2 = torch.cat([gmp(x, batch_vec), gap(x, batch_vec)], dim=1)

        x = torch.cat([x1, x2], dim=1)
        x = self.net.bn1(F.relu(self.net.fc1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.net.bn2(F.relu(self.net.fc2(x)))
        z = F.normalize(self.proj(x), dim=-1)
        return z, torch.sigmoid(score1).detach()


class BrainGNNPairModel(nn.Module):
    def __init__(self, encoder: BrainGNNEncoder, init_scale: float = 10.0):
        super().__init__()
        self.encoder = encoder
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale), dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def encode(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(batch)

    def forward(self, b1: Batch, b2: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        z1, s1 = self.encode(b1)
        z2, _s2 = self.encode(b2)
        cos = (z1 * z2).sum(dim=-1)
        return self.log_scale.exp().clamp(max=100.0) * cos + self.bias, s1


def edge_index_from_adjacency(adjacency: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    src, dst = np.nonzero(adjacency)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_attr = torch.tensor(adjacency[src, dst], dtype=torch.float32)
    return edge_index, edge_attr


def load_adjacency(fold_dir: Path, mode: str, seed: int) -> np.ndarray:
    base = np.load(fold_dir / "adjacency.npy").astype(np.float32)
    if mode == "identity":
        return np.eye(base.shape[0], dtype=np.float32)
    if mode == "shuffled":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(base.shape[0])
        base = base[perm][:, perm]
    base = np.abs(base)
    np.fill_diagonal(base, 1.0)
    return base


def tensors_to_batch(x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, pos: torch.Tensor) -> Batch:
    data_list = [Data(x=x[idx], edge_index=edge_index, edge_attr=edge_attr, pos=pos) for idx in range(x.shape[0])]
    return Batch.from_data_list(data_list)


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


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata_average(scores)
    pos_rank_sum = float(ranks[labels == 1].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float((precision * sorted_labels).sum() / n_pos)


def balanced_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    preds = preds.astype(np.int64)
    pos = labels == 1
    neg = labels == 0
    tpr = float((preds[pos] == 1).mean()) if pos.any() else float("nan")
    tnr = float((preds[neg] == 0).mean()) if neg.any() else float("nan")
    return float((tpr + tnr) / 2.0)


def accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    return float((labels.astype(np.int64) == preds.astype(np.int64)).mean())


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    candidates = np.unique(scores)
    if len(candidates) > 5000:
        candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 5000)))
    best = (float(candidates[0]), -1.0, -1.0)
    for threshold in candidates:
        pred = (scores >= threshold).astype(np.int64)
        bal = balanced_accuracy(labels, pred)
        acc = accuracy(labels, pred)
        if bal > best[1] or (math.isclose(bal, best[1]) and acc > best[2]):
            best = (float(threshold), float(bal), float(acc))
    return best


def collect_scores(
    model: BrainGNNPairModel,
    loader: DataLoader,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    pos: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for batch in loader:
            b1 = tensors_to_batch(batch["x1"], edge_index, edge_attr, pos).to(device)
            b2 = tensors_to_batch(batch["x2"], edge_index, edge_attr, pos).to(device)
            y = batch["same_image"].to(device)
            logits, _scores = model(b1, b2)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            labels.append(y.cpu().numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())
            total_loss += float(loss.item()) * y.numel()
            total_n += y.numel()
    return np.concatenate(labels), np.concatenate(scores), total_loss / max(total_n, 1)


def encode_positive_pairs(
    model: BrainGNNPairModel,
    pairs: list[dict[str, object]],
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    pos: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, object]]:
    positives = [pair for pair in pairs if int(pair["same_image"]) == 1]
    out: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(positives), batch_size):
            chunk = positives[start : start + batch_size]
            x1 = torch.stack([pair["x1"].float() for pair in chunk])
            x2 = torch.stack([pair["x2"].float() for pair in chunk])
            b1 = tensors_to_batch(x1, edge_index, edge_attr, pos).to(device)
            b2 = tensors_to_batch(x2, edge_index, edge_attr, pos).to(device)
            z1, _s1 = model.encode(b1)
            z2, _s2 = model.encode(b2)
            z1_np = z1.cpu().numpy()
            z2_np = z2.cpu().numpy()
            for idx, pair in enumerate(chunk):
                out.append(
                    {
                        "subject": int(pair["subject"]),
                        "repeat_1": int(pair["repeat_1"]),
                        "repeat_2": int(pair["repeat_2"]),
                        "nsdId_1": int(pair["nsdId_1"]),
                        "nsdId_2": int(pair["nsdId_2"]),
                        "z1": z1_np[idx],
                        "z2": z2_np[idx],
                    }
                )
    return out


def retrieval_metrics(model: BrainGNNPairModel, pairs: list[dict[str, object]], edge_index: torch.Tensor, edge_attr: torch.Tensor, pos: torch.Tensor, device: torch.device, batch_size: int) -> dict[str, float]:
    encoded = encode_positive_pairs(model, pairs, edge_index, edge_attr, pos, device, batch_size)
    groups: dict[tuple[int, int, int], list[dict[str, object]]] = {}
    for row in encoded:
        groups.setdefault((int(row["subject"]), int(row["repeat_1"]), int(row["repeat_2"])), []).append(row)

    recall1 = 0
    recall5 = 0
    rr: list[float] = []
    n_queries = 0
    for rows in groups.values():
        rows = sorted(rows, key=lambda row: int(row["nsdId_1"]))
        z1 = np.stack([row["z1"] for row in rows], axis=0)
        z2 = np.stack([row["z2"] for row in rows], axis=0)
        candidate_ids = np.array([int(row["nsdId_2"]) for row in rows])
        true_ids = np.array([int(row["nsdId_1"]) for row in rows])
        scores = z1 @ z2.T
        for idx in range(scores.shape[0]):
            order = np.argsort(-scores[idx], kind="mergesort")
            hit = np.where(candidate_ids[order] == true_ids[idx])[0]
            if len(hit) == 0:
                continue
            rank = int(hit[0]) + 1
            recall1 += int(rank <= 1)
            recall5 += int(rank <= 5)
            rr.append(1.0 / rank)
            n_queries += 1
    if n_queries == 0:
        return {"recall_at_1": float("nan"), "recall_at_5": float("nan"), "mrr": float("nan"), "n_queries": 0}
    return {
        "recall_at_1": float(recall1 / n_queries),
        "recall_at_5": float(recall5 / n_queries),
        "mrr": float(np.mean(rr)),
        "n_queries": int(n_queries),
    }


def load_pairs(fold_dir: Path, split: str) -> list[dict[str, object]]:
    return torch.load(fold_dir / f"{split}_pairs.pt", map_location="cpu", weights_only=False)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    fold_dir = root / args.dataset_root / args.fold
    out_dir = root / args.output_root / "braingnn" / args.adjacency / args.fold
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    train_pairs = load_pairs(fold_dir, "train")
    val_pairs = load_pairs(fold_dir, "val")
    test_pairs = load_pairs(fold_dir, "test")
    first_x = train_pairs[0]["x1"]
    n_nodes, in_dim = int(first_x.shape[0]), int(first_x.shape[1])  # type: ignore[union-attr]
    adjacency = load_adjacency(fold_dir, args.adjacency, args.seed)
    edge_index, edge_attr = edge_index_from_adjacency(adjacency)
    pos = torch.eye(n_nodes, dtype=torch.float32)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    pos = pos.to(device)

    model = BrainGNNPairModel(
        BrainGNNEncoder(root / args.braingnn_root, in_dim, args.ratio, n_nodes, args.embedding_dim)
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(PairDataset(train_pairs), batch_size=args.batch_size, shuffle=True, collate_fn=collate_pairs)
    val_loader = DataLoader(PairDataset(val_pairs), batch_size=args.batch_size, shuffle=False, collate_fn=collate_pairs)
    test_loader = DataLoader(PairDataset(test_pairs), batch_size=args.batch_size, shuffle=False, collate_fn=collate_pairs)

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_auroc = -1.0
    best_epoch = -1
    bad_epochs = 0
    curve: list[dict[str, object]] = []
    roi_score_sum = torch.zeros(n_nodes)
    roi_score_count = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_n = 0
        for batch in train_loader:
            b1 = tensors_to_batch(batch["x1"], edge_index.cpu(), edge_attr.cpu(), pos.cpu()).to(device)
            b2 = tensors_to_batch(batch["x2"], edge_index.cpu(), edge_attr.cpu(), pos.cpu()).to(device)
            y = batch["same_image"].to(device)
            logits, scores = model(b1, b2)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_loss += float(loss.item()) * y.numel()
            train_n += y.numel()
            if scores.numel() > 0 and scores.shape[-1] == n_nodes:
                roi_score_sum += scores.detach().cpu().mean(dim=0)
                roi_score_count += 1

        train_labels, train_scores, _train_eval_loss = collect_scores(model, train_loader, edge_index.cpu(), edge_attr.cpu(), pos.cpu(), device)
        val_labels, val_scores, val_loss = collect_scores(model, val_loader, edge_index.cpu(), edge_attr.cpu(), pos.cpu(), device)
        val_threshold, val_bal, _val_acc = best_threshold(val_labels, val_scores)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_n, 1),
            "train_auroc": auroc(train_labels, train_scores),
            "train_auprc": average_precision(train_labels, train_scores),
            "val_loss": val_loss,
            "val_auroc": auroc(val_labels, val_scores),
            "val_auprc": average_precision(val_labels, val_scores),
            "val_balanced_accuracy": val_bal,
        }
        curve.append(row)
        if row["val_auroc"] > best_val_auroc:
            best_val_auroc = float(row["val_auroc"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break

    model.load_state_dict(best_state)
    val_labels, val_scores, val_loss = collect_scores(model, val_loader, edge_index.cpu(), edge_attr.cpu(), pos.cpu(), device)
    threshold, val_bal, val_acc = best_threshold(val_labels, val_scores)
    test_labels, test_scores, test_loss = collect_scores(model, test_loader, edge_index.cpu(), edge_attr.cpu(), pos.cpu(), device)
    test_pred = (test_scores >= threshold).astype(np.int64)
    test_retrieval = retrieval_metrics(model, test_pairs, edge_index.cpu(), edge_attr.cpu(), pos.cpu(), device, args.batch_size)
    metrics = {
        "model": "braingnn",
        "fold": args.fold,
        "adjacency": args.adjacency,
        "loss_mode": "bce",
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "n_test_pairs": len(test_pairs),
        "n_nodes": n_nodes,
        "node_feature_dim": in_dim,
        "embedding_dim": args.embedding_dim,
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
        "val_loss": val_loss,
        "val_auroc": auroc(val_labels, val_scores),
        "val_auprc": average_precision(val_labels, val_scores),
        "val_threshold": threshold,
        "val_balanced_accuracy": val_bal,
        "val_accuracy": val_acc,
        "test_loss": test_loss,
        "auroc": auroc(test_labels, test_scores),
        "auprc": average_precision(test_labels, test_scores),
        "balanced_accuracy": balanced_accuracy(test_labels, test_pred),
        "accuracy": accuracy(test_labels, test_pred),
        "recall_at_1": test_retrieval["recall_at_1"],
        "recall_at_5": test_retrieval["recall_at_5"],
        "mrr": test_retrieval["mrr"],
        "n_retrieval_queries": test_retrieval["n_queries"],
        "status": "ok",
    }
    torch.save({"model_state": best_state, "args": vars(args), "metrics": metrics}, out_dir / "checkpoint.pt")
    np.savez_compressed(out_dir / "test_scores.npz", labels=test_labels, scores=test_scores)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out_dir / "learning_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "train_auroc", "train_auprc", "val_loss", "val_auroc", "val_auprc", "val_balanced_accuracy"])
        writer.writeheader()
        writer.writerows(curve)
    roi_importance = roi_score_sum / max(roi_score_count, 1)
    with (out_dir / "roi_importance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node_index", "importance"])
        writer.writeheader()
        for idx, score in enumerate(roi_importance.tolist()):
            writer.writerow({"node_index": idx, "importance": score})
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
