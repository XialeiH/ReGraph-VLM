#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.bnt_encoder import BNTNativeEncoder, BNTTokenEncoder, TokenMLPEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train repeat-pair encoders for same-image matching.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fold", required=True, choices=["fold_01", "fold_04"])
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "roi_mlp",
            "gcn",
            "gcn_roiid_mean",
            "gcn_roiid_flat",
            "gat_roiid_flat",
            "bnt_token",
            "bnt_native_cosine",
            "bnt_native_outer",
            "bnt_native_hybrid",
            "token_mlp",
        ],
    )
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
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--loss-mode", default="bce", choices=["bce", "infonce", "bce_infonce"])
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--infonce-weight", type=float, default=1.0)
    parser.add_argument("--readout", default="flat", choices=["cls", "mean", "flat"])
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--roi-id-mode", default="normal", choices=["normal", "none", "shuffled"])
    parser.add_argument("--no-roi-id", action="store_true")
    parser.add_argument("--hybrid-alpha", type=float, default=0.5)
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
        "subject": torch.tensor([int(item["subject"]) for item in batch], dtype=torch.int64),
        "nsdId_1": torch.tensor([int(item["nsdId_1"]) for item in batch], dtype=torch.int64),
        "nsdId_2": torch.tensor([int(item["nsdId_2"]) for item in batch], dtype=torch.int64),
        "repeat_1": torch.tensor([int(item["repeat_1"]) for item in batch], dtype=torch.int64),
        "repeat_2": torch.tensor([int(item["repeat_2"]) for item in batch], dtype=torch.int64),
    }


class RoiMLPEncoder(nn.Module):
    def __init__(self, n_nodes: int, in_dim: int, hidden_dim: int, embedding_dim: int, dropout: float):
        super().__init__()
        flat_dim = n_nodes * in_dim
        self.net = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        z = self.net(x.flatten(start_dim=1))
        return F.normalize(z, dim=-1)


class DenseGCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = torch.einsum("ij,bjf->bif", adjacency, x)
        h = F.gelu(self.lin1(h))
        h = self.dropout(h)
        h = torch.einsum("ij,bjf->bif", adjacency, h)
        h = self.norm(F.gelu(self.lin2(h)))
        z = self.head(h.mean(dim=1))
        return F.normalize(z, dim=-1)


class RoiIdGCNEncoder(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        in_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
        readout: str,
    ):
        super().__init__()
        self.readout = readout
        self.feature = nn.Linear(in_dim, hidden_dim)
        self.roi_embedding = nn.Embedding(n_nodes, hidden_dim)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        if readout == "mean":
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embedding_dim),
            )
        elif readout == "flat":
            self.head = nn.Sequential(
                nn.LayerNorm(n_nodes * hidden_dim),
                nn.Linear(n_nodes * hidden_dim, max(embedding_dim * 2, hidden_dim)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(embedding_dim * 2, hidden_dim), embedding_dim),
            )
        else:
            raise ValueError(f"Unsupported readout: {readout}")

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        node_ids = torch.arange(x.shape[1], device=x.device)
        h = self.feature(x) + self.roi_embedding(node_ids)[None, :, :]
        h = torch.einsum("ij,bjf->bif", adjacency, h)
        h = self.norm1(F.gelu(self.lin1(h)))
        h = self.dropout(h)
        h = torch.einsum("ij,bjf->bif", adjacency, h)
        h = self.norm2(F.gelu(self.lin2(h)))
        if self.readout == "mean":
            z = self.head(h.mean(dim=1))
        else:
            z = self.head(h.flatten(start_dim=1))
        return F.normalize(z, dim=-1)


class RoiIdGATEncoder(nn.Module):
    def __init__(self, n_nodes: int, in_dim: int, hidden_dim: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.feature = nn.Linear(in_dim, hidden_dim)
        self.roi_embedding = nn.Embedding(n_nodes, hidden_dim)
        self.att_src = nn.Linear(hidden_dim, 1, bias=False)
        self.att_dst = nn.Linear(hidden_dim, 1, bias=False)
        self.out1 = nn.Linear(hidden_dim, hidden_dim)
        self.out2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(n_nodes * hidden_dim),
            nn.Linear(n_nodes * hidden_dim, max(embedding_dim * 2, hidden_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(embedding_dim * 2, hidden_dim), embedding_dim),
        )

    def attention_layer(self, h: torch.Tensor, adjacency: torch.Tensor, out: nn.Linear, norm: nn.LayerNorm) -> torch.Tensor:
        scores = self.att_src(h) + self.att_dst(h).transpose(1, 2)
        scores = F.leaky_relu(scores, negative_slope=0.2)
        mask = adjacency > 0
        scores = scores.masked_fill(~mask[None, :, :], -1e9)
        attn = torch.softmax(scores, dim=-1)
        h_next = torch.bmm(attn, h)
        return norm(F.gelu(out(self.dropout(h_next))))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        node_ids = torch.arange(x.shape[1], device=x.device)
        h = self.feature(x) + self.roi_embedding(node_ids)[None, :, :]
        h = self.attention_layer(h, adjacency, self.out1, self.norm1)
        h = self.attention_layer(h, adjacency, self.out2, self.norm2)
        z = self.head(h.flatten(start_dim=1))
        return F.normalize(z, dim=-1)


class PairModel(nn.Module):
    def __init__(self, encoder: nn.Module, init_scale: float = 10.0):
        super().__init__()
        self.encoder = encoder
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale), dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def encode(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, adjacency)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        z1 = self.encode(x1, adjacency)
        z2 = self.encode(x2, adjacency)
        cos = (z1 * z2).sum(dim=-1)
        return self.log_scale.exp().clamp(max=100.0) * cos + self.bias


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
    tp = np.cumsum(sorted_labels)
    precision = tp / np.arange(1, len(labels) + 1)
    return float((precision * sorted_labels).sum() / n_pos)


def accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    return float((labels.astype(np.int64) == preds.astype(np.int64)).mean())


def balanced_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    preds = preds.astype(np.int64)
    pos = labels == 1
    neg = labels == 0
    tpr = float((preds[pos] == 1).mean()) if pos.any() else float("nan")
    tnr = float((preds[neg] == 0).mean()) if neg.any() else float("nan")
    return float((tpr + tnr) / 2.0)


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    candidates = np.unique(scores)
    if len(candidates) > 5000:
        candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 5000)))
    best = (float(candidates[0]), -1.0, -1.0)
    for threshold in candidates:
        preds = (scores >= threshold).astype(np.int64)
        bal = balanced_accuracy(labels, preds)
        acc = accuracy(labels, preds)
        if bal > best[1] or (math.isclose(bal, best[1]) and acc > best[2]):
            best = (float(threshold), float(bal), float(acc))
    return best


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (scores >= threshold).astype(np.int64)
    return {
        "auroc": auroc(labels, scores),
        "auprc": average_precision(labels, scores),
        "accuracy": accuracy(labels, preds),
        "balanced_accuracy": balanced_accuracy(labels, preds),
    }


def collect_scores(
    model: PairModel,
    loader: DataLoader,
    adjacency: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for batch in loader:
            x1 = batch["x1"].to(device)
            x2 = batch["x2"].to(device)
            y = batch["same_image"].to(device)
            logits = model(x1, x2, adjacency)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            probs = torch.sigmoid(logits)
            scores.append(probs.cpu().numpy())
            labels.append(y.cpu().numpy())
            total_loss += float(loss.item()) * y.numel()
            total_n += y.numel()
    return np.concatenate(labels), np.concatenate(scores), total_loss / max(total_n, 1)


def infonce_loss_from_batch(model: PairModel, x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor, adjacency: torch.Tensor, temperature: float) -> torch.Tensor:
    pos = y > 0.5
    if int(pos.sum().item()) < 2:
        return x1.sum() * 0.0
    z1 = model.encode(x1[pos], adjacency)
    z2 = model.encode(x2[pos], adjacency)
    logits = (z1 @ z2.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def encode_positive_pairs(
    model: PairModel,
    pairs: list[dict[str, object]],
    adjacency: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, object]]:
    positives = [pair for pair in pairs if int(pair["same_image"]) == 1]
    out: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(positives), batch_size):
            chunk = positives[start : start + batch_size]
            x1 = torch.stack([pair["x1"].float() for pair in chunk]).to(device)  # type: ignore[list-item]
            x2 = torch.stack([pair["x2"].float() for pair in chunk]).to(device)  # type: ignore[list-item]
            z1 = model.encode(x1, adjacency).cpu().numpy()
            z2 = model.encode(x2, adjacency).cpu().numpy()
            for idx, pair in enumerate(chunk):
                out.append(
                    {
                        "subject": int(pair["subject"]),
                        "repeat_1": int(pair["repeat_1"]),
                        "repeat_2": int(pair["repeat_2"]),
                        "nsdId_1": int(pair["nsdId_1"]),
                        "nsdId_2": int(pair["nsdId_2"]),
                        "z1": z1[idx],
                        "z2": z2[idx],
                    }
                )
    return out


def retrieval_metrics(
    model: PairModel,
    pairs: list[dict[str, object]],
    adjacency: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    encoded = encode_positive_pairs(model, pairs, adjacency, device, batch_size)
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
            pos = np.where(candidate_ids[order] == true_ids[idx])[0]
            if len(pos) == 0:
                continue
            rank = int(pos[0]) + 1
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


def write_learning_curve(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_auroc",
                "train_auprc",
                "val_loss",
                "val_auroc",
                "val_auprc",
                "val_balanced_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    fold_dir = root / args.dataset_root / args.fold
    model_dir = args.model
    if args.no_roi_id:
        args.roi_id_mode = "none"
    if args.model.startswith("bnt_") or args.model == "token_mlp":
        roi_tag = f"roi_{args.roi_id_mode}"
        model_dir = f"{args.model}_{args.readout}_{roi_tag}"
    out_dir = root / args.output_root / model_dir / args.adjacency / args.fold
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

    if args.model == "roi_mlp":
        encoder: nn.Module = RoiMLPEncoder(n_nodes, in_dim, args.hidden_dim, args.embedding_dim, args.dropout)
    elif args.model == "gcn":
        encoder = DenseGCNEncoder(in_dim, args.hidden_dim, args.embedding_dim, args.dropout)
    elif args.model == "gcn_roiid_mean":
        encoder = RoiIdGCNEncoder(n_nodes, in_dim, args.hidden_dim, args.embedding_dim, args.dropout, readout="mean")
    elif args.model == "gcn_roiid_flat":
        encoder = RoiIdGCNEncoder(n_nodes, in_dim, args.hidden_dim, args.embedding_dim, args.dropout, readout="flat")
    elif args.model == "gat_roiid_flat":
        encoder = RoiIdGATEncoder(n_nodes, in_dim, args.hidden_dim, args.embedding_dim, args.dropout)
    elif args.model == "bnt_token":
        encoder = BNTTokenEncoder(
            n_nodes=n_nodes,
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            readout=args.readout,
            roi_id_mode=args.roi_id_mode,
        )
    elif args.model == "token_mlp":
        encoder = TokenMLPEncoder(
            n_nodes=n_nodes,
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            roi_id_mode=args.roi_id_mode,
        )
    else:
        native_mode = args.model.replace("bnt_native_", "")
        encoder = BNTNativeEncoder(
            n_nodes=n_nodes,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            readout=args.readout,
            native_mode=native_mode,
            hybrid_alpha=args.hybrid_alpha,
        )
    model = PairModel(encoder).to(device)
    adjacency_np = load_adjacency(fold_dir, args.adjacency, args.seed)
    adjacency = torch.from_numpy(adjacency_np).to(device)

    train_loader = DataLoader(PairDataset(train_pairs), batch_size=args.batch_size, shuffle=True, collate_fn=collate_pairs)
    val_loader = DataLoader(PairDataset(val_pairs), batch_size=args.batch_size, shuffle=False, collate_fn=collate_pairs)
    test_loader = DataLoader(PairDataset(test_pairs), batch_size=args.batch_size, shuffle=False, collate_fn=collate_pairs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_auroc = -1.0
    best_epoch = -1
    bad_epochs = 0
    curve: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_n = 0
        for batch in train_loader:
            x1 = batch["x1"].to(device)
            x2 = batch["x2"].to(device)
            y = batch["same_image"].to(device)
            logits = model(x1, x2, adjacency)
            if args.loss_mode == "bce":
                loss = F.binary_cross_entropy_with_logits(logits, y)
            elif args.loss_mode == "infonce":
                loss = infonce_loss_from_batch(model, x1, x2, y, adjacency, args.temperature)
            else:
                bce = F.binary_cross_entropy_with_logits(logits, y)
                nce = infonce_loss_from_batch(model, x1, x2, y, adjacency, args.temperature)
                loss = bce + args.infonce_weight * nce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_loss += float(loss.item()) * y.numel()
            train_n += y.numel()

        train_labels, train_scores, train_eval_loss = collect_scores(model, train_loader, adjacency, device)
        val_labels, val_scores, val_loss = collect_scores(model, val_loader, adjacency, device)
        val_threshold, val_bal, _val_acc = best_threshold(val_labels, val_scores)
        train_row = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_n, 1),
            "train_auroc": auroc(train_labels, train_scores),
            "train_auprc": average_precision(train_labels, train_scores),
            "val_loss": val_loss,
            "val_auroc": auroc(val_labels, val_scores),
            "val_auprc": average_precision(val_labels, val_scores),
            "val_balanced_accuracy": val_bal,
        }
        curve.append(train_row)

        if train_row["val_auroc"] > best_val_auroc:
            best_val_auroc = float(train_row["val_auroc"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break

    model.load_state_dict(best_state)
    val_labels, val_scores, val_loss = collect_scores(model, val_loader, adjacency, device)
    threshold, val_bal, val_acc = best_threshold(val_labels, val_scores)
    test_labels, test_scores, test_loss = collect_scores(model, test_loader, adjacency, device)
    test_binary = binary_metrics(test_labels, test_scores, threshold)
    test_retrieval = retrieval_metrics(model, test_pairs, adjacency, device, args.batch_size)

    metrics = {
        "model": args.model,
        "fold": args.fold,
        "adjacency": args.adjacency,
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "n_test_pairs": len(test_pairs),
        "n_nodes": n_nodes,
        "node_feature_dim": in_dim,
        "embedding_dim": args.embedding_dim,
        "loss_mode": args.loss_mode,
        "bnt_input_type": "token" if args.model == "bnt_token" else ("native" if args.model.startswith("bnt_native") else ""),
        "readout": args.readout if args.model.startswith("bnt_") or args.model == "token_mlp" else "",
        "roi_id": args.roi_id_mode if args.model.startswith("bnt_") or args.model == "token_mlp" else "",
        "num_heads": args.num_heads if args.model.startswith("bnt_") else "",
        "num_layers": args.num_layers if args.model.startswith("bnt_") else "",
        "hybrid_alpha": args.hybrid_alpha if args.model == "bnt_native_hybrid" else "",
        "temperature": args.temperature,
        "infonce_weight": args.infonce_weight,
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
        "val_loss": val_loss,
        "val_auroc": auroc(val_labels, val_scores),
        "val_auprc": average_precision(val_labels, val_scores),
        "val_threshold": threshold,
        "val_balanced_accuracy": val_bal,
        "val_accuracy": val_acc,
        "test_loss": test_loss,
        "auroc": test_binary["auroc"],
        "auprc": test_binary["auprc"],
        "balanced_accuracy": test_binary["balanced_accuracy"],
        "accuracy": test_binary["accuracy"],
        "recall_at_1": test_retrieval["recall_at_1"],
        "recall_at_5": test_retrieval["recall_at_5"],
        "mrr": test_retrieval["mrr"],
        "n_retrieval_queries": test_retrieval["n_queries"],
        "status": "ok",
    }

    torch.save({"model_state": best_state, "args": vars(args), "metrics": metrics}, out_dir / "checkpoint.pt")
    np.savez_compressed(out_dir / "test_scores.npz", labels=test_labels, scores=test_scores)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_learning_curve(out_dir / "learning_curve.csv", curve)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
