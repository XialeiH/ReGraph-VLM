#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BrainGNN smoke training on an NSD ROI graph fold.")
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--fold-name", type=str, required=True)
    parser.add_argument("--braingnn-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def load_braingnn_network(root: Path):
    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from net.braingnn import Network  # noqa: PLC0415

    return Network


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    topk = torch.topk(logits, k=min(k, logits.shape[1]), dim=1).indices
    return float((topk == labels[:, None]).any(dim=1).float().mean().item())


def convert_graphs(path: Path, n_nodes: int) -> list[Data]:
    graphs = torch.load(path, map_location="cpu", weights_only=False)
    pos = torch.eye(n_nodes, dtype=torch.float32)
    data_list: list[Data] = []
    for graph in graphs:
        edge_attr = graph["edge_attr"].float()
        if edge_attr.ndim == 2 and edge_attr.shape[1] == 1:
            edge_attr = edge_attr.view(-1)
        data = Data(
            x=graph["x"].float(),
            edge_index=graph["edge_index"].long(),
            edge_attr=edge_attr,
            y=torch.tensor([int(graph["y"])], dtype=torch.long),
            pos=pos.clone(),
        )
        data.subject = graph["subject"]
        data.nsdId = int(graph["nsdId"])
        data_list.append(data)
    return data_list


def eval_model(model: torch.nn.Module, loader: DataLoader, device: torch.device, nclass: int) -> dict[str, object]:
    model.eval()
    losses: list[float] = []
    logits_parts: list[torch.Tensor] = []
    labels_parts: list[torch.Tensor] = []
    score_sums = None
    score_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output, _w1, _w2, s1, _s2 = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr, batch.pos)
            losses.append(float(F.nll_loss(output, batch.y).item()) * batch.num_graphs)
            logits_parts.append(output.cpu())
            labels_parts.append(batch.y.cpu())
            s1_cpu = s1.detach().cpu()
            if s1_cpu.ndim == 2:
                batch_score = s1_cpu.mean(dim=0)
                score_sums = batch_score if score_sums is None else score_sums + batch_score
                score_count += 1
    logits = torch.cat(logits_parts, dim=0)
    labels = torch.cat(labels_parts, dim=0)
    roi_scores = (score_sums / max(score_count, 1)).numpy().astype(np.float32) if score_sums is not None else np.zeros((0,), dtype=np.float32)
    return {
        "loss": sum(losses) / max(labels.numel(), 1),
        "top1": topk_accuracy(logits, labels, 1),
        "top5": topk_accuracy(logits, labels, min(5, nclass)),
        "logits": logits.numpy().astype(np.float32),
        "labels": labels.numpy().astype(np.int64),
        "roi_scores": roi_scores,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    qc = json.loads((args.graph_dir / "graph_dataset_qc.json").read_text(encoding="utf-8"))
    n_nodes = int(qc["n_nodes"])
    indim = int(qc["node_feature_dim"])
    nclass = int(qc["num_classes"])
    train_data = convert_graphs(args.graph_dir / "train_graphs.pt", n_nodes)
    val_data = convert_graphs(args.graph_dir / "val_graphs.pt", n_nodes)
    test_data = convert_graphs(args.graph_dir / "test_graphs.pt", n_nodes)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    Network = load_braingnn_network(args.braingnn_root)
    model = Network(indim=indim, ratio=args.ratio, nclass=nclass, R=n_nodes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_top1 = -1.0
    best_val_top5 = -1.0
    best_epoch = -1
    bad_epochs = 0
    curve: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0.0
        train_count = 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            output, _w1, _w2, _s1, _s2 = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr, batch.pos)
            loss = F.nll_loss(output, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += float(loss.item()) * batch.num_graphs
            train_correct += float((output.argmax(dim=1) == batch.y).float().sum().item())
            train_count += batch.num_graphs
        val = eval_model(model, val_loader, device, nclass)
        curve.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(train_count, 1),
                "train_top1": train_correct / max(train_count, 1),
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
    test = eval_model(model, test_loader, device, nclass)
    torch.save(best_state, args.output_dir / "checkpoint.pt")
    np.save(args.output_dir / "logits.npy", test["logits"])
    metrics = {
        "model": "braingnn",
        "fold": args.fold_name,
        "n_train": len(train_data),
        "n_val": len(val_data),
        "n_test": len(test_data),
        "n_nodes": n_nodes,
        "node_feature_dim": indim,
        "num_classes": nclass,
        "chance_level": float(1.0 / nclass),
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
    roi_scores = test["roi_scores"]
    with (args.output_dir / "roi_importance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node_index", "importance"])
        writer.writeheader()
        for idx, score in enumerate(roi_scores.tolist()):
            writer.writerow({"node_index": idx, "importance": f"{float(score):.8f}"})
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
