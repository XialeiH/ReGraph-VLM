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
    parser = argparse.ArgumentParser(description="Run the Stage 4 dynamic GNN pilot on one fold.")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--fold-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gnn-hidden-dim", type=int, default=64)
    parser.add_argument("--gnn-layers", type=int, default=1)
    parser.add_argument("--gru-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seeds", type=str, default="11,22")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


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


class GraphStep(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.neigh_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neigh = torch.einsum("ij,btjd->btid", adjacency, h)
        update = F.gelu(self.self_proj(h) + self.neigh_proj(neigh))
        return self.norm(h + self.dropout(update))


class DynamicGNNModel(nn.Module):
    def __init__(self, num_units: int, num_classes: int, hidden_dim: int, gnn_layers: int, gru_hidden_dim: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(1, hidden_dim)
        self.graph_layers = nn.ModuleList([GraphStep(hidden_dim=hidden_dim, dropout=dropout) for _ in range(gnn_layers)])
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(gru_hidden_dim),
            nn.Linear(gru_hidden_dim, gru_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_dim, num_classes),
        )
        self.num_units = num_units

    def forward(self, sequences: torch.Tensor, masks: torch.Tensor, adjacency: torch.Tensor) -> dict[str, torch.Tensor]:
        node_states = self.input_proj(sequences.unsqueeze(-1))
        for layer in self.graph_layers:
            node_states = layer(node_states, adjacency)
        step_repr = (sequences.unsqueeze(-1) * node_states).sum(dim=2)
        lengths = masks.sum(dim=1).long().clamp_min(1)
        packed = nn.utils.rnn.pack_padded_sequence(
            step_repr,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=step_repr.shape[1],
        )
        mask_f = masks.unsqueeze(-1)
        pooled = (out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        logits = self.head(pooled)
        return {"logits": logits, "sequence_repr": pooled}


def evaluate_model(
    model: DynamicGNNModel,
    x: torch.Tensor,
    m: torch.Tensor,
    y: torch.Tensor,
    adjacency: torch.Tensor,
    batch_size: int,
) -> dict[str, object]:
    model.eval()
    losses: list[float] = []
    logits_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size]
            mb = m[start : start + batch_size]
            yb = y[start : start + batch_size]
            outputs = model(xb, mb, adjacency)
            loss = F.cross_entropy(outputs["logits"], yb)
            losses.append(float(loss.item()) * xb.shape[0])
            logits_parts.append(outputs["logits"].cpu())
    logits = torch.cat(logits_parts, dim=0)
    total = max(x.shape[0], 1)
    return {
        "loss": sum(losses) / total,
        "top1": topk_accuracy(logits, y.cpu(), 1),
        "top5": topk_accuracy(logits, y.cpu(), min(5, logits.shape[1])),
        "logits": logits.numpy().astype(np.float32),
    }


def train_one_seed(
    seed: int,
    fit_x: np.ndarray,
    fit_m: np.ndarray,
    fit_y: np.ndarray,
    val_x: np.ndarray,
    val_m: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_m: np.ndarray,
    test_y: np.ndarray,
    adjacency: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    num_classes = int(max(fit_y.max(), val_y.max(), test_y.max()) + 1)
    model = DynamicGNNModel(
        num_units=fit_x.shape[2],
        num_classes=num_classes,
        hidden_dim=args.gnn_hidden_dim,
        gnn_layers=args.gnn_layers,
        gru_hidden_dim=args.gru_hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    fit_x_t = torch.from_numpy(fit_x).to(device)
    fit_m_t = torch.from_numpy(fit_m).to(device)
    fit_y_t = torch.from_numpy(fit_y).to(device)
    val_x_t = torch.from_numpy(val_x).to(device)
    val_m_t = torch.from_numpy(val_m).to(device)
    val_y_t = torch.from_numpy(val_y).to(device)
    test_x_t = torch.from_numpy(test_x).to(device)
    test_m_t = torch.from_numpy(test_m).to(device)
    test_y_t = torch.from_numpy(test_y).to(device)
    adjacency_t = torch.from_numpy(adjacency).to(device)

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_top1 = -1.0
    best_val_top5 = -1.0
    best_epoch = -1
    bad_epochs = 0
    learning_rows: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(fit_x.shape[0])
        train_loss = 0.0
        train_correct = 0.0
        train_count = 0
        for start in range(0, fit_x.shape[0], args.batch_size):
            idx = order[start : start + args.batch_size]
            xb = fit_x_t[idx]
            mb = fit_m_t[idx]
            yb = fit_y_t[idx]
            outputs = model(xb, mb, adjacency_t)
            loss = F.cross_entropy(outputs["logits"], yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += float(loss.item()) * xb.shape[0]
            train_correct += float((outputs["logits"].argmax(dim=1) == yb).float().sum().item())
            train_count += xb.shape[0]

        val_outputs = evaluate_model(model, val_x_t, val_m_t, val_y_t, adjacency_t, args.batch_size)
        learning_rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_loss / max(train_count, 1),
                "train_top1": train_correct / max(train_count, 1),
                "val_loss": val_outputs["loss"],
                "val_top1": val_outputs["top1"],
                "val_top5": val_outputs["top5"],
            }
        )

        improved = (val_outputs["top1"] > best_val_top1) or (
            np.isclose(val_outputs["top1"], best_val_top1) and val_outputs["top5"] > best_val_top5
        )
        if improved:
            best_val_top1 = float(val_outputs["top1"])
            best_val_top5 = float(val_outputs["top5"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break

    model.load_state_dict(best_state)
    val_outputs = evaluate_model(model, val_x_t, val_m_t, val_y_t, adjacency_t, args.batch_size)
    test_outputs = evaluate_model(model, test_x_t, test_m_t, test_y_t, adjacency_t, args.batch_size)
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_top1": best_val_top1,
        "best_val_top5": best_val_top5,
        "val_top1": val_outputs["top1"],
        "val_top5": val_outputs["top5"],
        "test_top1": test_outputs["top1"],
        "test_top5": test_outputs["top5"],
        "test_logits": test_outputs["logits"],
        "learning_rows": learning_rows,
        "state_dict": best_state,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.artifact_dir / f"{args.fold_name}_trial_sequences.npz")
    summary = json.loads((args.artifact_dir / f"{args.fold_name}_trial_sequence_summary.json").read_text(encoding="utf-8"))

    fit_x = data["fit_sequences"].astype(np.float32)
    fit_m = data["fit_masks"].astype(np.float32)
    fit_y = data["fit_labels"].astype(np.int64)
    val_x = data["val_sequences"].astype(np.float32)
    val_m = data["val_masks"].astype(np.float32)
    val_y = data["val_labels"].astype(np.int64)
    test_x = data["test_sequences"].astype(np.float32)
    test_m = data["test_masks"].astype(np.float32)
    test_y = data["test_labels"].astype(np.int64)
    adjacency = data["adjacency"].astype(np.float32)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    seeds = [int(token) for token in args.seeds.split(",") if token.strip()]
    results: list[dict[str, object]] = []
    learning_rows: list[dict[str, object]] = []
    for seed in seeds:
        result = train_one_seed(
            seed=seed,
            fit_x=fit_x,
            fit_m=fit_m,
            fit_y=fit_y,
            val_x=val_x,
            val_m=val_m,
            val_y=val_y,
            test_x=test_x,
            test_m=test_m,
            test_y=test_y,
            adjacency=adjacency,
            args=args,
            device=device,
        )
        results.append(result)
        learning_rows.extend(result["learning_rows"])
        torch.save(result["state_dict"], args.output_dir / f"dynamic_gnn_seed_{seed}_model.pt")

    best_result = max(results, key=lambda item: (float(item["best_val_top1"]), float(item["best_val_top5"])))
    logits = np.asarray(best_result["test_logits"], dtype=np.float32)

    metrics = {
        "model": "dynamic_gnn",
        "fold": args.fold_name,
        "held_out_subject": summary["held_out_subject"],
        "canonical_validation_subject": summary["canonical_validation_subject"],
        "n_train_sequences": int(fit_x.shape[0]),
        "n_val_sequences": int(val_x.shape[0]),
        "n_test_sequences": int(test_x.shape[0]),
        "n_train_trials": int(fit_m.sum()),
        "n_val_trials": int(val_m.sum()),
        "n_test_trials": int(test_m.sum()),
        "max_sequence_length": int(fit_x.shape[1]),
        "num_units": int(fit_x.shape[2]),
        "gnn_hidden_dim": int(args.gnn_hidden_dim),
        "gnn_layers": int(args.gnn_layers),
        "gru_hidden_dim": int(args.gru_hidden_dim),
        "chance_level": float(1.0 / logits.shape[1]),
        "top1_acc": float(np.mean([float(item["test_top1"]) for item in results])),
        "top1_std": float(np.std([float(item["test_top1"]) for item in results])),
        "top5_acc": float(np.mean([float(item["test_top5"]) for item in results])),
        "top5_std": float(np.std([float(item["test_top5"]) for item in results])),
        "best_seed": int(best_result["seed"]),
        "best_val_top1": float(best_result["best_val_top1"]),
        "best_val_top5": float(best_result["best_val_top5"]),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "status": "ok",
    }
    (args.output_dir / "dynamic_gnn_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    yaml_dump_simple(
        args.output_dir / "dynamic_gnn_run_config.yaml",
        {
            "fold": args.fold_name,
            "canonical_validation_subject": summary["canonical_validation_subject"],
            "gnn_hidden_dim": args.gnn_hidden_dim,
            "gnn_layers": args.gnn_layers,
            "gru_hidden_dim": args.gru_hidden_dim,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "seeds": seeds,
            "device": str(device),
        },
    )

    with (args.output_dir / "dynamic_gnn_learning_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seed", "epoch", "train_loss", "train_top1", "val_loss", "val_top1", "val_top5"],
        )
        writer.writeheader()
        writer.writerows(learning_rows)

    with (args.output_dir / "dynamic_gnn_seed_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
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

    np.save(args.output_dir / "dynamic_gnn_logits.npy", logits)


if __name__ == "__main__":
    main()
