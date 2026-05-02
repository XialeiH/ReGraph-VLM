#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stage 3A light interaction pilot on one fold.")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--fold-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--node-feature-mode",
        choices=["assignment_only", "assignment_weighted_embedding"],
        default="assignment_only",
    )
    parser.add_argument(
        "--readout-mode",
        choices=["fixed_pooling", "attention_pooling"],
        default="fixed_pooling",
    )
    parser.add_argument("--direct-path-hidden-dim", type=int, default=128)
    parser.add_argument("--disable-direct-path", action="store_true")
    parser.add_argument("--use-gated-message-passing", action="store_true")
    parser.add_argument("--message-gate-init", type=float, default=-2.0)
    parser.add_argument("--node-hidden-dim", type=int, default=256)
    parser.add_argument("--message-passing-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seeds", type=str, default="701,702,703")
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


class InteractionBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, use_gated_message_passing: bool, message_gate_init: float):
        super().__init__()
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.neigh_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.use_gated_message_passing = use_gated_message_passing
        if use_gated_message_passing:
            self.message_gate = nn.Parameter(torch.full((1,), float(message_gate_init), dtype=torch.float32))
        else:
            self.register_parameter("message_gate", None)

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neigh = torch.einsum("ij,bjd->bid", adjacency, h)
        update = F.gelu(self.self_proj(h) + self.neigh_proj(neigh))
        if self.use_gated_message_passing:
            update = torch.sigmoid(self.message_gate) * update
        return self.norm(h + self.dropout(update))


class LightInteractionModel(nn.Module):
    def __init__(
        self,
        prototype_embeddings: np.ndarray,
        num_classes: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        node_feature_mode: str,
        readout_mode: str,
        direct_path_hidden_dim: int,
        disable_direct_path: bool,
        use_gated_message_passing: bool,
        message_gate_init: float,
    ):
        super().__init__()
        proto = torch.from_numpy(prototype_embeddings.astype(np.float32))
        self.prototype_embeddings = nn.Parameter(proto.clone())
        self.node_feature_mode = node_feature_mode
        self.readout_mode = readout_mode
        self.disable_direct_path = disable_direct_path
        if node_feature_mode == "assignment_weighted_embedding":
            input_dim = int(proto.shape[1])
        else:
            input_dim = 1
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                InteractionBlock(
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    use_gated_message_passing=use_gated_message_passing,
                    message_gate_init=message_gate_init,
                )
                for _ in range(layers)
            ]
        )
        if readout_mode == "attention_pooling":
            self.attention_pool = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            readout_dim = hidden_dim
        else:
            self.attention_pool = None
            readout_dim = hidden_dim * 3
        if disable_direct_path:
            self.direct_path = None
            final_dim = readout_dim
        else:
            self.direct_path = nn.Sequential(
                nn.LayerNorm(proto.shape[0]),
                nn.Linear(proto.shape[0], direct_path_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            final_dim = readout_dim + direct_path_hidden_dim
        self.readout_norm = nn.LayerNorm(final_dim)
        self.head = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, assignments: torch.Tensor, adjacency: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.node_feature_mode == "assignment_weighted_embedding":
            h0 = assignments.unsqueeze(-1) * self.prototype_embeddings.unsqueeze(0)
        else:
            h0 = assignments.unsqueeze(-1)
        h = self.input_proj(h0)
        for block in self.blocks:
            h = block(h, adjacency)
        if self.readout_mode == "attention_pooling":
            attn_logits = self.attention_pool(h).squeeze(-1)
            attn_weights = torch.softmax(attn_logits, dim=1)
            pooled_parts = [torch.sum(attn_weights.unsqueeze(-1) * h, dim=1)]
        else:
            weighted_pool = (assignments.unsqueeze(-1) * h).sum(dim=1)
            mean_pool = h.mean(dim=1)
            max_pool = h.max(dim=1).values
            pooled_parts = [weighted_pool, mean_pool, max_pool]
        if self.direct_path is not None:
            pooled_parts.append(self.direct_path(assignments))
        pooled = torch.cat(pooled_parts, dim=1)
        logits = self.head(self.readout_norm(pooled))
        return {"logits": logits, "node_states": h}


@dataclass
class EvalOutputs:
    loss: float
    top1: float
    top5: float
    logits: np.ndarray


def evaluate_model(
    model: LightInteractionModel,
    x: torch.Tensor,
    y: torch.Tensor,
    adjacency: torch.Tensor,
    batch_size: int,
) -> EvalOutputs:
    model.eval()
    losses: list[float] = []
    logits_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size]
            yb = y[start : start + batch_size]
            outputs = model(xb, adjacency)
            loss = F.cross_entropy(outputs["logits"], yb)
            losses.append(float(loss.item()) * xb.shape[0])
            logits_parts.append(outputs["logits"].cpu())
    logits = torch.cat(logits_parts, dim=0)
    total = max(x.shape[0], 1)
    return EvalOutputs(
        loss=sum(losses) / total,
        top1=topk_accuracy(logits, y.cpu(), 1),
        top5=topk_accuracy(logits, y.cpu(), min(5, logits.shape[1])),
        logits=logits.numpy().astype(np.float32),
    )


def train_one_seed(
    seed: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    adjacency: np.ndarray,
    prototype_embeddings: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    num_classes = int(max(train_y.max(), val_y.max(), test_y.max()) + 1)
    model = LightInteractionModel(
        prototype_embeddings=prototype_embeddings,
        num_classes=num_classes,
        hidden_dim=args.node_hidden_dim,
        layers=args.message_passing_layers,
        dropout=args.dropout,
        node_feature_mode=args.node_feature_mode,
        readout_mode=args.readout_mode,
        direct_path_hidden_dim=args.direct_path_hidden_dim,
        disable_direct_path=args.disable_direct_path,
        use_gated_message_passing=args.use_gated_message_passing,
        message_gate_init=args.message_gate_init,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_x_t = torch.from_numpy(train_x).to(device)
    train_y_t = torch.from_numpy(train_y).to(device)
    val_x_t = torch.from_numpy(val_x).to(device)
    val_y_t = torch.from_numpy(val_y).to(device)
    test_x_t = torch.from_numpy(test_x).to(device)
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
        order = np.random.permutation(train_x.shape[0])
        epoch_loss = 0.0
        epoch_correct = 0.0
        epoch_count = 0
        for start in range(0, train_x.shape[0], args.batch_size):
            batch_idx = order[start : start + args.batch_size]
            xb = train_x_t[batch_idx]
            yb = train_y_t[batch_idx]
            outputs = model(xb, adjacency_t)
            loss = F.cross_entropy(outputs["logits"], yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += float(loss.item()) * xb.shape[0]
            epoch_correct += float((outputs["logits"].argmax(dim=1) == yb).float().sum().item())
            epoch_count += xb.shape[0]

        val_outputs = evaluate_model(model=model, x=val_x_t, y=val_y_t, adjacency=adjacency_t, batch_size=args.batch_size)
        learning_rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_loss": epoch_loss / max(epoch_count, 1),
                "train_top1": epoch_correct / max(epoch_count, 1),
                "val_loss": val_outputs.loss,
                "val_top1": val_outputs.top1,
                "val_top5": val_outputs.top5,
            }
        )

        improved = (val_outputs.top1 > best_val_top1) or (
            np.isclose(val_outputs.top1, best_val_top1) and val_outputs.top5 > best_val_top5
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
    val_outputs = evaluate_model(model=model, x=val_x_t, y=val_y_t, adjacency=adjacency_t, batch_size=args.batch_size)
    test_outputs = evaluate_model(model=model, x=test_x_t, y=test_y_t, adjacency=adjacency_t, batch_size=args.batch_size)
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
        "state_dict": best_state,
        "learning_rows": learning_rows,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = np.load(args.artifact_dir / f"{args.fold_name}_light_interaction_inputs.npz")
    graph = np.load(args.artifact_dir / f"{args.fold_name}_unit_graph.npz")
    graph_meta = json.loads((args.artifact_dir / f"{args.fold_name}_unit_graph_metadata.json").read_text(encoding="utf-8"))

    fit_assignments = inputs["fit_assignments"].astype(np.float32)
    val_assignments = inputs["val_assignments"].astype(np.float32)
    test_assignments = inputs["test_assignments"].astype(np.float32)
    fit_labels = inputs["fit_labels"].astype(np.int64)
    val_labels = inputs["val_labels"].astype(np.int64)
    test_labels = inputs["test_labels"].astype(np.int64)
    prototype_embeddings = inputs["prototype_embeddings"].astype(np.float32)
    adjacency = graph["adjacency"].astype(np.float32)

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
            train_x=fit_assignments,
            train_y=fit_labels,
            val_x=val_assignments,
            val_y=val_labels,
            test_x=test_assignments,
            test_y=test_labels,
            adjacency=adjacency,
            prototype_embeddings=prototype_embeddings,
            args=args,
            device=device,
        )
        results.append(result)
        learning_rows.extend(result["learning_rows"])
        torch.save(result["state_dict"], args.output_dir / f"light_interaction_seed_{seed}_model.pt")

    best_result = max(results, key=lambda item: (float(item["best_val_top1"]), float(item["best_val_top5"])))
    logits = np.asarray(best_result["test_logits"], dtype=np.float32)
    top1_preds = logits.argmax(axis=1)
    top5_idx = np.argpartition(-logits, kth=min(4, logits.shape[1] - 1), axis=1)[:, : min(5, logits.shape[1])]

    metrics = {
        "model": "light_interaction",
        "fold": args.fold_name,
        "held_out_subject": graph_meta["held_out_subject"],
        "canonical_validation_subject": graph_meta["canonical_validation_subject"],
        "n_train_samples": int(fit_assignments.shape[0]),
        "n_val_samples": int(val_assignments.shape[0]),
        "n_test_samples": int(test_assignments.shape[0]),
        "num_units": int(prototype_embeddings.shape[0]),
        "prototype_dim": int(prototype_embeddings.shape[1]),
        "node_feature_mode": args.node_feature_mode,
        "readout_mode": args.readout_mode,
        "node_hidden_dim": int(args.node_hidden_dim),
        "direct_path_enabled": bool(not args.disable_direct_path),
        "direct_path_hidden_dim": int(args.direct_path_hidden_dim),
        "use_gated_message_passing": bool(args.use_gated_message_passing),
        "message_gate_init": float(args.message_gate_init),
        "message_passing_layers": int(args.message_passing_layers),
        "graph_density_with_self_loops": float(graph_meta["graph_density_with_self_loops"]),
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
    (args.output_dir / "light_interaction_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    yaml_dump_simple(
        args.output_dir / "light_interaction_run_config.yaml",
        {
            "fold": args.fold_name,
            "canonical_validation_subject": graph_meta["canonical_validation_subject"],
            "node_feature_mode": args.node_feature_mode,
            "readout_mode": args.readout_mode,
            "node_hidden_dim": args.node_hidden_dim,
            "direct_path_enabled": bool(not args.disable_direct_path),
            "direct_path_hidden_dim": args.direct_path_hidden_dim,
            "use_gated_message_passing": bool(args.use_gated_message_passing),
            "message_gate_init": args.message_gate_init,
            "message_passing_layers": args.message_passing_layers,
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

    with (args.output_dir / "light_interaction_learning_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seed", "epoch", "train_loss", "train_top1", "val_loss", "val_top1", "val_top5"],
        )
        writer.writeheader()
        writer.writerows(learning_rows)

    with (args.output_dir / "light_interaction_seed_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
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

    with (args.output_dir / "light_interaction_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_index", "true_label", "pred_label", "top1_correct", "top5_correct"],
        )
        writer.writeheader()
        for idx, label in enumerate(test_labels.tolist()):
            writer.writerow(
                {
                    "row_index": idx,
                    "true_label": int(label),
                    "pred_label": int(top1_preds[idx]),
                    "top1_correct": int(top1_preds[idx] == label),
                    "top5_correct": int(label in top5_idx[idx]),
                }
            )

    np.save(args.output_dir / "light_interaction_logits.npy", logits)


if __name__ == "__main__":
    main()
