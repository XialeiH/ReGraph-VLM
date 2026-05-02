"""Shared model utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch


@dataclass
class ModelOutputs:
    logits: torch.Tensor
    loss_terms: dict[str, torch.Tensor]
    node_embeddings: list[torch.Tensor]
    assignments: list[torch.Tensor] | None


def task_output_dim(task: str) -> int:
    return 4 if task == "topology" else 1


def task_target(batch: Batch, task: str) -> torch.Tensor:
    if task == "topology":
        return batch.y_topology.view(-1).long()
    if task == "bridge_count":
        return batch.y_bridge_count.view(-1).float()
    if task == "attack_disconnect":
        return batch.y_attack_disconnect.view(-1).float()
    if task == "redundant":
        return batch.y_redundant.view(-1).float()
    raise ValueError(f"unsupported task: {task}")


def task_loss(logits: torch.Tensor, target: torch.Tensor, task: str) -> torch.Tensor:
    if task == "topology":
        return F.cross_entropy(logits, target)
    if task == "bridge_count":
        return F.l1_loss(logits.view(-1), target.view(-1))
    if task in {"attack_disconnect", "redundant"}:
        return F.binary_cross_entropy_with_logits(logits.view(-1), target.view(-1))
    raise ValueError(f"unsupported task: {task}")


def split_node_embeddings(x: torch.Tensor, batch_index: torch.Tensor) -> list[torch.Tensor]:
    outputs = []
    num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
    for graph_id in range(num_graphs):
        outputs.append(x[batch_index == graph_id])
    return outputs


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def estimate_num_clusters(
    num_nodes: int,
    target_cell_size: int = 12,
    min_clusters: int = 4,
    max_clusters: int = 48,
) -> int:
    estimated = max(min_clusters, round(num_nodes / target_cell_size))
    return min(max_clusters, estimated)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseReasoner(nn.Module):
    """Small dense message-passing block for pooled graphs."""

    def __init__(self, hidden_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.self_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.msg_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 3 or adj.dim() != 3:
            raise ValueError("DenseReasoner expects batched dense tensors")
        degree = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        norm_adj = adj / degree
        hidden = x
        for self_layer, msg_layer in zip(self.self_layers, self.msg_layers):
            msg = torch.bmm(norm_adj, msg_layer(hidden))
            hidden = F.relu(self_layer(hidden) + msg)
            hidden = self.dropout(hidden)
        if mask is None:
            return hidden.mean(dim=1)
        mask = mask.float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / denom

