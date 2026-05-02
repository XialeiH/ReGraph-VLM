"""Flat direct-reasoning baselines."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GINConv, GPSConv, global_add_pool

from .common import MLP, ModelOutputs, split_node_embeddings, task_output_dim


class GINModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        task: str,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.layers.append(GINConv(mlp))
        self.dropout = nn.Dropout(dropout)
        self.head = MLP(hidden_dim, hidden_dim, task_output_dim(task), dropout)

    def forward(self, batch, return_aux: bool = False) -> ModelOutputs:
        x = self.input_proj(batch.x)
        for layer in self.layers:
            x = layer(x, batch.edge_index).relu()
            x = self.dropout(x)
        pooled = global_add_pool(x, batch.batch)
        logits = self.head(pooled)
        return ModelOutputs(
            logits=logits,
            loss_terms={},
            node_embeddings=split_node_embeddings(x, batch.batch) if return_aux else [],
            assignments=None,
        )


class GATv2Model(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        task: str,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                )
            )
        self.dropout = nn.Dropout(dropout)
        self.head = MLP(hidden_dim, hidden_dim, task_output_dim(task), dropout)

    def forward(self, batch, return_aux: bool = False) -> ModelOutputs:
        x = self.input_proj(batch.x)
        for layer in self.layers:
            x = layer(x, batch.edge_index).relu()
            x = self.dropout(x)
        pooled = global_add_pool(x, batch.batch)
        logits = self.head(pooled)
        return ModelOutputs(
            logits=logits,
            loss_terms={},
            node_embeddings=split_node_embeddings(x, batch.batch) if return_aux else [],
            assignments=None,
        )


class GraphGPSModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        task: str,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            local_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            conv = GINConv(local_mlp)
            self.layers.append(
                GPSConv(
                    channels=hidden_dim,
                    conv=conv,
                    heads=heads,
                    dropout=dropout,
                )
            )
        self.dropout = nn.Dropout(dropout)
        self.head = MLP(hidden_dim, hidden_dim, task_output_dim(task), dropout)

    def forward(self, batch, return_aux: bool = False) -> ModelOutputs:
        x = self.input_proj(batch.x)
        for layer in self.layers:
            x = layer(x, batch.edge_index, batch=batch.batch).relu()
            x = self.dropout(x)
        pooled = global_add_pool(x, batch.batch)
        logits = self.head(pooled)
        return ModelOutputs(
            logits=logits,
            loss_terms={},
            node_embeddings=split_node_embeddings(x, batch.batch) if return_aux else [],
            assignments=None,
        )

