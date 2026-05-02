"""Pooling baselines and perception-then-reasoning prototype."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, dense_diff_pool, dense_mincut_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch

from .common import DenseReasoner, MLP, ModelOutputs, estimate_num_clusters, task_output_dim


class FineGINEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.layers.append(GINConv(mlp))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index).relu()
            h = self.dropout(h)
        return h


class BaseSingleGraphHierarchicalModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        task: str,
        fine_layers: int = 3,
        coarse_layers: int = 2,
        dropout: float = 0.2,
        max_clusters: int = 48,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_clusters = max_clusters
        self.task = task
        self.encoder = FineGINEncoder(input_dim, hidden_dim, num_layers=fine_layers, dropout=dropout)
        self.assign_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_clusters),
        )
        self.pool_embed = nn.Linear(hidden_dim, hidden_dim)
        self.reasoner = DenseReasoner(hidden_dim, num_layers=coarse_layers, dropout=dropout)
        self.head = MLP(hidden_dim, hidden_dim, task_output_dim(task), dropout)

    def _dense_inputs(self, graph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(graph.x, graph.edge_index)
        batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=h.device)
        dense_x, mask = to_dense_batch(h, batch)
        dense_adj = to_dense_adj(graph.edge_index, batch=batch)
        return h, dense_x, dense_adj, mask

    def _assignment(self, dense_x: torch.Tensor, graph) -> torch.Tensor:
        num_clusters = min(
            self.max_clusters,
            max(1, int(graph.cell_id.max().item()) + 1),
        )
        assign_logits = self.assign_head(dense_x)
        return torch.softmax(assign_logits[:, :, :num_clusters], dim=-1)

    def forward(self, batch, return_aux: bool = False) -> ModelOutputs:
        logits = []
        loss_terms = {}
        assignments = []
        node_embeddings = []
        for graph in batch.to_data_list():
            output = self.forward_single(graph, return_aux=return_aux)
            logits.append(output["logits"])
            node_embeddings.append(output["node_embeddings"])
            if return_aux:
                assignments.append(output["assignment"])
            for key, value in output["loss_terms"].items():
                loss_terms[key] = loss_terms.get(key, 0.0) + value
        if loss_terms:
            num_graphs = max(1, len(logits))
            loss_terms = {
                key: value / num_graphs if torch.is_tensor(value) else value
                for key, value in loss_terms.items()
            }
        return ModelOutputs(
            logits=torch.cat(logits, dim=0),
            loss_terms=loss_terms,
            node_embeddings=node_embeddings if return_aux else [],
            assignments=assignments if return_aux else None,
        )

    def forward_single(self, graph, return_aux: bool = False) -> dict:
        raise NotImplementedError


class DiffPoolModel(BaseSingleGraphHierarchicalModel):
    def forward_single(self, graph, return_aux: bool = False) -> dict:
        node_embeddings, dense_x, dense_adj, mask = self._dense_inputs(graph)
        assignment = self._assignment(dense_x, graph)
        pooled_x, pooled_adj, link_loss, ent_loss = dense_diff_pool(
            self.pool_embed(dense_x),
            dense_adj,
            assignment,
            mask=mask,
        )
        pooled_graph = self.reasoner(pooled_x, pooled_adj)
        logits = self.head(pooled_graph)
        return {
            "logits": logits,
            "loss_terms": {
                "link_loss": link_loss,
                "entropy_loss": ent_loss,
            },
            "node_embeddings": node_embeddings,
            "assignment": assignment.squeeze(0) if return_aux else None,
        }


class MinCutPoolModel(BaseSingleGraphHierarchicalModel):
    def forward_single(self, graph, return_aux: bool = False) -> dict:
        node_embeddings, dense_x, dense_adj, mask = self._dense_inputs(graph)
        assignment = self._assignment(dense_x, graph)
        pooled_x, pooled_adj, mincut_loss, ortho_loss = dense_mincut_pool(
            self.pool_embed(dense_x),
            dense_adj,
            assignment,
            mask=mask,
        )
        pooled_graph = self.reasoner(pooled_x, pooled_adj)
        logits = self.head(pooled_graph)
        return {
            "logits": logits,
            "loss_terms": {
                "mincut_loss": mincut_loss,
                "orthogonality_loss": ortho_loss,
            },
            "node_embeddings": node_embeddings,
            "assignment": assignment.squeeze(0) if return_aux else None,
        }


class PerceptionThenReasoningModel(BaseSingleGraphHierarchicalModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        task: str,
        fine_layers: int = 3,
        coarse_layers: int = 2,
        dropout: float = 0.2,
        max_clusters: int = 48,
        supervised_cells: bool = False,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            task=task,
            fine_layers=fine_layers,
            coarse_layers=coarse_layers,
            dropout=dropout,
            max_clusters=max_clusters,
        )
        self.supervised_cells = supervised_cells

    def forward_single(self, graph, return_aux: bool = False) -> dict:
        node_embeddings, dense_x, dense_adj, mask = self._dense_inputs(graph)
        assignment = self._assignment(dense_x, graph)
        pooled_x, pooled_adj, mincut_loss, ortho_loss = dense_mincut_pool(
            self.pool_embed(dense_x),
            dense_adj,
            assignment,
            mask=mask,
        )
        pooled_graph = self.reasoner(pooled_x, pooled_adj)
        logits = self.head(pooled_graph)
        entropy = -(assignment.clamp_min(1e-8) * assignment.clamp_min(1e-8).log()).sum(dim=-1).mean()
        reconstructed = torch.matmul(assignment, assignment.transpose(1, 2))
        coarse_edge_consistency = F.binary_cross_entropy(
            reconstructed.clamp(0.0, 1.0),
            (dense_adj > 0).float(),
        )
        loss_terms = {
            "cut_loss": mincut_loss,
            "orthogonality_loss": ortho_loss,
            "entropy_loss": entropy,
            "coarse_edge_consistency_loss": coarse_edge_consistency,
        }
        if self.supervised_cells:
            num_clusters = assignment.size(-1)
            targets = graph.cell_id.clamp_max(num_clusters - 1)
            assignment_logits = self.assign_head(node_embeddings)[:, :num_clusters]
            loss_terms["assignment_supervision_loss"] = F.cross_entropy(
                assignment_logits,
                targets,
            )
        return {
            "logits": logits,
            "loss_terms": loss_terms,
            "node_embeddings": node_embeddings,
            "assignment": assignment.squeeze(0) if return_aux else None,
        }

