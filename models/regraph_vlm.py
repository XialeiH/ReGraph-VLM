from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bnt_encoder import BNTTokenEncoder


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class RoiMLPEncoder(nn.Module):
    def __init__(self, n_nodes: int, node_feature_dim: int, hidden_dim: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        in_dim = n_nodes * node_feature_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, max(hidden_dim * 4, embedding_dim * 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim * 4, embedding_dim * 2), max(hidden_dim * 2, embedding_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim * 2, embedding_dim), embedding_dim),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        return F.normalize(self.net(x.flatten(start_dim=1)), dim=-1)


class FusionEncoder(nn.Module):
    """Concatenate non-graph ROI MLP and BNT/ReGraph embeddings before projection."""

    def __init__(
        self,
        n_nodes: int,
        node_feature_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
        readout: str,
        roi_id_mode: str,
        num_heads: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.roi_mlp = RoiMLPEncoder(
            n_nodes=n_nodes,
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.bnt = BNTTokenEncoder(
            n_nodes=n_nodes,
            in_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
            readout=readout,
            roi_id_mode=roi_id_mode,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        z_roi = self.roi_mlp(x, adjacency)
        z_bnt = self.bnt(x, adjacency)
        return F.normalize(self.fuse(torch.cat([z_roi, z_bnt], dim=-1)), dim=-1)


class ReGraphVLM(nn.Module):
    """ReGraph-VLM v0: BNT-token graph encoder plus frozen CLIP-image alignment."""

    def __init__(
        self,
        n_nodes: int,
        node_feature_dim: int,
        clip_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
        readout: str = "flat",
        roi_id_mode: str = "normal",
        num_heads: int = 4,
        num_layers: int = 2,
        graph_encoder: str = "bnt_token_flat",
        init_scale: float = 10.0,
    ) -> None:
        super().__init__()
        if graph_encoder == "bnt_token_flat":
            self.graph_encoder = BNTTokenEncoder(
                n_nodes=n_nodes,
                in_dim=node_feature_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                dropout=dropout,
                readout=readout,
                roi_id_mode=roi_id_mode,
                num_heads=num_heads,
                num_layers=num_layers,
            )
        elif graph_encoder == "roi_mlp":
            self.graph_encoder = RoiMLPEncoder(
                n_nodes=n_nodes,
                node_feature_dim=node_feature_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                dropout=dropout,
            )
        elif graph_encoder == "fusion":
            self.graph_encoder = FusionEncoder(
                n_nodes=n_nodes,
                node_feature_dim=node_feature_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                dropout=dropout,
                readout=readout,
                roi_id_mode=roi_id_mode,
                num_heads=num_heads,
                num_layers=num_layers,
            )
        else:
            raise ValueError(f"Unsupported ReGraph-VLM graph encoder: {graph_encoder}")
        self.brain_proj = ProjectionHead(embedding_dim, embedding_dim, dropout)
        self.image_proj = ProjectionHead(clip_dim, embedding_dim, dropout)
        self.log_scale = nn.Parameter(torch.tensor(float(init_scale)).log())
        self.bias = nn.Parameter(torch.tensor(0.0))

    def encode_brain(self, x: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        z = self.graph_encoder(x, adjacency)
        return self.brain_proj(z)

    def encode_image(self, clip: torch.Tensor) -> torch.Tensor:
        return self.image_proj(clip.float())

    def pair_logits(self, x1: torch.Tensor, x2: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        z1 = self.encode_brain(x1, adjacency)
        z2 = self.encode_brain(x2, adjacency)
        cos = (z1 * z2).sum(dim=-1)
        return self.log_scale.exp().clamp(max=100.0) * cos + self.bias
