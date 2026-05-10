from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BNTTokenEncoder(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        in_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        readout: str,
        roi_id_mode: str = "normal",
        use_graph_bias: bool = False,
        graph_bias_scale: float = 1.0,
    ):
        super().__init__()
        if readout not in {"cls", "mean", "flat", "gated_flat"}:
            raise ValueError(f"Unsupported BNT token readout: {readout}")
        if roi_id_mode not in {"normal", "none", "shuffled"}:
            raise ValueError(f"Unsupported ROI ID mode: {roi_id_mode}")
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.readout = readout
        self.roi_id_mode = roi_id_mode
        self.use_graph_bias = use_graph_bias
        self.graph_bias_scale = float(graph_bias_scale)
        self.feature = nn.Linear(in_dim, hidden_dim)
        self.roi_embedding = nn.Embedding(n_nodes, hidden_dim) if roi_id_mode != "none" else None
        if roi_id_mode == "shuffled":
            self.register_buffer("roi_permutation", torch.randperm(n_nodes), persistent=True)
        else:
            self.roi_permutation = None
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim)) if readout == "cls" else None
        self.gate = (
            nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1), nn.Sigmoid())
            if readout == "gated_flat"
            else None
        )
        self.graph_proj = nn.Linear(hidden_dim, hidden_dim) if use_graph_bias else None
        self.graph_norm = nn.LayerNorm(hidden_dim) if use_graph_bias else None
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        if readout in {"flat", "gated_flat"}:
            self.head = nn.Sequential(
                nn.LayerNorm(n_nodes * hidden_dim),
                nn.Linear(n_nodes * hidden_dim, max(embedding_dim * 2, hidden_dim)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(embedding_dim * 2, hidden_dim), embedding_dim),
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, max(embedding_dim * 2, hidden_dim)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(embedding_dim * 2, hidden_dim), embedding_dim),
            )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = self.feature(x)
        if self.roi_embedding is not None:
            node_ids = torch.arange(x.shape[1], device=x.device)
            if self.roi_permutation is not None:
                node_ids = self.roi_permutation.to(x.device)
            h = h + self.roi_embedding(node_ids)[None, :, :]
        if self.use_graph_bias and adjacency is not None:
            if self.graph_proj is None or self.graph_norm is None:
                raise RuntimeError("Graph-biased BNT encoder requires graph projection modules")
            mixed = torch.einsum("ij,bjd->bid", adjacency.to(h.device, h.dtype), h)
            h = self.graph_norm(h + self.graph_bias_scale * self.graph_proj(mixed))
        if self.cls is not None:
            cls = self.cls.expand(x.shape[0], -1, -1)
            h = torch.cat([cls, h], dim=1)
        h = self.transformer(h)
        if self.readout == "cls":
            pooled = h[:, 0]
        elif self.readout == "mean":
            pooled = h.mean(dim=1)
        elif self.readout == "gated_flat":
            if self.gate is None:
                raise RuntimeError("gated_flat readout requires a gate module")
            pooled = (h * self.gate(h)).flatten(start_dim=1)
        else:
            pooled = h.flatten(start_dim=1)
        return F.normalize(self.head(pooled), dim=-1)


class TokenMLPEncoder(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        in_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
        roi_id_mode: str = "normal",
        gated: bool = False,
    ):
        super().__init__()
        if roi_id_mode not in {"normal", "none", "shuffled"}:
            raise ValueError(f"Unsupported ROI ID mode: {roi_id_mode}")
        self.n_nodes = n_nodes
        self.roi_id_mode = roi_id_mode
        self.gated = gated
        self.feature = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.roi_embedding = nn.Embedding(n_nodes, hidden_dim) if roi_id_mode != "none" else None
        if roi_id_mode == "shuffled":
            self.register_buffer("roi_permutation", torch.randperm(n_nodes), persistent=True)
        else:
            self.roi_permutation = None
        self.gate = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1), nn.Sigmoid()) if gated else None
        self.head = nn.Sequential(
            nn.LayerNorm(n_nodes * hidden_dim),
            nn.Linear(n_nodes * hidden_dim, max(embedding_dim * 2, hidden_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(embedding_dim * 2, hidden_dim), embedding_dim),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = self.feature(x)
        if self.roi_embedding is not None:
            node_ids = torch.arange(x.shape[1], device=x.device)
            if self.roi_permutation is not None:
                node_ids = self.roi_permutation.to(x.device)
            h = h + self.roi_embedding(node_ids)[None, :, :]
        if self.gate is not None:
            h = h * self.gate(h)
        return F.normalize(self.head(h.flatten(start_dim=1)), dim=-1)


class BNTNativeEncoder(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        hidden_dim: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        readout: str,
        native_mode: str,
        hybrid_alpha: float = 0.5,
    ):
        super().__init__()
        if readout not in {"cls", "mean", "flat"}:
            raise ValueError(f"Unsupported BNT native readout: {readout}")
        if native_mode not in {"cosine", "outer", "hybrid"}:
            raise ValueError(f"Unsupported BNT native mode: {native_mode}")
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.readout = readout
        self.native_mode = native_mode
        self.hybrid_alpha = hybrid_alpha
        self.input_proj = nn.Linear(n_nodes, hidden_dim)
        self.roi_embedding = nn.Embedding(n_nodes, hidden_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim)) if readout == "cls" else None
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        if readout == "flat":
            self.head = nn.Sequential(
                nn.LayerNorm(n_nodes * hidden_dim),
                nn.Linear(n_nodes * hidden_dim, max(embedding_dim * 2, hidden_dim)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(embedding_dim * 2, hidden_dim), embedding_dim),
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, max(embedding_dim * 2, hidden_dim)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(embedding_dim * 2, hidden_dim), embedding_dim),
            )

    def build_connection_profile(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if self.native_mode == "cosine":
            feat = F.normalize(x, dim=-1)
            conn = torch.bmm(feat, feat.transpose(1, 2))
        elif self.native_mode == "outer":
            m = x[:, :, 0]
            m = (m - m.mean(dim=1, keepdim=True)) / (m.std(dim=1, keepdim=True).clamp_min(1e-6))
            conn = m[:, :, None] * m[:, None, :]
        else:
            feat = F.normalize(x, dim=-1)
            sample = torch.bmm(feat, feat.transpose(1, 2))
            base = adjacency.abs()
            base = base / base.max().clamp_min(1e-6)
            conn = self.hybrid_alpha * base[None, :, :] + (1.0 - self.hybrid_alpha) * sample
        eye = torch.eye(conn.shape[1], device=conn.device, dtype=torch.bool)
        conn = conn.masked_fill(eye[None, :, :], 0.0)
        return conn

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        conn = self.build_connection_profile(x, adjacency)
        node_ids = torch.arange(x.shape[1], device=x.device)
        h = self.input_proj(conn) + self.roi_embedding(node_ids)[None, :, :]
        if self.cls is not None:
            cls = self.cls.expand(x.shape[0], -1, -1)
            h = torch.cat([cls, h], dim=1)
        h = self.transformer(h)
        if self.readout == "cls":
            pooled = h[:, 0]
        elif self.readout == "mean":
            pooled = h.mean(dim=1)
        else:
            pooled = h.flatten(start_dim=1)
        return F.normalize(self.head(pooled), dim=-1)
