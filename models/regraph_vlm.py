from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bnt_encoder import BNTTokenEncoder, TokenMLPEncoder


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * grad_output, None


def gradient_reverse(x: torch.Tensor, scale: float) -> torch.Tensor:
    return GradientReverse.apply(x, scale)


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


class GatedRoiMLPEncoder(nn.Module):
    """ROI-wise MLP with learned gates but no token-token interaction."""

    def __init__(self, n_nodes: int, node_feature_dim: int, hidden_dim: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(node_feature_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.gate = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.LayerNorm(n_nodes * hidden_dim),
            nn.Linear(n_nodes * hidden_dim, max(hidden_dim * 4, embedding_dim * 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim * 4, embedding_dim * 2), embedding_dim),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        h = self.feature(x)
        h = h * self.gate(h)
        return F.normalize(self.head(h.flatten(start_dim=1)), dim=-1)


class SharedSubjectRoiMapperEncoder(nn.Module):
    """MindEye2-style baseline: subject-specific linear maps into one shared latent space."""

    def __init__(
        self,
        n_nodes: int,
        node_feature_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
        num_subjects: int,
    ) -> None:
        super().__init__()
        self.num_subjects = int(num_subjects)
        in_dim = n_nodes * node_feature_dim
        self.subject_maps = nn.ModuleList([nn.Linear(in_dim, hidden_dim) for _ in range(self.num_subjects + 1)])
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim * 2, embedding_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim * 2, embedding_dim), embedding_dim),
        )

    def subject_index(self, subject: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if subject is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        subject = subject.to(device=device, dtype=torch.long)
        return subject.clamp(min=0, max=self.num_subjects)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None = None, subject: torch.Tensor | None = None) -> torch.Tensor:
        flat = x.flatten(start_dim=1)
        subject_idx = self.subject_index(subject, flat.shape[0], flat.device)
        hidden = torch.empty(flat.shape[0], self.subject_maps[0].out_features, device=flat.device, dtype=flat.dtype)
        for idx in subject_idx.unique():
            mask = subject_idx == idx
            hidden[mask] = self.subject_maps[int(idx.item())](flat[mask])
        return F.normalize(self.shared(hidden), dim=-1)


class SubjectEmbeddingRoiEncoder(nn.Module):
    """UMBRAE-style universal ROI encoder with learned subject embeddings."""

    def __init__(
        self,
        n_nodes: int,
        node_feature_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
        num_subjects: int,
    ) -> None:
        super().__init__()
        self.num_subjects = int(num_subjects)
        in_dim = n_nodes * node_feature_dim
        self.feature = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.GELU())
        self.subject_embedding = nn.Embedding(self.num_subjects + 1, hidden_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, max(hidden_dim * 2, embedding_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim * 2, embedding_dim), embedding_dim),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None = None, subject: torch.Tensor | None = None) -> torch.Tensor:
        z = self.feature(x.flatten(start_dim=1))
        if subject is None:
            subject_idx = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        else:
            subject_idx = subject.to(device=x.device, dtype=torch.long).clamp(min=0, max=self.num_subjects)
        s = self.subject_embedding(subject_idx)
        return F.normalize(self.head(torch.cat([z, s], dim=-1)), dim=-1)


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

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None = None, subject: torch.Tensor | None = None) -> torch.Tensor:
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
        num_subjects: int = 8,
        graph_bias_scale: float = 1.0,
        attention_bias_scale: float = 1.0,
        attention_adjacency_scale: float = 0.0,
        init_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.num_subjects = int(num_subjects)
        if graph_encoder in {"bnt_token_flat", "roi_transformer_noadj"}:
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
                use_graph_bias=False,
            )
        elif graph_encoder in {"edge_bias_bnt", "edge_bias_graph_bnt"}:
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
                use_graph_bias=False,
                use_attention_bias=True,
                attention_bias_scale=attention_bias_scale,
                attention_adjacency_scale=attention_adjacency_scale if graph_encoder == "edge_bias_graph_bnt" else 0.0,
            )
        elif graph_encoder in {"graph_bnt", "regraph_graph"}:
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
                use_graph_bias=True,
                graph_bias_scale=graph_bias_scale,
            )
        elif graph_encoder == "roi_mlp":
            self.graph_encoder = RoiMLPEncoder(
                n_nodes=n_nodes,
                node_feature_dim=node_feature_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                dropout=dropout,
            )
        elif graph_encoder == "gated_roi_mlp":
            self.graph_encoder = GatedRoiMLPEncoder(
                n_nodes=n_nodes,
                node_feature_dim=node_feature_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                dropout=dropout,
            )
        elif graph_encoder in {"token_mlp", "gated_token_mlp"}:
            self.graph_encoder = TokenMLPEncoder(
                n_nodes=n_nodes,
                in_dim=node_feature_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                dropout=dropout,
                roi_id_mode=roi_id_mode,
                gated=graph_encoder == "gated_token_mlp",
            )
        elif graph_encoder == "mindeye2_shared":
            self.graph_encoder = SharedSubjectRoiMapperEncoder(
                n_nodes=n_nodes,
                node_feature_dim=node_feature_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                dropout=dropout,
                num_subjects=num_subjects,
            )
        elif graph_encoder == "umbrae_subject":
            self.graph_encoder = SubjectEmbeddingRoiEncoder(
                n_nodes=n_nodes,
                node_feature_dim=node_feature_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                dropout=dropout,
                num_subjects=num_subjects,
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
        self.subject_classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, max(embedding_dim, hidden_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(embedding_dim, hidden_dim), self.num_subjects),
        )
        self.log_scale = nn.Parameter(torch.tensor(float(init_scale)).log())
        self.bias = nn.Parameter(torch.tensor(0.0))

    def encode_brain(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        subject: torch.Tensor | None = None,
    ) -> torch.Tensor:
        try:
            z = self.graph_encoder(x, adjacency, subject)
        except TypeError:
            z = self.graph_encoder(x, adjacency)
        return self.brain_proj(z)

    def encode_image(self, clip: torch.Tensor) -> torch.Tensor:
        return self.image_proj(clip.float())

    def pair_logits(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        subject1: torch.Tensor | None = None,
        subject2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z1 = self.encode_brain(x1, adjacency, subject1)
        z2 = self.encode_brain(x2, adjacency, subject2)
        cos = (z1 * z2).sum(dim=-1)
        return self.log_scale.exp().clamp(max=100.0) * cos + self.bias

    def subject_logits(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
        subject: torch.Tensor | None = None,
        reverse_scale: float = 1.0,
    ) -> torch.Tensor:
        z = self.encode_brain(x, adjacency, subject)
        return self.subject_classifier(gradient_reverse(z, reverse_scale))
