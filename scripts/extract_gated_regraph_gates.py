#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from models.regraph_vlm import ReGraphVLM
from scripts.run_regraph_vlm_fold import normalize_adjacency


class PairDataset(Dataset):
    def __init__(self, pairs: list[dict[str, Any]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        p = self.pairs[idx]
        return {"x1": p["x1"].float(), "x2": p["x2"].float()}


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {"x1": torch.stack([b["x1"] for b in batch]), "x2": torch.stack([b["x2"] for b in batch])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract aggregate gate values from final gated ReGraph checkpoints.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/interpretability/gates"),
    )
    return parser.parse_args()


def find_checkpoints(root: Path) -> list[Path]:
    bases = [
        root / "preproc_v0/repetition_familiarity/results/cross_subject_gated_allfold_seed11",
        root / "preproc_v0/repetition_familiarity/results/cross_subject_allfold_final",
    ]
    paths: list[Path] = []
    for base in bases:
        paths.extend(sorted(base.glob("bnt_token_flat_gated_flat_clip/lambda_2/fold_*/seed_*/checkpoint.pt")))
    return paths


def build_model(checkpoint: dict[str, Any], sample: dict[str, Any], device: torch.device) -> ReGraphVLM:
    cfg = checkpoint["args"]
    model = ReGraphVLM(
        n_nodes=int(sample["x1"].shape[0]),
        node_feature_dim=int(sample["x1"].shape[1]),
        clip_dim=int(sample["clip_1"].shape[0]),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        embedding_dim=int(cfg.get("embedding_dim", 128)),
        dropout=float(cfg.get("dropout", 0.3)),
        readout="gated_flat",
        roi_id_mode=str(cfg.get("roi_id_mode", "normal")),
        num_heads=int(cfg.get("num_heads", 4)),
        num_layers=int(cfg.get("num_layers", 2)),
        graph_encoder="bnt_token_flat",
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def gate_values(model: ReGraphVLM, x: torch.Tensor) -> torch.Tensor:
    enc = model.graph_encoder
    h = enc.feature(x)
    if enc.roi_embedding is not None:
        node_ids = torch.arange(x.shape[1], device=x.device)
        if enc.roi_permutation is not None:
            node_ids = enc.roi_permutation.to(x.device)
        h = h + enc.roi_embedding(node_ids)[None, :, :]
    h = enc.transformer(h)
    if enc.gate is None:
        raise RuntimeError("Model does not have gated_flat gate module.")
    return enc.gate(h).squeeze(-1)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    device = torch.device(args.device)
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    checkpoints = find_checkpoints(root)
    if not checkpoints:
        raise FileNotFoundError("No final gated checkpoints found.")
    for ckpt_path in checkpoints:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = checkpoint["args"]
        fold = str(cfg["fold"])
        seed = int(cfg["seed"])
        dataset_root = root / cfg["dataset_root"] / fold
        pairs = torch.load(dataset_root / "test_pairs.pt", map_location="cpu", weights_only=False)
        sample = pairs[0]
        model = build_model(checkpoint, sample, device)
        _adjacency = torch.from_numpy(normalize_adjacency(np.load(dataset_root / "adjacency.npy"))).to(device)
        loader = DataLoader(PairDataset(pairs), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        n_nodes = int(sample["x1"].shape[0])
        sums = torch.zeros(n_nodes, device=device)
        sums2 = torch.zeros(n_nodes, device=device)
        count = 0
        for batch in loader:
            x = torch.cat([batch["x1"], batch["x2"]], dim=0).to(device)
            g = gate_values(model, x)
            sums += g.sum(dim=0)
            sums2 += (g * g).sum(dim=0)
            count += int(g.shape[0])
        mean = sums / max(count, 1)
        var = (sums2 / max(count, 1) - mean * mean).clamp_min(0.0)
        std = var.sqrt()
        for roi in range(n_nodes):
            rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "roi_id": roi + 1,
                    "gate_mean": float(mean[roi].cpu()),
                    "gate_std": float(std[roi].cpu()),
                    "n_graphs": count,
                    "checkpoint": str(ckpt_path),
                }
            )
    df = pd.DataFrame(rows)
    values_csv = out_dir / "roi_gate_values.csv"
    summary_csv = out_dir / "roi_gate_summary.csv"
    df.to_csv(values_csv, index=False)
    summary = df.groupby("roi_id").agg(
        gate_mean=("gate_mean", "mean"),
        gate_std_across_checkpoints=("gate_mean", "std"),
        mean_within_checkpoint_std=("gate_std", "mean"),
        n_checkpoints=("gate_mean", "count"),
    )
    summary = summary.reset_index().sort_values("gate_mean", ascending=False)
    summary.to_csv(summary_csv, index=False)
    print({"values_csv": str(values_csv), "summary_csv": str(summary_csv), "n_rows": len(df)})


if __name__ == "__main__":
    main()
