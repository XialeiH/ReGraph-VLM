#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.regraph_vlm import ReGraphVLM
from scripts.run_regraph_vlm_fold import (
    auroc,
    balanced_accuracy,
    best_threshold,
    collect_pair_scores,
    evaluate_pairs,
    load_adjacency,
)
from torch.utils.data import DataLoader
from scripts.run_regraph_vlm_fold import ClipPairDataset, collate_pairs


class ConstantGate(nn.Module):
    def __init__(self, value: float = 1.0) -> None:
        super().__init__()
        self.value = float(value)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.full((*h.shape[:-1], 1), self.value, dtype=h.dtype, device=h.device)


class FixedVectorGate(nn.Module):
    def __init__(self, values: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("values", values.float().view(1, -1, 1), persistent=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.values.to(device=h.device, dtype=h.dtype).expand(h.shape[0], -1, -1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate no-adj gated ROI-token layout/gate controls from checkpoints.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_allfold"),
    )
    parser.add_argument(
        "--checkpoint-template",
        default=(
            "preproc_v0/repetition_familiarity/results/phase3c_noadj_gated_final/"
            "roi_transformer_noadj_gated_flat_clip/lambda_2/{fold}/seed_{seed}/checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--fallback-checkpoint-template",
        default=(
            "preproc_v0/repetition_familiarity/results/phase3b_clean_graph_ablation/"
            "roi_transformer_noadj_gated_flat_clip/lambda_2/{fold}/seed_{seed}/checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/phase3d_roi_token_controls"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def clone_pairs_permute_rois(pairs: list[dict[str, Any]], perm: np.ndarray) -> list[dict[str, Any]]:
    idx = torch.as_tensor(perm, dtype=torch.long)
    out = []
    for p in pairs:
        q = dict(p)
        q["x1"] = p["x1"][idx, :].clone()
        q["x2"] = p["x2"][idx, :].clone()
        out.append(q)
    return out


def build_model(checkpoint: dict[str, Any], sample: dict[str, Any], device: torch.device) -> ReGraphVLM:
    cfg = checkpoint["args"]
    model = ReGraphVLM(
        n_nodes=int(sample["x1"].shape[0]),
        node_feature_dim=int(sample["x1"].shape[1]),
        clip_dim=int(sample["clip_1"].shape[0]),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        embedding_dim=int(cfg.get("embedding_dim", 128)),
        dropout=float(cfg.get("dropout", 0.3)),
        readout=str(cfg.get("readout", "gated_flat")),
        roi_id_mode=str(cfg.get("roi_id_mode", "normal")),
        num_heads=int(cfg.get("num_heads", 4)),
        num_layers=int(cfg.get("num_layers", 2)),
        graph_encoder=str(cfg.get("graph_encoder", "roi_transformer_noadj")),
        num_subjects=int(cfg.get("num_subjects", 8)),
        graph_bias_scale=float(cfg.get("graph_bias_scale", 1.0)),
        attention_bias_scale=float(cfg.get("attention_bias_scale", 1.0)),
        attention_adjacency_scale=float(cfg.get("attention_adjacency_scale", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def mean_gate_vector(model: ReGraphVLM, pairs: list[dict[str, Any]], device: torch.device, batch_size: int) -> torch.Tensor:
    enc = model.graph_encoder
    if getattr(enc, "gate", None) is None:
        raise RuntimeError("Model has no gate module.")
    sums = None
    count = 0
    loader = DataLoader(ClipPairDataset(pairs), batch_size=batch_size, shuffle=False, collate_fn=collate_pairs)
    for batch in loader:
        x = torch.cat([batch["x1"], batch["x2"]], dim=0).to(device)
        h = enc.feature(x)
        if enc.roi_embedding is not None:
            node_ids = torch.arange(x.shape[1], device=x.device)
            if enc.roi_permutation is not None:
                node_ids = enc.roi_permutation.to(x.device)
            h = h + enc.roi_embedding(node_ids)[None, :, :]
        h = enc.transformer(h)
        g = enc.gate(h).squeeze(-1)
        sums = g.sum(dim=0) if sums is None else sums + g.sum(dim=0)
        count += int(g.shape[0])
    return (sums / max(count, 1)).detach().cpu()


def evaluate_control(
    checkpoint: dict[str, Any],
    sample: dict[str, Any],
    pairs: list[dict[str, Any]],
    val_pairs: list[dict[str, Any]],
    adjacency: torch.Tensor,
    device: torch.device,
    batch_size: int,
    mode: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    model = build_model(checkpoint, sample, device)
    eval_pairs = pairs
    if mode == "roi_order_shuffle":
        perm = rng.permutation(int(sample["x1"].shape[0]))
        eval_pairs = clone_pairs_permute_rois(pairs, perm)
    elif mode == "zero_roi_embedding":
        enc = model.graph_encoder
        if getattr(enc, "roi_embedding", None) is not None:
            enc.roi_embedding.weight.data.zero_()
    elif mode == "uniform_gate":
        model.graph_encoder.gate = ConstantGate(1.0).to(device)
    elif mode == "random_fixed_gate":
        learned = mean_gate_vector(model, val_pairs, device, batch_size)
        perm = torch.as_tensor(rng.permutation(len(learned)), dtype=torch.long)
        model.graph_encoder.gate = FixedVectorGate(learned[perm]).to(device)
    elif mode != "baseline":
        raise ValueError(f"Unknown control mode: {mode}")

    val_loader = DataLoader(ClipPairDataset(val_pairs), batch_size=batch_size, shuffle=False, collate_fn=collate_pairs)
    labels, scores, _ = collect_pair_scores(model, val_loader, adjacency, device)
    threshold, _ = best_threshold(labels, scores)
    metrics = evaluate_pairs(model, eval_pairs, threshold, adjacency, device, batch_size)
    return {"control_mode": mode, "val_AUROC": auroc(labels, scores), "val_balanced_accuracy": balanced_accuracy(labels, (scores >= threshold).astype(np.int64)), **metrics}


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    device = torch.device(args.device)
    ckpt_path = root / args.checkpoint_template.format(fold=args.fold, seed=args.seed)
    if not ckpt_path.exists() and args.fallback_checkpoint_template:
        ckpt_path = root / args.fallback_checkpoint_template.format(fold=args.fold, seed=args.seed)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    fold_dir = root / args.dataset_root / args.fold
    val_pairs = torch.load(fold_dir / "val_pairs.pt", map_location="cpu", weights_only=False)
    test_pairs = torch.load(fold_dir / "test_pairs.pt", map_location="cpu", weights_only=False)
    adjacency = torch.from_numpy(load_adjacency(fold_dir, "no_adjacency", args.seed)).to(device)
    sample = test_pairs[0]
    rng = np.random.default_rng(20260519 + args.seed * 100 + int(args.fold.split("_")[-1]))
    rows = []
    for mode in ["baseline", "roi_order_shuffle", "zero_roi_embedding", "uniform_gate", "random_fixed_gate"]:
        row = evaluate_control(
            checkpoint,
            sample,
            test_pairs,
            val_pairs,
            adjacency,
            device,
            args.batch_size,
            mode,
            rng,
        )
        row.update({"fold": args.fold, "seed": args.seed, "checkpoint": str(ckpt_path)})
        rows.append(row)
        print(json.dumps(row), flush=True)
    out_dir = root / args.output_dir / args.fold / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "roi_token_control_metrics.csv", index=False)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
