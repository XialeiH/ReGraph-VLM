#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.regraph_vlm import ReGraphVLM
from scripts.run_regraph_vlm_fold import (
    ClipPairDataset,
    auroc,
    balanced_accuracy,
    best_threshold,
    collect_pair_scores,
    collate_pairs,
    evaluate_pairs,
    load_adjacency,
    normalize_adjacency,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained adjacency-based checkpoints under adjacency perturbations.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_allfold"),
    )
    parser.add_argument(
        "--checkpoint-template",
        default=(
            "preproc_v0/repetition_familiarity/results/phase3b_clean_graph_ablation/"
            "graph_bnt_gated_flat_clip_adj_topk20_corr/lambda_2/{fold}/seed_{seed}/checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--fallback-checkpoint-template",
        default=(
            "preproc_v0/repetition_familiarity/results/phase3_graph_ablation/"
            "graph_bnt_gated_flat_clip/lambda_2/{fold}/seed_{seed}/checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/phase4_adjacency_perturbation"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


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
        graph_encoder=str(cfg.get("graph_encoder", "graph_bnt")),
        num_subjects=int(cfg.get("num_subjects", 8)),
        graph_bias_scale=float(cfg.get("graph_bias_scale", 1.0)),
        attention_bias_scale=float(cfg.get("attention_bias_scale", 1.0)),
        attention_adjacency_scale=float(cfg.get("attention_adjacency_scale", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def dropout_adjacency(fold_dir: Path, drop_rate: float, seed: int) -> np.ndarray:
    base = np.load(fold_dir / "adjacency.npy").astype(np.float32)
    rng = np.random.default_rng(seed)
    upper = np.triu(np.abs(base) > 0, k=1)
    keep = rng.random(base.shape) >= drop_rate
    keep = np.triu(keep, k=1)
    mask = upper & keep
    adj = np.zeros_like(base, dtype=np.float32)
    adj[mask] = base[mask]
    adj = adj + adj.T
    return normalize_adjacency(adj)


def load_control_adjacency(fold_dir: Path, mode: str, seed: int) -> np.ndarray:
    if mode.startswith("edge_dropout_"):
        pct = int(mode.rsplit("_", 1)[1])
        return dropout_adjacency(fold_dir, pct / 100.0, seed)
    return load_adjacency(fold_dir, mode, seed)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    device = torch.device(args.device)
    fold_dir = root / args.dataset_root / args.fold
    ckpt_path = root / args.checkpoint_template.format(fold=args.fold, seed=args.seed)
    if not ckpt_path.exists() and args.fallback_checkpoint_template:
        ckpt_path = root / args.fallback_checkpoint_template.format(fold=args.fold, seed=args.seed)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    val_pairs = torch.load(fold_dir / "val_pairs.pt", map_location="cpu", weights_only=False)
    test_pairs = torch.load(fold_dir / "test_pairs.pt", map_location="cpu", weights_only=False)
    model = build_model(checkpoint, test_pairs[0], device)

    modes = [
        "default",
        "topk20_corr",
        "dense_corr",
        "identity",
        "no_adjacency",
        "shuffled",
        "random",
        "edge_dropout_10",
        "edge_dropout_30",
        "edge_dropout_50",
    ]
    rows = []
    val_loader = DataLoader(ClipPairDataset(val_pairs), batch_size=args.batch_size, shuffle=False, collate_fn=collate_pairs)
    for mode in modes:
        adjacency = torch.from_numpy(load_control_adjacency(fold_dir, mode, args.seed)).to(device)
        labels, scores, _ = collect_pair_scores(model, val_loader, adjacency, device)
        threshold, _ = best_threshold(labels, scores)
        metrics = evaluate_pairs(model, test_pairs, threshold, adjacency, device, args.batch_size)
        rows.append(
            {
                "fold": args.fold,
                "seed": args.seed,
                "control_mode": mode,
                "checkpoint": str(ckpt_path),
                "val_AUROC": auroc(labels, scores),
                "val_balanced_accuracy": balanced_accuracy(labels, (scores >= threshold).astype(np.int64)),
                **metrics,
            }
        )
        print(json.dumps(rows[-1]), flush=True)

    out_dir = root / args.output_dir / args.fold / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "adjacency_perturbation_metrics.csv", index=False)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
