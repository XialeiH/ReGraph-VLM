#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from models.regraph_vlm import ReGraphVLM
from scripts.run_gate_deletion_test import clone_pairs_with_zeroed_rois
from scripts.run_regraph_vlm_fold import evaluate_pairs, normalize_adjacency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matched gate deletion controls from ROI property-matched sets.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--seed", type=int, required=True)
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
        "--roi-sets",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/interpretability/roi_confound_controls/matched_deletion_roi_sets.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/interpretability/gates_noadj/matched_deletion_tests"),
    )
    parser.add_argument("--max-random-repeats", type=int, default=10)
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
        graph_encoder=str(cfg.get("graph_encoder", "roi_transformer_noadj")),
        graph_bias_scale=float(cfg.get("graph_bias_scale", 1.0)),
        attention_bias_scale=float(cfg.get("attention_bias_scale", 1.0)),
        attention_adjacency_scale=float(cfg.get("attention_adjacency_scale", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def parse_roi_ids(value: str) -> list[int]:
    return [int(x) for x in str(value).split() if x]


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    device = torch.device(args.device)
    ckpt = root / args.checkpoint_template.format(fold=args.fold, seed=args.seed)
    if not ckpt.exists() and args.fallback_checkpoint_template:
        ckpt = root / args.fallback_checkpoint_template.format(fold=args.fold, seed=args.seed)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = checkpoint["args"]
    dataset_root = root / cfg["dataset_root"] / args.fold
    pairs = torch.load(dataset_root / "test_pairs.pt", map_location="cpu", weights_only=False)
    adjacency = torch.from_numpy(normalize_adjacency(np.load(dataset_root / "adjacency.npy"))).to(device)
    model = build_model(checkpoint, pairs[0], device)
    sets = pd.read_csv(root / args.roi_sets)
    sets = sets[(sets["mode"].eq("top_gate")) | (sets["repeat"].astype(int).between(1, args.max_random_repeats))]

    rows = []
    base = evaluate_pairs(model, pairs, threshold=0.5, adjacency=adjacency, device=device, batch_size=args.batch_size)
    rows.append({"fold": args.fold, "seed": args.seed, "mode": "baseline", "k": 0, "repeat": 0, **base})
    for _, row in sets.iterrows():
        rois = parse_roi_ids(row["roi_ids"])
        metrics = evaluate_pairs(
            model,
            clone_pairs_with_zeroed_rois(pairs, rois),
            threshold=0.5,
            adjacency=adjacency,
            device=device,
            batch_size=args.batch_size,
        )
        rows.append({"fold": args.fold, "seed": args.seed, "mode": row["mode"], "k": int(row["k"]), "repeat": int(row["repeat"]), **metrics})

    out_dir = root / args.output_dir / args.fold / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "gate_deletion_metrics.csv", index=False)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    print({"out": str(out_dir / "gate_deletion_metrics.csv"), "n_rows": len(rows)})


if __name__ == "__main__":
    main()
