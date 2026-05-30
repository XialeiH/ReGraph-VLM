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
from scripts.run_regraph_vlm_fold import evaluate_pairs, normalize_adjacency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate deletion test for gated ReGraph-VLM checkpoints.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--checkpoint-template",
        default=(
            "preproc_v0/repetition_familiarity/results/cross_subject_allfold_final/"
            "bnt_token_flat_gated_flat_clip/lambda_2/{fold}/seed_{seed}/checkpoint.pt"
        ),
        help="Checkpoint path template relative to --root. Supports {fold} and {seed}.",
    )
    parser.add_argument(
        "--fallback-checkpoint-template",
        default=(
            "preproc_v0/repetition_familiarity/results/cross_subject_gated_allfold_seed11/"
            "bnt_token_flat_gated_flat_clip/lambda_2/{fold}/seed_{seed}/checkpoint.pt"
        ),
        help="Optional fallback checkpoint template relative to --root.",
    )
    parser.add_argument("--random-repeats", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--gate-summary",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/interpretability/gates/roi_gate_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/interpretability/gates/deletion_tests"),
    )
    return parser.parse_args()


def clone_pairs_with_zeroed_rois(pairs: list[dict[str, Any]], roi_ids_1based: list[int]) -> list[dict[str, Any]]:
    idx = [i - 1 for i in roi_ids_1based]
    out = []
    for p in pairs:
        q = dict(p)
        q["x1"] = p["x1"].clone()
        q["x2"] = p["x2"].clone()
        q["x1"][idx, :] = 0.0
        q["x2"][idx, :] = 0.0
        out.append(q)
    return out


def build_model(checkpoint: dict[str, Any], sample: dict[str, Any], device: torch.device) -> ReGraphVLM:
    cfg = checkpoint["args"]
    graph_encoder = str(cfg.get("graph_encoder", "bnt_token_flat"))
    readout = str(cfg.get("readout", "gated_flat"))
    model = ReGraphVLM(
        n_nodes=int(sample["x1"].shape[0]),
        node_feature_dim=int(sample["x1"].shape[1]),
        clip_dim=int(sample["clip_1"].shape[0]),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        embedding_dim=int(cfg.get("embedding_dim", 128)),
        dropout=float(cfg.get("dropout", 0.3)),
        readout=readout,
        roi_id_mode=str(cfg.get("roi_id_mode", "normal")),
        num_heads=int(cfg.get("num_heads", 4)),
        num_layers=int(cfg.get("num_layers", 2)),
        graph_encoder=graph_encoder,
        graph_bias_scale=float(cfg.get("graph_bias_scale", 1.0)),
        attention_bias_scale=float(cfg.get("attention_bias_scale", 1.0)),
        attention_adjacency_scale=float(cfg.get("attention_adjacency_scale", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    device = torch.device(args.device)
    ckpt = root / args.checkpoint_template.format(fold=args.fold, seed=args.seed)
    if not ckpt.exists() and args.fallback_checkpoint_template:
        ckpt = root / args.fallback_checkpoint_template.format(fold=args.fold, seed=args.seed)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found for {args.fold} seed {args.seed}: {ckpt}")
    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = checkpoint["args"]
    dataset_root = root / cfg["dataset_root"] / args.fold
    pairs = torch.load(dataset_root / "test_pairs.pt", map_location="cpu", weights_only=False)
    adjacency = torch.from_numpy(normalize_adjacency(np.load(dataset_root / "adjacency.npy"))).to(device)
    model = build_model(checkpoint, pairs[0], device)
    gates = pd.read_csv(root / args.gate_summary).sort_values("gate_mean", ascending=False)
    n_nodes = int(pairs[0]["x1"].shape[0])
    rng = np.random.default_rng(20260510 + args.seed + int(args.fold.split("_")[-1]))

    rows = []
    base = evaluate_pairs(model, pairs, threshold=0.5, adjacency=adjacency, device=device, batch_size=args.batch_size)
    rows.append({"fold": args.fold, "seed": args.seed, "mode": "baseline", "k": 0, "repeat": 0, **base})
    for k in [5, 10, 20, 40]:
        top = gates.head(k)["roi_id"].astype(int).tolist()
        bottom = gates.tail(k)["roi_id"].astype(int).tolist()
        for mode, rois in [("top_gate", top), ("bottom_gate", bottom)]:
            metrics = evaluate_pairs(
                model,
                clone_pairs_with_zeroed_rois(pairs, rois),
                threshold=0.5,
                adjacency=adjacency,
                device=device,
                batch_size=args.batch_size,
            )
            rows.append({"fold": args.fold, "seed": args.seed, "mode": mode, "k": k, "repeat": 0, **metrics})
        for repeat in range(args.random_repeats):
            rois = (rng.choice(np.arange(1, n_nodes + 1), size=k, replace=False)).astype(int).tolist()
            metrics = evaluate_pairs(
                model,
                clone_pairs_with_zeroed_rois(pairs, rois),
                threshold=0.5,
                adjacency=adjacency,
                device=device,
                batch_size=args.batch_size,
            )
            rows.append({"fold": args.fold, "seed": args.seed, "mode": "random", "k": k, "repeat": repeat + 1, **metrics})
    out_dir = root / args.output_dir / args.fold / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "gate_deletion_metrics.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    print({"out": str(out), "n_rows": len(rows)})


if __name__ == "__main__":
    main()
