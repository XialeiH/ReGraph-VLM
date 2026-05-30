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
from scripts.run_regraph_vlm_fold import load_adjacency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-shot held-out-subject ridge calibration on frozen brain embeddings.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", default=[f"fold_{i:02d}" for i in range(1, 9)])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--shots", nargs="+", type=int, default=[0, 10, 30, 60, 120])
    parser.add_argument("--ridge", type=float, default=1.0)
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
        "--out-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/lowshot_subject_adaptation"),
    )
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


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def ridge_map(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    xtx = x.T @ x
    reg = ridge * np.eye(xtx.shape[0], dtype=xtx.dtype)
    return np.linalg.solve(xtx + reg, x.T @ y)


def retrieval(query: np.ndarray, candidates: np.ndarray, query_ids: np.ndarray, candidate_ids: np.ndarray) -> dict[str, float]:
    scores = normalize(query) @ normalize(candidates).T
    hit1 = hit5 = hit10 = 0
    rr = []
    for i in range(scores.shape[0]):
        order = np.argsort(-scores[i], kind="mergesort")
        pos = np.where(candidate_ids[order] == query_ids[i])[0]
        if len(pos) == 0:
            continue
        rank = int(pos[0]) + 1
        hit1 += int(rank <= 1)
        hit5 += int(rank <= 5)
        hit10 += int(rank <= 10)
        rr.append(1.0 / rank)
    n = max(len(rr), 1)
    return {"image_R@1": hit1 / n, "image_R@5": hit5 / n, "image_R@10": hit10 / n, "image_MRR": float(np.mean(rr)) if rr else float("nan"), "n_eval": len(rr)}


@torch.no_grad()
def encode_fold(root: Path, args: argparse.Namespace, fold: str, seed: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fold_dir = root / args.dataset_root / fold
    pairs = torch.load(fold_dir / "test_pairs.pt", map_location="cpu", weights_only=False)
    positives = [p for p in pairs if int(p["same_image"]) == 1]
    ckpt = root / args.checkpoint_template.format(fold=fold, seed=seed)
    if not ckpt.exists() and args.fallback_checkpoint_template:
        ckpt = root / args.fallback_checkpoint_template.format(fold=fold, seed=seed)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    model = build_model(checkpoint, positives[0], device)
    adjacency = torch.from_numpy(load_adjacency(fold_dir, "no_adjacency", seed)).to(device)
    by_img: dict[int, list[tuple[torch.Tensor, torch.Tensor, int]]] = {}
    for p in positives:
        by_img.setdefault(int(p["nsdId_1"]), []).append((p["x1"].float(), p["clip_1"].float(), int(p.get("subject_1", p.get("subject", 0)))))
    ids = np.array(sorted(by_img), dtype=np.int64)
    brain = []
    image = []
    for image_id in ids:
        xs = torch.stack([x for x, _, _ in by_img[int(image_id)]]).to(device)
        cs = torch.stack([c for _, c, _ in by_img[int(image_id)]]).to(device)
        subjects = torch.tensor([s for _, _, s in by_img[int(image_id)]], dtype=torch.long, device=device)
        brain.append(model.encode_brain(xs, adjacency, subjects).mean(dim=0).cpu().numpy())
        image.append(model.encode_image(cs).mean(dim=0).cpu().numpy())
    return np.stack(brain), np.stack(image), ids


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    device = torch.device(args.device)
    rows = []
    for fold in args.folds:
        for seed in args.seeds:
            try:
                brain, image, ids = encode_fold(root, args, fold, seed, device)
            except FileNotFoundError:
                continue
            rng = np.random.default_rng(20260523 + seed * 100 + int(fold.split("_")[-1]))
            order = rng.permutation(len(ids))
            for shot in args.shots:
                n_cal = min(int(shot), max(len(ids) - 20, 0))
                cal_idx = order[:n_cal]
                eval_idx = order[n_cal:]
                if len(eval_idx) < 5:
                    continue
                q = brain[eval_idx]
                if n_cal > 0:
                    w = ridge_map(normalize(brain[cal_idx]), normalize(image[cal_idx]), args.ridge)
                    q = normalize(q) @ w
                metrics = retrieval(q, image[eval_idx], ids[eval_idx], ids[eval_idx])
                rows.append({"fold": fold, "seed": seed, "shots": n_cal, "n_images": len(ids), **metrics})
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "lowshot_subject_adaptation.csv", index=False)
    summary = df.groupby("shots").agg(
        n=("image_R@5", "count"),
        image_R5_mean=("image_R@5", "mean"),
        image_R5_std=("image_R@5", "std"),
        image_MRR_mean=("image_MRR", "mean"),
        image_MRR_std=("image_MRR", "std"),
    ).reset_index()
    summary.to_csv(out_dir / "lowshot_subject_adaptation_summary.csv", index=False)
    (out_dir / "lowshot_subject_adaptation_summary.md").write_text(summary.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    print({"out": str(out_dir / "lowshot_subject_adaptation_summary.csv"), "n_rows": len(df)})


if __name__ == "__main__":
    main()
