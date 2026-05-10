#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from models.regraph_vlm import ReGraphVLM
from scripts.run_regraph_vlm_fold import normalize_adjacency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze brain embedding RSM alignment with CLIP image RSM.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", default=["fold_01", "fold_04"])
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    return float(np.corrcoef(ra, rb)[0, 1])


def model_path(root: Path, result_root: str, enc_dir: str, fold: str, seed: int) -> Path:
    return root / "preproc_v0/repetition_familiarity/results" / result_root / enc_dir / "lambda_2" / fold / f"seed_{seed}" / "checkpoint.pt"


def build_model(checkpoint: dict[str, Any], sample: dict[str, Any], device: torch.device) -> ReGraphVLM:
    cfg = checkpoint["args"]
    model = ReGraphVLM(
        n_nodes=int(sample["x1"].shape[0]),
        node_feature_dim=int(sample["x1"].shape[1]),
        clip_dim=int(sample["clip_1"].shape[0]),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        embedding_dim=int(cfg.get("embedding_dim", 128)),
        dropout=float(cfg.get("dropout", 0.3)),
        readout=str(cfg.get("readout", "flat")),
        roi_id_mode=str(cfg.get("roi_id_mode", "normal")),
        num_heads=int(cfg.get("num_heads", 4)),
        num_layers=int(cfg.get("num_layers", 2)),
        graph_encoder=str(cfg.get("graph_encoder", "bnt_token_flat")),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def embeddings_for_model(root: Path, spec: dict[str, str], fold: str, seed: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[int]]:
    dataset_root = root / "preproc_v0/repetition_familiarity/datasets" / spec["dataset"] / fold
    pairs = torch.load(dataset_root / "test_pairs.pt", map_location="cpu", weights_only=False)
    ckpt_path = model_path(root, spec["result_root"], spec["enc_dir"], fold, seed)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(checkpoint, pairs[0], device)
    adjacency = torch.from_numpy(normalize_adjacency(np.load(dataset_root / "adjacency.npy"))).to(device)
    by_img: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for p in pairs:
        if int(p["same_image"]) != 1:
            continue
        by_img.setdefault(int(p["nsdId_1"]), []).append((p["x1"].float(), p["clip_1"].float()))
    nsd = sorted(by_img)
    brain_vecs = []
    clip_vecs = []
    for image_id in nsd:
        xs = torch.stack([x for x, _ in by_img[image_id]]).to(device)
        cs = torch.stack([c for _, c in by_img[image_id]]).to(device)
        brain_vecs.append(model.encode_brain(xs, adjacency).mean(dim=0).cpu())
        clip_vecs.append(model.encode_image(cs).mean(dim=0).cpu())
    brain = torch.stack(brain_vecs).numpy()
    clip = torch.stack(clip_vecs).numpy()
    return brain, clip, nsd


def upper_cosine(vecs: np.ndarray) -> np.ndarray:
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    sim = vecs @ vecs.T
    iu = np.triu_indices(sim.shape[0], k=1)
    return sim[iu]


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    device = torch.device(args.device)
    specs = [
        {
            "label": "ROI-MLP+CLIP",
            "dataset": "scalar4_T3_clip_cross_subject_imageheldout",
            "result_root": "heldout_image",
            "enc_dir": "roi_mlp_clip",
        },
        {
            "label": "Flat ReGraph+CLIP",
            "dataset": "scalar4_T3_clip_cross_subject_imageheldout",
            "result_root": "heldout_image",
            "enc_dir": "bnt_token_flat_clip",
        },
        {
            "label": "Gated ReGraph+CLIP",
            "dataset": "scalar4_T3_clip_cross_subject_imageheldout",
            "result_root": "heldout_image",
            "enc_dir": "bnt_token_flat_gated_flat_clip",
        },
        {
            "label": "Gated random embedding",
            "dataset": "scalar4_T3_clip_cross_subject_imageheldout_random_embedding",
            "result_root": "heldout_image_random_embedding",
            "enc_dir": "bnt_token_flat_gated_flat_clip",
        },
    ]
    rows = []
    for spec in specs:
        for fold in args.folds:
            try:
                brain, clip, nsd = embeddings_for_model(root, spec, fold, args.seed, device)
            except FileNotFoundError:
                continue
            rows.append(
                {
                    "model": spec["label"],
                    "fold": fold,
                    "seed": args.seed,
                    "n_images": len(nsd),
                    "brain_clip_rsm_spearman": spearman(upper_cosine(brain), upper_cosine(clip)),
                }
            )
    out_dir = root / "preproc_v0/repetition_familiarity/results/semantic_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "rsm_alignment_summary.csv", index=False)
    df.groupby("model")["brain_clip_rsm_spearman"].agg(["mean", "std", "count"]).to_csv(
        out_dir / "rsm_alignment_model_summary.csv"
    )
    print({"out": str(out_dir / "rsm_alignment_summary.csv"), "n_rows": len(df)})


if __name__ == "__main__":
    main()
