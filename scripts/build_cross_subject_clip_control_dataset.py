#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create CLIP-control cross-subject datasets.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--input-dataset-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject"),
    )
    parser.add_argument(
        "--output-dataset-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_shuffled_clip"),
    )
    parser.add_argument("--folds", nargs="+", default=["fold_01", "fold_04"])
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--mode", choices=["shuffled_clip", "random_embedding"], default="shuffled_clip")
    parser.add_argument("--seed", type=int, default=20260507)
    return parser.parse_args()


def collect_clip_map(fold_dirs: list[Path], splits: list[str]) -> dict[int, torch.Tensor]:
    clip_map: dict[int, torch.Tensor] = {}
    for fold_dir in fold_dirs:
        for split in splits:
            path = fold_dir / f"{split}_pairs.pt"
            if not path.exists():
                continue
            pairs = torch.load(path, map_location="cpu", weights_only=False)
            for pair in pairs:
                clip_map.setdefault(int(pair["nsdId_1"]), pair["clip_1"].float().clone())
                clip_map.setdefault(int(pair["nsdId_2"]), pair["clip_2"].float().clone())
    return clip_map


def make_control_map(clip_map: dict[int, torch.Tensor], mode: str, seed: int) -> dict[int, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    ids = sorted(clip_map)
    clips = torch.stack([clip_map[i] for i in ids])
    if mode == "shuffled_clip":
        perm = torch.randperm(len(ids), generator=generator)
        shuffled = clips[perm]
        return {ids[i]: shuffled[i].clone() for i in range(len(ids))}
    random = torch.randn(clips.shape, generator=generator)
    random = random / random.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return {ids[i]: random[i].clone() for i in range(len(ids))}


def rewrite_pairs(pairs: list[dict[str, Any]], control_map: dict[int, torch.Tensor]) -> list[dict[str, Any]]:
    out = []
    for pair in pairs:
        item = dict(pair)
        item["clip_1"] = control_map[int(pair["nsdId_1"])].clone()
        item["clip_2"] = control_map[int(pair["nsdId_2"])].clone()
        out.append(item)
    return out


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    input_root = root / args.input_dataset_root
    output_root = root / args.output_dataset_root
    output_root.mkdir(parents=True, exist_ok=True)
    fold_dirs = [input_root / fold for fold in args.folds]
    clip_map = collect_clip_map(fold_dirs, args.splits)
    control_map = make_control_map(clip_map, args.mode, args.seed)
    qc: dict[str, Any] = {"mode": args.mode, "seed": args.seed, "folds": {}, "n_clip_ids": len(clip_map)}
    for fold in args.folds:
        in_fold = input_root / fold
        out_fold = output_root / fold
        out_fold.mkdir(parents=True, exist_ok=True)
        for name in ["adjacency.npy", "adjacency_dense_corr.npy", "adjacency_topk20_corr.npy", "dataset_qc.json"]:
            src = in_fold / name
            if src.exists():
                shutil.copy2(src, out_fold / name)
        fold_qc: dict[str, Any] = {}
        for split in args.splits:
            src = in_fold / f"{split}_pairs.pt"
            if not src.exists():
                continue
            pairs = torch.load(src, map_location="cpu", weights_only=False)
            out_pairs = rewrite_pairs(pairs, control_map)
            torch.save(out_pairs, out_fold / f"{split}_pairs.pt")
            fold_qc[split] = {
                "n_pairs": len(out_pairs),
                "n_positive": sum(int(p["same_image"]) == 1 for p in out_pairs),
                "n_negative": sum(int(p["same_image"]) == 0 for p in out_pairs),
            }
        qc["folds"][fold] = fold_qc
    (output_root / "clip_control_dataset_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "qc": qc}, indent=2))


if __name__ == "__main__":
    main()
