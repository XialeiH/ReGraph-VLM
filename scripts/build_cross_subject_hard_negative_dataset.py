#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cross-subject hard-negative pair datasets.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--input-dataset-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject"),
    )
    parser.add_argument(
        "--output-dataset-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_hardneg"),
    )
    parser.add_argument("--folds", nargs="+", default=["fold_01", "fold_04"])
    parser.add_argument(
        "--negative-mode",
        choices=["random", "hard_pearson_top10", "hard_clip_top10", "mixed_50_random_50_clip_hard"],
        default="mixed_50_random_50_clip_hard",
    )
    parser.add_argument("--hard-top-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260508)
    return parser.parse_args()


def zflat(x: torch.Tensor) -> np.ndarray:
    y = x.float().flatten().numpy()
    std = float(y.std())
    if std < 1e-8:
        return y * 0.0
    return (y - float(y.mean())) / std


def clone_negative(pos: dict[str, Any], neg_source: dict[str, Any]) -> dict[str, Any]:
    out = dict(pos)
    out["x2"] = neg_source["x2"].clone()
    out["clip_2"] = neg_source["clip_2"].clone()
    out["same_image"] = 0
    out["nsdId_2"] = int(neg_source["nsdId_2"])
    out["n_ref_subjects"] = int(neg_source.get("n_ref_subjects", -1))
    out["negative_source"] = "cross_subject_hard_builder"
    return out


def build_split(pairs: list[dict[str, Any]], mode: str, hard_top_frac: float, rng: np.random.Generator) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    positives = [p for p in pairs if int(p["same_image"]) == 1]
    by_key: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for p in positives:
        by_key.setdefault((int(p["subject"]), int(p["repeat_1"])), []).append(p)
    cache: dict[tuple[int, int], dict[str, Any]] = {}
    for key, rows in by_key.items():
        x2 = np.stack([zflat(row["x2"]) for row in rows], axis=0).astype(np.float32)
        clip = np.stack([row["clip_2"].float().numpy() for row in rows], axis=0).astype(np.float32)
        clip /= np.maximum(np.linalg.norm(clip, axis=1, keepdims=True), 1e-12)
        nsd = np.array([int(row["nsdId_2"]) for row in rows], dtype=np.int64)
        cache[key] = {"rows": rows, "x2": x2, "clip": clip, "nsd": nsd}

    out: list[dict[str, Any]] = []
    n_hard = n_random = n_no_candidate = 0
    for idx, pos in enumerate(positives):
        key = (int(pos["subject"]), int(pos["repeat_1"]))
        c = cache.get(key)
        if c is None:
            n_no_candidate += 1
            continue
        valid_idx = np.flatnonzero(c["nsd"] != int(pos["nsdId_1"]))
        if len(valid_idx) == 0:
            n_no_candidate += 1
            continue
        use_hard = mode.startswith("hard") or (mode == "mixed_50_random_50_clip_hard" and idx % 2 == 0)
        if use_hard:
            if mode == "hard_pearson_top10":
                anchor = zflat(pos["x1"]).astype(np.float32)
                scores = c["x2"][valid_idx] @ anchor
            else:
                anchor = pos["clip_1"].float().numpy().astype(np.float32)
                anchor /= max(float(np.linalg.norm(anchor)), 1e-12)
                scores = c["clip"][valid_idx] @ anchor
            order = np.argsort(-scores, kind="mergesort")
            k = max(1, int(np.ceil(len(order) * hard_top_frac)))
            neg = c["rows"][int(valid_idx[order[int(rng.integers(0, k))]])]
            n_hard += 1
        else:
            neg = c["rows"][int(valid_idx[int(rng.integers(0, len(valid_idx)))])]
            n_random += 1
        out.append(pos)
        out.append(clone_negative(pos, neg))
    qc = {
        "n_input_pairs": len(pairs),
        "n_input_positive": len(positives),
        "n_output_pairs": len(out),
        "n_output_positive": sum(int(p["same_image"]) == 1 for p in out),
        "n_output_negative": sum(int(p["same_image"]) == 0 for p in out),
        "n_hard_negative": n_hard,
        "n_random_negative": n_random,
        "n_no_candidate": n_no_candidate,
    }
    return out, qc


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    input_root = root / args.input_dataset_root
    output_root = root / args.output_dataset_root / args.negative_mode
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    all_qc: dict[str, Any] = {"negative_mode": args.negative_mode, "seed": args.seed, "folds": {}}
    for fold in args.folds:
        in_fold = input_root / fold
        out_fold = output_root / fold
        out_fold.mkdir(parents=True, exist_ok=True)
        for name in ["adjacency.npy", "adjacency_dense_corr.npy", "adjacency_topk20_corr.npy"]:
            src = in_fold / name
            if src.exists():
                (out_fold / name).write_bytes(src.read_bytes())
        fold_qc: dict[str, Any] = {}
        for split in ["train", "val", "test"]:
            pairs = torch.load(in_fold / f"{split}_pairs.pt", map_location="cpu", weights_only=False)
            new_pairs, qc = build_split(pairs, args.negative_mode, args.hard_top_frac, rng)
            torch.save(new_pairs, out_fold / f"{split}_pairs.pt")
            fold_qc[split] = qc
        (out_fold / "dataset_qc.json").write_text(json.dumps(fold_qc, indent=2), encoding="utf-8")
        all_qc["folds"][fold] = fold_qc
    (output_root / "dataset_qc.json").write_text(json.dumps(all_qc, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "qc": all_qc}, indent=2))


if __name__ == "__main__":
    main()
