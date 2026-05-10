#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hard-negative same-image repeat pair datasets.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--input-dataset-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip"),
    )
    parser.add_argument(
        "--output-dataset-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_hardneg"),
    )
    parser.add_argument("--folds", nargs="+", default=["fold_01", "fold_04"])
    parser.add_argument(
        "--negative-mode",
        default="mixed_50_random_50_hard",
        choices=["random", "hard_pearson_top10", "mixed_50_random_50_hard"],
    )
    parser.add_argument("--hard-top-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def flatten_x(pair: dict[str, Any], key: str) -> np.ndarray:
    x = pair[key].float().flatten().numpy()
    std = float(x.std())
    if std < 1e-8:
        return x * 0.0
    return (x - float(x.mean())) / std


def pearson_flat(anchor: np.ndarray, candidate: np.ndarray) -> float:
    denom = float(np.linalg.norm(anchor) * np.linalg.norm(candidate))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(anchor, candidate) / denom)


def load_split(fold_dir: Path, split: str) -> list[dict[str, Any]]:
    return torch.load(fold_dir / f"{split}_pairs.pt", map_location="cpu", weights_only=False)


def clone_pair_with_negative(pos: dict[str, Any], neg_source: dict[str, Any]) -> dict[str, Any]:
    out = dict(pos)
    out["x2"] = neg_source["x2"].clone()
    out["clip_2"] = neg_source["clip_2"].clone()
    out["same_image"] = 0
    out["nsdId_2"] = int(neg_source["nsdId_2"])
    out["repeat_2"] = int(neg_source["repeat_2"])
    out["session_2"] = int(neg_source.get("session_2", -1))
    if "trial_2" in neg_source:
        out["trial_2"] = int(neg_source["trial_2"])
    out["negative_source"] = "hard_builder"
    return out


def build_split(pairs: list[dict[str, Any]], mode: str, hard_top_frac: float, rng: np.random.Generator) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    positives = [p for p in pairs if int(p["same_image"]) == 1]
    by_key: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for p in positives:
        key = (int(p["subject"]), int(p["repeat_1"]), int(p["repeat_2"]))
        by_key.setdefault(key, []).append(p)
    key_cache: dict[tuple[int, int, int], dict[str, Any]] = {}
    for key, rows in by_key.items():
        x2 = np.stack([flatten_x(row, "x2") for row in rows], axis=0).astype(np.float32)
        nsd = np.array([int(row["nsdId_2"]) for row in rows], dtype=np.int64)
        key_cache[key] = {"rows": rows, "x2": x2, "nsd": nsd}

    out: list[dict[str, Any]] = []
    n_hard = 0
    n_random = 0
    n_no_candidate = 0
    for idx, pos in enumerate(positives):
        key = (int(pos["subject"]), int(pos["repeat_1"]), int(pos["repeat_2"]))
        cache = key_cache.get(key)
        if cache is None:
            n_no_candidate += 1
            continue
        valid = cache["nsd"] != int(pos["nsdId_1"])
        valid_idx = np.flatnonzero(valid)
        if len(valid_idx) == 0:
            n_no_candidate += 1
            continue

        use_hard = mode == "hard_pearson_top10" or (mode == "mixed_50_random_50_hard" and idx % 2 == 0)
        if use_hard:
            anchor = flatten_x(pos, "x1")
            scores = cache["x2"][valid_idx] @ anchor.astype(np.float32)
            order = np.argsort(-scores, kind="mergesort")
            k = max(1, int(np.ceil(len(order) * hard_top_frac)))
            neg_source = cache["rows"][int(valid_idx[order[int(rng.integers(0, k))]])]
            n_hard += 1
        else:
            neg_source = cache["rows"][int(valid_idx[int(rng.integers(0, len(valid_idx)))])]
            n_random += 1

        out.append(pos)
        out.append(clone_pair_with_negative(pos, neg_source))

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
    rng = np.random.default_rng(args.seed)
    output_root.mkdir(parents=True, exist_ok=True)

    all_qc: dict[str, Any] = {
        "negative_mode": args.negative_mode,
        "hard_top_frac": args.hard_top_frac,
        "seed": args.seed,
        "folds": {},
    }
    for fold in args.folds:
        in_fold = input_root / fold
        out_fold = output_root / fold
        out_fold.mkdir(parents=True, exist_ok=True)
        fold_qc: dict[str, Any] = {}

        for name in ["adjacency.npy", "adjacency_dense_corr.npy", "adjacency_topk20_corr.npy"]:
            src = in_fold / name
            if src.exists():
                dst = out_fold / name
                dst.write_bytes(src.read_bytes())

        for split in ["train", "val", "test"]:
            pairs = load_split(in_fold, split)
            new_pairs, split_qc = build_split(pairs, args.negative_mode, args.hard_top_frac, rng)
            torch.save(new_pairs, out_fold / f"{split}_pairs.pt")
            fold_qc[split] = split_qc

        (out_fold / "dataset_qc.json").write_text(json.dumps(fold_qc, indent=2), encoding="utf-8")
        all_qc["folds"][fold] = fold_qc

    (output_root / "dataset_qc.json").write_text(json.dumps(all_qc, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "qc": all_qc}, indent=2))


if __name__ == "__main__":
    main()
