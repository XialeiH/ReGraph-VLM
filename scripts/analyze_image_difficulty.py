#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize image difficulty proxies from cross-subject datasets.")
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    dataset = root / "preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_allfold"
    out_dir = root / "preproc_v0/repetition_familiarity/results/error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold_dir in sorted(dataset.glob("fold_*")):
        path = fold_dir / "test_pairs.pt"
        if not path.exists():
            continue
        pairs = torch.load(path, map_location="cpu", weights_only=False)
        for p in pairs:
            if int(p["same_image"]) != 1:
                continue
            x1 = p["x1"].float().flatten()
            x2 = p["x2"].float().flatten()
            sim = torch.nn.functional.cosine_similarity(x1[None], x2[None]).item()
            rows.append(
                {
                    "fold": fold_dir.name,
                    "subject": int(p["subject"]),
                    "nsdId": int(p["nsdId_1"]),
                    "repeat_1": int(p["repeat_1"]),
                    "repeat_2": int(p["repeat_2"]),
                    "raw_cosine_same_image": sim,
                }
            )
    df = pd.DataFrame(rows)
    summary = df.groupby("nsdId").agg(
        mean_same_image_similarity=("raw_cosine_same_image", "mean"),
        std_same_image_similarity=("raw_cosine_same_image", "std"),
        n_pairs=("raw_cosine_same_image", "count"),
    )
    summary = summary.reset_index().sort_values("mean_same_image_similarity")
    df.to_csv(out_dir / "image_difficulty_pairs.csv", index=False)
    summary.to_csv(out_dir / "image_difficulty.csv", index=False)
    print({"out": str(out_dir / "image_difficulty.csv"), "n_images": len(summary)})


if __name__ == "__main__":
    main()
