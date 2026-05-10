#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize held-out semantic generalization real-CLIP vs random.")
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    results = root / "preproc_v0/repetition_familiarity/results"
    held = pd.read_csv(results / "heldout_image/regraph_vlm_summary.csv")
    rand = pd.read_csv(results / "heldout_image_random_embedding/regraph_vlm_summary.csv")
    held["embedding_source"] = "real_clip"
    rand["embedding_source"] = "random_embedding"
    df = pd.concat([held, rand], ignore_index=True)
    metrics = [m for m in ["AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "image_MRR", "brain_R@5", "brain_MRR"] if m in df.columns]
    summary = df.groupby(["graph_encoder", "readout", "embedding_source"])[metrics].agg(["mean", "std", "count"]).reset_index()
    out_dir = results / "semantic_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "heldout_semantic_generalization.csv", index=False)
    summary.to_csv(out_dir / "heldout_semantic_generalization_summary.csv", index=False)
    print({"out": str(out_dir / "heldout_semantic_generalization_summary.csv"), "n_rows": len(summary)})


if __name__ == "__main__":
    main()
