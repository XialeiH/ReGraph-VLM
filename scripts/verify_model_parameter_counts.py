#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.regraph_vlm import ReGraphVLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify model_parameter_counts.csv against instantiated ReGraphVLM modules.")
    parser.add_argument(
        "--parameter-counts",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables/model_parameter_counts.csv"),
    )
    return parser.parse_args()


def count_trainable_parameters(graph_encoder: str, readout: str) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ReGraphVLM(
            n_nodes=180,
            node_feature_dim=4,
            clip_dim=512,
            hidden_dim=64,
            embedding_dim=128,
            dropout=0.3,
            readout=readout,
            roi_id_mode="normal",
            num_heads=4,
            num_layers=2,
            graph_encoder=graph_encoder,
            num_subjects=8,
            graph_bias_scale=1.0,
            attention_bias_scale=1.0,
            attention_adjacency_scale=0.1,
        )
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    rows = read_rows(args.parameter_counts)
    if not rows:
        raise SystemExit(f"No rows found in {args.parameter_counts}")

    problems: list[str] = []
    for row in rows:
        model = row["model"]
        graph_encoder = row["graph_encoder"]
        readout = row["readout"]
        expected = int(row["trainable_parameters"])
        observed = count_trainable_parameters(graph_encoder, readout)
        if observed != expected:
            problems.append(f"{model}: observed {observed}, expected {expected}")

    if problems:
        raise SystemExit("; ".join(problems))

    print(f"Verified {len(rows)} model parameter-count rows against instantiated ReGraphVLM modules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
