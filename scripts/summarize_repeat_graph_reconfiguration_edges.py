#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train-only and fold-consistent repeat graph edge summaries.")
    parser.add_argument("--root", type=Path, required=True, help="v0_shared_unit root.")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/analysis/scalar4_T3/graph_reconfiguration"),
    )
    parser.add_argument("--folds", nargs="+", default=["fold_01", "fold_04"])
    parser.add_argument("--deltas", nargs="+", default=["delta_A_21", "delta_A_31"])
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = args.root.resolve() / args.analysis_dir
    edge_path = out_dir / "repeat_graph_reconfiguration_edges.csv"
    if not edge_path.exists():
        raise FileNotFoundError(f"Missing edge table: {edge_path}")

    edges = pd.read_csv(edge_path)
    train = edges[
        (edges["split"] == "train")
        & (edges["fold"].isin(args.folds))
        & (edges["delta"].isin(args.deltas))
        & (edges["rank"] <= args.top_k)
    ].copy()
    train = train.sort_values(["delta", "fold", "rank"]).reset_index(drop=True)

    train_path = out_dir / "repeat_graph_reconfiguration_train_only_edges.csv"
    train.to_csv(train_path, index=False)

    grouped: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in train.to_dict("records"):
        roi_i, roi_j = sorted((int(row["roi_i"]), int(row["roi_j"])))
        grouped[(str(row["delta"]), roi_i, roi_j)].append(row)

    consistent_rows: list[dict[str, object]] = []
    for (delta_name, roi_i, roi_j), rows in grouped.items():
        folds = sorted({str(row["fold"]) for row in rows})
        if folds != sorted(args.folds):
            continue
        signed_values = [float(row["delta_value"]) for row in rows]
        abs_values = [float(row["abs_delta_value"]) for row in rows]
        directions = ["increase" if value > 0 else "decrease" for value in signed_values]
        consistent_direction = len(set(directions)) == 1
        consistent_rows.append(
            {
                "delta": delta_name,
                "roi_i": roi_i,
                "roi_j": roi_j,
                "folds": ",".join(folds),
                "n_folds": len(folds),
                "mean_delta_value": sum(signed_values) / len(signed_values),
                "mean_abs_delta_value": sum(abs_values) / len(abs_values),
                "max_abs_delta_value": max(abs_values),
                "consistent_direction": bool(consistent_direction),
                "direction": directions[0] if consistent_direction else "mixed",
                "per_fold_delta_values": ";".join(f"{row['fold']}:{float(row['delta_value']):.6g}" for row in rows),
                "per_fold_ranks": ";".join(f"{row['fold']}:{int(row['rank'])}" for row in rows),
            }
        )

    consistent_rows.sort(key=lambda row: (str(row["delta"]), -float(row["mean_abs_delta_value"])))
    consistent_path = out_dir / "repeat_graph_reconfiguration_fold_consistent_edges.csv"
    write_csv(
        consistent_path,
        consistent_rows,
        [
            "delta",
            "roi_i",
            "roi_j",
            "folds",
            "n_folds",
            "mean_delta_value",
            "mean_abs_delta_value",
            "max_abs_delta_value",
            "consistent_direction",
            "direction",
            "per_fold_delta_values",
            "per_fold_ranks",
        ],
    )

    summary = {
        "source": str(edge_path),
        "train_only_edges": str(train_path),
        "fold_consistent_edges": str(consistent_path),
        "folds": args.folds,
        "deltas": args.deltas,
        "top_k_per_fold_delta": args.top_k,
        "n_train_only_rows": int(len(train)),
        "n_fold_consistent_rows": int(len(consistent_rows)),
        "n_fold_consistent_delta_A_21": int(sum(row["delta"] == "delta_A_21" for row in consistent_rows)),
        "n_fold_consistent_delta_A_31": int(sum(row["delta"] == "delta_A_31" for row in consistent_rows)),
    }
    (out_dir / "repeat_graph_reconfiguration_edge_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
