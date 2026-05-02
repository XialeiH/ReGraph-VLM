#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze repeat-specific ROI graph reconfiguration.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3"))
    parser.add_argument("--output-dir", type=Path, default=Path("preproc_v0/repetition_familiarity/analysis/scalar4_T3"))
    parser.add_argument("--folds", nargs="+", default=["fold_01", "fold_04"])
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--top-edges", type=int, default=10)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def corr_adj(mat: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(mat.astype(np.float64), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    return corr.astype(np.float32)


def flat_corr(a: np.ndarray, b: np.ndarray) -> float:
    iu = np.triu_indices(a.shape[0], k=1)
    av = a[iu].astype(np.float64)
    bv = b[iu].astype(np.float64)
    av -= av.mean()
    bv -= bv.mean()
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    return float(np.dot(av, bv) / denom) if denom > 0 else 0.0


def load_repeat_mats(root: Path, dataset_root: Path, fold: str, split: str) -> list[np.ndarray]:
    seqs = torch.load(root / dataset_root / fold / f"{split}_sequences.pt", map_location="cpu", weights_only=False)
    mats = [[], [], []]
    for seq in seqs:
        x = seq["x_seq"].to(torch.float32).numpy()[:, :, 0]  # [3,180], mean_beta
        for r in range(3):
            mats[r].append(x[r])
    return [np.stack(m, axis=0) for m in mats]


def summarize_adjs(a1: np.ndarray, a2: np.ndarray, a3: np.ndarray) -> dict[str, float]:
    d21 = a2 - a1
    d31 = a3 - a1
    d32 = a3 - a2
    return {
        "mean_abs_delta_A_21": float(np.mean(np.abs(d21))),
        "mean_abs_delta_A_31": float(np.mean(np.abs(d31))),
        "mean_abs_delta_A_32": float(np.mean(np.abs(d32))),
        "corr_A1_A2": flat_corr(a1, a2),
        "corr_A1_A3": flat_corr(a1, a3),
        "corr_A2_A3": flat_corr(a2, a3),
    }


def top_edges(delta: np.ndarray, fold: str, split: str, delta_name: str, topn: int) -> list[dict[str, object]]:
    rows = []
    iu = np.triu_indices(delta.shape[0], k=1)
    vals = delta[iu]
    order = np.argsort(np.abs(vals))[::-1][:topn]
    for rank, idx in enumerate(order, start=1):
        i = int(iu[0][idx])
        j = int(iu[1][idx])
        value = float(vals[idx])
        rows.append(
            {
                "fold": fold,
                "split": split,
                "delta": delta_name,
                "rank": rank,
                "roi_i": i,
                "roi_j": j,
                "delta_value": value,
                "abs_delta_value": abs(value),
                "direction": "increase" if value > 0 else "decrease",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = root / args.output_dir / "graph_reconfiguration"
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    for fold in args.folds:
        summary[fold] = {}
        fold_dir = out_dir / fold
        fold_dir.mkdir(parents=True, exist_ok=True)
        for split in args.splits:
            split_dir = fold_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            mats = load_repeat_mats(root, args.dataset_root, fold, split)
            a1, a2, a3 = [corr_adj(m) for m in mats]
            d21, d31, d32 = a2 - a1, a3 - a1, a3 - a2
            np.save(split_dir / "A_repeat1.npy", a1)
            np.save(split_dir / "A_repeat2.npy", a2)
            np.save(split_dir / "A_repeat3.npy", a3)
            np.save(split_dir / "delta_A_21.npy", d21)
            np.save(split_dir / "delta_A_31.npy", d31)
            np.save(split_dir / "delta_A_32.npy", d32)
            summary[fold][split] = summarize_adjs(a1, a2, a3)
            edge_rows.extend(top_edges(d21, fold, split, "delta_A_21", args.top_edges))
            edge_rows.extend(top_edges(d31, fold, split, "delta_A_31", args.top_edges))
            edge_rows.extend(top_edges(d32, fold, split, "delta_A_32", args.top_edges))

    write_csv(
        out_dir / "repeat_graph_reconfiguration_edges.csv",
        edge_rows,
        ["fold", "split", "delta", "rank", "roi_i", "roi_j", "delta_value", "abs_delta_value", "direction"],
    )
    (out_dir / "repeat_graph_reconfiguration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
