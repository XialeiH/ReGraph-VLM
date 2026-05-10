#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export ROI point/value CSV for iELVis plotPialSurf rendering.")
    p.add_argument("--dataset_root", type=Path, required=True)
    p.add_argument("--example_csv", type=Path, required=True)
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--criterion", default="strong_delta")
    p.add_argument("--fold", default="")
    p.add_argument("--feature_idx", type=int, default=0)
    p.add_argument("--mode", choices=["repeat1", "abs_delta31"], default="abs_delta31")
    return p.parse_args()


def load_sequences(dataset_root: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    out = {}
    for fold_dir in sorted(dataset_root.glob("fold_*")):
        for split in ("train", "val", "test"):
            path = fold_dir / f"{split}_sequences.pt"
            if not path.exists():
                continue
            for seq in torch.load(path, map_location="cpu", weights_only=False):
                out[(fold_dir.name, int(seq["subject"]), int(seq["nsdId"]))] = dict(seq)
    return out


def choose_example(example_csv: Path, criterion: str, fold: str) -> dict[str, str]:
    with example_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row.get("criterion") == criterion and (not fold or row.get("fold") == fold):
            return row
    for row in rows:
        if row.get("criterion") == criterion:
            return row
    return rows[0]


def fsaverage_like_roi_points() -> np.ndarray:
    """Return deterministic RAS-like ROI points for 180 HCP-MMP nodes.

    This is a plotting scaffold for iELVis until an exact HCP-MMP ROI centroid
    table is available. Coordinates are distributed over fsaverage-like left
    and right hemisphere shells and are suitable for visual proposal figures,
    not anatomical measurement.
    """
    pts = []
    for hemi_sign in (-1, 1):
        for i in range(90):
            row = i // 15
            col = i % 15
            theta = (col + 0.5) / 15 * np.pi - np.pi / 2
            phi = (row + 0.5) / 6 * 0.78 * np.pi + 0.11 * np.pi
            x = hemi_sign * (35 + 38 * abs(np.cos(theta)))
            y = 55 * np.sin(theta)
            z = 55 * np.cos(phi)
            pts.append([x, y, z, 1 if hemi_sign == -1 else 0])
    return np.asarray(pts, dtype=float)


def main() -> None:
    args = parse_args()
    example = choose_example(args.example_csv, args.criterion, args.fold)
    seqs = load_sequences(args.dataset_root)
    seq = seqs[(example["fold"], int(example["subject"]), int(example["nsdId"]))]
    x_seq = seq["x_seq"].float().numpy()[:, :, args.feature_idx]
    if args.mode == "repeat1":
        values = x_seq[0]
    else:
        values = np.abs(x_seq[2] - x_seq[0])
    pts = fsaverage_like_roi_points()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["roi_id", "x", "y", "z", "isLeft", "value", "nsdId", "subject", "fold", "criterion", "mode"])
        for idx, (pt, value) in enumerate(zip(pts, values), start=1):
            writer.writerow(
                [
                    idx,
                    float(pt[0]),
                    float(pt[1]),
                    float(pt[2]),
                    int(pt[3]),
                    float(value),
                    int(seq["nsdId"]),
                    int(seq["subject"]),
                    example["fold"],
                    example["criterion"],
                    args.mode,
                ]
            )
    print(args.out_csv)


if __name__ == "__main__":
    main()
