#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


FEATURES = {"mean_beta": 0, "std_beta": 1, "q90_beta": 2, "positive_fraction": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot stimulus-specific HCP-MMP ROI brain reaction examples.")
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--clip_dataset_root", type=Path, default=None)
    parser.add_argument("--atlas", type=Path, default=None)
    parser.add_argument("--example_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--feature", choices=sorted(FEATURES), default="mean_beta")
    parser.add_argument("--plot_repeats", action="store_true")
    parser.add_argument("--plot_delta31", action="store_true")
    parser.add_argument("--plot_subject_grid", action="store_true")
    parser.add_argument("--plot_roi_heatmap", action="store_true")
    parser.add_argument("--plot_similarity_matrix", action="store_true")
    parser.add_argument("--make_png", action="store_true")
    parser.add_argument("--make_svg", action="store_true")
    return parser.parse_args()


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(a.dot(b) / denom)


def load_all_sequences(dataset_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold_dir in sorted(dataset_root.glob("fold_*")):
        for split in ["train", "val", "test"]:
            path = fold_dir / f"{split}_sequences.pt"
            if not path.exists():
                continue
            seqs = torch.load(path, map_location="cpu", weights_only=False)
            for seq in seqs:
                item = dict(seq)
                item["fold"] = fold_dir.name
                item["split"] = split
                rows.append(item)
    return rows


def read_examples(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def roi_grid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size != 180:
        raise ValueError(f"Expected 180 ROI values, got {values.size}")
    return values.reshape(10, 18)


def symmetric_limits(arrays: list[np.ndarray]) -> tuple[float, float]:
    mx = max(float(np.nanmax(np.abs(a))) for a in arrays if a.size)
    mx = max(mx, 1e-6)
    return -mx, mx


def savefig(fig: plt.Figure, stem: Path, make_png: bool, make_svg: bool) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    if make_png or not make_svg:
        fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    if make_svg:
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_repeat_maps(seq: dict[str, Any], criterion: str, out_stem: Path, feature_idx: int, make_png: bool, make_svg: bool) -> None:
    x = seq["x_seq"].float().numpy()[:, :, feature_idx]
    delta = x[2] - x[0]
    vmin, vmax = symmetric_limits([x[0], x[1], x[2], delta])
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2), constrained_layout=True)
    axes[0].axis("off")
    axes[0].text(
        0.02,
        0.75,
        f"nsdId {int(seq['nsdId'])}\nsubject {int(seq['subject'])}\n{criterion}",
        fontsize=13,
        va="top",
        ha="left",
    )
    for idx, title in enumerate(["repeat1", "repeat2", "repeat3"]):
        im = axes[idx + 1].imshow(roi_grid(x[idx]), cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")
        axes[idx + 1].set_title(title)
        axes[idx + 1].set_xticks([])
        axes[idx + 1].set_yticks([])
    im = axes[4].imshow(roi_grid(delta), cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")
    axes[4].set_title("repeat3 - repeat1")
    axes[4].set_xticks([])
    axes[4].set_yticks([])
    fig.colorbar(im, ax=axes[1:], fraction=0.025, pad=0.02)
    savefig(fig, out_stem, make_png, make_svg)


def plot_delta(seq: dict[str, Any], out_stem: Path, feature_idx: int, make_png: bool, make_svg: bool) -> None:
    delta = seq["x_seq"].float().numpy()[2, :, feature_idx] - seq["x_seq"].float().numpy()[0, :, feature_idx]
    vmin, vmax = symmetric_limits([delta])
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    im = ax.imshow(roi_grid(delta), cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(f"nsdId {int(seq['nsdId'])}: repeat3 - repeat1")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    savefig(fig, out_stem, make_png, make_svg)


def plot_subject_grid(seqs: list[dict[str, Any]], nsd_id: int, out_stem: Path, feature_idx: int, make_png: bool, make_svg: bool) -> None:
    seqs = sorted(seqs, key=lambda s: int(s["subject"]))
    arrays = []
    for seq in seqs:
        x = seq["x_seq"].float().numpy()[:, :, feature_idx]
        arrays.extend([x[0], x[1], x[2], x[2] - x[0]])
    vmin, vmax = symmetric_limits(arrays)
    fig, axes = plt.subplots(len(seqs), 4, figsize=(10, max(2.4, 1.4 * len(seqs))), constrained_layout=True)
    if len(seqs) == 1:
        axes = np.asarray([axes])
    for r, seq in enumerate(seqs):
        x = seq["x_seq"].float().numpy()[:, :, feature_idx]
        vals = [x[0], x[1], x[2], x[2] - x[0]]
        for c, val in enumerate(vals):
            axes[r, c].imshow(roi_grid(val), cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(["r1", "r2", "r3", "r3-r1"][c])
            if c == 0:
                axes[r, c].set_ylabel(f"subj {int(seq['subject'])}")
    fig.suptitle(f"Stimulus-specific ROI reaction grid: nsdId {nsd_id}", y=1.02)
    savefig(fig, out_stem, make_png, make_svg)


def plot_roi_heatmap(seqs: list[dict[str, Any]], nsd_id: int, out_stem: Path, feature_idx: int, make_png: bool, make_svg: bool) -> None:
    rows = []
    labels = []
    for seq in sorted(seqs, key=lambda s: int(s["subject"])):
        x = seq["x_seq"].float().numpy()[:, :, feature_idx]
        for ridx in range(3):
            rows.append(x[ridx])
            labels.append(f"s{int(seq['subject'])}-r{ridx+1}")
    mat = np.stack(rows)
    vmin, vmax = symmetric_limits([mat])
    fig, ax = plt.subplots(figsize=(13, max(3.5, 0.28 * len(labels))), constrained_layout=True)
    im = ax.imshow(mat, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(f"ROI response heatmap across subjects/repeats: nsdId {nsd_id}")
    ax.set_xlabel("HCP-MMP ROI index")
    ax.set_ylabel("subject-repeat")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    savefig(fig, out_stem, make_png, make_svg)


def plot_similarity(seqs: list[dict[str, Any]], nsd_id: int, out_stem: Path, make_png: bool, make_svg: bool) -> None:
    rows = []
    labels = []
    for seq in sorted(seqs, key=lambda s: int(s["subject"])):
        x = seq["x_seq"].float().numpy()
        for ridx in range(3):
            rows.append(x[ridx].reshape(-1))
            labels.append(f"s{int(seq['subject'])}-r{ridx+1}")
    n = len(rows)
    sim = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            sim[i, j] = corr(rows[i], rows[j])
    fig, ax = plt.subplots(figsize=(max(5, 0.33 * n), max(4.5, 0.33 * n)), constrained_layout=True)
    im = ax.imshow(sim, cmap="viridis", vmin=-0.2, vmax=1.0)
    ax.set_title(f"Same-image subject/repeat similarity: nsdId {nsd_id}")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    savefig(fig, out_stem, make_png, make_svg)


def main() -> None:
    args = parse_args()
    feature_idx = FEATURES[args.feature]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    examples = read_examples(args.example_csv)
    seqs = load_all_sequences(args.dataset_root)
    by_key = {(str(s["fold"]), int(s["subject"]), int(s["nsdId"])): s for s in seqs}
    by_fold_image: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for seq in seqs:
        by_fold_image.setdefault((str(seq["fold"]), int(seq["nsdId"])), []).append(seq)

    for ex in examples:
        fold = ex["fold"]
        subject = int(ex["subject"])
        nsd_id = int(ex["nsdId"])
        criterion = ex["criterion"]
        seq = by_key.get((fold, subject, nsd_id))
        if seq is None:
            continue
        prefix = args.out_dir / f"{criterion}_fold_{fold}_subj_{subject}_stimulus_{nsd_id}"
        if args.plot_repeats:
            plot_repeat_maps(seq, criterion, prefix.with_name(prefix.name + "_repeat_maps"), feature_idx, args.make_png, args.make_svg)
        if args.plot_delta31:
            plot_delta(seq, prefix.with_name(prefix.name + "_delta31_map"), feature_idx, args.make_png, args.make_svg)
        same_image = by_fold_image.get((fold, nsd_id), [seq])
        if args.plot_subject_grid:
            plot_subject_grid(same_image, nsd_id, prefix.with_name(prefix.name + "_subject_repeat_grid"), feature_idx, args.make_png, args.make_svg)
        if args.plot_roi_heatmap:
            plot_roi_heatmap(same_image, nsd_id, prefix.with_name(prefix.name + "_roi_heatmap"), feature_idx, args.make_png, args.make_svg)
        if args.plot_similarity_matrix:
            plot_similarity(same_image, nsd_id, prefix.with_name(prefix.name + "_similarity_matrix"), args.make_png, args.make_svg)
    print({"out_dir": str(args.out_dir), "n_examples": len(examples)})


if __name__ == "__main__":
    main()
