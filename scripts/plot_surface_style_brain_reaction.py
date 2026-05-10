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
from matplotlib import cm


FEATURES = {"mean_beta": 0, "std_beta": 1, "q90_beta": 2, "positive_fraction": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render FreeSurfer-like cortical surface-style stimulus brain reaction examples from ROI vectors."
    )
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--example_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--feature", choices=sorted(FEATURES), default="mean_beta")
    parser.add_argument("--make_png", action="store_true")
    parser.add_argument("--make_svg", action="store_true")
    parser.add_argument("--max_examples", type=int, default=8)
    return parser.parse_args()


def load_all_sequences(dataset_root: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    out: dict[tuple[str, int, int], dict[str, Any]] = {}
    for fold_dir in sorted(dataset_root.glob("fold_*")):
        for split in ["train", "val", "test"]:
            path = fold_dir / f"{split}_sequences.pt"
            if not path.exists():
                continue
            seqs = torch.load(path, map_location="cpu", weights_only=False)
            for seq in seqs:
                out[(fold_dir.name, int(seq["subject"]), int(seq["nsdId"]))] = dict(seq)
    return out


def read_examples(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def cortex_mesh(n_theta: int = 96, n_phi: int = 56) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi = np.linspace(0.08 * np.pi, 0.92 * np.pi, n_phi)
    tt, pp = np.meshgrid(theta, phi)
    # Two-lobed, inflated cortex-like shape. The sinusoidal term gives weak sulcal texture.
    x = 1.15 * np.sin(pp) * np.cos(tt)
    y = 0.78 * np.sin(pp) * np.sin(tt)
    z = 0.92 * np.cos(pp)
    sulci = 0.035 * np.sin(8 * tt + 1.3 * np.sin(3 * pp)) * np.sin(pp) ** 2
    x = x * (1 + sulci)
    y = y * (1 + sulci)
    z = z * (1 + 0.5 * sulci)
    return x, y, z


def roi_centers() -> np.ndarray:
    """Deterministic pseudo-surface centers for 180 HCP-MMP ROI indices.

    This does not claim exact anatomical parcel placement. It gives a stable
    cortical-surface layout for proposal visualization when HCP-MMP surface
    annotations are unavailable in the runtime.
    """
    centers = []
    for hemi in [-1, 1]:
        for i in range(90):
            row = i // 15
            col = i % 15
            theta = (col + 0.5) / 15 * np.pi - np.pi / 2
            phi = (row + 0.5) / 6 * 0.78 * np.pi + 0.11 * np.pi
            lateral = 0.55 + 0.45 * abs(np.sin(theta))
            x = hemi * (0.52 + 0.55 * lateral * abs(np.cos(theta)))
            y = 0.78 * np.sin(theta)
            z = 0.88 * np.cos(phi)
            centers.append([x, y, z])
    return np.array(centers, dtype=np.float64)


def activation_field(values: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals - np.nanmedian(vals)
    scale = np.nanpercentile(np.abs(vals), 95)
    if scale < 1e-8:
        scale = 1.0
    vals = vals / scale
    vals = np.clip(vals, -2.5, 2.5)
    centers = roi_centers()
    points = np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=1)
    field = np.zeros(points.shape[0], dtype=np.float64)
    sigma2 = 0.18**2
    for c, v in zip(centers, vals):
        if v <= 0:
            continue
        d2 = ((points - c) ** 2).sum(axis=1)
        field += float(v) * np.exp(-0.5 * d2 / sigma2)
    if np.nanmax(field) > 0:
        field = field / np.nanpercentile(field, 99)
    return np.clip(field.reshape(x.shape), 0, 1)


def draw_brain(ax: plt.Axes, values: np.ndarray, view: tuple[float, float], title: str) -> None:
    x, y, z = cortex_mesh()
    act = activation_field(values, x, y, z)
    base = np.ones((*x.shape, 4), dtype=float)
    base[..., :3] = 0.72
    base[..., 3] = 1.0
    hot = cm.get_cmap("hot")(act)
    alpha = np.clip(act**0.8, 0, 0.88)
    face = base.copy()
    face[..., :3] = (1 - alpha[..., None]) * base[..., :3] + alpha[..., None] * hot[..., :3]
    ax.plot_surface(x, y, z, facecolors=face, linewidth=0, antialiased=False, shade=True)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_title(title, fontsize=9, pad=1)
    ax.set_axis_off()
    ax.set_box_aspect((2.3, 1.5, 1.5))
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.9, 0.9)
    ax.set_zlim(-1.0, 1.0)


def savefig(fig: plt.Figure, stem: Path, make_png: bool, make_svg: bool) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    if make_png or not make_svg:
        fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    if make_svg:
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_example(seq: dict[str, Any], criterion: str, out_stem: Path, feature_idx: int, make_png: bool, make_svg: bool) -> None:
    x_seq = seq["x_seq"].float().numpy()[:, :, feature_idx]
    panels = [
        ("repeat1", x_seq[0]),
        ("repeat2", x_seq[1]),
        ("repeat3", x_seq[2]),
        ("repeat3 - repeat1", x_seq[2] - x_seq[0]),
    ]
    views = [("left lateral", (7, 205)), ("posterior", (5, 270)), ("right lateral", (7, -25))]
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    fig.suptitle(
        f"Stimulus-specific cortical reaction: nsdId {int(seq['nsdId'])}, subject {int(seq['subject'])}, {criterion}\n"
        "Surface-style ROI projection; illustrative, not exact FreeSurfer/HCP-MMP parcel rendering",
        fontsize=12,
    )
    for r, (row_name, vals) in enumerate(panels):
        for c, (view_name, view) in enumerate(views):
            ax = fig.add_subplot(4, 3, r * 3 + c + 1, projection="3d")
            draw_brain(ax, vals, view, f"{row_name} | {view_name}")
    savefig(fig, out_stem, make_png, make_svg)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seqs = load_all_sequences(args.dataset_root)
    examples = read_examples(args.example_csv)[: args.max_examples]
    feature_idx = FEATURES[args.feature]
    made = []
    for ex in examples:
        key = (ex["fold"], int(ex["subject"]), int(ex["nsdId"]))
        seq = seqs.get(key)
        if seq is None:
            continue
        stem = args.out_dir / (
            f"{ex['criterion']}_fold_{ex['fold']}_subj_{ex['subject']}_stimulus_{ex['nsdId']}_surface_style"
        )
        render_example(seq, ex["criterion"], stem, feature_idx, args.make_png, args.make_svg)
        made.append(str(stem))
    print({"out_dir": str(args.out_dir), "n_surface_style_examples": len(made)})


if __name__ == "__main__":
    main()
