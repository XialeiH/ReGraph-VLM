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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make a FreeSurfer-style camera-view selection sheet.")
    p.add_argument("--dataset_root", type=Path, required=True)
    p.add_argument("--example_csv", type=Path, required=True)
    p.add_argument("--clip_metadata_csv", type=Path, required=True)
    p.add_argument("--out_png", type=Path, required=True)
    p.add_argument("--criterion", default="strong_delta")
    p.add_argument("--face_stride", type=int, default=10)
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


def choose_example(example_csv: Path, criterion: str) -> dict[str, str]:
    with example_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row.get("criterion") == criterion:
            return row
    return rows[0]


def image_path_for_nsd(metadata_csv: Path, nsd_id: int, project_root: Path) -> Path | None:
    marker = "preproc_v0/repetition_familiarity/"
    with metadata_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["nsdId"]) != nsd_id:
                continue
            raw = Path(row["image_path"])
            candidates = [raw]
            raw_s = str(raw)
            if marker in raw_s:
                candidates.append(project_root / "preproc_v0" / "repetition_familiarity" / raw_s.split(marker, 1)[1])
            for cand in candidates:
                if cand.exists():
                    return cand
    return None


def load_meshes(face_stride: int):
    from nilearn import datasets, surface

    fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    meshes = {}
    for hemi, key in (("left", "infl_left"), ("right", "infl_right")):
        coords, faces = surface.load_surf_mesh(fsavg[key])
        sulc = surface.load_surf_data(fsavg["sulc_left" if hemi == "left" else "sulc_right"])
        coords = np.asarray(coords, dtype=float)
        coords = coords - coords.mean(axis=0, keepdims=True)
        coords = coords / np.percentile(np.linalg.norm(coords, axis=1), 98)
        coords[:, 0] += -0.9 if hemi == "left" else 0.9
        faces = np.asarray(faces, dtype=np.int64)
        if face_stride > 1:
            faces = faces[::face_stride]
        meshes[hemi] = {"coords": coords, "faces": faces, "sulc": np.asarray(sulc, dtype=float)}
    return meshes


def roi_anchors(meshes):
    anchors = []
    for hemi in ("left", "right"):
        coords = meshes[hemi]["coords"]
        sign = -1 if hemi == "left" else 1
        cand = np.where(coords[:, 0] * sign > np.percentile(coords[:, 0] * sign, 35))[0]
        sub = coords[cand]
        order = np.lexsort((sub[:, 2], sub[:, 1]))
        ordered = cand[order]
        take = np.linspace(0, len(ordered) - 1, 90).round().astype(int)
        anchors.extend((hemi, int(v)) for v in ordered[take])
    return anchors


def activation(values, meshes, anchors):
    vals = np.asarray(values, dtype=float)
    vals = vals - np.nanmedian(vals)
    scale = np.nanpercentile(np.abs(vals), 95)
    if scale < 1e-8:
        scale = 1.0
    vals = np.clip(vals / scale, -2.5, 2.5)
    out = {h: np.zeros(len(meshes[h]["coords"]), dtype=float) for h in ("left", "right")}
    sigma2 = 0.12**2
    for value, (hemi, idx) in zip(vals, anchors):
        if value <= 0:
            continue
        coords = meshes[hemi]["coords"]
        d2 = ((coords - coords[idx]) ** 2).sum(axis=1)
        out[hemi] += float(value) * np.exp(-0.5 * d2 / sigma2)
    for hemi in out:
        vmax = np.nanpercentile(out[hemi], 99)
        if vmax > 1e-8:
            out[hemi] = np.clip(out[hemi] / vmax, 0, 1)
    return out


def facecolors(mesh, act):
    faces = mesh["faces"]
    sulc = mesh["sulc"][faces].mean(axis=1)
    sulc = (sulc - np.percentile(sulc, 2)) / (np.percentile(sulc, 98) - np.percentile(sulc, 2) + 1e-8)
    grey = 0.55 + 0.23 * sulc
    base = np.stack([grey, grey, grey, np.ones_like(grey)], axis=1)
    af = act[faces].mean(axis=1)
    hot = cm.get_cmap("hot")(af)
    alpha = np.clip(af**0.75, 0, 0.9)
    base[:, :3] = (1 - alpha[:, None]) * base[:, :3] + alpha[:, None] * hot[:, :3]
    return base


def draw(ax, meshes, act, view, hemis=("left", "right")):
    for hemi in hemis:
        coords = meshes[hemi]["coords"]
        faces = meshes[hemi]["faces"]
        coll = Poly3DCollection(
            coords[faces],
            facecolors=facecolors(meshes[hemi], act[hemi]),
            linewidths=0,
            antialiased=False,
            shade=False,
        )
        ax.add_collection3d(coll)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_axis_off()
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.1, 1.1)
    ax.set_box_aspect((3.6, 2.0, 1.8))


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    examples = choose_example(args.example_csv, args.criterion)
    seqs = load_sequences(args.dataset_root)
    seq = seqs[(examples["fold"], int(examples["subject"]), int(examples["nsdId"]))]
    nsd_id = int(seq["nsdId"])
    values = np.abs(seq["x_seq"].float().numpy()[2, :, 0] - seq["x_seq"].float().numpy()[0, :, 0])
    img_path = image_path_for_nsd(args.clip_metadata_csv, nsd_id, project_root)
    meshes = load_meshes(args.face_stride)
    act = activation(values, meshes, roi_anchors(meshes))

    views = [
        ("LH lateral", (0, 180), ("left",)),
        ("LH medial", (0, 0), ("left",)),
        ("RH lateral", (0, 0), ("right",)),
        ("RH medial", (0, 180), ("right",)),
        ("anterior", (0, 90), ("left", "right")),
        ("posterior", (0, 270), ("left", "right")),
        ("dorsal", (90, 270), ("left", "right")),
        ("ventral", (-90, 270), ("left", "right")),
    ]
    fig = plt.figure(figsize=(13, 7.2), constrained_layout=True)
    fig.suptitle(f"FreeSurfer/fsaverage view candidates for nsdId {nsd_id} | overlay: |repeat3 - repeat1|", fontsize=14)
    ax0 = fig.add_subplot(2, 5, 1)
    ax0.set_axis_off()
    if img_path is not None:
        ax0.imshow(plt.imread(img_path))
        ax0.set_title("stimulus", fontsize=10)
    else:
        ax0.text(0.5, 0.5, f"stimulus\nnsdId {nsd_id}", ha="center", va="center")
    for i, (name, view, hemis) in enumerate(views, start=2):
        ax = fig.add_subplot(2, 5, i, projection="3d")
        draw(ax, meshes, act, view, hemis)
        ax.set_title(name, fontsize=10)
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=240, bbox_inches="tight", facecolor="white")
    print(args.out_png)


if __name__ == "__main__":
    main()
