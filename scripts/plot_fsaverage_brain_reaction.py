#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


FEATURES = {"mean_beta": 0, "std_beta": 1, "q90_beta": 2, "positive_fraction": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render FreeSurfer/fsaverage-style cortical overlays for selected "
            "stimulus-specific ROI responses."
        )
    )
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--example_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--feature", choices=sorted(FEATURES), default="mean_beta")
    parser.add_argument("--max_examples", type=int, default=8)
    parser.add_argument("--clip_metadata_csv", type=Path, default=None)
    parser.add_argument("--make_png", action="store_true")
    parser.add_argument("--make_svg", action="store_true")
    parser.add_argument("--face_stride", type=int, default=8, help="Use every Nth surface face for faster report figures.")
    parser.add_argument(
        "--mesh_source",
        choices=["auto", "freesurfer", "nilearn"],
        default="auto",
        help="Prefer installed FreeSurfer fsaverage surfaces when available.",
    )
    return parser.parse_args()


def load_all_sequences(dataset_root: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    seqs: dict[tuple[str, int, int], dict[str, Any]] = {}
    for fold_dir in sorted(dataset_root.glob("fold_*")):
        for split in ("train", "val", "test"):
            path = fold_dir / f"{split}_sequences.pt"
            if not path.exists():
                continue
            for seq in torch.load(path, map_location="cpu", weights_only=False):
                seqs[(fold_dir.name, int(seq["subject"]), int(seq["nsdId"]))] = dict(seq)
    return seqs


def read_examples(path: Path, max_examples: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))[:max_examples]


def load_image_paths(metadata_csv: Path | None, project_root: Path) -> dict[int, Path]:
    if metadata_csv is None or not metadata_csv.exists():
        return {}
    out: dict[int, Path] = {}
    with metadata_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                nsd_id = int(row["nsdId"])
            except Exception:
                continue
            raw = Path(row.get("image_path", ""))
            candidates = [raw]
            marker = "preproc_v0/repetition_familiarity/"
            raw_s = str(raw)
            if marker in raw_s:
                rel = raw_s.split(marker, 1)[1]
                candidates.append(project_root / "preproc_v0" / "repetition_familiarity" / rel)
            for cand in candidates:
                if cand.exists():
                    out[nsd_id] = cand
                    break
    return out


def _load_freesurfer_meshes(face_stride: int) -> dict[str, dict[str, np.ndarray]]:
    from nibabel.freesurfer.io import read_geometry, read_morph_data

    subjects_dir = Path(os.environ.get("SUBJECTS_DIR", ""))
    if not subjects_dir:
        fs_home = Path(os.environ.get("FREESURFER_HOME", ""))
        subjects_dir = fs_home / "subjects"
    fsaverage = subjects_dir / "fsaverage"
    if not fsaverage.exists():
        raise FileNotFoundError(f"FreeSurfer fsaverage not found under {subjects_dir}")

    meshes: dict[str, dict[str, np.ndarray]] = {}
    for hemi_name, hemi_prefix, offset in (("left", "lh", -0.15), ("right", "rh", 0.15)):
        coords, faces = read_geometry(str(fsaverage / "surf" / f"{hemi_prefix}.inflated"))
        sulc = read_morph_data(str(fsaverage / "surf" / f"{hemi_prefix}.sulc"))
        coords = np.asarray(coords, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        coords = coords - coords.mean(axis=0, keepdims=True)
        coords = coords / np.percentile(np.linalg.norm(coords, axis=1), 98)
        coords[:, 0] += offset
        if face_stride > 1:
            faces = faces[::face_stride]
        meshes[hemi_name] = {"coords": coords, "faces": faces, "sulc": np.asarray(sulc, dtype=np.float64)}
    return meshes


def _load_nilearn_meshes(face_stride: int) -> dict[str, dict[str, np.ndarray]]:
    from nilearn import datasets, surface

    fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    meshes: dict[str, dict[str, np.ndarray]] = {}
    for hemi, key in (("left", "infl_left"), ("right", "infl_right")):
        coords, faces = surface.load_surf_mesh(fsavg[key])
        sulc_key = "sulc_left" if hemi == "left" else "sulc_right"
        sulc = surface.load_surf_data(fsavg[sulc_key])
        coords = np.asarray(coords, dtype=np.float64)
        coords = coords - coords.mean(axis=0, keepdims=True)
        coords = coords / np.percentile(np.linalg.norm(coords, axis=1), 98)
        if hemi == "left":
            coords[:, 0] -= 0.9
        else:
            coords[:, 0] += 0.9
        faces = np.asarray(faces, dtype=np.int64)
        if face_stride > 1:
            faces = faces[::face_stride]
        meshes[hemi] = {
            "coords": coords,
            "faces": faces,
            "sulc": np.asarray(sulc, dtype=np.float64),
        }
    return meshes


def load_fsaverage_meshes(face_stride: int, mesh_source: str) -> dict[str, dict[str, np.ndarray]]:
    if mesh_source in {"auto", "freesurfer"}:
        try:
            return _load_freesurfer_meshes(face_stride)
        except Exception:
            if mesh_source == "freesurfer":
                raise
    return _load_nilearn_meshes(face_stride)


def roi_anchor_vertices(meshes: dict[str, dict[str, np.ndarray]]) -> list[tuple[str, int]]:
    """Assign 180 ROI values to stable fsaverage5 vertices.

    This is an illustrative HCP-MMP ROI-to-surface projection because the exact
    HCP-MMP surface annotation is not available in the current runtime. The
    mapping is deterministic and spatially distributed over both hemispheres.
    """
    anchors: list[tuple[str, int]] = []
    for hemi in ("left", "right"):
        coords = meshes[hemi]["coords"]
        # Use lateral cortical vertices and spread anchors across posterior/anterior
        # and inferior/superior axes so overlays look like surface parcels rather
        # than random speckles.
        lateral_sign = -1 if hemi == "left" else 1
        lateral = coords[:, 0] * lateral_sign
        candidate = np.where(lateral > np.percentile(lateral, 35))[0]
        sub = coords[candidate]
        order = np.lexsort((sub[:, 2], sub[:, 1]))
        ordered = candidate[order]
        take = np.linspace(0, len(ordered) - 1, 90).round().astype(int)
        anchors.extend((hemi, int(v)) for v in ordered[take])
    return anchors


def vertex_activation(values: np.ndarray, meshes: dict[str, dict[str, np.ndarray]], anchors: list[tuple[str, int]]) -> dict[str, np.ndarray]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals - np.nanmedian(vals)
    scale = np.nanpercentile(np.abs(vals), 95)
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    vals = np.clip(vals / scale, -2.5, 2.5)
    out = {hemi: np.zeros(len(meshes[hemi]["coords"]), dtype=np.float64) for hemi in ("left", "right")}
    sigma2 = 0.12**2
    for value, (hemi, anchor_idx) in zip(vals, anchors):
        if value <= 0:
            continue
        coords = meshes[hemi]["coords"]
        center = coords[anchor_idx]
        d2 = ((coords - center) ** 2).sum(axis=1)
        out[hemi] += float(value) * np.exp(-0.5 * d2 / sigma2)
    for hemi in ("left", "right"):
        vmax = np.nanpercentile(out[hemi], 99)
        if np.isfinite(vmax) and vmax > 1e-8:
            out[hemi] = np.clip(out[hemi] / vmax, 0, 1)
    return out


def face_colors(mesh: dict[str, np.ndarray], activation: np.ndarray) -> np.ndarray:
    faces = mesh["faces"]
    sulc = mesh["sulc"]
    sulc_face = sulc[faces].mean(axis=1)
    sulc_face = (sulc_face - np.nanpercentile(sulc_face, 2)) / (
        np.nanpercentile(sulc_face, 98) - np.nanpercentile(sulc_face, 2) + 1e-8
    )
    grey = 0.55 + 0.23 * sulc_face
    base = np.stack([grey, grey, grey, np.ones_like(grey)], axis=1)
    act_face = activation[faces].mean(axis=1)
    hot = cm.get_cmap("hot")(act_face)
    alpha = np.clip(act_face**0.75, 0, 0.9)
    base[:, :3] = (1 - alpha[:, None]) * base[:, :3] + alpha[:, None] * hot[:, :3]
    return base


def draw_surface(
    ax: plt.Axes,
    meshes: dict[str, dict[str, np.ndarray]],
    activation: dict[str, np.ndarray],
    view: tuple[float, float],
    title: str,
) -> None:
    for hemi in ("left", "right"):
        coords = meshes[hemi]["coords"]
        faces = meshes[hemi]["faces"]
        polys = coords[faces]
        coll = Poly3DCollection(
            polys,
            facecolors=face_colors(meshes[hemi], activation[hemi]),
            linewidths=0.0,
            antialiased=False,
            shade=False,
        )
        ax.add_collection3d(coll)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_title(title, fontsize=9, pad=1)
    ax.set_axis_off()
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.1, 1.1)
    ax.set_box_aspect((3.6, 2.0, 1.8))


def draw_stimulus_panel(ax: plt.Axes, image_path: Path | None, nsd_id: int) -> None:
    ax.set_axis_off()
    if image_path is not None and image_path.exists():
        img = plt.imread(image_path)
        ax.imshow(img)
        ax.set_title(f"stimulus\nnsdId {nsd_id}", fontsize=9)
    else:
        ax.text(0.5, 0.5, f"stimulus\nnsdId {nsd_id}\nimage not found", ha="center", va="center", fontsize=9)
        ax.set_title("stimulus", fontsize=9)


def savefig(fig: plt.Figure, stem: Path, make_png: bool, make_svg: bool) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    if make_png or not make_svg:
        fig.savefig(stem.with_suffix(".png"), dpi=260, bbox_inches="tight", facecolor="white")
    if make_svg:
        fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_example(
    seq: dict[str, Any],
    criterion: str,
    out_stem: Path,
    feature_idx: int,
    meshes: dict[str, dict[str, np.ndarray]],
    anchors: list[tuple[str, int]],
    make_png: bool,
    make_svg: bool,
) -> None:
    x_seq = seq["x_seq"].float().numpy()[:, :, feature_idx]
    panels = [
        ("repeat1", x_seq[0]),
        ("repeat2", x_seq[1]),
        ("repeat3", x_seq[2]),
        ("repeat3 - repeat1", x_seq[2] - x_seq[0]),
    ]
    views = [("left lateral", (0, 180)), ("posterior", (0, 270)), ("right lateral", (0, 0))]
    fig = plt.figure(figsize=(10.5, 10.5), constrained_layout=True)
    fig.suptitle(
        f"Stimulus-specific cortical reaction: nsdId {int(seq['nsdId'])}, subject {int(seq['subject'])}, {criterion}\n"
        "FreeSurfer/fsaverage surface-style ROI projection; illustrative until exact HCP-MMP surface annotation is attached",
        fontsize=12,
    )
    for r, (repeat_name, values) in enumerate(panels):
        activation = vertex_activation(values, meshes, anchors)
        for c, (view_name, view) in enumerate(views):
            ax = fig.add_subplot(4, 3, r * 3 + c + 1, projection="3d")
            draw_surface(ax, meshes, activation, view, f"{repeat_name} | {view_name}")
    savefig(fig, out_stem, make_png, make_svg)


def render_compact_three_view(
    values: np.ndarray,
    title: str,
    out_stem: Path,
    meshes: dict[str, dict[str, np.ndarray]],
    anchors: list[tuple[str, int]],
    image_path: Path | None,
    nsd_id: int,
    make_png: bool,
    make_svg: bool,
) -> None:
    views = [("left", (0, 180)), ("posterior", (0, 270)), ("right", (0, 0))]
    activation = vertex_activation(values, meshes, anchors)
    fig = plt.figure(figsize=(10.8, 2.6), constrained_layout=True)
    fig.suptitle(title, fontsize=10)
    ax0 = fig.add_subplot(1, 4, 1)
    draw_stimulus_panel(ax0, image_path, nsd_id)
    for i, (_, view) in enumerate(views):
        ax = fig.add_subplot(1, 4, i + 2, projection="3d")
        draw_surface(ax, meshes, activation, view, "")
    savefig(fig, out_stem, make_png, make_svg)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seqs = load_all_sequences(args.dataset_root)
    examples = read_examples(args.example_csv, args.max_examples)
    meshes = load_fsaverage_meshes(args.face_stride, args.mesh_source)
    anchors = roi_anchor_vertices(meshes)
    project_root = Path.cwd()
    image_paths = load_image_paths(args.clip_metadata_csv, project_root)
    made: list[str] = []
    for ex in examples:
        key = (ex["fold"], int(ex["subject"]), int(ex["nsdId"]))
        seq = seqs.get(key)
        if seq is None:
            continue
        stem = args.out_dir / (
            f"{ex['criterion']}_{ex['fold']}_subj_{ex['subject']}_stimulus_{ex['nsdId']}_fsaverage_surface"
        )
        render_example(seq, ex["criterion"], stem, FEATURES[args.feature], meshes, anchors, args.make_png, args.make_svg)
        x_seq = seq["x_seq"].float().numpy()[:, :, FEATURES[args.feature]]
        compact_base = args.out_dir / (
            f"{ex['criterion']}_{ex['fold']}_subj_{ex['subject']}_stimulus_{ex['nsdId']}"
        )
        image_path = image_paths.get(int(seq["nsdId"]))
        render_compact_three_view(
            x_seq[0],
            f"nsdId {int(seq['nsdId'])}: repeat1 cortical response",
            compact_base.with_name(compact_base.name + "_repeat1_3view"),
            meshes,
            anchors,
            image_path,
            int(seq["nsdId"]),
            args.make_png,
            args.make_svg,
        )
        render_compact_three_view(
            np.abs(x_seq[2] - x_seq[0]),
            f"nsdId {int(seq['nsdId'])}: |repeat3 - repeat1| cortical change",
            compact_base.with_name(compact_base.name + "_abs_delta31_3view"),
            meshes,
            anchors,
            image_path,
            int(seq["nsdId"]),
            args.make_png,
            args.make_svg,
        )
        made.append(str(stem))
    print({"out_dir": str(args.out_dir), "n_fsaverage_surface_examples": len(made)})


if __name__ == "__main__":
    main()
