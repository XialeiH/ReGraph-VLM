#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


DEFAULT_INPUT_ROOT = Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3")
DEFAULT_OUTPUT_ROOT = Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip")
DEFAULT_CLIP_PATH = Path("preproc_v0/repetition_familiarity/vlm/clip_embeddings/clip_image_embeddings.pt")
FOLDS = ["fold_01", "fold_04"]
SPLITS = ["train", "val", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach CLIP image embeddings to strict T=3 repetition datasets.")
    parser.add_argument("--root", type=Path, required=True, help="v0_shared_unit root.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--clip-path", type=Path, default=DEFAULT_CLIP_PATH)
    parser.add_argument("--folds", nargs="+", default=FOLDS)
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    return parser.parse_args()


def load_clip(path: Path) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CLIP embedding file: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    clip_emb = data["clip_emb"].float()
    nsd_ids = data["nsdId"].long().tolist()
    if clip_emb.shape[0] != len(nsd_ids):
        raise ValueError(f"CLIP rows {clip_emb.shape[0]} != nsdId rows {len(nsd_ids)}")
    clip_map = {int(nsd_id): clip_emb[idx].clone() for idx, nsd_id in enumerate(nsd_ids)}
    meta = {
        "clip_model_name": data.get("clip_model_name", ""),
        "clip_backend": data.get("clip_backend", ""),
        "resolved_model_name": data.get("resolved_model_name", ""),
        "embedding_dim": int(clip_emb.shape[1]),
        "n_clip_images": int(clip_emb.shape[0]),
        "normalized": bool(data.get("normalized", False)),
        "source": data.get("source", ""),
    }
    return clip_map, meta


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def attach_sequences(items: list[dict[str, Any]], clip_map: dict[int, torch.Tensor]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        nsd_id = int(item["nsdId"])
        if nsd_id not in clip_map:
            raise KeyError(f"Missing CLIP embedding for sequence nsdId={nsd_id}")
        new = dict(item)
        new["clip"] = clip_map[nsd_id].clone()
        out.append(new)
        meta.append(
            {
                "sequence_index": idx,
                "subject": int(item["subject"]),
                "nsdId": nsd_id,
                "y_image": int(item.get("y_image", -1)),
                "repeat_seq": " ".join(str(int(v)) for v in item["repeat_seq"].tolist()),
                "clip_available": True,
            }
        )
    return out, meta


def attach_pairs(items: list[dict[str, Any]], clip_map: dict[int, torch.Tensor]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    out: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    pos_match_errors = 0
    neg_same_id_errors = 0
    for idx, item in enumerate(items):
        nsd1 = int(item["nsdId_1"])
        nsd2 = int(item["nsdId_2"])
        if nsd1 not in clip_map:
            raise KeyError(f"Missing CLIP embedding for pair nsdId_1={nsd1}")
        if nsd2 not in clip_map:
            raise KeyError(f"Missing CLIP embedding for pair nsdId_2={nsd2}")
        new = dict(item)
        clip1 = clip_map[nsd1].clone()
        clip2 = clip_map[nsd2].clone()
        new["clip_1"] = clip1
        new["clip_2"] = clip2
        out.append(new)

        same_image = int(item["same_image"])
        same_nsd = nsd1 == nsd2
        if same_image == 1 and not same_nsd:
            pos_match_errors += 1
        if same_image == 0 and same_nsd:
            neg_same_id_errors += 1
        meta.append(
            {
                "pair_index": idx,
                "subject": int(item["subject"]),
                "nsdId_1": nsd1,
                "nsdId_2": nsd2,
                "same_image": same_image,
                "repeat_1": int(item["repeat_1"]),
                "repeat_2": int(item["repeat_2"]),
                "session_1": int(item["session_1"]),
                "session_2": int(item["session_2"]),
                "clip_1_available": True,
                "clip_2_available": True,
                "positive_has_matching_nsdId": bool(same_image == 0 or same_nsd),
                "negative_has_different_nsdId": bool(same_image == 1 or not same_nsd),
                "clip_cosine": float(torch.dot(clip1, clip2).item()),
            }
        )
    return out, meta, {"pos_match_errors": pos_match_errors, "neg_same_id_errors": neg_same_id_errors}


def tensor_nan_inf_count(items: list[dict[str, Any]]) -> tuple[int, int]:
    nan_count = 0
    inf_count = 0
    for item in items:
        for value in item.values():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                nan_count += int(torch.isnan(value).sum().item())
                inf_count += int(torch.isinf(value).sum().item())
    return nan_count, inf_count


def copy_adjacency_files(in_dir: Path, out_dir: Path) -> None:
    for name in ["adjacency.npy", "adjacency_dense_corr.npy", "adjacency_topk20_corr.npy"]:
        src = in_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)


def attach_fold(
    in_fold_dir: Path,
    out_fold_dir: Path,
    clip_map: dict[int, torch.Tensor],
    clip_meta: dict[str, Any],
    splits: list[str],
) -> dict[str, Any]:
    out_fold_dir.mkdir(parents=True, exist_ok=True)
    copy_adjacency_files(in_fold_dir, out_fold_dir)

    qc: dict[str, Any] = {
        "fold": out_fold_dir.name,
        "clip": clip_meta,
        "sequence_counts": {},
        "pair_counts": {},
        "positive_pair_counts": {},
        "negative_pair_counts": {},
        "nan_count": 0,
        "inf_count": 0,
        "pos_match_errors": 0,
        "neg_same_id_errors": 0,
    }
    all_seq_meta: list[dict[str, Any]] = []
    all_pair_meta: list[dict[str, Any]] = []

    for split in splits:
        seq_path = in_fold_dir / f"{split}_sequences.pt"
        pair_path = in_fold_dir / f"{split}_pairs.pt"
        if not seq_path.exists():
            raise FileNotFoundError(seq_path)
        if not pair_path.exists():
            raise FileNotFoundError(pair_path)

        sequences = torch.load(seq_path, map_location="cpu", weights_only=False)
        pairs = torch.load(pair_path, map_location="cpu", weights_only=False)
        seq_out, seq_meta = attach_sequences(sequences, clip_map)
        pair_out, pair_meta, pair_errors = attach_pairs(pairs, clip_map)

        torch.save(seq_out, out_fold_dir / f"{split}_sequences.pt")
        torch.save(pair_out, out_fold_dir / f"{split}_pairs.pt")

        positive = sum(1 for item in pair_out if int(item["same_image"]) == 1)
        negative = sum(1 for item in pair_out if int(item["same_image"]) == 0)
        seq_nan, seq_inf = tensor_nan_inf_count(seq_out)
        pair_nan, pair_inf = tensor_nan_inf_count(pair_out)

        qc["sequence_counts"][split] = int(len(seq_out))
        qc["pair_counts"][split] = int(len(pair_out))
        qc["positive_pair_counts"][split] = int(positive)
        qc["negative_pair_counts"][split] = int(negative)
        qc["nan_count"] += int(seq_nan + pair_nan)
        qc["inf_count"] += int(seq_inf + pair_inf)
        qc["pos_match_errors"] += int(pair_errors["pos_match_errors"])
        qc["neg_same_id_errors"] += int(pair_errors["neg_same_id_errors"])

        for row in seq_meta:
            row["split"] = split
        for row in pair_meta:
            row["split"] = split
        all_seq_meta.extend(seq_meta)
        all_pair_meta.extend(pair_meta)

    if all_seq_meta:
        write_csv(
            out_fold_dir / "metadata_sequences.csv",
            all_seq_meta,
            ["split", "sequence_index", "subject", "nsdId", "y_image", "repeat_seq", "clip_available"],
        )
    if all_pair_meta:
        write_csv(
            out_fold_dir / "metadata_pairs.csv",
            all_pair_meta,
            [
                "split",
                "pair_index",
                "subject",
                "nsdId_1",
                "nsdId_2",
                "same_image",
                "repeat_1",
                "repeat_2",
                "session_1",
                "session_2",
                "clip_1_available",
                "clip_2_available",
                "positive_has_matching_nsdId",
                "negative_has_different_nsdId",
                "clip_cosine",
            ],
        )

    adj_path = out_fold_dir / "adjacency.npy"
    if adj_path.exists():
        adj = np.load(adj_path)
        qc["adjacency_shape"] = list(adj.shape)
        qc["adjacency_density"] = float(np.count_nonzero(adj) / adj.size)
    else:
        qc["adjacency_shape"] = None
        qc["adjacency_density"] = None

    qc["status"] = (
        "ok"
        if qc["nan_count"] == 0
        and qc["inf_count"] == 0
        and qc["pos_match_errors"] == 0
        and qc["neg_same_id_errors"] == 0
        else "check"
    )
    (out_fold_dir / "dataset_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    return qc


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    input_root = (root / args.input_root).resolve()
    output_root = (root / args.output_root).resolve()
    clip_path = (root / args.clip_path).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    clip_map, clip_meta = load_clip(clip_path)
    all_qc = []
    for fold in args.folds:
        qc = attach_fold(input_root / fold, output_root / fold, clip_map, clip_meta, args.splits)
        all_qc.append(qc)
        print(json.dumps(qc, indent=2), flush=True)

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "clip_path": str(clip_path),
        "folds": args.folds,
        "n_clip_images": int(len(clip_map)),
        "clip": clip_meta,
        "fold_qc": all_qc,
        "status": "ok" if all(q["status"] == "ok" for q in all_qc) else "check",
    }
    (output_root / "dataset_qc.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
