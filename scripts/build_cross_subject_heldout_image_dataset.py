#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from build_cross_subject_repeat_pair_dataset import build_reference, expand_trials, load_sequences, make_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cross-subject pair datasets with held-out image splits.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--input-dataset-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip"),
    )
    parser.add_argument(
        "--output-dataset-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_imageheldout"),
    )
    parser.add_argument("--folds", nargs="+", default=["fold_01", "fold_04"])
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260508)
    return parser.parse_args()


def write_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def filter_trials(rows: list[dict[str, Any]], image_ids: set[int]) -> list[dict[str, Any]]:
    return [row for row in rows if int(row["nsdId"]) in image_ids]


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    input_root = root / args.input_dataset_root
    output_root = root / args.output_dataset_root
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    all_qc: dict[str, Any] = {"seed": args.seed, "folds": {}}
    count_rows: list[dict[str, Any]] = []

    for fold in args.folds:
        in_fold = input_root / fold
        out_fold = output_root / fold
        out_fold.mkdir(parents=True, exist_ok=True)
        for name in ["adjacency.npy", "adjacency_dense_corr.npy", "adjacency_topk20_corr.npy"]:
            src = in_fold / name
            if src.exists():
                (out_fold / name).write_bytes(src.read_bytes())

        train_trials_all = expand_trials(load_sequences(in_fold, "train"))
        val_trials_all = expand_trials(load_sequences(in_fold, "val"))
        test_trials_all = expand_trials(load_sequences(in_fold, "test"))
        image_ids = sorted({int(row["nsdId"]) for row in train_trials_all + val_trials_all + test_trials_all})
        perm = rng.permutation(image_ids)
        n_train = int(round(len(perm) * args.train_frac))
        n_val = int(round(len(perm) * args.val_frac))
        train_images = set(map(int, perm[:n_train]))
        val_images = set(map(int, perm[n_train : n_train + n_val]))
        test_images = set(map(int, perm[n_train + n_val :]))

        split_specs = {
            "train": (filter_trials(train_trials_all, train_images), train_images),
            "val": (filter_trials(val_trials_all, val_images), val_images),
            "test": (filter_trials(test_trials_all, test_images), test_images),
        }
        fold_qc: dict[str, Any] = {
            "n_images_total": len(image_ids),
            "n_train_images": len(train_images),
            "n_val_images": len(val_images),
            "n_test_images": len(test_images),
            "train_val_overlap": len(train_images & val_images),
            "train_test_overlap": len(train_images & test_images),
            "val_test_overlap": len(val_images & test_images),
            "splits": {},
        }
        for split, (anchors, split_images) in split_specs.items():
            if split == "train":
                refs_trials = filter_trials(train_trials_all, split_images)
                pairs, meta = [], []
                by_subject: dict[int, list[dict[str, Any]]] = {}
                for row in anchors:
                    by_subject.setdefault(int(row["subject"]), []).append(row)
                for subject, subject_anchors in by_subject.items():
                    refs = build_reference(refs_trials, exclude_subject=subject)
                    subject_pairs, subject_meta = make_pairs(subject_anchors, refs, rng)
                    pairs.extend(subject_pairs)
                    meta.extend(subject_meta)
            else:
                refs = build_reference(filter_trials(train_trials_all, split_images), exclude_subject=None)
                pairs, meta = make_pairs(anchors, refs, rng)
            torch.save(pairs, out_fold / f"{split}_pairs.pt")
            write_metadata(out_fold / f"metadata_{split}_pairs.csv", meta)
            n_pos = sum(int(p["same_image"]) == 1 for p in pairs)
            n_neg = sum(int(p["same_image"]) == 0 for p in pairs)
            qc = {
                "n_anchor_trials": len(anchors),
                "n_pairs": len(pairs),
                "n_positive": n_pos,
                "n_negative": n_neg,
                "n_unique_anchor_images": len({int(row["nsdId"]) for row in anchors}),
                "status": "ok" if n_pos == n_neg and n_pos > 0 else "check",
            }
            fold_qc["splits"][split] = qc
            count_rows.append({"fold": fold, "split": split, **qc})
        (out_fold / "dataset_qc.json").write_text(json.dumps(fold_qc, indent=2), encoding="utf-8")
        all_qc["folds"][fold] = fold_qc

    (output_root / "heldout_image_dataset_qc.json").write_text(json.dumps(all_qc, indent=2), encoding="utf-8")
    with (output_root / "heldout_image_pair_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(count_rows[0].keys()))
        writer.writeheader()
        writer.writerows(count_rows)
    print(json.dumps({"output_root": str(output_root), "qc": all_qc}, indent=2))


if __name__ == "__main__":
    main()
