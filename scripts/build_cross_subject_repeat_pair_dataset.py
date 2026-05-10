#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cross-subject same-image repeat matching pairs.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--input-dataset-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip"),
    )
    parser.add_argument(
        "--output-dataset-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject"),
    )
    parser.add_argument("--folds", nargs="+", default=["fold_01", "fold_04"])
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_sequences(fold_dir: Path, split: str) -> list[dict[str, Any]]:
    return torch.load(fold_dir / f"{split}_sequences.pt", map_location="cpu", weights_only=False)


def expand_trials(sequences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seq in sequences:
        repeat_seq = seq["repeat_seq"].tolist()
        session_seq = seq["session_seq"].tolist()
        trial_seq = seq["trial_seq"].tolist()
        for idx, repeat_index in enumerate(repeat_seq):
            rows.append(
                {
                    "x": seq["x_seq"][idx].clone(),
                    "clip": seq["clip"].clone(),
                    "subject": int(seq["subject"]),
                    "nsdId": int(seq["nsdId"]),
                    "repeat_index": int(repeat_index),
                    "session_index": int(session_seq[idx]),
                    "trial_index": int(trial_seq[idx]),
                }
            )
    return rows


def build_reference(train_trials: list[dict[str, Any]], exclude_subject: int | None = None) -> dict[tuple[int, int], dict[str, Any]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in train_trials:
        if exclude_subject is not None and int(row["subject"]) == int(exclude_subject):
            continue
        groups.setdefault((int(row["nsdId"]), int(row["repeat_index"])), []).append(row)
    refs: dict[tuple[int, int], dict[str, Any]] = {}
    for key, rows in groups.items():
        if not rows:
            continue
        refs[key] = {
            "x": torch.stack([row["x"].float() for row in rows], dim=0).mean(dim=0),
            "clip": rows[0]["clip"].float().clone(),
            "subjects": sorted({int(row["subject"]) for row in rows}),
            "n_ref_subjects": len({int(row["subject"]) for row in rows}),
        }
    return refs


def make_pairs(
    anchors: list[dict[str, Any]],
    refs: dict[tuple[int, int], dict[str, Any]],
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_repeat: dict[int, list[int]] = {}
    for nsd_id, repeat_index in refs:
        by_repeat.setdefault(int(repeat_index), []).append(int(nsd_id))

    pairs: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    for anchor in anchors:
        nsd_id = int(anchor["nsdId"])
        repeat_index = int(anchor["repeat_index"])
        ref = refs.get((nsd_id, repeat_index))
        candidates = [cid for cid in by_repeat.get(repeat_index, []) if cid != nsd_id]
        if ref is None or not candidates:
            continue
        neg_id = int(candidates[int(rng.integers(0, len(candidates)))])
        neg_ref = refs[(neg_id, repeat_index)]

        common = {
            "x1": anchor["x"].float().clone(),
            "clip_1": anchor["clip"].float().clone(),
            "subject": int(anchor["subject"]),
            "repeat_1": repeat_index,
            "repeat_2": repeat_index,
            "session_1": int(anchor["session_index"]),
            "session_2": -1,
            "anchor_nsdId": nsd_id,
        }
        pos = {
            **common,
            "x2": ref["x"].float().clone(),
            "clip_2": ref["clip"].float().clone(),
            "same_image": 1,
            "nsdId_1": nsd_id,
            "nsdId_2": nsd_id,
            "n_ref_subjects": int(ref["n_ref_subjects"]),
        }
        neg = {
            **common,
            "x2": neg_ref["x"].float().clone(),
            "clip_2": neg_ref["clip"].float().clone(),
            "same_image": 0,
            "nsdId_1": nsd_id,
            "nsdId_2": neg_id,
            "n_ref_subjects": int(neg_ref["n_ref_subjects"]),
        }
        pairs.extend([pos, neg])
        meta.append(
            {
                "subject": int(anchor["subject"]),
                "nsdId": nsd_id,
                "repeat_index": repeat_index,
                "positive_ref_subjects": ";".join(map(str, ref["subjects"])),
                "negative_nsdId": neg_id,
                "negative_ref_subjects": ";".join(map(str, neg_ref["subjects"])),
            }
        )
    return pairs, meta


def write_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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

        train_trials = expand_trials(load_sequences(in_fold, "train"))
        val_trials = expand_trials(load_sequences(in_fold, "val"))
        test_trials = expand_trials(load_sequences(in_fold, "test"))

        split_trials = {"train": train_trials, "val": val_trials, "test": test_trials}
        fold_qc: dict[str, Any] = {}
        for split, anchors in split_trials.items():
            pairs: list[dict[str, Any]] = []
            meta: list[dict[str, Any]] = []
            if split == "train":
                by_subject: dict[int, list[dict[str, Any]]] = {}
                for row in anchors:
                    by_subject.setdefault(int(row["subject"]), []).append(row)
                for subject, subject_anchors in by_subject.items():
                    refs = build_reference(train_trials, exclude_subject=subject)
                    subject_pairs, subject_meta = make_pairs(subject_anchors, refs, rng)
                    pairs.extend(subject_pairs)
                    meta.extend(subject_meta)
            else:
                refs = build_reference(train_trials, exclude_subject=None)
                pairs, meta = make_pairs(anchors, refs, rng)

            torch.save(pairs, out_fold / f"{split}_pairs.pt")
            write_metadata(out_fold / f"metadata_{split}_pairs.csv", meta)
            n_pos = sum(int(p["same_image"]) == 1 for p in pairs)
            n_neg = sum(int(p["same_image"]) == 0 for p in pairs)
            fold_qc[split] = {
                "n_anchor_trials": len(anchors),
                "n_pairs": len(pairs),
                "n_positive": n_pos,
                "n_negative": n_neg,
                "n_unique_anchor_images": len({int(row["nsdId"]) for row in anchors}),
                "n_unique_pair_images": len({int(p["nsdId_1"]) for p in pairs}),
                "status": "ok" if n_pos == n_neg and len(pairs) > 0 else "check",
            }
            count_rows.append({"fold": fold, "split": split, **fold_qc[split]})

        (out_fold / "dataset_qc.json").write_text(json.dumps(fold_qc, indent=2), encoding="utf-8")
        all_qc["folds"][fold] = fold_qc

    (output_root / "cross_subject_pair_dataset_qc.json").write_text(json.dumps(all_qc, indent=2), encoding="utf-8")
    with (output_root / "cross_subject_pair_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(count_rows[0].keys()))
        writer.writeheader()
        writer.writerows(count_rows)
    print(json.dumps({"output_root": str(output_root), "qc": all_qc}, indent=2))


if __name__ == "__main__":
    main()
