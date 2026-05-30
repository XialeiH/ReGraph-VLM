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
    parser = argparse.ArgumentParser(description="Build single-reference cross-subject pairs with anchor/reference session controls.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--input-dataset-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_allfold"),
    )
    parser.add_argument(
        "--output-dataset-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject_single_ref_matched"),
    )
    parser.add_argument("--folds", nargs="+", default=[f"fold_{idx:02d}" for idx in range(1, 9)])
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
                    "x": seq["x_seq"][idx].float().clone(),
                    "clip": seq["clip"].float().clone(),
                    "subject": int(seq["subject"]),
                    "nsdId": int(seq["nsdId"]),
                    "repeat_index": int(repeat_index),
                    "session_index": int(session_seq[idx]),
                    "trial_index": int(trial_seq[idx]),
                }
            )
    return rows


def index_trials(rows: list[dict[str, Any]]) -> tuple[dict[tuple[int, int], list[dict[str, Any]]], dict[tuple[int, int, int], list[dict[str, Any]]]]:
    by_image_repeat: dict[tuple[int, int], list[dict[str, Any]]] = {}
    by_subject_repeat: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        by_image_repeat.setdefault((int(row["nsdId"]), int(row["repeat_index"])), []).append(row)
        by_subject_repeat.setdefault((int(row["subject"]), int(row["repeat_index"])), []).append(row)
    return by_image_repeat, by_subject_repeat


def choose_negative(
    candidates: list[dict[str, Any]],
    anchor_nsd: int,
    target_session: int,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    valid = [row for row in candidates if int(row["nsdId"]) != int(anchor_nsd)]
    if not valid:
        return None
    distances = np.array([abs(int(row["session_index"]) - int(target_session)) for row in valid])
    best_distance = int(distances.min())
    best = [row for row, distance in zip(valid, distances) if int(distance) == best_distance]
    return best[int(rng.integers(0, len(best)))]


def make_pairs(
    anchors: list[dict[str, Any]],
    reference_trials: list[dict[str, Any]],
    rng: np.random.Generator,
    exclude_anchor_subject_from_refs: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_image_repeat, by_subject_repeat = index_trials(reference_trials)
    pairs: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []

    for anchor in anchors:
        nsd_id = int(anchor["nsdId"])
        repeat_index = int(anchor["repeat_index"])
        positive_candidates = by_image_repeat.get((nsd_id, repeat_index), [])
        if exclude_anchor_subject_from_refs:
            positive_candidates = [row for row in positive_candidates if int(row["subject"]) != int(anchor["subject"])]
        if not positive_candidates:
            continue
        pos_ref = positive_candidates[int(rng.integers(0, len(positive_candidates)))]
        ref_subject = int(pos_ref["subject"])
        neg_ref = choose_negative(
            by_subject_repeat.get((ref_subject, repeat_index), []),
            anchor_nsd=nsd_id,
            target_session=int(pos_ref["session_index"]),
            rng=rng,
        )
        if neg_ref is None:
            continue

        common = {
            "x1": anchor["x"].clone(),
            "clip_1": anchor["clip"].clone(),
            "subject": int(anchor["subject"]),
            "subject_1": int(anchor["subject"]),
            "subject_2": ref_subject,
            "repeat_1": repeat_index,
            "repeat_2": repeat_index,
            "session_1": int(anchor["session_index"]),
            "trial_1": int(anchor["trial_index"]),
            "anchor_nsdId": nsd_id,
            "reference_subject": ref_subject,
        }
        pos = {
            **common,
            "x2": pos_ref["x"].clone(),
            "clip_2": pos_ref["clip"].clone(),
            "same_image": 1,
            "nsdId_1": nsd_id,
            "nsdId_2": nsd_id,
            "session_2": int(pos_ref["session_index"]),
            "trial_2": int(pos_ref["trial_index"]),
        }
        neg = {
            **common,
            "x2": neg_ref["x"].clone(),
            "clip_2": neg_ref["clip"].clone(),
            "same_image": 0,
            "nsdId_1": nsd_id,
            "nsdId_2": int(neg_ref["nsdId"]),
            "session_2": int(neg_ref["session_index"]),
            "trial_2": int(neg_ref["trial_index"]),
        }
        pairs.extend([pos, neg])
        meta.append(
            {
                "anchor_subject": int(anchor["subject"]),
                "reference_subject": ref_subject,
                "anchor_nsdId": nsd_id,
                "positive_nsdId": nsd_id,
                "negative_nsdId": int(neg_ref["nsdId"]),
                "repeat_index": repeat_index,
                "anchor_session": int(anchor["session_index"]),
                "positive_reference_session": int(pos_ref["session_index"]),
                "negative_reference_session": int(neg_ref["session_index"]),
                "reference_session_abs_diff": abs(int(pos_ref["session_index"]) - int(neg_ref["session_index"])),
                "anchor_trial": int(anchor["trial_index"]),
                "positive_reference_trial": int(pos_ref["trial_index"]),
                "negative_reference_trial": int(neg_ref["trial_index"]),
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


def summarize_metadata(meta: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    session_diffs = [int(row["reference_session_abs_diff"]) for row in meta]
    return {
        "n_pairs": len(pairs),
        "n_positive": sum(int(pair["same_image"]) == 1 for pair in pairs),
        "n_negative": sum(int(pair["same_image"]) == 0 for pair in pairs),
        "n_anchor_groups": len(meta),
        "n_unique_anchor_images": len({int(row["anchor_nsdId"]) for row in meta}),
        "n_unique_reference_subjects": len({int(row["reference_subject"]) for row in meta}),
        "reference_session_exact_match_rate": float(np.mean([diff == 0 for diff in session_diffs])) if session_diffs else float("nan"),
        "reference_session_mean_abs_diff": float(np.mean(session_diffs)) if session_diffs else float("nan"),
        "reference_session_max_abs_diff": int(max(session_diffs)) if session_diffs else 0,
        "status": "ok" if len(pairs) > 0 and sum(int(pair["same_image"]) == 1 for pair in pairs) == sum(int(pair["same_image"]) == 0 for pair in pairs) else "check",
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    input_root = root / args.input_dataset_root
    output_root = root / args.output_dataset_root
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    qc: dict[str, Any] = {"seed": args.seed, "folds": {}}
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
            if split == "train":
                reference_trials = train_trials
                exclude_anchor_subject = True
            else:
                reference_trials = train_trials
                exclude_anchor_subject = False
            pairs, meta = make_pairs(anchors, reference_trials, rng, exclude_anchor_subject)
            torch.save(pairs, out_fold / f"{split}_pairs.pt")
            write_metadata(out_fold / f"metadata_{split}_pairs.csv", meta)
            split_qc = summarize_metadata(meta, pairs)
            fold_qc[split] = split_qc
            count_rows.append({"fold": fold, "split": split, **split_qc})

        (out_fold / "dataset_qc.json").write_text(json.dumps(fold_qc, indent=2), encoding="utf-8")
        qc["folds"][fold] = fold_qc

    (output_root / "single_reference_matched_dataset_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    with (output_root / "single_reference_matched_pair_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(count_rows[0].keys()))
        writer.writeheader()
        writer.writerows(count_rows)
    print(json.dumps({"output_root": str(output_root), "n_rows": len(count_rows)}, indent=2))


if __name__ == "__main__":
    main()
