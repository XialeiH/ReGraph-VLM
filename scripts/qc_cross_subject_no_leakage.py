#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QC cross-subject pair datasets for leakage and pair semantics.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3_clip_cross_subject"),
    )
    parser.add_argument("--folds", nargs="+", default=["fold_01", "fold_04"])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/frozen_cross_subject"),
    )
    return parser.parse_args()


def subject_for_fold(fold: str) -> int:
    return int(fold.split("_")[-1])


def load_pairs(fold_dir: Path, split: str) -> list[dict[str, Any]]:
    return torch.load(fold_dir / f"{split}_pairs.pt", map_location="cpu", weights_only=False)


def pair_key(pair: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        int(pair["subject"]),
        int(pair["nsdId_1"]),
        int(pair["nsdId_2"]),
        int(pair["repeat_1"]),
        int(pair["same_image"]),
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    dataset_root = root / args.dataset_root
    out_dir = root / args.output_root
    out_dir.mkdir(parents=True, exist_ok=True)

    qc: dict[str, Any] = {"dataset_root": str(dataset_root), "folds": {}, "status": "ok"}
    md_lines = ["# Cross-Subject Leakage QC", ""]
    for fold in args.folds:
        fold_dir = dataset_root / fold
        test_subject = subject_for_fold(fold)
        train_pairs = load_pairs(fold_dir, "train")
        val_pairs = load_pairs(fold_dir, "val")
        test_pairs = load_pairs(fold_dir, "test")
        split_pairs = {"train": train_pairs, "val": val_pairs, "test": test_pairs}

        train_subjects = {int(p["subject"]) for p in train_pairs}
        val_subjects = {int(p["subject"]) for p in val_pairs}
        test_subjects = {int(p["subject"]) for p in test_pairs}
        duplicate_overlap = {
            "train_val": len({pair_key(p) for p in train_pairs} & {pair_key(p) for p in val_pairs}),
            "train_test": len({pair_key(p) for p in train_pairs} & {pair_key(p) for p in test_pairs}),
            "val_test": len({pair_key(p) for p in val_pairs} & {pair_key(p) for p in test_pairs}),
        }
        fold_qc: dict[str, Any] = {
            "test_subject": test_subject,
            "train_subjects": sorted(train_subjects),
            "val_subjects": sorted(val_subjects),
            "test_subjects": sorted(test_subjects),
            "test_subject_in_train": test_subject in train_subjects,
            "split_duplicate_overlap": duplicate_overlap,
            "splits": {},
        }
        for split, pairs in split_pairs.items():
            n_pos = n_neg = pos_bad = neg_bad = repeat_bad = subject_bad = nan_count = inf_count = 0
            nsd_missing = 0
            for p in pairs:
                same = int(p["same_image"])
                nsd1 = int(p["nsdId_1"])
                nsd2 = int(p["nsdId_2"])
                rep1 = int(p["repeat_1"])
                rep2 = int(p["repeat_2"])
                if same == 1:
                    n_pos += 1
                    pos_bad += int(nsd1 != nsd2)
                else:
                    n_neg += 1
                    neg_bad += int(nsd1 == nsd2)
                repeat_bad += int(rep1 != rep2)
                subject_bad += int(int(p.get("n_ref_subjects", 1)) <= 0)
                for key in ["x1", "x2", "clip_1", "clip_2"]:
                    value = p.get(key)
                    if isinstance(value, torch.Tensor) and value.is_floating_point():
                        nan_count += int(torch.isnan(value).sum().item())
                        inf_count += int(torch.isinf(value).sum().item())
                nsd_missing += int("nsdId_1" not in p or "nsdId_2" not in p)
            split_qc = {
                "n_pairs": len(pairs),
                "n_positive": n_pos,
                "n_negative": n_neg,
                "positive_nsdId_mismatch": pos_bad,
                "negative_nsdId_match": neg_bad,
                "repeat_mismatch": repeat_bad,
                "bad_ref_subject_count": subject_bad,
                "nan_count": nan_count,
                "inf_count": inf_count,
                "missing_nsdId_fields": nsd_missing,
            }
            split_qc["status"] = "ok" if all(split_qc[k] == 0 for k in [
                "positive_nsdId_mismatch",
                "negative_nsdId_match",
                "repeat_mismatch",
                "bad_ref_subject_count",
                "nan_count",
                "inf_count",
                "missing_nsdId_fields",
            ]) else "check"
            fold_qc["splits"][split] = split_qc
        fold_status = (
            not fold_qc["test_subject_in_train"]
            and all(v == 0 for v in duplicate_overlap.values())
            and all(v["status"] == "ok" for v in fold_qc["splits"].values())
        )
        fold_qc["status"] = "ok" if fold_status else "check"
        qc["folds"][fold] = fold_qc
        if fold_qc["status"] != "ok":
            qc["status"] = "check"
        md_lines.extend([f"## {fold}", "", "```json", json.dumps(fold_qc, indent=2), "```", ""])

    (out_dir / "cross_subject_leakage_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    (out_dir / "cross_subject_leakage_qc.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "status": qc["status"]}, indent=2))


if __name__ == "__main__":
    main()
