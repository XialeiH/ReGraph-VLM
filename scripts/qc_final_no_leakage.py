#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final split/leakage QC for ReGraph-VLM datasets.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("preproc_v0/repetition_familiarity/results/final_audit"))
    return parser.parse_args()


def load(path: Path) -> list[dict[str, Any]]:
    return torch.load(path, map_location="cpu", weights_only=False)


def subject_set(pairs: list[dict[str, Any]]) -> set[int]:
    vals = set()
    for p in pairs:
        vals.add(int(p["subject"]))
        if "subject_2" in p:
            vals.add(int(p["subject_2"]))
    return vals


def check_pair_rules(pairs: list[dict[str, Any]], require_matched_repeat: bool = True) -> dict[str, int]:
    bad_pos_nsd = bad_neg_nsd = bad_repeat = missing_clip = nan_clip = 0
    for p in pairs:
        same = int(p["same_image"])
        n1, n2 = int(p["nsdId_1"]), int(p["nsdId_2"])
        if same == 1 and n1 != n2:
            bad_pos_nsd += 1
        if same == 0 and n1 == n2:
            bad_neg_nsd += 1
        if require_matched_repeat and int(p["repeat_1"]) != int(p["repeat_2"]):
            bad_repeat += 1
        if "clip_1" not in p or "clip_2" not in p:
            missing_clip += 1
        else:
            if bool(torch.isnan(p["clip_1"]).any()) or bool(torch.isnan(p["clip_2"]).any()):
                nan_clip += 1
    return {
        "bad_positive_nsd": bad_pos_nsd,
        "bad_negative_nsd": bad_neg_nsd,
        "bad_repeat_match": bad_repeat,
        "missing_clip": missing_clip,
        "nan_clip": nan_clip,
    }


def check_dataset(name: str, dataset_root: Path, heldout_image: bool = False) -> dict[str, Any]:
    fold_rows = []
    for fold_dir in sorted(dataset_root.glob("fold_*")):
        split_pairs = {}
        for split in ["train", "val", "test"]:
            path = fold_dir / f"{split}_pairs.pt"
            if path.exists():
                split_pairs[split] = load(path)
        if not split_pairs:
            continue
        train_subjects = subject_set(split_pairs.get("train", []))
        val_subjects = subject_set(split_pairs.get("val", []))
        test_subjects = subject_set(split_pairs.get("test", []))
        train_nsd = {int(p["nsdId_1"]) for p in split_pairs.get("train", [])} | {
            int(p["nsdId_2"]) for p in split_pairs.get("train", [])
        }
        test_nsd = {int(p["nsdId_1"]) for p in split_pairs.get("test", [])} | {
            int(p["nsdId_2"]) for p in split_pairs.get("test", [])
        }
        rule_counts = {split: check_pair_rules(pairs) for split, pairs in split_pairs.items()}
        fold_rows.append(
            {
                "dataset": name,
                "fold": fold_dir.name,
                "train_pairs": len(split_pairs.get("train", [])),
                "val_pairs": len(split_pairs.get("val", [])),
                "test_pairs": len(split_pairs.get("test", [])),
                "train_test_subject_overlap": sorted(train_subjects & test_subjects),
                "train_val_subject_overlap": sorted(train_subjects & val_subjects),
                "heldout_image_train_test_nsd_overlap": len(train_nsd & test_nsd) if heldout_image else None,
                "rule_counts": rule_counts,
            }
        )
    status = "ok"
    for row in fold_rows:
        if row["train_test_subject_overlap"]:
            status = "fail"
        if heldout_image and row["heldout_image_train_test_nsd_overlap"]:
            status = "fail"
        for counts in row["rule_counts"].values():
            if any(int(v) != 0 for v in counts.values()):
                status = "fail"
    return {"dataset": name, "status": status, "folds": fold_rows}


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    base = root / "preproc_v0/repetition_familiarity/datasets"
    specs = [
        ("cross_subject_allfold", base / "scalar4_T3_clip_cross_subject_allfold", False),
        ("heldout_image", base / "scalar4_T3_clip_cross_subject_imageheldout", True),
        ("heldout_image_random", base / "scalar4_T3_clip_cross_subject_imageheldout_random_embedding", True),
        ("hardneg_allfold", base / "scalar4_T3_clip_cross_subject_hardneg_allfold/mixed_50_random_50_clip_hard", False),
    ]
    results = [check_dataset(name, path, heldout) for name, path, heldout in specs if path.exists()]
    overall = "ok" if all(r["status"] == "ok" for r in results) else "fail"
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"status": overall, "datasets": results}
    (out_dir / "final_leakage_qc.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = []
    for result in results:
        for row in result["folds"]:
            rows.append(
                {
                    "dataset": result["dataset"],
                    "fold": row["fold"],
                    "train_pairs": row["train_pairs"],
                    "val_pairs": row["val_pairs"],
                    "test_pairs": row["test_pairs"],
                    "train_test_subject_overlap": row["train_test_subject_overlap"],
                    "heldout_image_train_test_nsd_overlap": row["heldout_image_train_test_nsd_overlap"],
                    "status": result["status"],
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "final_leakage_qc.csv", index=False)
    (out_dir / "final_leakage_qc.md").write_text("# Final Leakage QC\n\n" + df.to_markdown(index=False), encoding="utf-8")
    print({"out_dir": str(out_dir), "status": overall})


if __name__ == "__main__":
    main()
