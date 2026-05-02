#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ROI repetition suppression/enhancement from strict T=3 sequences.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3"))
    parser.add_argument("--output-dir", type=Path, default=Path("preproc_v0/repetition_familiarity/analysis/scalar4_T3"))
    parser.add_argument("--folds", nargs="+", default=["fold_01", "fold_04"])
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--n-shuffles", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260501)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normal_sf_approx(z: float) -> float:
    return 0.5 * math.erfc(abs(z) / math.sqrt(2.0))


def paired_t_pvalue(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    n = values.size
    if n < 2:
        return float("nan"), float("nan")
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    if std == 0:
        return float("inf") if mean != 0 else 0.0, 0.0 if mean != 0 else 1.0
    t_value = mean / (std / math.sqrt(n))
    # n=8 here; normal approximation is sufficient for screening, not final inference.
    p_value = min(1.0, 2.0 * normal_sf_approx(t_value))
    return float(t_value), float(p_value)


def bh_fdr(pvalues: list[float]) -> list[float]:
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: (math.inf if not math.isfinite(pvalues[i]) else pvalues[i]))
    q = [math.nan] * n
    running = 1.0
    for rank_from_end, idx in enumerate(reversed(order), start=1):
        rank = n - rank_from_end + 1
        p = pvalues[idx]
        val = 1.0 if not math.isfinite(p) else min(1.0, p * n / rank)
        running = min(running, val)
        q[idx] = running
    return q


def load_sequences(root: Path, dataset_root: Path, folds: list[str], splits: list[str]) -> list[dict[str, object]]:
    rows = []
    seen = set()
    for fold in folds:
        for split in splits:
            path = root / dataset_root / fold / f"{split}_sequences.pt"
            seqs = torch.load(path, map_location="cpu", weights_only=False)
            for seq in seqs:
                subject = int(seq["subject"])
                nsd_id = int(seq["nsdId"])
                key = (subject, nsd_id)
                x = seq["x_seq"].to(torch.float32).numpy()
                rows.append({"fold": fold, "split": split, "subject": subject, "nsdId": nsd_id, "x": x, "dedup": key not in seen})
                seen.add(key)
    return rows


def subject_roi_stats(rows: list[dict[str, object]], dedup_only: bool) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[np.ndarray]] = defaultdict(list)
    for row in rows:
        if dedup_only and not row["dedup"]:
            continue
        key = ("dedup", "all", int(row["subject"])) if dedup_only else (str(row["fold"]), str(row["split"]), int(row["subject"]))
        grouped[key].append(row["x"][:, :, 0])  # [3, 180], mean_beta only

    out = []
    for (fold, split, subject), values in grouped.items():
        arr = np.stack(values, axis=0)  # [n_images, 3, 180]
        means = arr.mean(axis=0)  # [3, 180]
        for roi_idx in range(means.shape[1]):
            r1 = float(means[0, roi_idx])
            r2 = float(means[1, roi_idx])
            r3 = float(means[2, roi_idx])
            out.append(
                {
                    "fold": fold,
                    "split": split,
                    "subject": subject,
                    "roi_id": roi_idx,
                    "repeat1_mean": r1,
                    "repeat2_mean": r2,
                    "repeat3_mean": r3,
                    "delta_21": r2 - r1,
                    "delta_31": r3 - r1,
                    "delta_32": r3 - r2,
                    "n_images": int(arr.shape[0]),
                }
            )
    return out


def group_summary(by_subject_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rois = sorted({int(row["roi_id"]) for row in by_subject_rows})
    out = []
    p21, p31, p32 = [], [], []
    temp = []
    for roi_id in rois:
        rows = [row for row in by_subject_rows if int(row["roi_id"]) == roi_id]
        d21 = np.array([float(row["delta_21"]) for row in rows], dtype=np.float64)
        d31 = np.array([float(row["delta_31"]) for row in rows], dtype=np.float64)
        d32 = np.array([float(row["delta_32"]) for row in rows], dtype=np.float64)
        t21, pv21 = paired_t_pvalue(d21)
        t31, pv31 = paired_t_pvalue(d31)
        t32, pv32 = paired_t_pvalue(d32)
        p21.append(pv21); p31.append(pv31); p32.append(pv32)
        temp.append((roi_id, d21, d31, d32, t21, pv21, t31, pv31, t32, pv32))
    q21, q31, q32 = bh_fdr(p21), bh_fdr(p31), bh_fdr(p32)
    for idx, (roi_id, d21, d31, d32, t21, pv21, t31, pv31, t32, pv32) in enumerate(temp):
        m21 = float(d21.mean()); m31 = float(d31.mean()); m32 = float(d32.mean())
        out.append(
            {
                "roi_id": roi_id,
                "n_subjects": int(d21.size),
                "mean_delta_21": m21,
                "mean_delta_31": m31,
                "mean_delta_32": m32,
                "t_value_21": t21,
                "p_value_21": pv21,
                "fdr_q_value_21": q21[idx],
                "t_value_31": t31,
                "p_value_31": pv31,
                "fdr_q_value_31": q31[idx],
                "t_value_32": t32,
                "p_value_32": pv32,
                "fdr_q_value_32": q32[idx],
                "effect_direction_21": "suppression" if m21 < 0 else "enhancement",
                "effect_direction_31": "suppression" if m31 < 0 else "enhancement",
            }
        )
    return out


def shuffle_null(rows: list[dict[str, object]], n_shuffles: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    dedup = [row for row in rows if row["dedup"]]
    real = subject_roi_stats(rows, dedup_only=True)
    real_group = group_summary(real)
    real_abs21 = float(np.mean([abs(float(row["mean_delta_21"])) for row in real_group]))
    real_abs31 = float(np.mean([abs(float(row["mean_delta_31"])) for row in real_group]))
    null21, null31 = [], []
    for _ in range(n_shuffles):
        shuffled = []
        for row in dedup:
            x = np.array(row["x"], copy=True)
            x = x[rng.permutation(3)]
            shuffled.append({**row, "x": x})
        stats = subject_roi_stats(shuffled, dedup_only=True)
        grp = group_summary(stats)
        null21.append(float(np.mean([abs(float(row["mean_delta_21"])) for row in grp])))
        null31.append(float(np.mean([abs(float(row["mean_delta_31"])) for row in grp])))
    return {
        "real_mean_abs_delta_21": real_abs21,
        "real_mean_abs_delta_31": real_abs31,
        "shuffle_mean_abs_delta_21_mean": float(np.mean(null21)),
        "shuffle_mean_abs_delta_31_mean": float(np.mean(null31)),
        "shuffle_p_abs_delta_21": float((np.sum(np.array(null21) >= real_abs21) + 1) / (n_shuffles + 1)),
        "shuffle_p_abs_delta_31": float((np.sum(np.array(null31) >= real_abs31) + 1) / (n_shuffles + 1)),
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_sequences(root, args.dataset_root, args.folds, args.splits)

    by_subject_dedup = subject_roi_stats(rows, dedup_only=True)
    by_subject_split = subject_roi_stats(rows, dedup_only=False)
    group = group_summary(by_subject_dedup)
    null = shuffle_null(rows, args.n_shuffles, args.seed)

    by_subject_rows = by_subject_split + by_subject_dedup
    write_csv(
        out_dir / "roi_repetition_suppression_by_subject.csv",
        by_subject_rows,
        ["fold", "split", "subject", "roi_id", "repeat1_mean", "repeat2_mean", "repeat3_mean", "delta_21", "delta_31", "delta_32", "n_images"],
    )
    write_csv(
        out_dir / "roi_repetition_suppression_group.csv",
        group,
        [
            "roi_id", "n_subjects", "mean_delta_21", "mean_delta_31", "mean_delta_32",
            "t_value_21", "p_value_21", "fdr_q_value_21",
            "t_value_31", "p_value_31", "fdr_q_value_31",
            "t_value_32", "p_value_32", "fdr_q_value_32",
            "effect_direction_21", "effect_direction_31",
        ],
    )
    summary = {
        "n_rois": len(group),
        "n_subjects": len({row["subject"] for row in by_subject_dedup}),
        "n_dedup_sequences": len({(row["subject"], row["nsdId"]) for row in rows if row["dedup"]}),
        "n_fdr_sig_delta_21": int(sum(float(row["fdr_q_value_21"]) < 0.05 for row in group)),
        "n_fdr_sig_delta_31": int(sum(float(row["fdr_q_value_31"]) < 0.05 for row in group)),
        "n_suppression_delta_21": int(sum(float(row["mean_delta_21"]) < 0 for row in group)),
        "n_suppression_delta_31": int(sum(float(row["mean_delta_31"]) < 0 for row in group)),
        "top10_suppression_delta_31": sorted(group, key=lambda r: float(r["mean_delta_31"]))[:10],
        "top10_enhancement_delta_31": sorted(group, key=lambda r: float(r["mean_delta_31"]), reverse=True)[:10],
        "shuffle_control": null,
        "visual_roi_note": "HCP-MMP label names/groups are not available in the current node-set JSON; visual-vs-nonvisual grouping is deferred.",
    }
    (out_dir / "roi_repetition_suppression_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
