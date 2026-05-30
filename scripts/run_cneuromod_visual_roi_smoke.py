#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a raw external CNeuroMod visual-ROI retrieval smoke test.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/cneuromod_things/visual_roi_scalar4_smoke"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/gpfsnyu/scratch/xh2906/ReGraph-VLM/external_validation/cneuromod_things/visual_roi_scalar4_smoke/results"),
    )
    parser.add_argument("--subject-a", default="sub-01")
    parser.add_argument("--subject-b", default="sub-02")
    return parser.parse_args()


def load_subject(data_dir: Path, subject: str) -> tuple[pd.DataFrame, np.ndarray]:
    pt = torch.load(data_dir / f"{subject}_cneuromod_visual_roi_scalar4.pt", map_location="cpu")
    x = pt["x"].numpy().reshape(pt["x"].shape[0], -1).astype(np.float32)
    image_label = np.asarray(pt["image_label"], dtype=str)
    repetition = pt["repetition"].numpy().astype(int)
    meta = pd.DataFrame({"image_label": image_label, "repetition": repetition})
    return meta, x


def zscore_rows(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, 1e-8)


def rank_metrics(scores: np.ndarray) -> tuple[float, float]:
    ranks = []
    for i, row in enumerate(scores):
        order = np.argsort(-row)
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)
    ranks_arr = np.asarray(ranks)
    return float((ranks_arr <= 5).mean()), float((1.0 / ranks_arr).mean())


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta_a, x_a = load_subject(args.data_dir, args.subject_a)
    meta_b, x_b = load_subject(args.data_dir, args.subject_b)
    x_a = zscore_rows(x_a)
    x_b = zscore_rows(x_b)

    rows = []
    y_all = []
    score_all = []
    forward_r5 = []
    forward_mrr = []
    reverse_r5 = []
    reverse_mrr = []

    for repetition in sorted(set(meta_a["repetition"]).intersection(set(meta_b["repetition"]))):
        a_idx = np.flatnonzero(meta_a["repetition"].to_numpy() == repetition)
        b_idx = np.flatnonzero(meta_b["repetition"].to_numpy() == repetition)
        a = meta_a.iloc[a_idx].reset_index(drop=True)
        b = meta_b.iloc[b_idx].reset_index(drop=True)
        shared = sorted(set(a["image_label"]).intersection(set(b["image_label"])))
        a_pos = {label: idx for idx, label in enumerate(a["image_label"])}
        b_pos = {label: idx for idx, label in enumerate(b["image_label"])}
        xa = x_a[a_idx[np.asarray([a_pos[label] for label in shared], dtype=np.int64)]]
        xb = x_b[b_idx[np.asarray([b_pos[label] for label in shared], dtype=np.int64)]]

        scores = xa @ xb.T
        labels = np.eye(len(shared), dtype=np.int8)
        y_all.append(labels.reshape(-1))
        score_all.append(scores.reshape(-1))

        f_r5, f_mrr = rank_metrics(scores)
        r_r5, r_mrr = rank_metrics(scores.T)
        forward_r5.append(f_r5)
        forward_mrr.append(f_mrr)
        reverse_r5.append(r_r5)
        reverse_mrr.append(r_mrr)
        rows.append(
            {
                "repetition": int(repetition),
                "n_images": int(len(shared)),
                "AUROC": float(roc_auc_score(labels.reshape(-1), scores.reshape(-1))),
                "AUPRC": float(average_precision_score(labels.reshape(-1), scores.reshape(-1))),
                "A_to_B_R@5": f_r5,
                "A_to_B_MRR": f_mrr,
                "B_to_A_R@5": r_r5,
                "B_to_A_MRR": r_mrr,
            }
        )

    y = np.concatenate(y_all)
    score = np.concatenate(score_all)
    summary = {
        "subject_a": args.subject_a,
        "subject_b": args.subject_b,
        "n_pairs": int(len(y)),
        "n_positive_pairs": int(y.sum()),
        "AUROC": float(roc_auc_score(y, score)),
        "AUPRC": float(average_precision_score(y, score)),
        "R@5": float(np.mean(forward_r5 + reverse_r5)),
        "MRR": float(np.mean(forward_mrr + reverse_mrr)),
        "A_to_B_R@5": float(np.mean(forward_r5)),
        "A_to_B_MRR": float(np.mean(forward_mrr)),
        "B_to_A_R@5": float(np.mean(reverse_r5)),
        "B_to_A_MRR": float(np.mean(reverse_mrr)),
        "chance_R@5": 5.0 / rows[0]["n_images"] if rows else np.nan,
        "chance_AUPRC": float(y.mean()),
    }

    per_repeat = pd.DataFrame(rows)
    per_repeat.to_csv(args.out_dir / "cneuromod_visual_roi_smoke_per_repeat.csv", index=False)
    pd.DataFrame([summary]).to_csv(args.out_dir / "cneuromod_visual_roi_smoke_summary.csv", index=False)

    lines = [
        "# CNeuroMod Visual-ROI External Smoke Test",
        "",
        "This evaluates raw cosine similarity between public CNeuroMod visual-ROI scalar4 features for two subjects.",
        "It is a lightweight external sanity check, not a full ReGraph-VLM retraining result.",
        "",
        "## Summary",
        "",
        pd.DataFrame([summary]).to_markdown(index=False),
        "",
        "## Per Repeat",
        "",
        per_repeat.to_markdown(index=False),
        "",
    ]
    (args.out_dir / "cneuromod_visual_roi_smoke_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
