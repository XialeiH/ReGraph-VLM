#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch


PAIR_TYPES = [(1, 2), (1, 3), (2, 3)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze same-image repeat representational stability.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3"))
    parser.add_argument("--output-dir", type=Path, default=Path("preproc_v0/repetition_familiarity/analysis/scalar4_T3"))
    parser.add_argument("--folds", nargs="+", default=["fold_01", "fold_04"])
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser.parse_args()


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    am = a - a.mean()
    bm = b - b.mean()
    denom = float(np.linalg.norm(am) * np.linalg.norm(bm))
    if denom == 0:
        return 0.0
    return float(np.dot(am, bm) / denom)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normal_sf_approx(z: float) -> float:
    return 0.5 * math.erfc(abs(z) / math.sqrt(2.0))


def two_sample_summary(values: list[float], controls: list[float]) -> dict[str, float]:
    v = np.array(values, dtype=np.float64)
    c = np.array(controls, dtype=np.float64)
    gap = float(v.mean() - c.mean())
    pooled = math.sqrt(float(v.var(ddof=1) / max(len(v), 1) + c.var(ddof=1) / max(len(c), 1))) if len(v) > 1 and len(c) > 1 else 0.0
    z = gap / pooled if pooled > 0 else 0.0
    return {
        "same_mean": float(v.mean()),
        "same_std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "diff_mean": float(c.mean()),
        "diff_std": float(c.std(ddof=1)) if len(c) > 1 else 0.0,
        "same_minus_diff": gap,
        "z_value": float(z),
        "p_value_normal_approx": float(min(1.0, 2.0 * normal_sf_approx(z))),
        "n_same": int(len(v)),
        "n_diff": int(len(c)),
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for fold in args.folds:
        for split in args.splits:
            pairs = torch.load(root / args.dataset_root / fold / f"{split}_pairs.pt", map_location="cpu", weights_only=False)
            for item in pairs:
                r1 = int(item["repeat_1"])
                r2 = int(item["repeat_2"])
                pair_type = f"{r1}-{r2}"
                sim = corr(item["x1"].numpy(), item["x2"].numpy())
                rows.append(
                    {
                        "fold": fold,
                        "split": split,
                        "subject": int(item["subject"]),
                        "nsdId_1": int(item["nsdId_1"]),
                        "nsdId_2": int(item["nsdId_2"]),
                        "same_image": int(item["same_image"]),
                        "repeat_pair": pair_type,
                        "similarity": sim,
                        "session_1": int(item["session_1"]),
                        "session_2": int(item["session_2"]),
                    }
                )

    write_csv(
        out_dir / "repeat_representational_stability_pairs.csv",
        rows,
        ["fold", "split", "subject", "nsdId_1", "nsdId_2", "same_image", "repeat_pair", "similarity", "session_1", "session_2"],
    )

    summary_rows: list[dict[str, object]] = []
    summary_json: dict[str, object] = {}
    for pair_type in ["1-2", "1-3", "2-3"]:
        same = [float(r["similarity"]) for r in rows if r["repeat_pair"] == pair_type and int(r["same_image"]) == 1]
        diff = [float(r["similarity"]) for r in rows if r["repeat_pair"] == pair_type and int(r["same_image"]) == 0]
        s = two_sample_summary(same, diff)
        summary_json[pair_type] = s
        summary_rows.append({"scope": "all", "repeat_pair": pair_type, **s})
        for fold in args.folds:
            fsame = [float(r["similarity"]) for r in rows if r["fold"] == fold and r["repeat_pair"] == pair_type and int(r["same_image"]) == 1]
            fdiff = [float(r["similarity"]) for r in rows if r["fold"] == fold and r["repeat_pair"] == pair_type and int(r["same_image"]) == 0]
            summary_rows.append({"scope": fold, "repeat_pair": pair_type, **two_sample_summary(fsame, fdiff)})

    write_csv(
        out_dir / "repeat_representational_stability_summary.csv",
        summary_rows,
        ["scope", "repeat_pair", "same_mean", "same_std", "diff_mean", "diff_std", "same_minus_diff", "z_value", "p_value_normal_approx", "n_same", "n_diff"],
    )
    (out_dir / "repeat_representational_stability_summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    print(json.dumps(summary_json, indent=2))


if __name__ == "__main__":
    main()
