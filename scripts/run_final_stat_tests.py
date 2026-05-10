#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired final statistical tests for ReGraph-VLM summaries.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    return parser.parse_args()


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def paired_t_pvalue(diff: np.ndarray) -> tuple[float, float]:
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n < 2:
        return float("nan"), float("nan")
    mean = float(diff.mean())
    std = float(diff.std(ddof=1))
    if std < 1e-12:
        return float("inf"), 0.0
    t = mean / (std / math.sqrt(n))
    p = 2.0 * (1.0 - norm_cdf(abs(t)))
    return float(t), float(p)


def bootstrap_ci(diff: np.ndarray, seed: int = 2026, n_boot: int = 5000) -> tuple[float, float]:
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = [float(rng.choice(diff, size=len(diff), replace=True).mean()) for _ in range(n_boot)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def compare(df: pd.DataFrame, name: str, a_filter: dict[str, object], b_filter: dict[str, object], metrics: list[str]) -> list[dict[str, object]]:
    def filt(spec: dict[str, object]) -> pd.DataFrame:
        out = df.copy()
        for key, value in spec.items():
            out = out[out[key] == value]
        return out

    a = filt(a_filter)
    b = filt(b_filter)
    if a.empty or b.empty:
        return []
    key_cols = [c for c in ["fold", "seed"] if c in a.columns and c in b.columns]
    merged = a.merge(b, on=key_cols, suffixes=("_a", "_b"))
    rows = []
    for metric in metrics:
        ca, cb = f"{metric}_a", f"{metric}_b"
        if ca not in merged.columns or cb not in merged.columns:
            continue
        diff = (merged[ca] - merged[cb]).to_numpy(float)
        t, p = paired_t_pvalue(diff)
        lo, hi = bootstrap_ci(diff)
        rows.append(
            {
                "comparison": name,
                "metric": metric,
                "n_pairs": int(np.isfinite(diff).sum()),
                "mean_diff": float(np.nanmean(diff)),
                "t_approx": t,
                "p_norm_approx": p,
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    results = root / "preproc_v0/repetition_familiarity/results"
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    metrics = ["AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "brain_R@5", "brain_MRR"]
    rows = []

    final_path = results / "cross_subject_allfold_final/regraph_vlm_summary.csv"
    if final_path.exists():
        df = pd.read_csv(final_path)
        rows += compare(
            df,
            "gated_vs_roi_mlp_allfold_final",
            {"graph_encoder": "bnt_token_flat", "readout": "gated_flat", "lambda_clip": 2.0},
            {"graph_encoder": "roi_mlp", "readout": "flat", "lambda_clip": 2.0},
            metrics,
        )
        rows += compare(
            df,
            "gated_vs_flat_allfold_final",
            {"graph_encoder": "bnt_token_flat", "readout": "gated_flat", "lambda_clip": 2.0},
            {"graph_encoder": "bnt_token_flat", "readout": "flat", "lambda_clip": 2.0},
            metrics,
        )

    heldout = results / "heldout_image_final/regraph_vlm_summary.csv"
    if not heldout.exists():
        heldout = results / "heldout_image/regraph_vlm_summary.csv"
    heldout_rand = results / "heldout_image_random_embedding_final/regraph_vlm_summary.csv"
    if not heldout_rand.exists():
        heldout_rand = results / "heldout_image_random_embedding/regraph_vlm_summary.csv"
    if heldout.exists() and heldout_rand.exists():
        df = pd.concat([pd.read_csv(heldout), pd.read_csv(heldout_rand)], ignore_index=True)
        df["source"] = ["real"] * len(pd.read_csv(heldout)) + ["random"] * len(pd.read_csv(heldout_rand))
        rows += compare(
            df,
            "heldout_gated_real_clip_vs_random_embedding",
            {"graph_encoder": "bnt_token_flat", "readout": "gated_flat", "source": "real"},
            {"graph_encoder": "bnt_token_flat", "readout": "gated_flat", "source": "random"},
            metrics,
        )

    out_csv = out / "final_stat_tests.csv"
    out_md = out / "final_stat_tests.md"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    if rows:
        out_md.write_text(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")
    else:
        out_md.write_text("No paired comparisons available yet.\n", encoding="utf-8")
    print({"out_csv": str(out_csv), "n_rows": len(rows)})


if __name__ == "__main__":
    main()
