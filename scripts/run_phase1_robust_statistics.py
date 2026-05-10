#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METRICS = ["AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "image_MRR", "brain_R@5", "brain_MRR"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-1 robust statistics for ReGraph-VLM final comparisons.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/cross_subject_allfold_final/regraph_vlm_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/phase1_robust_statistics"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260510)
    return parser.parse_args()


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def paired_t(diff: np.ndarray) -> tuple[float, float]:
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2:
        return float("nan"), float("nan")
    std = float(diff.std(ddof=1))
    if std < 1e-12:
        return float("inf"), 0.0
    t = float(diff.mean() / (std / math.sqrt(len(diff))))
    return t, float(2.0 * (1.0 - norm_cdf(abs(t))))


def ci(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def row_filter(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for key, value in spec.items():
        if key not in out.columns:
            return out.iloc[0:0]
        out = out[out[key] == value]
    return out


def paired_rows(df: pd.DataFrame, a: dict[str, Any], b: dict[str, Any], metric: str, unit_cols: list[str]) -> pd.DataFrame:
    aa = row_filter(df, a)
    bb = row_filter(df, b)
    if aa.empty or bb.empty or metric not in aa.columns or metric not in bb.columns:
        return pd.DataFrame()
    keep = unit_cols + [metric]
    return aa[keep].merge(bb[keep], on=unit_cols, suffixes=("_a", "_b"))


def bootstrap_mean(diff: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return float("nan"), float("nan")
    vals = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        vals[i] = rng.choice(diff, size=len(diff), replace=True).mean()
    return ci(vals)


def hierarchical_bootstrap(
    merged: pd.DataFrame,
    metric: str,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float]:
    if merged.empty:
        return float("nan"), float("nan")
    folds = sorted(merged["fold"].unique())
    vals = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sampled_diffs: list[float] = []
        sampled_folds = rng.choice(folds, size=len(folds), replace=True)
        for fold in sampled_folds:
            sub = merged[merged["fold"] == fold]
            if "seed" in sub.columns:
                seeds = sorted(sub["seed"].unique())
                seed = rng.choice(seeds)
                sub = sub[sub["seed"] == seed]
            sampled_diffs.extend((sub[f"{metric}_a"] - sub[f"{metric}_b"]).to_numpy(float).tolist())
        vals[i] = float(np.mean(sampled_diffs)) if sampled_diffs else float("nan")
    return ci(vals)


def comparison_specs() -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    return [
        (
            "gated_regraph_clip_vs_roi_mlp_clip",
            {"graph_encoder": "bnt_token_flat", "readout": "gated_flat", "lambda_clip": 2.0},
            {"graph_encoder": "roi_mlp", "readout": "flat", "lambda_clip": 2.0},
        ),
        (
            "gated_regraph_clip_vs_flat_regraph_clip",
            {"graph_encoder": "bnt_token_flat", "readout": "gated_flat", "lambda_clip": 2.0},
            {"graph_encoder": "bnt_token_flat", "readout": "flat", "lambda_clip": 2.0},
        ),
    ]


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No statistics available.\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.5f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def summary_statistics(df: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, a, b in comparison_specs():
        for metric in METRICS:
            if metric not in df.columns:
                continue
            run = paired_rows(df, a, b, metric, ["fold", "seed"])
            if not run.empty:
                diff = (run[f"{metric}_a"] - run[f"{metric}_b"]).to_numpy(float)
                t_run, p_run = paired_t(diff)
                lo_run, hi_run = bootstrap_mean(diff, rng, n_boot)
                lo_h, hi_h = hierarchical_bootstrap(run, metric, rng, n_boot)
                rows.append(
                    {
                        "comparison": name,
                        "metric": metric,
                        "unit": "fold_seed",
                        "n": int(np.isfinite(diff).sum()),
                        "mean_diff": float(np.nanmean(diff)),
                        "paired_t": t_run,
                        "p_norm_approx": p_run,
                        "bootstrap_ci_low": lo_run,
                        "bootstrap_ci_high": hi_run,
                        "hierarchical_ci_low": lo_h,
                        "hierarchical_ci_high": hi_h,
                    }
                )
            subject_df = df.groupby([c for c in ["graph_encoder", "readout", "lambda_clip", "fold"] if c in df.columns], dropna=False)[
                metric
            ].mean().reset_index()
            subject = paired_rows(subject_df, a, b, metric, ["fold"])
            if not subject.empty:
                diff = (subject[f"{metric}_a"] - subject[f"{metric}_b"]).to_numpy(float)
                t_subj, p_subj = paired_t(diff)
                lo_subj, hi_subj = bootstrap_mean(diff, rng, n_boot)
                rows.append(
                    {
                        "comparison": name,
                        "metric": metric,
                        "unit": "held_out_subject",
                        "n": int(np.isfinite(diff).sum()),
                        "mean_diff": float(np.nanmean(diff)),
                        "paired_t": t_subj,
                        "p_norm_approx": p_subj,
                        "bootstrap_ci_low": lo_subj,
                        "bootstrap_ci_high": hi_subj,
                        "hierarchical_ci_low": float("nan"),
                        "hierarchical_ci_high": float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def load_rank_rows(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for _, row in row_filter(df, spec).iterrows():
        metrics_path = row.get("metrics_path")
        if not isinstance(metrics_path, str) or not metrics_path:
            continue
        rank_path = Path(metrics_path).with_name("test_retrieval_ranks.csv")
        if not rank_path.exists():
            continue
        ranks = pd.read_csv(rank_path)
        ranks["fold"] = row.get("fold")
        ranks["seed"] = row.get("seed")
        rows.append(ranks)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def image_level_bootstrap(df: pd.DataFrame, rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, a, b in comparison_specs():
        aa = load_rank_rows(df, a)
        bb = load_rank_rows(df, b)
        if aa.empty or bb.empty:
            rows.append(
                {
                    "comparison": name,
                    "status": "missing_rank_files",
                    "note": "Rerun eval with --save-eval-details to create test_retrieval_ranks.csv.",
                }
            )
            continue
        key_cols = ["fold", "seed", "mode", "subject", "repeat_1", "repeat_2", "nsdId"]
        merged = aa.merge(bb, on=key_cols, suffixes=("_a", "_b"))
        for mode in sorted(merged["mode"].unique()):
            sub = merged[merged["mode"] == mode]
            for metric in ["hit5", "reciprocal_rank"]:
                diff_by_image = sub.groupby("nsdId").apply(lambda x: float((x[f"{metric}_a"] - x[f"{metric}_b"]).mean()))
                vals = diff_by_image.to_numpy(float)
                lo, hi = bootstrap_mean(vals, rng, n_boot)
                rows.append(
                    {
                        "comparison": name,
                        "status": "ok",
                        "mode": mode,
                        "metric": metric,
                        "n_images": int(len(vals)),
                        "mean_diff": float(np.nanmean(vals)),
                        "image_bootstrap_ci_low": lo,
                        "image_bootstrap_ci_high": hi,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    summary_path = root / args.summary_csv
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(summary_path)
    stats = summary_statistics(df, rng, args.n_bootstrap)
    image_stats = image_level_bootstrap(df, rng, args.n_bootstrap)
    stats.to_csv(out_dir / "phase1_subject_hierarchical_statistics.csv", index=False)
    image_stats.to_csv(out_dir / "phase1_image_level_bootstrap.csv", index=False)
    (out_dir / "phase1_subject_hierarchical_statistics.md").write_text(
        markdown_table(stats),
        encoding="utf-8",
    )
    manifest = {
        "summary_csv": str(summary_path),
        "n_rows": int(len(df)),
        "n_bootstrap": int(args.n_bootstrap),
        "outputs": [
            "phase1_subject_hierarchical_statistics.csv",
            "phase1_image_level_bootstrap.csv",
            "phase1_subject_hierarchical_statistics.md",
        ],
    }
    (out_dir / "phase1_statistics_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "n_stat_rows": len(stats), "n_image_rows": len(image_stats)}, indent=2))


if __name__ == "__main__":
    main()
