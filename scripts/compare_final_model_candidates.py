#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy may be absent on some systems.
    stats = None


METRICS = ["AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "image_MRR", "brain_R@5", "brain_MRR"]
KEYS = ["fold", "seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired comparison of final model candidates.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_model_comparison"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260519)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "readout" not in df.columns:
        df["readout"] = ""
    if "lambda_clip" not in df.columns:
        df["lambda_clip"] = np.nan
    return df


def filter_model(df: pd.DataFrame, encoder: str, readout: str | None = None, lambda_clip: float = 2.0) -> pd.DataFrame:
    out = df[df["graph_encoder"].eq(encoder)].copy()
    if readout is not None:
        out = out[out["readout"].eq(readout)]
    if "lambda_clip" in out.columns and lambda_clip is not None:
        out = out[np.isclose(out["lambda_clip"].astype(float), float(lambda_clip))]
    if "metrics_path" in out.columns:
        out = out.sort_values("metrics_path")
    return out.drop_duplicates(KEYS, keep="last")


def paired_table(
    out_dir: Path,
    rng: np.random.Generator,
    n_bootstrap: int,
    name: str,
    a_name: str,
    a: pd.DataFrame,
    b_name: str,
    b: pd.DataFrame,
) -> pd.DataFrame:
    aa = a[KEYS + METRICS].copy().rename(columns={metric: f"{metric}__a" for metric in METRICS})
    bb = b[KEYS + METRICS].copy().rename(columns={metric: f"{metric}__b" for metric in METRICS})
    merged = aa.merge(bb, on=KEYS, how="inner")
    merged["comparison"] = name
    merged["a_model"] = a_name
    merged["b_model"] = b_name
    merged.to_csv(out_dir / f"{name}_paired_units.csv", index=False)

    rows: list[dict[str, object]] = []
    for metric in METRICS:
        avals = merged[f"{metric}__a"].to_numpy(float)
        bvals = merged[f"{metric}__b"].to_numpy(float)
        diff = avals - bvals
        n = int(len(diff))
        if n == 0:
            continue
        boot = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, n)
            boot.append(float(diff[idx].mean()))
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        t_stat = p_t = np.nan
        w_stat = p_w = np.nan
        if stats is not None and n > 1:
            ttest = stats.ttest_rel(avals, bvals, nan_policy="omit")
            t_stat = float(ttest.statistic)
            p_t = float(ttest.pvalue)
            try:
                wilcox = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                w_stat = float(wilcox.statistic)
                p_w = float(wilcox.pvalue)
            except Exception:
                pass
        diff_std = float(np.std(diff, ddof=1)) if n > 1 else np.nan
        rows.append(
            {
                "comparison": name,
                "a_model": a_name,
                "b_model": b_name,
                "metric": metric,
                "n_pairs": n,
                "a_mean": float(np.mean(avals)),
                "b_mean": float(np.mean(bvals)),
                "mean_diff_a_minus_b": float(np.mean(diff)),
                "diff_std": diff_std,
                "diff_sem": float(diff_std / math.sqrt(n)) if n > 1 else np.nan,
                "bootstrap_ci_low": float(ci_low),
                "bootstrap_ci_high": float(ci_high),
                "paired_t": t_stat,
                "paired_t_p": p_t,
                "wilcoxon_stat": w_stat,
                "wilcoxon_p": p_w,
                "n_a_gt_b": int((diff > 0).sum()),
                "n_a_lt_b": int((diff < 0).sum()),
                "n_equal": int((diff == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def add_summary_row(name: str, df: pd.DataFrame) -> dict[str, object]:
    row: dict[str, object] = {"model_setting": name, "n": int(len(df))}
    for metric in METRICS:
        row[f"{metric}_mean"] = float(df[metric].mean())
        row[f"{metric}_std"] = float(df[metric].std()) if len(df) > 1 else np.nan
    return row


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    results = root / "preproc_v0/repetition_familiarity/results"
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    allfold = read_csv(results / "cross_subject_allfold_final/regraph_vlm_summary.csv")
    noadj = read_csv(results / "phase3c_noadj_gated_final/regraph_vlm_summary.csv")
    hard = read_csv(results / "cross_subject_hardneg_allfold_seed11/regraph_vlm_summary.csv")
    noadj_hard = read_csv(results / "phase3c_noadj_gated_hardneg/regraph_vlm_summary.csv")
    held = read_csv(results / "heldout_image/regraph_vlm_summary.csv")
    noadj_held = read_csv(results / "phase3c_noadj_gated_heldout/regraph_vlm_summary.csv")

    bnt_gated = filter_model(allfold, "bnt_token_flat", "gated_flat")
    roi = filter_model(allfold, "roi_mlp", "flat")
    noadj_gated = filter_model(noadj, "roi_transformer_noadj", "gated_flat")

    hard_bnt = filter_model(hard, "bnt_token_flat", "gated_flat")
    hard_roi = filter_model(hard, "roi_mlp", "flat")
    hard_noadj = filter_model(noadj_hard, "roi_transformer_noadj", "gated_flat")
    hard_bnt = hard_bnt[hard_bnt["seed"].eq(11)].drop_duplicates(KEYS, keep="last")
    hard_roi = hard_roi[hard_roi["seed"].eq(11)].drop_duplicates(KEYS, keep="last")

    held_bnt = filter_model(held, "bnt_token_flat", "gated_flat")
    held_roi = filter_model(held, "roi_mlp", "flat")
    held_noadj = filter_model(noadj_held, "roi_transformer_noadj", "gated_flat")
    held_keys = held_noadj[KEYS].drop_duplicates()
    held_bnt = held_bnt.merge(held_keys, on=KEYS, how="inner").drop_duplicates(KEYS, keep="last")
    held_roi = held_roi.merge(held_keys, on=KEYS, how="inner").drop_duplicates(KEYS, keep="last")

    comparisons = [
        paired_table(
            out_dir,
            rng,
            args.n_bootstrap,
            "main_noadj_gated_vs_gated_regraph",
            "ROI Transformer no-adj + gated",
            noadj_gated,
            "Gated ReGraph/BNT + CLIP",
            bnt_gated,
        ),
        paired_table(
            out_dir,
            rng,
            args.n_bootstrap,
            "main_noadj_gated_vs_roi_mlp",
            "ROI Transformer no-adj + gated",
            noadj_gated,
            "ROI-MLP + CLIP",
            roi,
        ),
        paired_table(
            out_dir,
            rng,
            args.n_bootstrap,
            "hardneg_noadj_gated_vs_gated_regraph_seed11",
            "ROI Transformer no-adj + gated",
            hard_noadj,
            "Gated ReGraph/BNT + CLIP",
            hard_bnt,
        ),
        paired_table(
            out_dir,
            rng,
            args.n_bootstrap,
            "hardneg_noadj_gated_vs_roi_mlp_seed11",
            "ROI Transformer no-adj + gated",
            hard_noadj,
            "ROI-MLP + CLIP",
            hard_roi,
        ),
        paired_table(
            out_dir,
            rng,
            args.n_bootstrap,
            "heldout_noadj_gated_vs_gated_regraph",
            "ROI Transformer no-adj + gated",
            held_noadj,
            "Gated ReGraph/BNT + CLIP",
            held_bnt,
        ),
        paired_table(
            out_dir,
            rng,
            args.n_bootstrap,
            "heldout_noadj_gated_vs_roi_mlp",
            "ROI Transformer no-adj + gated",
            held_noadj,
            "ROI-MLP + CLIP",
            held_roi,
        ),
    ]
    stats_df = pd.concat(comparisons, ignore_index=True)
    stats_df.to_csv(out_dir / "paired_model_candidate_tests.csv", index=False)

    summary = pd.DataFrame(
        [
            add_summary_row("main_gated_regraph", bnt_gated),
            add_summary_row("main_noadj_gated", noadj_gated),
            add_summary_row("main_roi_mlp", roi),
            add_summary_row("hardneg_gated_regraph_seed11", hard_bnt),
            add_summary_row("hardneg_noadj_gated_seed11", hard_noadj),
            add_summary_row("hardneg_roi_mlp_seed11", hard_roi),
            add_summary_row("heldout_gated_regraph", held_bnt),
            add_summary_row("heldout_noadj_gated", held_noadj),
            add_summary_row("heldout_roi_mlp", held_roi),
        ]
    )
    summary.to_csv(out_dir / "model_candidate_metric_summary.csv", index=False)

    selected = stats_df[
        stats_df["metric"].isin(["AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "brain_R@5", "brain_MRR"])
    ]
    md = [
        "# Final Model Candidate Paired Comparison",
        "",
        "Positive differences mean the ROI Transformer no-adjacency gated model is better than the comparison model.",
        "",
        "## Metric Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Paired Tests",
        "",
        selected[
            [
                "comparison",
                "metric",
                "n_pairs",
                "a_mean",
                "b_mean",
                "mean_diff_a_minus_b",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "paired_t_p",
                "wilcoxon_p",
                "n_a_gt_b",
                "n_a_lt_b",
            ]
        ].to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Readout",
        "",
        "- Main all-fold: no-adj gated ROI transformer is statistically indistinguishable from gated ReGraph/BNT; CIs are small and centered near zero.",
        "- Against ROI-MLP, no-adj gated improves AUPRC/retrieval more clearly than AUROC.",
        "- Hard-negative: gated ReGraph/BNT remains slightly stronger than no-adj gated on most retrieval metrics.",
        "- Held-out image: no-adj gated has higher AUROC/AUPRC but weaker image/brain retrieval than gated ReGraph/BNT in the matched fold01/fold04 comparison.",
        "- Conclusion: explicit fixed adjacency is not the main source of the all-fold gain, but the BNT/ReGraph variant may retain a small advantage for harder semantic/retrieval settings.",
    ]
    (out_dir / "paired_model_candidate_tests.md").write_text("\n".join(md), encoding="utf-8")

    print({"out_dir": str(out_dir), "n_tests": int(len(stats_df))})
    print(summary.to_string(index=False))
    focus = stats_df[stats_df["metric"].isin(["AUROC", "R@5", "brain_R@5"])][
        [
            "comparison",
            "metric",
            "n_pairs",
            "a_mean",
            "b_mean",
            "mean_diff_a_minus_b",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "paired_t_p",
            "wilcoxon_p",
        ]
    ]
    print(focus.to_string(index=False))


if __name__ == "__main__":
    main()
