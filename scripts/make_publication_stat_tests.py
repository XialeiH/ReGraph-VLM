#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


METRICS = ["AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "brain_R@5", "brain_MRR"]


@dataclass(frozen=True)
class ModelSpec:
    label: str
    graph_encoder: str | None = None
    readout: str | None = None
    lambda_clip: float | None = None
    lambda_subject_adv: float | None = None
    model: str | None = None


@dataclass(frozen=True)
class Comparison:
    setting: str
    path: str
    model_a: ModelSpec
    model_b: ModelSpec


@dataclass(frozen=True)
class CrossTableComparison:
    setting: str
    path_a: str
    path_b: str
    model_a: ModelSpec
    model_b: ModelSpec


COMPARISONS = [
    Comparison(
        setting="main_allfold",
        path="table_allfold_final.csv",
        model_a=ModelSpec("Gated ReGraph/BNT+CLIP", graph_encoder="bnt_token_flat", readout="gated_flat", lambda_clip=2.0),
        model_b=ModelSpec("ROI-MLP+CLIP", graph_encoder="roi_mlp", readout="flat", lambda_clip=2.0),
    ),
    Comparison(
        setting="main_allfold",
        path="table_allfold_final.csv",
        model_a=ModelSpec("Gated ReGraph/BNT+CLIP", graph_encoder="bnt_token_flat", readout="gated_flat", lambda_clip=2.0),
        model_b=ModelSpec("Flat ReGraph+CLIP", graph_encoder="bnt_token_flat", readout="flat", lambda_clip=2.0),
    ),
    Comparison(
        setting="hard_negative",
        path="table_hard_negative_allfold.csv",
        model_a=ModelSpec("Gated ReGraph/BNT+CLIP", graph_encoder="bnt_token_flat", readout="gated_flat", lambda_clip=2.0),
        model_b=ModelSpec("ROI-MLP+CLIP", graph_encoder="roi_mlp", readout="flat", lambda_clip=2.0),
    ),
    Comparison(
        setting="hard_negative",
        path="table_hard_negative_allfold.csv",
        model_a=ModelSpec("Gated ReGraph/BNT+CLIP", graph_encoder="bnt_token_flat", readout="gated_flat", lambda_clip=2.0),
        model_b=ModelSpec("Flat ReGraph+CLIP", graph_encoder="bnt_token_flat", readout="flat", lambda_clip=2.0),
    ),
    Comparison(
        setting="heldout_image_available_raw",
        path="table_heldout_image.csv",
        model_a=ModelSpec("Gated ReGraph/BNT+CLIP", graph_encoder="bnt_token_flat", readout="gated_flat", lambda_clip=2.0),
        model_b=ModelSpec("ROI-MLP+CLIP", graph_encoder="roi_mlp", readout="flat", lambda_clip=2.0),
    ),
    Comparison(
        setting="heldout_image_available_raw",
        path="table_heldout_image.csv",
        model_a=ModelSpec("Gated ReGraph/BNT+CLIP", graph_encoder="bnt_token_flat", readout="gated_flat", lambda_clip=2.0),
        model_b=ModelSpec("Flat ReGraph+CLIP", graph_encoder="bnt_token_flat", readout="flat", lambda_clip=2.0),
    ),
    Comparison(
        setting="heldout_real_vs_random_available_raw",
        path="table_heldout_image.csv",
        model_a=ModelSpec("Gated ReGraph/BNT+CLIP", graph_encoder="bnt_token_flat", readout="gated_flat", lambda_clip=2.0),
        model_b=ModelSpec("Gated random embedding", model="__heldout_random__"),
    ),
    Comparison(
        setting="single_ref_eval_existing",
        path="single_ref_matched_all_runs.csv",
        model_a=ModelSpec("Gated ReGraph/BNT+CLIP", model="Gated ReGraph/BNT+CLIP"),
        model_b=ModelSpec("ROI-MLP+CLIP", model="ROI-MLP+CLIP"),
    ),
    Comparison(
        setting="single_ref_eval_existing",
        path="single_ref_matched_all_runs.csv",
        model_a=ModelSpec("No-adj gated ROI Transformer+CLIP", model="No-adj gated ROI Transformer+CLIP"),
        model_b=ModelSpec("ROI-MLP+CLIP", model="ROI-MLP+CLIP"),
    ),
    Comparison(
        setting="single_ref_eval_existing",
        path="single_ref_matched_all_runs.csv",
        model_a=ModelSpec("Gated ReGraph/BNT+CLIP", model="Gated ReGraph/BNT+CLIP"),
        model_b=ModelSpec("No-adj gated ROI Transformer+CLIP", model="No-adj gated ROI Transformer+CLIP"),
    ),
    Comparison(
        setting="single_ref_retrained",
        path="single_ref_matched_allseed_all_runs.csv",
        model_a=ModelSpec("Gated ReGraph/BNT+CLIP", model="Gated ReGraph/BNT+CLIP"),
        model_b=ModelSpec("ROI-MLP+CLIP", model="ROI-MLP+CLIP"),
    ),
    Comparison(
        setting="single_ref_retrained",
        path="single_ref_matched_allseed_all_runs.csv",
        model_a=ModelSpec("No-adj gated ROI Transformer+CLIP", model="No-adj gated ROI Transformer+CLIP"),
        model_b=ModelSpec("ROI-MLP+CLIP", model="ROI-MLP+CLIP"),
    ),
    Comparison(
        setting="single_ref_retrained",
        path="single_ref_matched_allseed_all_runs.csv",
        model_a=ModelSpec("Gated ReGraph/BNT+CLIP", model="Gated ReGraph/BNT+CLIP"),
        model_b=ModelSpec("No-adj gated ROI Transformer+CLIP", model="No-adj gated ROI Transformer+CLIP"),
    ),
]

CROSS_TABLE_COMPARISONS = [
    CrossTableComparison(
        setting="component_baselines",
        path_a="table_allfold_final.csv",
        path_b="table_phase2_sota_graph_baselines.csv",
        model_a=ModelSpec("Gated ReGraph/BNT+CLIP", graph_encoder="bnt_token_flat", readout="gated_flat", lambda_clip=2.0),
        model_b=ModelSpec("MindEye2-style shared ROI mapper", graph_encoder="mindeye2_shared", readout="flat", lambda_clip=2.0),
    ),
    CrossTableComparison(
        setting="component_baselines",
        path_a="table_allfold_final.csv",
        path_b="table_phase2_sota_graph_baselines.csv",
        model_a=ModelSpec("Gated ReGraph/BNT+CLIP", graph_encoder="bnt_token_flat", readout="gated_flat", lambda_clip=2.0),
        model_b=ModelSpec("UMBRAE-style subject encoder", graph_encoder="umbrae_subject", readout="flat", lambda_clip=2.0),
    ),
    CrossTableComparison(
        setting="component_baselines",
        path_a="table_allfold_final.csv",
        path_b="table_phase2_sota_graph_baselines.csv",
        model_a=ModelSpec("Gated ReGraph/BNT+CLIP", graph_encoder="bnt_token_flat", readout="gated_flat", lambda_clip=2.0),
        model_b=ModelSpec(
            "MindLink-style subject-adversarial ROI-MLP",
            graph_encoder="roi_mlp",
            readout="flat",
            lambda_clip=2.0,
            lambda_subject_adv=0.1,
        ),
    ),
    CrossTableComparison(
        setting="component_baselines",
        path_a="table_allfold_final.csv",
        path_b="table_phase2_sota_graph_baselines.csv",
        model_a=ModelSpec("Gated ReGraph/BNT+CLIP", graph_encoder="bnt_token_flat", readout="gated_flat", lambda_clip=2.0),
        model_b=ModelSpec(
            "MindLink-style subject-adversarial ReGraph",
            graph_encoder="graph_bnt",
            readout="gated_flat",
            lambda_clip=2.0,
            lambda_subject_adv=0.1,
        ),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create standardized publication paired statistical tests.")
    parser.add_argument("--final-tables-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="publication_paired_stats")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def filter_model(df: pd.DataFrame, spec: ModelSpec, random_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if spec.model == "__heldout_random__":
        if random_df is None:
            return pd.DataFrame()
        return random_df.copy()
    out = df.copy()
    if spec.model is not None and "model" in out.columns:
        out = out[out["model"] == spec.model]
    if spec.graph_encoder is not None and "graph_encoder" in out.columns:
        out = out[out["graph_encoder"] == spec.graph_encoder]
    if spec.readout is not None and "readout" in out.columns:
        out = out[out["readout"] == spec.readout]
    if spec.lambda_clip is not None and "lambda_clip" in out.columns:
        out = out[np.isclose(out["lambda_clip"].astype(float), spec.lambda_clip)]
    if spec.lambda_subject_adv is not None and "lambda_subject_adv" in out.columns:
        out = out[np.isclose(out["lambda_subject_adv"].astype(float), spec.lambda_subject_adv)]
    return out.copy()


def bootstrap_ci(diffs: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float]:
    if diffs.size == 0:
        return math.nan, math.nan
    indices = rng.integers(0, diffs.size, size=(n_bootstrap, diffs.size))
    means = diffs[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_rows(
    *,
    setting: str,
    model_a_label: str,
    model_b_label: str,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if df_a.empty or df_b.empty:
        return []
    keys = ["fold", "seed"]
    merged = df_a[keys + [metric for metric in METRICS if metric in df_a.columns]].merge(
        df_b[keys + [metric for metric in METRICS if metric in df_b.columns]],
        on=keys,
        suffixes=("_a", "_b"),
    )
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        left = f"{metric}_a"
        right = f"{metric}_b"
        if left not in merged.columns or right not in merged.columns:
            continue
        diffs = (merged[left] - merged[right]).astype(float).to_numpy()
        diffs = diffs[np.isfinite(diffs)]
        n = int(diffs.size)
        if n == 0:
            continue
        mean_diff = float(diffs.mean())
        std_diff = float(diffs.std(ddof=1)) if n > 1 else math.nan
        ci_low, ci_high = bootstrap_ci(diffs, n_bootstrap, rng)
        if stats is not None and n > 1 and np.isfinite(std_diff) and std_diff > 0:
            p_value = float(stats.ttest_1samp(diffs, popmean=0.0).pvalue)
        else:
            p_value = math.nan
        rows.append(
            {
                "setting": setting,
                "comparison": f"{model_a_label} - {model_b_label}",
                "metric": metric,
                "n": n,
                "mean_diff": mean_diff,
                "std_diff": std_diff,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "paired_t_p": p_value,
            }
        )
    return rows


def load_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def format_float(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.3g}" if abs(value) < 0.001 and value != 0 else f"{value:.4f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Publication paired statistical tests",
        "",
        "All comparisons are paired by held-out fold and seed. Positive differences mean the first model in the comparison is better. Confidence intervals are bootstrap percentile intervals over paired fold-by-seed differences.",
        "",
    ]
    if not rows:
        lines.append("No paired rows were available.")
    else:
        columns = ["setting", "comparison", "metric", "n", "mean_diff", "bootstrap_ci_low", "bootstrap_ci_high", "paired_t_p"]
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(format_float(row[column]) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    random_df = load_optional(args.final_tables_dir / "table_heldout_image_random.csv")
    all_rows: list[dict[str, Any]] = []
    for comparison in COMPARISONS:
        table_path = args.final_tables_dir / comparison.path
        if not table_path.exists():
            continue
        df = pd.read_csv(table_path)
        df_a = filter_model(df, comparison.model_a, random_df=random_df)
        df_b = filter_model(df, comparison.model_b, random_df=random_df)
        all_rows.extend(
            paired_rows(
                setting=comparison.setting,
                model_a_label=comparison.model_a.label,
                model_b_label=comparison.model_b.label,
                df_a=df_a,
                df_b=df_b,
                n_bootstrap=args.n_bootstrap,
                rng=rng,
            )
        )
    for comparison in CROSS_TABLE_COMPARISONS:
        table_a = args.final_tables_dir / comparison.path_a
        table_b = args.final_tables_dir / comparison.path_b
        if not table_a.exists() or not table_b.exists():
            continue
        df_a = filter_model(pd.read_csv(table_a), comparison.model_a, random_df=random_df)
        df_b = filter_model(pd.read_csv(table_b), comparison.model_b, random_df=random_df)
        all_rows.extend(
            paired_rows(
                setting=comparison.setting,
                model_a_label=comparison.model_a.label,
                model_b_label=comparison.model_b.label,
                df_a=df_a,
                df_b=df_b,
                n_bootstrap=args.n_bootstrap,
                rng=rng,
            )
        )
    write_csv(args.final_tables_dir / f"{args.output_prefix}.csv", all_rows)
    write_markdown(args.final_tables_dir / f"{args.output_prefix}.md", all_rows)
    print((args.final_tables_dir / f"{args.output_prefix}.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
