#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None


METRICS = ["test_AUROC", "test_AUPRC", "test_R@5", "test_MRR"]
MODELS = {
    "roi_mlp": "ROI-MLP",
    "roi_transformer_gated": "Gated ROI Transformer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CNeuroMod-THINGS HCP-MMP external validation runs.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("sub-*_sub-*/*/seed*/summary.csv")):
        row = pd.read_csv(path).iloc[0].to_dict()
        row["pair"] = path.parents[2].name
        row["summary_path"] = str(path)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No summary.csv files found under {root}")
    return pd.DataFrame(rows)


def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline = "roi_mlp"
    candidate = "roi_transformer_gated"
    for metric in METRICS:
        wide = df.pivot_table(index=["pair", "seed"], columns="model", values=metric).dropna(
            subset=[baseline, candidate]
        )
        if wide.empty:
            continue
        diff = wide[candidate] - wide[baseline]
        n = int(diff.shape[0])
        mean_diff = float(diff.mean())
        std_diff = float(diff.std(ddof=1)) if n > 1 else float("nan")
        sem = std_diff / math.sqrt(n) if n > 1 else float("nan")
        if stats is not None and n > 1:
            p_value = float(stats.ttest_rel(wide[candidate], wide[baseline]).pvalue)
            ci_delta = float(stats.t.ppf(0.975, n - 1) * sem)
        elif n > 1 and sem > 0:
            z = abs(mean_diff / sem)
            p_value = float(math.erfc(z / math.sqrt(2.0)))
            ci_delta = float(1.96 * sem)
        else:
            p_value = float("nan")
            ci_delta = float("nan")
        rows.append(
            {
                "comparison": "Gated ROI Transformer - ROI-MLP",
                "metric": metric,
                "n": n,
                "mean_diff": mean_diff,
                "std_diff": std_diff,
                "ci95_low": mean_diff - ci_delta,
                "ci95_high": mean_diff + ci_delta,
                "paired_p": p_value,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_runs = collect(args.root)
    summary = all_runs.groupby("model")[METRICS].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns]
    summary.insert(1, "model_name", summary["model"].map(MODELS).fillna(summary["model"]))

    by_pair = all_runs.groupby(["pair", "model"])[METRICS].agg(["mean", "std", "count"]).reset_index()
    by_pair.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in by_pair.columns]
    tests = paired_tests(all_runs)

    all_runs.to_csv(args.out_dir / "cneuromod_hcp_mmp180_all_runs.csv", index=False)
    summary.to_csv(args.out_dir / "cneuromod_hcp_mmp180_summary.csv", index=False)
    by_pair.to_csv(args.out_dir / "cneuromod_hcp_mmp180_by_pair.csv", index=False)
    tests.to_csv(args.out_dir / "cneuromod_hcp_mmp180_pairwise_tests.csv", index=False)

    lines = [
        "# CNeuroMod-THINGS HCP-MMP180 External Validation",
        "",
        "Values are mean/std/count over subject-pair x seed runs.",
        "",
        "## Overall",
        "",
        markdown_table(summary),
        "",
        "## Paired Tests",
        "",
        markdown_table(tests) if len(tests) else "No paired tests available.",
        "",
    ]
    (args.out_dir / "cneuromod_hcp_mmp180_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print({"all_runs": int(len(all_runs)), "summary": str(args.out_dir / "cneuromod_hcp_mmp180_summary.csv")})


if __name__ == "__main__":
    main()
