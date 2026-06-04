#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd


METRICS = ["AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "brain_R@5"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize repeat-dynamic LNN/GRU VLM experiments.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--dynamic-source",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/repeat_dynamic_lnn"),
    )
    parser.add_argument(
        "--static-sources",
        nargs="*",
        default=[
            "preproc_v0/repetition_familiarity/results/phase3c_noadj_gated_final",
            "preproc_v0/repetition_familiarity/results/phase3b_clean_graph_ablation",
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/repeat_dynamic_lnn_summary"),
    )
    return parser.parse_args()


def read_metrics(root: Path, source: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted((root / source).glob("**/metrics.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["metrics_path"] = str(path)
        rows.append(row)
    return pd.DataFrame(rows)


def model_label(row: pd.Series) -> str:
    temporal = row.get("temporal_module")
    if isinstance(temporal, str) and temporal:
        return f"{row.get('base_encoder', row.get('graph_encoder'))}+{temporal}"
    return f"{row.get('graph_encoder')}+{row.get('readout')}"


def mean_std(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["model_label"] = df.apply(model_label, axis=1)
    rows: list[dict[str, object]] = []
    for label, group in df.groupby("model_label", sort=True):
        row: dict[str, object] = {"model_label": label, "n": int(len(group))}
        for metric in METRICS:
            if metric in group.columns:
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def paired_vs_static(dynamic: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    if dynamic.empty or static.empty:
        return pd.DataFrame()
    static = static.copy()
    if "adjacency_mode" not in static.columns:
        static["adjacency_mode"] = "no_adjacency"
    static = static[
        (static["graph_encoder"] == "roi_transformer_noadj")
        & (static["readout"] == "gated_flat")
        & (static["adjacency_mode"].fillna("no_adjacency").isin(["default", "no_adjacency"]))
    ]
    if static.empty:
        return pd.DataFrame()
    static = static.sort_values("metrics_path").drop_duplicates(["fold", "seed"], keep="last")
    rows: list[dict[str, object]] = []
    dynamic = dynamic.copy()
    dynamic["model_label"] = dynamic.apply(model_label, axis=1)
    for label, group in dynamic.groupby("model_label", sort=True):
        merged = group.merge(static, on=["fold", "seed"], suffixes=("_dynamic", "_static"))
        row: dict[str, object] = {"model_label": label, "n_paired": int(len(merged))}
        for metric in METRICS:
            dyn_col = f"{metric}_dynamic"
            sta_col = f"{metric}_static"
            if dyn_col not in merged.columns or sta_col not in merged.columns or merged.empty:
                continue
            diff = merged[dyn_col] - merged[sta_col]
            row[f"{metric}_diff_mean"] = float(diff.mean())
            row[f"{metric}_diff_std"] = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dynamic = read_metrics(root, args.dynamic_source)
    if dynamic.empty:
        raise FileNotFoundError(f"No dynamic metrics found under {root / args.dynamic_source}")
    static_parts = [read_metrics(root, Path(source)) for source in args.static_sources]
    static = pd.concat([part for part in static_parts if not part.empty], ignore_index=True) if static_parts else pd.DataFrame()
    dynamic.to_csv(out_dir / "repeat_dynamic_lnn_all_metrics.csv", index=False)
    mean_std(dynamic).to_csv(out_dir / "repeat_dynamic_lnn_mean_std.csv", index=False)
    paired = paired_vs_static(dynamic, static)
    if not paired.empty:
        paired.to_csv(out_dir / "repeat_dynamic_lnn_paired_vs_static.csv", index=False)
    print({"dynamic_rows": int(len(dynamic)), "static_rows": int(len(static)), "output_dir": str(out_dir)})


if __name__ == "__main__":
    main()
