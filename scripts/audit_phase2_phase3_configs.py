#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Phase 2/3 configuration differences.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/phase_audit"),
    )
    return parser.parse_args()


def load_configs(source: Path, phase: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(source.glob("**/run_config.json")):
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        cfg["phase"] = phase
        cfg["config_path"] = str(path)
        cfg["run_dir"] = str(path.parent)
        rows.append(cfg)
    return rows


def compact_name(row: pd.Series) -> str:
    parts = [
        str(row.get("phase", "")),
        str(row.get("graph_encoder", "")),
        str(row.get("readout", "")),
        f"adj={row.get('adjacency_mode', 'default')}",
        f"clip={row.get('lambda_clip', 0)}",
        f"adv={row.get('lambda_subject_adv', 0)}",
    ]
    return " | ".join(parts)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    results = root / "preproc_v0/repetition_familiarity/results"
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(load_configs(results / "phase2_sota_graph_baselines", "phase2"))
    rows.extend(load_configs(results / "phase3_graph_ablation", "phase3_1"))
    rows.extend(load_configs(results / "cross_subject_allfold_final", "final_allfold"))

    if not rows:
        raise FileNotFoundError("No run_config.json files found for audit.")

    df = pd.DataFrame(rows)
    if "adjacency_mode" not in df.columns:
        df["adjacency_mode"] = "default"
    df["config_label"] = df.apply(compact_name, axis=1)

    key_cols = [
        "phase",
        "graph_encoder",
        "readout",
        "adjacency_mode",
        "lambda_clip",
        "lambda_cross",
        "lambda_subject_adv",
        "roi_id_mode",
        "hidden_dim",
        "embedding_dim",
        "num_heads",
        "num_layers",
        "dropout",
        "batch_size",
        "lr",
        "weight_decay",
        "epochs",
        "patience",
        "seed",
        "fold",
        "dataset_root",
        "output_root",
        "config_path",
    ]
    key_cols = [c for c in key_cols if c in df.columns]
    detail = df[key_cols].copy()
    detail.to_csv(out_dir / "phase2_phase3_config_diff.csv", index=False)

    summary_cols = [
        "phase",
        "graph_encoder",
        "readout",
        "adjacency_mode",
        "lambda_clip",
        "lambda_subject_adv",
        "hidden_dim",
        "embedding_dim",
        "num_heads",
        "num_layers",
        "dropout",
        "dataset_root",
    ]
    summary_cols = [c for c in summary_cols if c in df.columns]
    summary = df.groupby(summary_cols, dropna=False).size().reset_index(name="n_runs")
    summary.to_csv(out_dir / "phase2_phase3_config_summary.csv", index=False)

    md = ["# Phase 2 / Phase 3 Configuration Audit", ""]
    md.append("## Unique Configurations")
    md.append("")
    md.append(summary.to_markdown(index=False))
    md.append("")
    md.append("## Interpretation Notes")
    md.append("")
    md.append("- Phase 2 `roi_transformer_noadj` disables graph bias at the model level.")
    md.append("- Phase 3.1 `no_adjacency` used `graph_bnt` with graph bias enabled but supplied a zero adjacency matrix.")
    md.append("- These are not equivalent code paths; a clean unified ablation is required before concluding whether adjacency helps.")
    md.append("- The Phase 3b clean ablation should compare all variants through one training script and output root.")
    (out_dir / "phase2_phase3_config_diff.md").write_text("\n".join(md), encoding="utf-8")

    print({"out_dir": str(out_dir), "n_configs": int(len(df)), "n_unique": int(len(summary))})


if __name__ == "__main__":
    main()
