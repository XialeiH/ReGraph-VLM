#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect ReGraph-VLM metrics.json files under one or more result roots.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", required=True, help="Result directories relative to --root.")
    parser.add_argument("--out-csv", type=Path, required=True, help="Output CSV path relative to --root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    rows: list[dict[str, object]] = []
    for source in args.sources:
        source_path = root / source
        for path in sorted(source_path.glob("**/metrics.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            row["metrics_path"] = str(path)
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No metrics.json files found under {args.sources}")
    df = pd.DataFrame(rows)
    if "readout" not in df.columns:
        df["readout"] = pd.NA
    if "lambda_cross" not in df.columns:
        df["lambda_cross"] = 0.0
    if "lambda_subject_adv" not in df.columns:
        df["lambda_subject_adv"] = 0.0
    key_cols = [
        c
        for c in [
            "graph_encoder",
            "readout",
            "lambda_clip",
            "lambda_cross",
            "lambda_subject_adv",
            "adjacency_mode",
            "fold",
            "seed",
        ]
        if c in df.columns
    ]
    if key_cols:
        df = df.sort_values("metrics_path").drop_duplicates(key_cols, keep="last")
    sort_cols = [
        c
        for c in ["graph_encoder", "readout", "lambda_clip", "lambda_subject_adv", "adjacency_mode", "fold", "seed"]
        if c in df.columns
    ]
    if sort_cols:
        df = df.sort_values(sort_cols)
    out = root / args.out_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print({"out_csv": str(out), "n_rows": int(len(df))})


if __name__ == "__main__":
    main()
