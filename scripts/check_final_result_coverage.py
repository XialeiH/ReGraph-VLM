#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit final result coverage for ReGraph-VLM.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("preproc_v0/repetition_familiarity/results/final_audit"))
    return parser.parse_args()


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def model_count(df: pd.DataFrame, graph_encoder: str, readout: str, lambda_clip: float) -> int:
    if df.empty:
        return 0
    x = df[
        (df.get("graph_encoder") == graph_encoder)
        & (df.get("readout") == readout)
        & (df.get("lambda_clip").astype(float) == float(lambda_clip))
    ]
    return int(len(x))


def status(n: int, expected: int) -> str:
    if n >= expected:
        return "ok"
    if n > 0:
        return "partial"
    return "missing"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    results = root / "preproc_v0/repetition_familiarity/results"
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    allfold = read(results / "cross_subject_allfold_final/regraph_vlm_summary.csv")
    held = read(results / "heldout_image/regraph_vlm_summary.csv")
    held_rand = read(results / "heldout_image_random_embedding/regraph_vlm_summary.csv")
    hard = read(results / "cross_subject_hardneg_allfold_seed11/regraph_vlm_summary.csv")

    checks = [
        {
            "section": "allfold_main",
            "model": "ROI-MLP+CLIP",
            "observed": model_count(allfold, "roi_mlp", "flat", 2.0),
            "expected": 24,
        },
        {
            "section": "allfold_main",
            "model": "Flat ReGraph+CLIP",
            "observed": model_count(allfold, "bnt_token_flat", "flat", 2.0),
            "expected": 24,
        },
        {
            "section": "allfold_main",
            "model": "Gated ReGraph+CLIP",
            "observed": model_count(allfold, "bnt_token_flat", "gated_flat", 2.0),
            "expected": 24,
        },
        {
            "section": "allfold_graph_only",
            "model": "Flat ReGraph graph-only",
            "observed": model_count(allfold, "bnt_token_flat", "flat", 0.0),
            "expected": 8,
        },
        {
            "section": "heldout_image",
            "model": "ROI-MLP+CLIP",
            "observed": model_count(held, "roi_mlp", "flat", 2.0),
            "expected": 6,
        },
        {
            "section": "heldout_image",
            "model": "Flat ReGraph+CLIP",
            "observed": model_count(held, "bnt_token_flat", "flat", 2.0),
            "expected": 6,
        },
        {
            "section": "heldout_image",
            "model": "Gated ReGraph+CLIP",
            "observed": model_count(held, "bnt_token_flat", "gated_flat", 2.0),
            "expected": 6,
        },
        {
            "section": "heldout_image",
            "model": "Gated random embedding",
            "observed": model_count(held_rand, "bnt_token_flat", "gated_flat", 2.0),
            "expected": 6,
        },
        {
            "section": "hard_negative_allfold",
            "model": "ROI-MLP+CLIP",
            "observed": model_count(hard, "roi_mlp", "flat", 2.0),
            "expected": 8,
        },
        {
            "section": "hard_negative_allfold",
            "model": "Gated ReGraph+CLIP",
            "observed": model_count(hard, "bnt_token_flat", "gated_flat", 2.0),
            "expected": 8,
        },
    ]
    for row in checks:
        row["status"] = status(int(row["observed"]), int(row["expected"]))
    summary = {
        "status": "ok" if all(row["status"] == "ok" for row in checks) else "partial",
        "checks": checks,
    }
    (out_dir / "final_result_coverage.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    df = pd.DataFrame(checks)
    df.to_csv(out_dir / "final_result_coverage.csv", index=False)
    md = ["# Final Result Coverage", "", df.to_markdown(index=False)]
    (out_dir / "final_result_coverage.md").write_text("\n".join(md), encoding="utf-8")
    print({"out_dir": str(out_dir), "status": summary["status"]})


if __name__ == "__main__":
    main()
