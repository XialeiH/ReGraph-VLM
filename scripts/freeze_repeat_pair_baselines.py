#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_RESULTS_ROOT = Path("preproc_v0/repetition_familiarity/results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze repeat-pair baseline summaries for ReGraph-VLM v0 comparison.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    return parser.parse_args()


def model_label(row: pd.Series) -> str:
    model = str(row.get("model", ""))
    loss = str(row.get("loss_mode", ""))
    readout = str(row.get("readout", ""))
    roi_id = str(row.get("roi_id", ""))
    adjacency = str(row.get("adjacency", ""))
    if model == "roi_mlp":
        return f"ROI MLP + {loss.upper()}"
    if model == "gcn":
        return f"GCN ({adjacency})"
    if model == "gcn_roiid_flat":
        return f"GCN + ROI ID + flat ({adjacency})"
    if model == "gat_roiid_flat":
        return f"GAT + ROI ID + flat ({adjacency})"
    if model == "bnt_token":
        return f"BNT-token + {readout} + ROI {roi_id} + {loss.upper()}"
    if model.startswith("bnt_native"):
        return f"{model} + {readout} + {loss.upper()}"
    if model == "token_mlp":
        return f"Token MLP + {readout} + ROI {roi_id} + {loss.upper()}"
    return f"{model} + {loss}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for value in row.tolist():
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    results_root = (root / args.results_root).resolve()
    encoder_summary = results_root / "repeat_pair_encoder_results" / "repeat_pair_encoder_summary.csv"
    similarity_summary = results_root / "repeat_pair_similarity_baseline" / "repeat_pair_similarity_baseline_summary.csv"
    output_dir = results_root / "frozen_baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if similarity_summary.exists():
        sim = pd.read_csv(similarity_summary)
        method_col = "method" if "method" in sim.columns else "similarity"
        if method_col not in sim.columns:
            raise ValueError(f"{similarity_summary} has neither `method` nor `similarity` column")
        sim = sim[sim.get("split", "test") == "test"].copy() if "split" in sim.columns else sim
        for method, group in sim.groupby(method_col, dropna=False):
            rows.append(
                {
                    "baseline": f"raw {method}",
                    "source": "raw_similarity",
                    "n_runs": int(len(group)),
                    "auroc_mean": float(group["auroc"].mean()),
                    "auroc_std": float(group["auroc"].std(ddof=1)) if len(group) > 1 else 0.0,
                    "auprc_mean": float(group["auprc"].mean()) if "auprc" in group else float("nan"),
                    "recall_at_1_mean": float(group["recall_at_1"].mean()) if "recall_at_1" in group else float("nan"),
                    "recall_at_5_mean": float(group["recall_at_5"].mean()) if "recall_at_5" in group else float("nan"),
                    "mrr_mean": float(group["mrr"].mean()) if "mrr" in group else float("nan"),
                }
            )

    if encoder_summary.exists():
        enc = pd.read_csv(encoder_summary)
        enc["baseline"] = enc.apply(model_label, axis=1)
        for baseline, group in enc.groupby("baseline", dropna=False):
            rows.append(
                {
                    "baseline": str(baseline),
                    "source": "learned_encoder",
                    "n_runs": int(len(group)),
                    "auroc_mean": float(group["auroc"].mean()),
                    "auroc_std": float(group["auroc"].std(ddof=1)) if len(group) > 1 else 0.0,
                    "auprc_mean": float(group["auprc"].mean()) if "auprc" in group else float("nan"),
                    "recall_at_1_mean": float(group["recall_at_1"].mean()) if "recall_at_1" in group else float("nan"),
                    "recall_at_5_mean": float(group["recall_at_5"].mean()) if "recall_at_5" in group else float("nan"),
                    "mrr_mean": float(group["mrr"].mean()) if "mrr" in group else float("nan"),
                }
            )

    if not rows:
        raise FileNotFoundError(f"No baseline summaries found under {results_root}")

    out = pd.DataFrame(rows).sort_values(["source", "auroc_mean"], ascending=[True, False]).reset_index(drop=True)
    csv_path = output_dir / "frozen_repeat_pair_baselines.csv"
    md_path = output_dir / "frozen_repeat_pair_baselines.md"
    out.to_csv(csv_path, index=False)
    md_path.write_text(
        "# Frozen Repeat-Pair Baselines\n\n"
        "Generated from current repeat-pair raw-similarity and learned-encoder summaries.\n\n"
        + dataframe_to_markdown(out)
        + "\n",
        encoding="utf-8",
    )
    print({"csv": str(csv_path), "md": str(md_path), "n_rows": int(len(out))})


if __name__ == "__main__":
    main()
