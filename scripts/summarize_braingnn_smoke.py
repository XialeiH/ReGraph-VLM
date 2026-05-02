#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize NSD BrainGNN smoke metrics.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sanity-summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_sanity(path: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["fold"], row["model"]): row for row in rows}


def main() -> None:
    args = parse_args()
    sanity = read_sanity(args.sanity_summary)
    rows: list[dict[str, object]] = []
    for path in sorted(args.root.glob("fold_*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        fold = metrics["fold"]
        gcn = sanity.get((fold, "gcn"), {})
        gat = sanity.get((fold, "gat"), {})
        mlp = sanity.get((fold, "roi_mlp"), {})
        rows.append(
            {
                "fold": fold,
                "top1": metrics["top1"],
                "top5": metrics["top5"],
                "chance_level": metrics["chance_level"],
                "best_val_top1": metrics["best_val_top1"],
                "best_val_top5": metrics["best_val_top5"],
                "best_epoch": metrics["best_epoch"],
                "n_nodes": metrics["n_nodes"],
                "node_feature_dim": metrics["node_feature_dim"],
                "roi_mlp_top1": mlp.get("top1", ""),
                "roi_mlp_top5": mlp.get("top5", ""),
                "gcn_top1": gcn.get("top1", ""),
                "gcn_top5": gcn.get("top5", ""),
                "gat_top1": gat.get("top1", ""),
                "gat_top5": gat.get("top5", ""),
                "status": metrics["status"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold",
                "top1",
                "top5",
                "chance_level",
                "best_val_top1",
                "best_val_top5",
                "best_epoch",
                "n_nodes",
                "node_feature_dim",
                "roi_mlp_top1",
                "roi_mlp_top5",
                "gcn_top1",
                "gcn_top5",
                "gat_top1",
                "gat_top5",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
