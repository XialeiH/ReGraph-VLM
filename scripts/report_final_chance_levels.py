#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report chance levels for final retrieval datasets.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("preproc_v0/repetition_familiarity/results/final_tables/final_chance_levels.csv"))
    return parser.parse_args()


def load_pairs(path: Path) -> list[dict]:
    return torch.load(path, map_location="cpu", weights_only=False)


def summarize_dataset(root: Path, dataset_name: str, dataset_root: Path) -> list[dict[str, object]]:
    rows = []
    for fold_dir in sorted(dataset_root.glob("fold_*")):
        path = fold_dir / "test_pairs.pt"
        if not path.exists():
            continue
        pairs = load_pairs(path)
        nsd = sorted({int(p["nsdId_1"]) for p in pairs} | {int(p["nsdId_2"]) for p in pairs})
        n_candidates = max(1, len(nsd))
        rows.append(
            {
                "dataset": dataset_name,
                "fold": fold_dir.name,
                "n_test_pairs": len(pairs),
                "n_candidate_images": n_candidates,
                "chance_R@1": min(1.0, 1.0 / n_candidates),
                "chance_R@5": min(1.0, 5.0 / n_candidates),
                "chance_R@10": min(1.0, 10.0 / n_candidates),
                "chance_MRR_approx": sum(1.0 / r for r in range(1, n_candidates + 1)) / n_candidates,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    base = root / "preproc_v0/repetition_familiarity/datasets"
    specs = {
        "cross_subject_allfold": base / "scalar4_T3_clip_cross_subject_allfold",
        "heldout_image": base / "scalar4_T3_clip_cross_subject_imageheldout",
        "heldout_image_random": base / "scalar4_T3_clip_cross_subject_imageheldout_random_embedding",
        "hardneg_allfold": base / "scalar4_T3_clip_cross_subject_hardneg_allfold/mixed_50_random_50_clip_hard",
    }
    rows = []
    for name, path in specs.items():
        if path.exists():
            rows.extend(summarize_dataset(root, name, path))
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    out.with_suffix(".md").write_text(df.to_markdown(index=False, floatfmt=".6f"), encoding="utf-8")
    print({"out": str(out), "n_rows": len(df)})


if __name__ == "__main__":
    main()
