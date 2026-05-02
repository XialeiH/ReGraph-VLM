#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np


SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect NSD ROI masks and choose an atlas ROI node set.")
    parser.add_argument("--root", type=Path, required=True, help="v0_shared_unit root on HPC.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preferred-roi-file", type=str, default="HCP_MMP1.nii.gz")
    return parser.parse_args()


def roi_family(filename: str) -> str:
    name = filename.replace(".nii.gz", "")
    if name.startswith(("lh.", "rh.")):
        name = name[3:]
    return name


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    family_labels: dict[str, list[set[int]]] = {}
    family_files: dict[str, set[str]] = {}

    for subject in SUBJECTS:
        roi_dir = args.root / f"data/nsddata/ppdata/{subject}/func1pt8mm/roi"
        for path in sorted(roi_dir.glob("*.nii.gz")):
            try:
                arr = np.asanyarray(nib.load(str(path)).dataobj)
                labels = sorted(int(v) for v in np.unique(arr.astype(np.int32)) if int(v) > 0)
                usable = len(labels) > 0
            except Exception:
                labels = []
                usable = False
            family = roi_family(path.name)
            if usable:
                family_labels.setdefault(family, []).append(set(labels))
                family_files.setdefault(family, set()).add(path.name)
            rows.append(
                {
                    "subject": subject,
                    "roi_file": path.name,
                    "roi_family": family,
                    "n_labels": len(labels),
                    "label_ids": " ".join(str(v) for v in labels),
                    "label_names_if_available": "",
                    "usable": str(usable).lower(),
                }
            )

    inventory_path = args.output_dir / "roi_inventory.csv"
    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "subject",
                "roi_file",
                "roi_family",
                "n_labels",
                "label_ids",
                "label_names_if_available",
                "usable",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    preferred_family = roi_family(args.preferred_roi_file)
    label_sets = family_labels.get(preferred_family, [])
    if not label_sets:
        raise ValueError(f"No usable labels found for preferred ROI family {preferred_family}")
    common_labels = sorted(set.intersection(*label_sets))
    if len(common_labels) < 10:
        raise ValueError(f"Preferred ROI family {preferred_family} has only {len(common_labels)} common labels")

    node_set = {
        "node_set_name": f"{preferred_family.lower()}_{len(common_labels)}_v1",
        "n_nodes": len(common_labels),
        "roi_source_files": sorted(family_files[preferred_family]),
        "roi_family": preferred_family,
        "node_labels": [
            {
                "node_index": idx,
                "label_id": int(label_id),
                "node_name": f"{preferred_family}_{label_id}",
            }
            for idx, label_id in enumerate(common_labels)
        ],
        "include_hemisphere_split": False,
        "label_rule": "positive integer labels common across all subjects",
        "decision_rule": "<10 nodes stop; 10-20 cautious smoke; 20+ BrainGNN path; 50+ preferred",
    }
    (args.output_dir / "roi_node_set_v1.json").write_text(json.dumps(node_set, indent=2), encoding="utf-8")
    print(json.dumps({"inventory": str(inventory_path), "node_set": node_set["node_set_name"], "n_nodes": len(common_labels)}))


if __name__ == "__main__":
    main()
