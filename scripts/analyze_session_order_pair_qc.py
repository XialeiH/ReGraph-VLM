#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit anchor-side session/order matching in cross-subject pair tensors.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", default=[f"fold_{idx:02d}" for idx in range(1, 9)])
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    return parser.parse_args()


def anchor_key(pair: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(pair.get("subject", pair.get("subject_1", -1))),
        int(pair.get("anchor_nsdId", pair.get("nsdId_1", -1))),
        int(pair.get("repeat_1", -1)),
        int(pair.get("session_1", -999)),
    )


def audit_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        groups[anchor_key(pair)].append(pair)

    n_complete = 0
    n_problem = 0
    n_pos = 0
    n_neg = 0
    same_subject = 0
    same_anchor_image = 0
    same_repeat = 0
    same_session = 0
    session_pos = Counter()
    session_neg = Counter()
    repeat_pos = Counter()
    repeat_neg = Counter()
    example_problems: list[dict[str, Any]] = []

    for key, group in groups.items():
        labels = [int(pair["same_image"]) for pair in group]
        n_pos += labels.count(1)
        n_neg += labels.count(0)
        for pair in group:
            label = int(pair["same_image"])
            session = int(pair.get("session_1", -999))
            repeat = int(pair.get("repeat_1", -1))
            if label == 1:
                session_pos[session] += 1
                repeat_pos[repeat] += 1
            else:
                session_neg[session] += 1
                repeat_neg[repeat] += 1

        if labels.count(1) == 1 and labels.count(0) == 1 and len(group) == 2:
            n_complete += 1
            first, second = group
            same_subject += int(
                int(first.get("subject", first.get("subject_1", -1)))
                == int(second.get("subject", second.get("subject_1", -1)))
            )
            same_anchor_image += int(
                int(first.get("anchor_nsdId", first.get("nsdId_1", -1)))
                == int(second.get("anchor_nsdId", second.get("nsdId_1", -1)))
            )
            same_repeat += int(
                int(first.get("repeat_1", -1)) == int(second.get("repeat_1", -1))
                and int(first.get("repeat_2", -1)) == int(second.get("repeat_2", -1))
            )
            same_session += int(int(first.get("session_1", -999)) == int(second.get("session_1", -999)))
        else:
            n_problem += 1
            if len(example_problems) < 10:
                example_problems.append({"key": key, "labels": labels, "n": len(group)})

    def max_counter_gap(a: Counter, b: Counter) -> int:
        return max((abs(a[key] - b[key]) for key in set(a) | set(b)), default=0)

    denom = max(n_complete, 1)
    return {
        "n_pairs": len(pairs),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_anchor_groups": len(groups),
        "n_complete_pos_neg_anchor_groups": n_complete,
        "n_problem_anchor_groups": n_problem,
        "pct_pos_neg_share_anchor_subject": same_subject / denom,
        "pct_pos_neg_share_anchor_image": same_anchor_image / denom,
        "pct_pos_neg_share_repeat_index": same_repeat / denom,
        "pct_pos_neg_share_anchor_session": same_session / denom,
        "max_positive_negative_count_gap_by_anchor_session": max_counter_gap(session_pos, session_neg),
        "max_positive_negative_count_gap_by_repeat_index": max_counter_gap(repeat_pos, repeat_neg),
        "example_problem_groups": example_problems,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in rows[0] if key != "example_problem_groups"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key != "example_problem_groups"})


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []

    for fold in args.folds:
        fold_dir = args.dataset_root / fold
        for split in args.splits:
            path = fold_dir / f"{split}_pairs.pt"
            if not path.exists():
                continue
            pairs = torch.load(path, map_location="cpu", weights_only=False)
            row = {"fold": fold, "split": split, **audit_pairs(pairs)}
            problems.extend({"fold": fold, "split": split, **item} for item in row["example_problem_groups"])
            rows.append(row)

    write_csv(args.output_dir / "session_order_pair_qc.csv", rows)

    summary_rows: list[dict[str, Any]] = []
    for split in [*args.splits, "all"]:
        subset = rows if split == "all" else [row for row in rows if row["split"] == split]
        if not subset:
            continue
        complete = sum(int(row["n_complete_pos_neg_anchor_groups"]) for row in subset)

        def weighted_average(key: str) -> float:
            return sum(float(row[key]) * int(row["n_complete_pos_neg_anchor_groups"]) for row in subset) / max(complete, 1)

        summary_rows.append(
            {
                "split": split,
                "n_pairs": sum(int(row["n_pairs"]) for row in subset),
                "n_positive": sum(int(row["n_positive"]) for row in subset),
                "n_negative": sum(int(row["n_negative"]) for row in subset),
                "n_complete_anchor_groups": complete,
                "n_problem_anchor_groups": sum(int(row["n_problem_anchor_groups"]) for row in subset),
                "pct_pos_neg_share_anchor_subject": weighted_average("pct_pos_neg_share_anchor_subject"),
                "pct_pos_neg_share_anchor_image": weighted_average("pct_pos_neg_share_anchor_image"),
                "pct_pos_neg_share_repeat_index": weighted_average("pct_pos_neg_share_repeat_index"),
                "pct_pos_neg_share_anchor_session": weighted_average("pct_pos_neg_share_anchor_session"),
                "max_positive_negative_count_gap_by_anchor_session": max(
                    int(row["max_positive_negative_count_gap_by_anchor_session"]) for row in subset
                ),
                "max_positive_negative_count_gap_by_repeat_index": max(
                    int(row["max_positive_negative_count_gap_by_repeat_index"]) for row in subset
                ),
            }
        )

    write_csv(args.output_dir / "session_order_pair_qc_summary.csv", summary_rows)
    md = [
        "# Session/order pair QC",
        "",
        "Cross-subject pair construction was audited by grouping the positive and negative example generated from each anchor trial. A complete anchor group contains exactly one positive and one negative with the same anchor subject, anchor image, repeat index, and anchor session. The reference side is a training-subject average, so this is an anchor-side session/order control rather than a full reference-session control.",
        "",
    ]
    columns = list(summary_rows[0])
    md.append("| " + " | ".join(columns) + " |")
    md.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in summary_rows:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        md.append("| " + " | ".join(values) + " |")
    if problems:
        md.extend(["", "Example problematic groups:", "", "```json", json.dumps(problems[:10], indent=2), "```"])
    (args.output_dir / "session_order_pair_qc.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print((args.output_dir / "session_order_pair_qc_summary.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
