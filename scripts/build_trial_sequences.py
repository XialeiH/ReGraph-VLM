#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build trial sequences for Stage 4 dynamic GNN.")
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--fold-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_index_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_metadata_row(index_row: dict[str, str], cache: dict[Path, list[dict[str, str]]]) -> dict[str, str]:
    feature_path = Path(index_row["source_feature_path"])
    metadata_path = feature_path.with_name(feature_path.name.replace("_roi_features.npz", "_metadata.csv"))
    if metadata_path not in cache:
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            cache[metadata_path] = list(csv.DictReader(handle))
    sample_index = int(index_row["sample_index"])
    return cache[metadata_path][sample_index]


def group_split(
    rows: list[dict[str, str]],
    assignments: np.ndarray,
    labels: np.ndarray,
    metadata_cache: dict[Path, list[dict[str, str]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    grouped: dict[tuple[str, int], list[tuple[int, np.ndarray, int, dict[str, str]]]] = defaultdict(list)
    for row_idx, row in enumerate(rows):
        meta = load_metadata_row(row, metadata_cache)
        subject = row["subject"]
        nsd_id = int(row["nsdId"])
        sort_key = int(meta.get("global_trial_index", meta.get("sample_index", row["sample_index"])))
        grouped[(subject, nsd_id)].append((sort_key, assignments[row_idx], int(labels[row_idx]), meta))

    sequence_meta: list[dict[str, object]] = []
    max_len = 0
    for key in grouped:
        grouped[key].sort(key=lambda item: item[0])
        max_len = max(max_len, len(grouped[key]))

    seqs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    seq_labels: list[int] = []
    for seq_idx, ((subject, nsd_id), items) in enumerate(sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1]))):
        length = len(items)
        seq = np.zeros((max_len, assignments.shape[1]), dtype=np.float32)
        mask = np.zeros((max_len,), dtype=np.float32)
        trial_order: list[int] = []
        sessions: list[int] = []
        trials_in_session: list[int] = []
        rep_indices: list[int] = []
        label = int(items[0][2])
        for t, (sort_key, assignment, item_label, meta) in enumerate(items):
            if int(item_label) != label:
                raise ValueError(f"Inconsistent labels within sequence {(subject, nsd_id)}")
            seq[t] = assignment.astype(np.float32)
            mask[t] = 1.0
            trial_order.append(int(sort_key))
            sessions.append(int(meta["session"]))
            trials_in_session.append(int(meta["trial_in_session"]))
            rep_indices.append(int(meta["rep_index_for_subject"]))
        seqs.append(seq)
        masks.append(mask)
        seq_labels.append(label)
        sequence_meta.append(
            {
                "sequence_index": seq_idx,
                "subject": subject,
                "nsdId": nsd_id,
                "label": label,
                "length": length,
                "global_trial_indices": trial_order,
                "sessions": sessions,
                "trials_in_session": trials_in_session,
                "rep_indices": rep_indices,
            }
        )

    return (
        np.stack(seqs, axis=0).astype(np.float32),
        np.stack(masks, axis=0).astype(np.float32),
        np.asarray(seq_labels, dtype=np.int64),
        sequence_meta,
    )


def write_sequence_index(path: Path, rows: list[dict[str, object]], split: str) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "sequence_index",
                "subject",
                "nsdId",
                "label",
                "length",
                "global_trial_indices",
                "sessions",
                "trials_in_session",
                "rep_indices",
            ],
        )
        if handle.tell() == 0:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "split": split,
                    "sequence_index": row["sequence_index"],
                    "subject": row["subject"],
                    "nsdId": row["nsdId"],
                    "label": row["label"],
                    "length": row["length"],
                    "global_trial_indices": " ".join(str(v) for v in row["global_trial_indices"]),
                    "sessions": " ".join(str(v) for v in row["sessions"]),
                    "trials_in_session": " ".join(str(v) for v in row["trials_in_session"]),
                    "rep_indices": " ".join(str(v) for v in row["rep_indices"]),
                }
            )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_index = read_index_csv(args.fold_root / f"{args.fold_name}_train_features_index.csv")
    test_index = read_index_csv(args.fold_root / f"{args.fold_name}_test_features_index.csv")
    graph_meta = json.loads((args.graph_root / f"{args.fold_name}_unit_graph_metadata.json").read_text(encoding="utf-8"))
    canonical_val_subject = str(graph_meta["canonical_validation_subject"])
    graph_inputs = np.load(args.graph_root / f"{args.fold_name}_light_interaction_inputs.npz")
    adjacency = np.load(args.graph_root / f"{args.fold_name}_unit_graph.npz")["adjacency"].astype(np.float32)

    fit_mask = np.array([row["subject"] != canonical_val_subject for row in train_index], dtype=bool)
    val_mask = ~fit_mask
    fit_rows = [row for row, keep in zip(train_index, fit_mask.tolist()) if keep]
    val_rows = [row for row, keep in zip(train_index, val_mask.tolist()) if keep]

    metadata_cache: dict[Path, list[dict[str, str]]] = {}
    fit_seq, fit_seq_mask, fit_labels, fit_meta = group_split(
        rows=fit_rows,
        assignments=graph_inputs["fit_assignments"].astype(np.float32),
        labels=graph_inputs["fit_labels"].astype(np.int64),
        metadata_cache=metadata_cache,
    )
    val_seq, val_seq_mask, val_labels, val_meta = group_split(
        rows=val_rows,
        assignments=graph_inputs["val_assignments"].astype(np.float32),
        labels=graph_inputs["val_labels"].astype(np.int64),
        metadata_cache=metadata_cache,
    )
    test_seq, test_seq_mask, test_labels, test_meta = group_split(
        rows=test_index,
        assignments=graph_inputs["test_assignments"].astype(np.float32),
        labels=graph_inputs["test_labels"].astype(np.int64),
        metadata_cache=metadata_cache,
    )

    max_seq_len = max(fit_seq.shape[1], val_seq.shape[1], test_seq.shape[1])
    if not (fit_seq.shape[1] == val_seq.shape[1] == test_seq.shape[1]):
        raise ValueError("Expected identical padded sequence lengths across fit/val/test splits")

    np.savez_compressed(
        args.output_dir / f"{args.fold_name}_trial_sequences.npz",
        fit_sequences=fit_seq,
        fit_masks=fit_seq_mask,
        fit_labels=fit_labels,
        val_sequences=val_seq,
        val_masks=val_seq_mask,
        val_labels=val_labels,
        test_sequences=test_seq,
        test_masks=test_seq_mask,
        test_labels=test_labels,
        adjacency=adjacency,
    )

    index_path = args.output_dir / f"{args.fold_name}_trial_sequence_index.csv"
    if index_path.exists():
        index_path.unlink()
    write_sequence_index(index_path, fit_meta, "fit")
    write_sequence_index(index_path, val_meta, "val")
    write_sequence_index(index_path, test_meta, "test")

    summary = {
        "fold": args.fold_name,
        "held_out_subject": str(graph_meta["held_out_subject"]),
        "canonical_validation_subject": canonical_val_subject,
        "num_units": int(adjacency.shape[0]),
        "fit_sequences": int(fit_seq.shape[0]),
        "val_sequences": int(val_seq.shape[0]),
        "test_sequences": int(test_seq.shape[0]),
        "fit_trials": int(fit_seq_mask.sum()),
        "val_trials": int(val_seq_mask.sum()),
        "test_trials": int(test_seq_mask.sum()),
        "max_sequence_length": int(max_seq_len),
    }
    (args.output_dir / f"{args.fold_name}_trial_sequence_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
