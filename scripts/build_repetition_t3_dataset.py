#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
PAIR_TYPES = [(1, 2), (1, 3), (2, 3)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strict T=3 repetition/familiarity scalar4 datasets.")
    parser.add_argument("--root", type=Path, required=True, help="v0_shared_unit root.")
    parser.add_argument("--folds", nargs="+", default=["fold_01", "fold_04"])
    parser.add_argument("--inventory", type=Path, default=Path("preproc_v0/repetition_familiarity/repetition_inventory.csv"))
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/trial_roi_features_scalar4"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3"),
    )
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260501)
    return parser.parse_args()


def subject_to_int(subject: str) -> int:
    return int(subject.replace("subj", ""))


def fold_subjects(fold_name: str) -> dict[str, list[str] | str]:
    fold_idx = int(fold_name.split("_")[-1])
    test_subject = f"subj{fold_idx:02d}"
    train_candidates = [s for s in SUBJECTS if s != test_subject]
    val_subject = train_candidates[-1]
    train_subjects = [s for s in train_candidates if s != val_subject]
    return {"test": test_subject, "val": val_subject, "train": train_subjects}


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_subject_features(feature_dir: Path) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for subject in SUBJECTS:
        path = feature_dir / f"{subject}_trial_scalar4.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing subject feature file: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        x = data["x"]
        key_to_index = {}
        for idx in range(x.shape[0]):
            key = (
                int(data["nsdId"][idx]),
                int(data["repeat_index"][idx]),
                int(data["session_index"][idx]),
                int(data["trial_index"][idx]),
            )
            key_to_index[key] = idx
        out[subject] = {"data": data, "key_to_index": key_to_index}
    return out


def class_map_from_inventory(inventory: pd.DataFrame) -> dict[int, int]:
    nsd_ids = sorted(inventory["nsdId"].astype(int).unique().tolist())
    return {nsd_id: idx for idx, nsd_id in enumerate(nsd_ids)}


def strict_t3_groups(inventory: pd.DataFrame) -> pd.DataFrame:
    rows = inventory[
        (inventory["usable_T3"] == True)
        & (inventory["repeat_index"].isin([1, 2, 3]))
        & (inventory["has_beta"] == True)
        & (inventory["has_roi_mask"] == True)
    ].copy()
    counts = rows.groupby(["subject", "nsdId"])["repeat_index"].nunique()
    good_index = counts[counts == 3].index
    good = rows.set_index(["subject", "nsdId"]).loc[good_index].reset_index()
    return good.sort_values(["subject", "nsdId", "repeat_index", "session_index", "trial_index"]).reset_index(drop=True)


def build_sequences(
    rows: pd.DataFrame,
    features: dict[str, dict[str, object]],
    cmap: dict[int, int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    items: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    for (subject, nsd_id), group in rows.groupby(["subject", "nsdId"], sort=True):
        group = group.sort_values("repeat_index")
        repeat_seq = group["repeat_index"].astype(int).tolist()
        if repeat_seq != [1, 2, 3]:
            raise ValueError(f"{subject} {nsd_id} repeat sequence is {repeat_seq}, expected [1, 2, 3]")
        x_parts = []
        session_seq = []
        trial_seq = []
        for row in group.itertuples(index=False):
            key = (int(row.nsdId), int(row.repeat_index), int(row.session_index), int(row.trial_index))
            idx = features[subject]["key_to_index"][key]  # type: ignore[index]
            x_parts.append(features[subject]["data"]["x"][idx])  # type: ignore[index]
            session_seq.append(int(row.session_index))
            trial_seq.append(int(row.trial_index))
        x_seq = torch.stack(x_parts, dim=0).to(torch.float32)
        item = {
            "x_seq": x_seq,
            "subject": subject_to_int(subject),
            "nsdId": int(nsd_id),
            "y_image": int(cmap[int(nsd_id)]),
            "repeat_seq": torch.tensor([1, 2, 3], dtype=torch.int64),
            "session_seq": torch.tensor(session_seq, dtype=torch.int64),
            "trial_seq": torch.tensor(trial_seq, dtype=torch.int64),
        }
        items.append(item)
        metadata.append(
            {
                "subject": subject,
                "nsdId": int(nsd_id),
                "y_image": int(cmap[int(nsd_id)]),
                "repeat_seq": "1 2 3",
                "session_seq": " ".join(str(v) for v in session_seq),
                "trial_seq": " ".join(str(v) for v in trial_seq),
            }
        )
    return items, metadata


def build_single_trials(sequences: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    items: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    for seq_idx, seq in enumerate(sequences):
        x_seq = seq["x_seq"]
        assert isinstance(x_seq, torch.Tensor)
        for step, repeat_index in enumerate([1, 2, 3]):
            item = {
                "x": x_seq[step].clone(),
                "subject": int(seq["subject"]),
                "nsdId": int(seq["nsdId"]),
                "repeat_index": repeat_index,
                "y_first_vs_repeated": 0 if repeat_index == 1 else 1,
                "y_repeat_index": repeat_index - 1,
                "session_index": int(seq["session_seq"][step]),
                "trial_index": int(seq["trial_seq"][step]),
            }
            items.append(item)
            metadata.append(
                {
                    "sequence_index": seq_idx,
                    "subject": int(seq["subject"]),
                    "nsdId": int(seq["nsdId"]),
                    "repeat_index": repeat_index,
                    "y_first_vs_repeated": item["y_first_vs_repeated"],
                    "y_repeat_index": item["y_repeat_index"],
                    "session_index": item["session_index"],
                    "trial_index": item["trial_index"],
                }
            )
    return items, metadata


def build_pairs(
    sequences: list[dict[str, object]],
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int, int]:
    by_pair_type: dict[tuple[int, int], list[int]] = {pair_type: [] for pair_type in PAIR_TYPES}
    for idx, _ in enumerate(sequences):
        for pair_type in PAIR_TYPES:
            by_pair_type[pair_type].append(idx)

    items: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    positive_count = 0
    negative_count = 0

    for pair_type in PAIR_TYPES:
        seq_indices = by_pair_type[pair_type]
        for seq_idx in seq_indices:
            seq = sequences[seq_idx]
            r1, r2 = pair_type
            x_seq = seq["x_seq"]
            assert isinstance(x_seq, torch.Tensor)
            pos = {
                "x1": x_seq[r1 - 1].clone(),
                "x2": x_seq[r2 - 1].clone(),
                "same_image": 1,
                "subject": int(seq["subject"]),
                "nsdId_1": int(seq["nsdId"]),
                "nsdId_2": int(seq["nsdId"]),
                "repeat_1": r1,
                "repeat_2": r2,
                "session_1": int(seq["session_seq"][r1 - 1]),
                "session_2": int(seq["session_seq"][r2 - 1]),
            }
            items.append(pos)
            metadata.append({k: v for k, v in pos.items() if not isinstance(v, torch.Tensor)})
            positive_count += 1

            candidates = [
                idx
                for idx in seq_indices
                if int(sequences[idx]["subject"]) == int(seq["subject"])
                and int(sequences[idx]["nsdId"]) != int(seq["nsdId"])
            ]
            if not candidates:
                raise ValueError("Cannot sample negative pair; no different-image candidates")
            neg_idx = int(rng.choice(candidates))
            neg_seq = sequences[neg_idx]
            neg_x_seq = neg_seq["x_seq"]
            assert isinstance(neg_x_seq, torch.Tensor)
            neg = {
                "x1": x_seq[r1 - 1].clone(),
                "x2": neg_x_seq[r2 - 1].clone(),
                "same_image": 0,
                "subject": int(seq["subject"]),
                "nsdId_1": int(seq["nsdId"]),
                "nsdId_2": int(neg_seq["nsdId"]),
                "repeat_1": r1,
                "repeat_2": r2,
                "session_1": int(seq["session_seq"][r1 - 1]),
                "session_2": int(neg_seq["session_seq"][r2 - 1]),
            }
            items.append(neg)
            metadata.append({k: v for k, v in neg.items() if not isinstance(v, torch.Tensor)})
            negative_count += 1

    return items, metadata, positive_count, negative_count


def split_rows(rows: pd.DataFrame, split_subjects: set[str]) -> pd.DataFrame:
    return rows[rows["subject"].isin(split_subjects)].copy()


def compute_adjacency(train_single_trials: list[dict[str, object]], topk: int) -> tuple[np.ndarray, np.ndarray]:
    roi_mean = np.stack([item["x"][:, 0].numpy() for item in train_single_trials], axis=0).astype(np.float64)
    corr = np.corrcoef(roi_mean, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    dense = corr.astype(np.float32)

    topk_adj = np.zeros_like(dense)
    k = min(topk, dense.shape[0] - 1)
    for i in range(dense.shape[0]):
        idx = np.argsort(np.abs(dense[i]))[-k:]
        topk_adj[i, idx] = dense[i, idx]
    topk_adj = np.maximum(topk_adj, topk_adj.T).astype(np.float32)
    np.fill_diagonal(topk_adj, 0.0)
    return dense, topk_adj


def count_nan_inf_tensors(items: list[dict[str, object]]) -> tuple[int, int]:
    nan_count = 0
    inf_count = 0
    for item in items:
        for value in item.values():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                nan_count += int(torch.isnan(value).sum().item())
                inf_count += int(torch.isinf(value).sum().item())
    return nan_count, inf_count


def save_split(
    out_dir: Path,
    split: str,
    rows: pd.DataFrame,
    features: dict[str, dict[str, object]],
    cmap: dict[int, int],
    rng: np.random.Generator,
) -> dict[str, object]:
    sequences, sequence_meta = build_sequences(rows, features, cmap)
    single_trials, single_meta = build_single_trials(sequences)
    pairs, pair_meta, n_pos, n_neg = build_pairs(sequences, rng)

    torch.save(sequences, out_dir / f"{split}_sequences.pt")
    torch.save(single_trials, out_dir / f"{split}_single_trials.pt")
    torch.save(pairs, out_dir / f"{split}_pairs.pt")

    return {
        "sequences": sequences,
        "single_trials": single_trials,
        "pairs": pairs,
        "sequence_meta": sequence_meta,
        "single_meta": single_meta,
        "pair_meta": pair_meta,
        "n_positive_pairs": n_pos,
        "n_negative_pairs": n_neg,
    }


def build_fold(
    root: Path,
    fold_name: str,
    rows: pd.DataFrame,
    features: dict[str, dict[str, object]],
    cmap: dict[int, int],
    output_root: Path,
    topk: int,
    seed: int,
) -> dict[str, object]:
    out_dir = output_root / fold_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_def = fold_subjects(fold_name)
    split_subjects_map = {
        "train": set(fold_def["train"]),  # type: ignore[arg-type]
        "val": {str(fold_def["val"])},
        "test": {str(fold_def["test"])},
    }
    rng = np.random.default_rng(seed + int(fold_name.split("_")[-1]))
    split_payloads: dict[str, dict[str, object]] = {}
    for split, subjects in split_subjects_map.items():
        split_payloads[split] = save_split(out_dir, split, split_rows(rows, subjects), features, cmap, rng)

    dense_adj, topk_adj = compute_adjacency(split_payloads["train"]["single_trials"], topk)  # type: ignore[arg-type]
    np.save(out_dir / "adjacency.npy", topk_adj)
    np.save(out_dir / "adjacency_dense_corr.npy", dense_adj)
    np.save(out_dir / f"adjacency_topk{topk}_corr.npy", topk_adj)

    all_sequence_meta: list[dict[str, object]] = []
    all_single_meta: list[dict[str, object]] = []
    all_pair_meta: list[dict[str, object]] = []
    for split in ["train", "val", "test"]:
        for row in split_payloads[split]["sequence_meta"]:  # type: ignore[union-attr]
            row = dict(row)
            row["split"] = split
            all_sequence_meta.append(row)
        for row in split_payloads[split]["single_meta"]:  # type: ignore[union-attr]
            row = dict(row)
            row["split"] = split
            all_single_meta.append(row)
        for row in split_payloads[split]["pair_meta"]:  # type: ignore[union-attr]
            row = dict(row)
            row["split"] = split
            all_pair_meta.append(row)

    write_csv(
        out_dir / "metadata_sequences.csv",
        all_sequence_meta,
        ["split", "subject", "nsdId", "y_image", "repeat_seq", "session_seq", "trial_seq"],
    )
    write_csv(
        out_dir / "metadata_single_trials.csv",
        all_single_meta,
        [
            "split",
            "sequence_index",
            "subject",
            "nsdId",
            "repeat_index",
            "y_first_vs_repeated",
            "y_repeat_index",
            "session_index",
            "trial_index",
        ],
    )
    write_csv(
        out_dir / "metadata_pairs.csv",
        all_pair_meta,
        ["split", "same_image", "subject", "nsdId_1", "nsdId_2", "repeat_1", "repeat_2", "session_1", "session_2"],
    )

    nan_count = 0
    inf_count = 0
    for split in ["train", "val", "test"]:
        for key in ["sequences", "single_trials", "pairs"]:
            n, i = count_nan_inf_tensors(split_payloads[split][key])  # type: ignore[arg-type]
            nan_count += n
            inf_count += i

    sequence_counts = {split: len(split_payloads[split]["sequences"]) for split in ["train", "val", "test"]}
    single_counts = {split: len(split_payloads[split]["single_trials"]) for split in ["train", "val", "test"]}
    pos_counts = {split: int(split_payloads[split]["n_positive_pairs"]) for split in ["train", "val", "test"]}
    neg_counts = {split: int(split_payloads[split]["n_negative_pairs"]) for split in ["train", "val", "test"]}

    repeat_ok = all(
        torch.equal(item["repeat_seq"], torch.tensor([1, 2, 3], dtype=torch.int64))
        for split in ["train", "val", "test"]
        for item in split_payloads[split]["sequences"]  # type: ignore[union-attr]
    )
    qc = {
        "fold": fold_name,
        "feature_set": "scalar4",
        "T": 3,
        "test_subject": fold_def["test"],
        "val_subject": fold_def["val"],
        "train_subjects": fold_def["train"],
        "sequence_counts": sequence_counts,
        "single_trial_counts": single_counts,
        "positive_pair_counts": pos_counts,
        "negative_pair_counts": neg_counts,
        "nan_count": int(nan_count),
        "inf_count": int(inf_count),
        "repeat_seq_check": "all_[1,2,3]" if repeat_ok else "failed",
        "adjacency_shape": list(topk_adj.shape),
        "adjacency_dense_density": float(np.count_nonzero(dense_adj) / dense_adj.size),
        "adjacency_topk_density": float(np.count_nonzero(topk_adj) / topk_adj.size),
        "adjacency_topk": int(topk),
        "status": "ok" if nan_count == 0 and inf_count == 0 and repeat_ok else "failed",
    }
    (out_dir / "dataset_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc), flush=True)
    return qc


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    inventory = pd.read_csv(root / args.inventory)
    t3_rows = strict_t3_groups(inventory)
    features = load_subject_features(root / args.feature_dir)
    cmap = class_map_from_inventory(inventory)
    output_root = root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for fold_name in args.folds:
        summaries.append(build_fold(root, fold_name, t3_rows, features, cmap, output_root, args.topk, args.seed))
    (output_root / "dataset_qc_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
