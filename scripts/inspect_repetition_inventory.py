#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
TRIALS_PER_SESSION = 750


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect NSD repeated-image availability for repetition/familiarity ROI-graph tasks."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="v0_shared_unit root.")
    parser.add_argument("--shared-manifest", type=Path, default=Path("preproc_v0/shared1000_manifest.csv"))
    parser.add_argument("--dataset-manifest", type=Path, default=Path("preproc_v0/all8_ge2_766_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("preproc_v0/repetition_familiarity"))
    parser.add_argument("--folds", nargs="+", default=["fold_01", "fold_04"])
    parser.add_argument(
        "--trials-per-run",
        type=int,
        default=63,
        help="Approximate run derivation from trial_in_session. Used only for inventory metadata.",
    )
    parser.add_argument(
        "--min-t3-sequences-per-split",
        type=int,
        default=100,
        help="Gate threshold used to recommend strict T=3 vs T=2 for smoke folds.",
    )
    return parser.parse_args()


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def fold_subjects(fold_name: str) -> dict[str, list[str] | str]:
    fold_idx = int(fold_name.split("_")[-1])
    test_subject = f"subj{fold_idx:02d}"
    train_candidates = [s for s in SUBJECTS if s != test_subject]
    val_subject = train_candidates[-1]
    train_subjects = [s for s in train_candidates if s != val_subject]
    return {"test": test_subject, "val": val_subject, "train": train_subjects}


def safe_int(value: object, default: int | None = None) -> int | None:
    if pd.isna(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def beta_path_exists(root: Path, row: pd.Series) -> bool:
    path_value = str(row.get("beta_path", "")).strip()
    if path_value and path_value.lower() != "nan":
        path = Path(path_value)
        if not path.is_absolute():
            path = root / path
        if path.exists():
            return True
    session = safe_int(row.get("session"))
    subject = str(row.get("subject"))
    if session is None:
        return False
    fallback = (
        root
        / "data/nsddata_betas/ppdata"
        / subject
        / "func1pt8mm/betas_fithrf_GLMdenoise_RR"
        / f"betas_session{session:02d}.hdf5"
    )
    return fallback.exists()


def roi_mask_exists(root: Path, subject: str) -> bool:
    return (
        root
        / "data/nsddata/ppdata"
        / subject
        / "func1pt8mm/roi/HCP_MMP1.nii.gz"
    ).exists()


def numeric_nan_inf_counts(df: pd.DataFrame) -> dict[str, int]:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return {"nan_count": 0, "inf_count": 0}
    values = numeric.to_numpy(dtype=np.float64, copy=False)
    return {
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
    }


def counter_to_sorted_dict(counter: Counter) -> dict[str, int]:
    return {str(k): int(counter[k]) for k in sorted(counter)}


def split_for_subject(subject: str, fold_def: dict[str, list[str] | str]) -> str:
    if subject == fold_def["test"]:
        return "test"
    if subject == fold_def["val"]:
        return "val"
    return "train"


def build_inventory(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    root = args.root.resolve()
    shared_manifest = pd.read_csv(root / args.shared_manifest)
    dataset_manifest = pd.read_csv(root / args.dataset_manifest)

    if "usable" in shared_manifest.columns:
        shared_manifest = shared_manifest[bool_series(shared_manifest["usable"])].copy()

    selected_nsd_ids = sorted(dataset_manifest["nsdId"].astype(int).unique().tolist())
    class_map = {nsd_id: idx for idx, nsd_id in enumerate(selected_nsd_ids)}
    shared_manifest = shared_manifest[shared_manifest["nsdId"].astype(int).isin(class_map)].copy()
    shared_manifest["nsdId"] = shared_manifest["nsdId"].astype(int)
    shared_manifest["subject"] = shared_manifest["subject"].astype(str)
    shared_manifest["session"] = shared_manifest["session"].apply(safe_int)
    shared_manifest["trial_in_session"] = shared_manifest["trial_in_session"].apply(safe_int)
    shared_manifest["rep_index_for_subject"] = shared_manifest["rep_index_for_subject"].apply(safe_int)
    if "global_trial_index" not in shared_manifest.columns:
        shared_manifest["global_trial_index"] = np.nan
    shared_manifest["global_trial_index"] = shared_manifest["global_trial_index"].apply(safe_int)

    roi_exists_by_subject = {subject: roi_mask_exists(root, subject) for subject in SUBJECTS}
    group_cols = ["subject", "nsdId"]

    rows: list[dict[str, object]] = []
    missing_beta_count = 0
    repeat_position_counter: Counter = Counter()
    session_counter: Counter = Counter()

    for (subject, nsd_id), group in shared_manifest.groupby(group_cols, sort=True):
        group = group.sort_values(["rep_index_for_subject", "session", "trial_in_session"]).reset_index(drop=True)
        n_repeats = len(group)
        usable_beta_count = 0
        first_global = None
        if group["global_trial_index"].notna().any():
            first_global = int(group["global_trial_index"].dropna().min())
        else:
            first_session = safe_int(group.loc[0, "session"], 0) or 0
            first_trial = safe_int(group.loc[0, "trial_in_session"], 0) or 0
            first_global = (first_session - 1) * TRIALS_PER_SESSION + first_trial if first_session and first_trial else None

        for order_idx, row in group.iterrows():
            session = safe_int(row["session"])
            trial = safe_int(row["trial_in_session"])
            raw_rep_index = safe_int(row["rep_index_for_subject"], order_idx + 1) or (order_idx + 1)
            # Some manifest rows do not encode repeats as clean 1/2/3 positions.
            # For familiarity analyses, repeat_index means ordinal presentation order.
            rep_index = order_idx + 1
            run_index = (
                int(math.floor((trial - 1) / args.trials_per_run) + 1)
                if trial is not None and args.trials_per_run > 0
                else None
            )
            global_trial = safe_int(row.get("global_trial_index"))
            if global_trial is None and session is not None and trial is not None:
                global_trial = (session - 1) * TRIALS_PER_SESSION + trial
            time_gap = (
                int(global_trial - first_global)
                if global_trial is not None and first_global is not None
                else None
            )
            has_beta = beta_path_exists(root, row)
            has_roi = bool(roi_exists_by_subject.get(subject, False))
            if has_beta:
                usable_beta_count += 1
            else:
                missing_beta_count += 1
            if session is not None:
                session_counter[(subject, session)] += 1
            repeat_position_counter[rep_index] += 1

            rows.append(
                {
                    "subject": subject,
                    "nsdId": int(nsd_id),
                    "image_label_766": int(class_map[int(nsd_id)]),
                    "n_repeats_available": int(n_repeats),
                    "repeat_index": int(rep_index),
                    "raw_rep_index_for_subject": int(raw_rep_index),
                    "session_index": int(session) if session is not None else "",
                    "trial_index": int(trial) if trial is not None else "",
                    "run_index": int(run_index) if run_index is not None else "",
                    "time_gap_from_first_repeat": int(time_gap) if time_gap is not None else "",
                    "has_beta": bool(has_beta),
                    "has_roi_mask": bool(has_roi),
                    "usable_T2": bool(n_repeats >= 2 and order_idx < 2 and has_beta and has_roi),
                    "usable_T3": bool(n_repeats >= 3 and order_idx < 3 and has_beta and has_roi),
                }
            )

    inventory = pd.DataFrame(rows)
    nan_inf = numeric_nan_inf_counts(shared_manifest)

    per_subject: dict[str, dict[str, object]] = {}
    for subject in SUBJECTS:
        sub = inventory[inventory["subject"] == subject]
        image_repeats = sub.groupby("nsdId")["n_repeats_available"].max()
        per_subject[subject] = {
            "n_images_ge1": int((image_repeats >= 1).sum()),
            "n_images_ge2": int((image_repeats >= 2).sum()),
            "n_images_ge3": int((image_repeats >= 3).sum()),
            "n_trial_rows": int(len(sub)),
            "n_usable_T2_trial_rows": int(sub["usable_T2"].sum()),
            "n_usable_T3_trial_rows": int(sub["usable_T3"].sum()),
            "n_same_image_pairs_T2": int(sum(math.comb(min(int(v), 2), 2) for v in image_repeats)),
            "n_same_image_pairs_T3": int(sum(math.comb(min(int(v), 3), 2) for v in image_repeats)),
        }

    folds: dict[str, object] = {}
    for fold_name in args.folds:
        fold_def = fold_subjects(fold_name)
        fold_payload: dict[str, object] = {
            "test_subject": fold_def["test"],
            "val_subject": fold_def["val"],
            "train_subjects": fold_def["train"],
            "splits": {},
        }
        for split in ["train", "val", "test"]:
            if split == "train":
                split_subjects = set(fold_def["train"])  # type: ignore[arg-type]
            else:
                split_subjects = {str(fold_def[split])}
            split_df = inventory[inventory["subject"].isin(split_subjects)]
            grouped = split_df.groupby(["subject", "nsdId"])["n_repeats_available"].max()
            n_t2 = int((grouped >= 2).sum())
            n_t3 = int((grouped >= 3).sum())
            fold_payload["splits"][split] = {
                "n_subject_image_sequences_T2": n_t2,
                "n_subject_image_sequences_T3": n_t3,
                "n_single_trial_graphs_T2": int(split_df["usable_T2"].sum()),
                "n_single_trial_graphs_T3": int(split_df["usable_T3"].sum()),
                "n_same_image_pairs_T2": int(sum(math.comb(min(int(v), 2), 2) for v in grouped)),
                "n_same_image_pairs_T3": int(sum(math.comb(min(int(v), 3), 2) for v in grouped)),
            }
        t3_ok = all(
            fold_payload["splits"][split]["n_subject_image_sequences_T3"] >= args.min_t3_sequences_per_split
            for split in ["train", "val", "test"]
        )
        fold_payload["recommended_T"] = 3 if t3_ok else 2
        folds[fold_name] = fold_payload

    by_repeat_session = defaultdict(Counter)
    for row in rows:
        rep = row["repeat_index"]
        ses = row["session_index"]
        if ses != "":
            by_repeat_session[str(rep)][int(ses)] += 1

    summary: dict[str, object] = {
        "dataset_view": "all8_ge2_766",
        "n_subjects": len(SUBJECTS),
        "n_images": len(selected_nsd_ids),
        "n_inventory_rows": int(len(inventory)),
        "n_subject_image_groups": int(inventory.groupby(["subject", "nsdId"]).ngroups),
        "per_subject": per_subject,
        "folds": folds,
        "missing_beta_count": int(missing_beta_count),
        "nan_count": int(nan_inf["nan_count"]),
        "inf_count": int(nan_inf["inf_count"]),
        "session_distribution": {
            f"{subject}_session{session:02d}": int(count)
            for (subject, session), count in sorted(session_counter.items())
        },
        "session_distribution_by_repeat_index": {
            rep: counter_to_sorted_dict(counter) for rep, counter in sorted(by_repeat_session.items())
        },
        "repeat_position_distribution": counter_to_sorted_dict(repeat_position_counter),
        "roi_mask_exists_by_subject": roi_exists_by_subject,
        "run_index_note": (
            f"run_index is derived as floor((trial_in_session - 1) / {args.trials_per_run}) + 1 "
            "for inventory only; it is not used as ground-truth NSD run metadata."
        ),
        "decision_gate": {
            "min_t3_sequences_per_split": int(args.min_t3_sequences_per_split),
            "smoke_folds": args.folds,
            "recommendation": (
                "use_T3"
                if all(folds[f]["recommended_T"] == 3 for f in args.folds)
                else "use_T2_first"
            ),
        },
    }
    return inventory, summary


def main() -> None:
    args = parse_args()
    inventory, summary = build_inventory(args)
    output_dir = (args.root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory_path = output_dir / "repetition_inventory.csv"
    summary_path = output_dir / "repetition_inventory_summary.json"
    inventory.to_csv(inventory_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "inventory_path": str(inventory_path),
                "summary_path": str(summary_path),
                "n_rows": int(len(inventory)),
                "recommendation": summary["decision_gate"]["recommendation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
