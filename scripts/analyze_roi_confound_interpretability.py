#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


VISUAL_MODULES = {
    6: ("visual_intermediate", "V4"),
    7: ("visual_intermediate", "V8"),
    18: ("ventral_object", "FFC"),
    20: ("lateral_object", "LO1"),
    21: ("lateral_object", "LO2"),
    22: ("ventral_object", "PIT"),
    154: ("ventral_visual", "VMV3"),
    158: ("dorsal_lateral_visual", "V3CD"),
    160: ("ventral_visual", "VMV2"),
    163: ("ventral_visual", "VVC"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROI reliability/confound controls for gate interpretation.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--sequence-root",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3_allfold"),
    )
    parser.add_argument(
        "--gate-summary",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/interpretability/gates_noadj/roi_gate_summary.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/interpretability/roi_confound_controls"),
    )
    return parser.parse_args()


def corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    a = a[ok]
    b = b[ok]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return corr(rankdata(a[ok]), rankdata(b[ok]))


def residualize(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    ok = np.isfinite(y) & np.isfinite(controls).all(axis=1)
    out = np.full_like(y, np.nan, dtype=float)
    if ok.sum() < controls.shape[1] + 3:
        return out
    x = controls[ok]
    x = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x, y[ok], rcond=None)
    out[ok] = y[ok] - x @ beta
    return out


def partial_spearman(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> float:
    rx = rankdata(x)
    ry = rankdata(y)
    rc = np.column_stack([rankdata(controls[:, i]) for i in range(controls.shape[1])])
    return corr(residualize(rx, rc), residualize(ry, rc))


def compute_roi_properties(root: Path, sequence_root: Path) -> pd.DataFrame:
    rows = []
    by_roi_values: dict[int, list[np.ndarray]] = {roi: [] for roi in range(180)}
    for fold_dir in sorted((root / sequence_root).glob("fold_*")):
        for split in ["train_sequences.pt", "val_sequences.pt", "test_sequences.pt"]:
            path = fold_dir / split
            if not path.exists():
                continue
            seqs = torch.load(path, map_location="cpu", weights_only=False)
            for seq in seqs:
                x = seq["x_seq"].float().numpy()
                for roi0 in range(x.shape[1]):
                    by_roi_values[roi0].append(x[:, roi0, :])
    for roi0, chunks in by_roi_values.items():
        arr = np.stack(chunks, axis=0)
        mean_beta = arr[:, :, 0]
        std_beta = arr[:, :, 1]
        q90_beta = arr[:, :, 2]
        pos_frac = arr[:, :, 3]
        d21 = mean_beta[:, 1] - mean_beta[:, 0]
        d31 = mean_beta[:, 2] - mean_beta[:, 0]
        d32 = mean_beta[:, 2] - mean_beta[:, 1]
        reliability = np.nanmean([corr(mean_beta[:, 0], mean_beta[:, 1]), corr(mean_beta[:, 0], mean_beta[:, 2]), corr(mean_beta[:, 1], mean_beta[:, 2])])
        rows.append(
            {
                "roi_id": roi0 + 1,
                "n_sequences": int(arr.shape[0]),
                "mean_beta_mean": float(np.nanmean(mean_beta)),
                "mean_beta_abs_mean": float(np.nanmean(np.abs(mean_beta))),
                "mean_beta_variance": float(np.nanvar(mean_beta)),
                "std_beta_mean": float(np.nanmean(std_beta)),
                "q90_beta_mean": float(np.nanmean(q90_beta)),
                "positive_frac_mean": float(np.nanmean(pos_frac)),
                "repeat_reliability_mean_beta": float(reliability),
                "abs_delta21_mean": float(np.nanmean(np.abs(d21))),
                "abs_delta31_mean": float(np.nanmean(np.abs(d31))),
                "abs_delta32_mean": float(np.nanmean(np.abs(d32))),
                "mean_delta21": float(np.nanmean(d21)),
                "mean_delta31": float(np.nanmean(d31)),
                "mean_delta32": float(np.nanmean(d32)),
            }
        )
    return pd.DataFrame(rows)


def add_modules(df: pd.DataFrame) -> pd.DataFrame:
    modules = []
    labels = []
    for roi_id in df["roi_id"].astype(int):
        module, label = VISUAL_MODULES.get(roi_id, ("other_or_unlabeled", f"ROI {roi_id}"))
        modules.append(module)
        labels.append(label)
    out = df.copy()
    out["approx_module"] = modules
    out["approx_label"] = labels
    out["is_named_visual_roi"] = out["approx_module"].ne("other_or_unlabeled").astype(int)
    return out


def matched_sets(merged: pd.DataFrame, ks: list[int], repeats: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(20260523)
    prop = merged.copy()
    for col in ["repeat_reliability_mean_beta", "mean_beta_variance", "positive_frac_mean"]:
        prop[f"{col}_bin"] = pd.qcut(prop[col].rank(method="first"), q=4, labels=False, duplicates="drop")
    rows = []
    for k in ks:
        top = prop.sort_values("gate_mean", ascending=False).head(k)
        top_ids = set(top["roi_id"].astype(int))
        for repeat in range(1, repeats + 1):
            chosen = []
            used = set(top_ids)
            for _, target in top.iterrows():
                pool = prop[
                    (prop["roi_id"].astype(int).map(lambda x: x not in used))
                    & (prop["repeat_reliability_mean_beta_bin"].eq(target["repeat_reliability_mean_beta_bin"]))
                    & (prop["mean_beta_variance_bin"].eq(target["mean_beta_variance_bin"]))
                    & (prop["positive_frac_mean_bin"].eq(target["positive_frac_mean_bin"]))
                ]
                if pool.empty:
                    pool = prop[prop["roi_id"].astype(int).map(lambda x: x not in used)]
                pick = int(pool.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1)))["roi_id"].iloc[0])
                chosen.append(pick)
                used.add(pick)
            rows.append({"k": k, "repeat": repeat, "mode": "matched_property_random", "roi_ids": " ".join(map(str, chosen))})
        rows.append({"k": k, "repeat": 0, "mode": "top_gate", "roi_ids": " ".join(map(str, sorted(top_ids)))})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    props = add_modules(compute_roi_properties(root, args.sequence_root))
    gates = pd.read_csv(root / args.gate_summary)
    merged = props.merge(gates, on="roi_id", how="inner")

    props.to_csv(out_dir / "roi_property_table.csv", index=False)
    merged.to_csv(out_dir / "roi_gate_property_table.csv", index=False)

    controls = merged[["repeat_reliability_mean_beta", "mean_beta_variance", "positive_frac_mean"]].to_numpy(float)
    corr_rows = []
    for target in ["abs_delta21_mean", "abs_delta31_mean", "abs_delta32_mean"]:
        corr_rows.append(
            {
                "target": target,
                "spearman_gate_vs_target": spearman(merged["gate_mean"].to_numpy(float), merged[target].to_numpy(float)),
                "partial_spearman_control_reliability_variance_posfrac": partial_spearman(
                    merged["gate_mean"].to_numpy(float), merged[target].to_numpy(float), controls
                ),
                "controls": "repeat_reliability_mean_beta,mean_beta_variance,positive_frac_mean",
                "n_roi": int(len(merged)),
            }
        )
    partial = pd.DataFrame(corr_rows)
    partial.to_csv(out_dir / "gate_partial_correlations.csv", index=False)

    module = (
        merged.groupby("approx_module")
        .agg(
            n=("roi_id", "count"),
            gate_mean=("gate_mean", "mean"),
            gate_std=("gate_mean", "std"),
            abs_delta31_mean=("abs_delta31_mean", "mean"),
            repeat_reliability_mean_beta=("repeat_reliability_mean_beta", "mean"),
        )
        .reset_index()
        .sort_values("gate_mean", ascending=False)
    )
    module.to_csv(out_dir / "module_gate_summary.csv", index=False)

    sets = matched_sets(merged, [5, 10, 20, 40], repeats=20)
    sets.to_csv(out_dir / "matched_deletion_roi_sets.csv", index=False)

    md = [
        "# ROI Confound-Control Interpretability Summary",
        "",
        "## Partial Correlations",
        "",
        partial.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Approximate Module Summary",
        "",
        module.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Notes: module labels are approximate and only assigned for the verified visual ROI IDs used in the report table; all other ROIs are grouped as other_or_unlabeled.",
        "",
    ]
    (out_dir / "roi_confound_interpretability_summary.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "run_manifest.json").write_text(json.dumps({"gate_summary": str(args.gate_summary), "sequence_root": str(args.sequence_root)}, indent=2), encoding="utf-8")
    print({"out": str(out_dir), "n_roi": len(merged)})


if __name__ == "__main__":
    main()
