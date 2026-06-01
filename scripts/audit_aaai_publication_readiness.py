#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CheckResult:
    item: str
    status: str
    evidence: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit publication-facing artifacts for the AAAI-style ReGraph-VLM story.")
    parser.add_argument(
        "--final-tables-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument(
        "--external-summary-dir",
        type=Path,
        default=Path("external_validation/summary"),
    )
    parser.add_argument("--output-prefix", default="aaai_publication_readiness_audit")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def has_rows(df: pd.DataFrame, filters: dict[str, object], min_n: int | None = None) -> tuple[bool, str]:
    if df.empty:
        return False, "file empty or missing"
    sub = df.copy()
    for col, value in filters.items():
        if col not in sub.columns:
            return False, f"missing column {col}"
        if isinstance(value, float):
            sub = sub[(sub[col].astype(float) - value).abs() < 1e-8]
        else:
            sub = sub[sub[col] == value]
    if sub.empty:
        return False, f"no rows match {filters}"
    if min_n is not None:
        n = len(sub)
        if n < min_n:
            return False, f"{n} matching rows, expected at least {min_n}"
    return True, f"{len(sub)} matching rows"


def support_n(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    if "n" in df.columns:
        values = pd.to_numeric(df["n"], errors="coerce").dropna()
        if not values.empty:
            return int(values.sum())
    return len(df)


def finite_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.map(lambda value: pd.notna(value) and math.isfinite(float(value)))


def check_table(
    item: str,
    path: Path,
    filters: dict[str, object] | None = None,
    min_n: int | None = None,
) -> CheckResult:
    if not path.exists():
        return CheckResult(item, "missing", f"{path} not found")
    df = read_csv(path)
    if filters:
        ok, evidence = has_rows(df, filters, min_n=min_n)
        return CheckResult(item, "ready" if ok else "incomplete", f"{path.name}: {evidence}")
    if df.empty:
        return CheckResult(item, "incomplete", f"{path.name}: empty")
    if min_n is not None:
        support = support_n(df)
        ok = support >= min_n
        return CheckResult(
            item,
            "ready" if ok else "incomplete",
            f"{path.name}: {len(df)} rows, support n={support}, expected at least {min_n}",
        )
    return CheckResult(item, "ready", f"{path.name}: {len(df)} rows")


def check_split_invariants(path: Path) -> CheckResult:
    item = "split accounting invariants"
    if not path.exists():
        return CheckResult(item, "missing", f"{path} not found")
    df = read_csv(path)
    required = {"fold", "test_subject", "val_subject", "train_seq", "val_seq", "test_seq", "train_pairs", "val_pairs", "test_pairs", "test_imgs"}
    if df.empty or not required.issubset(df.columns):
        return CheckResult(item, "incomplete", f"{path.name}: missing expected columns")
    problems: list[str] = []
    if len(df) != 8:
        problems.append(f"fold rows={len(df)}, expected 8")
    if df["test_subject"].nunique() != 8:
        problems.append(f"unique test subjects={df['test_subject'].nunique()}, expected 8")
    if (df["test_subject"] == df["val_subject"]).any():
        problems.append("validation subject overlaps test subject")
    for split in ("train", "val", "test"):
        seq = pd.to_numeric(df[f"{split}_seq"], errors="coerce")
        pairs = pd.to_numeric(df[f"{split}_pairs"], errors="coerce")
        if not pairs.eq(seq * 6).all():
            problems.append(f"{split}_pairs do not equal {split}_seq x 6")
    test_seq = pd.to_numeric(df["test_seq"], errors="coerce")
    test_imgs = pd.to_numeric(df["test_imgs"], errors="coerce")
    if not test_imgs.eq(test_seq).all():
        problems.append("test_imgs does not equal test_seq")
    return CheckResult(
        item,
        "ready" if not problems else "incomplete",
        f"{path.name}: 8 unique held-out folds; strict T=3 pair-count invariant holds" if not problems else "; ".join(problems[:6]),
    )


def check_session_qc_invariants(path: Path) -> CheckResult:
    item = "session/order QC invariants"
    if not path.exists():
        return CheckResult(item, "missing", f"{path} not found")
    df = read_csv(path)
    required = {"split", "pairs", "positive", "negative", "complete_groups", "problem_groups", "anchor_match"}
    if df.empty or not required.issubset(df.columns):
        return CheckResult(item, "incomplete", f"{path.name}: missing expected columns")
    problems: list[str] = []
    by_split = {row["split"]: row for row in df.to_dict("records")}
    if {"Train", "Val", "Test", "All"} - set(by_split):
        problems.append("expected Train/Val/Test/All rows")
    for split, row in by_split.items():
        pairs = int(row["pairs"])
        positive = int(row["positive"])
        negative = int(row["negative"])
        complete_groups = int(row["complete_groups"])
        problem_groups = int(row["problem_groups"])
        if pairs != positive + negative:
            problems.append(f"{split}: pairs != positive + negative")
        if positive != negative:
            problems.append(f"{split}: positive != negative")
        if complete_groups != positive:
            problems.append(f"{split}: complete_groups != positive")
        if problem_groups != 0:
            problems.append(f"{split}: problem_groups={problem_groups}")
        if row["anchor_match"] != "100%":
            problems.append(f"{split}: anchor_match={row['anchor_match']}")
    if "All" in by_split:
        for column in ("pairs", "positive", "negative", "complete_groups", "problem_groups"):
            observed = int(by_split["All"][column])
            expected = sum(int(by_split[split][column]) for split in ("Train", "Val", "Test") if split in by_split)
            if observed != expected:
                problems.append(f"All {column}={observed}, expected {expected}")
    return CheckResult(
        item,
        "ready" if not problems else "incomplete",
        f"{path.name}: balanced positives/negatives, 100% anchor match, split totals consistent" if not problems else "; ".join(problems[:6]),
    )


def check_publication_stats(path: Path, setting: str, comparison_substr: str, min_metrics: int = 3) -> CheckResult:
    item = f"paired stats: {setting} / {comparison_substr}"
    if not path.exists():
        return CheckResult(item, "missing", f"{path} not found")
    df = read_csv(path)
    if df.empty or "setting" not in df.columns or "comparison" not in df.columns:
        return CheckResult(item, "incomplete", f"{path.name}: missing expected columns")
    sub = df[(df["setting"] == setting) & (df["comparison"].str.contains(comparison_substr, regex=False))]
    if len(sub) < min_metrics:
        return CheckResult(item, "incomplete", f"{path.name}: {len(sub)} metrics, expected at least {min_metrics}")
    required = ["n", "mean_diff", "std_diff", "bootstrap_ci_low", "bootstrap_ci_high", "paired_t_p"]
    if not set(required).issubset(sub.columns):
        return CheckResult(item, "incomplete", f"{path.name}: missing numeric columns")
    problems: list[str] = []
    for column in required[:-1]:
        mask = finite_numeric(sub[column])
        if not mask.all():
            problems.append(f"{column} invalid={int((~mask).sum())}")
    p_values = pd.to_numeric(sub["paired_t_p"], errors="coerce")
    p_finite = finite_numeric(sub["paired_t_p"])
    zero_tie = pd.to_numeric(sub["mean_diff"], errors="coerce").eq(0.0) & pd.to_numeric(sub["std_diff"], errors="coerce").eq(0.0)
    if ((~p_finite) & ~zero_tie).any():
        problems.append(f"paired_t_p invalid={int(((~p_finite) & ~zero_tie).sum())}")
    if not pd.to_numeric(sub["n"], errors="coerce").gt(0).all():
        problems.append("n must be positive")
    if not pd.to_numeric(sub["std_diff"], errors="coerce").ge(0).all():
        problems.append("std_diff must be nonnegative")
    if not pd.to_numeric(sub["bootstrap_ci_low"], errors="coerce").le(pd.to_numeric(sub["bootstrap_ci_high"], errors="coerce")).all():
        problems.append("bootstrap CI low must be <= high")
    if not p_values[p_finite].between(0.0, 1.0).all():
        problems.append("paired_t_p must be in [0, 1]")
    tie_note = f", exact-zero ties={int(((~p_finite) & zero_tie).sum())}" if ((~p_finite) & zero_tie).any() else ""
    return CheckResult(
        item,
        "ready" if not problems else "incomplete",
        f"{path.name}: {len(sub)} numeric paired metric rows valid{tie_note}" if not problems else "; ".join(problems[:6]),
    )


def check_model_parameter_counts(path: Path) -> CheckResult:
    item = "model parameter counts"
    if not path.exists():
        return CheckResult(item, "missing", f"{path} not found")
    df = read_csv(path)
    required = {"model", "graph_encoder", "readout", "trainable_parameters", "source"}
    if df.empty or not required.issubset(df.columns):
        return CheckResult(item, "incomplete", f"{path.name}: missing expected columns")
    expected = {
        "Final gated BNT/ReGraph+CLIP": 3251595,
        "No-adj gated ROI Transformer+CLIP": 3251595,
        "Graph-bias BNT/ReGraph+CLIP": 3255883,
        "Learned edge-bias BNT+CLIP": 3283995,
        "ROI-MLP+CLIP": 369706,
    }
    rows = {str(row.get("model")): row for row in df.to_dict("records")}
    problems: list[str] = []
    for model, expected_count in expected.items():
        if model not in rows:
            problems.append(f"missing {model}")
            continue
        observed = int(float(rows[model]["trainable_parameters"]))
        if observed != expected_count:
            problems.append(f"{model}: {observed} != {expected_count}")
        source = str(rows[model].get("source", ""))
        if "tab:implementation_details" not in source:
            problems.append(f"{model}: source does not cite implementation table")
    return CheckResult(
        item,
        "ready" if not problems else "incomplete",
        f"{path.name}: {len(expected)} expected model counts verified" if not problems else "; ".join(problems[:6]),
    )




def check_text_file(item: str, path: Path, required_text: str | None = None) -> CheckResult:
    if not path.exists():
        return CheckResult(item, "missing", f"{path} not found")
    text = path.read_text(encoding="utf-8", errors="replace")
    if required_text and required_text not in text:
        return CheckResult(item, "incomplete", f"{path.name}: missing marker {required_text!r}")
    return CheckResult(item, "ready", f"{path.name}: {len(text.splitlines())} lines")


def write_outputs(out_prefix: Path, rows: list[CheckResult]) -> None:
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item", "status", "evidence"], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({"item": row.item, "status": row.status, "evidence": row.evidence})

    counts = pd.Series([row.status for row in rows]).value_counts().to_dict()
    lines = [
        "# AAAI Publication Readiness Audit",
        "",
        "This audit checks whether the current publication-facing result artifacts exist and contain the expected evidence. It does not replace manuscript review.",
        "",
        f"Status counts: {counts}",
        "",
        "| Item | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row.item} | {row.status} | {row.evidence} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    final = args.final_tables_dir
    stats = final / "publication_paired_stats.csv"
    rows = [
        check_table("split accounting table", final / "split_accounting.csv"),
        check_split_invariants(final / "split_accounting.csv"),
        check_table("within-subject smoke table", final / "table_within_subject.csv"),
        check_table("main all-fold final table", final / "table_allfold_final.csv", min_n=24),
        check_table("hard-negative all-fold table", final / "table_hard_negative_allfold.csv", min_n=24),
        check_table("held-out-image table", final / "table_heldout_image.csv", min_n=24),
        check_model_parameter_counts(final / "model_parameter_counts.csv"),
        check_table("component baseline table", final / "table_phase2_sota_graph_baselines.csv", min_n=100),
        check_table("graph-only CLIP ablation table", final / "table_graph_only.csv", min_n=48),
        check_table("adjacency ablation table", final / "table_adjacency_ablation.csv"),
        check_table("ROI-token control table", final / "table_roi_token_controls.csv"),
        check_table("static adjacency perturbation table", final / "table_adjacency_perturbation.csv"),
        check_table("edge-bias follow-up table", final / "table_edge_bias_followup.csv"),
        check_table("session/order pair QC", final / "session_order_pair_qc.csv"),
        check_session_qc_invariants(final / "session_order_pair_qc.csv"),
        check_table("fold difficulty QC", final / "fold_difficulty_qc.csv"),
        check_table("single-reference eval-existing summary", final / "single_ref_matched_summary.csv", min_n=72),
        check_table("single-reference retrained summary", final / "single_ref_matched_allseed_summary.csv", min_n=72),
        check_table("low-shot calibration table", final / "table_lowshot_calibration.csv", min_n=120),
        check_table("external visual-ROI smoke table", final / "table_external_visual_roi_smoke.csv", min_n=100),
        check_table("gate confound table", final / "table_gate_confound.csv"),
        check_table("matched deletion table", final / "table_matched_deletion.csv"),
        check_text_file(
            "single-reference eval-existing LaTeX rows",
            final / "single_ref_matched_latex.txt",
            "Auto-generated by scripts/summarize_single_ref_matched_results.py",
        ),
        check_text_file(
            "single-reference retrained LaTeX rows",
            final / "single_ref_matched_allseed_latex.txt",
            "Auto-generated by scripts/summarize_single_ref_matched_results.py",
        ),
        check_publication_stats(stats, "main_allfold", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP"),
        check_publication_stats(stats, "hard_negative", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP"),
        check_publication_stats(stats, "heldout_real_vs_random_available_raw", "Gated ReGraph/BNT+CLIP - Gated random embedding"),
        check_publication_stats(stats, "component_baselines", "MindLink-style subject-adversarial ROI-MLP"),
        check_publication_stats(stats, "single_ref_retrained", "No-adj gated ROI Transformer+CLIP - ROI-MLP+CLIP"),
        check_text_file("AAAI ROI-token story summary", final / "aaai_roi_token_story_summary.md", "Recommended claim"),
        check_text_file(
            "external visual-ROI smoke summary",
            args.external_summary_dir / "external_visual_roi_all4_summary.md",
            "not full HCP-MMP",
        ),
        check_text_file(
            "LAION trial-wise external validation summary",
            args.external_summary_dir / "laion_fmri_visual_roi_summary.md",
            "Trial-wise public LAION-fMRI beta maps",
        ),
        check_text_file(
            "LAION trial-wise external validation LaTeX rows",
            args.external_summary_dir / "laion_fmri_visual_roi_latex.txt",
            "LAION-fMRI external visual-ROI validation",
        ),
    ]
    write_outputs(final / args.output_prefix, rows)


if __name__ == "__main__":
    main()
