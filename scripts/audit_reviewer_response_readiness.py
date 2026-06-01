#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuditRow:
    item: str
    status: str
    evidence: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit reviewer-response readiness for the publication manuscript.")
    parser.add_argument("--tex", type=Path, default=Path("reports/neurips_report/may30.tex"))
    parser.add_argument(
        "--final-tables-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument("--external-summary-dir", type=Path, default=Path("external_validation/summary"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument("--output-prefix", default="reviewer_response_readiness_audit")
    return parser.parse_args()


def ready(ok: bool) -> str:
    return "ready" if ok else "incomplete"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def has_all(text: str, fragments: tuple[str, ...]) -> bool:
    return all(fragment in text for fragment in fragments)


def has_rows(rows: list[dict[str, str]], column: str, values: tuple[str, ...]) -> bool:
    present = {row.get(column, "") for row in rows}
    return all(value in present for value in values)


def audit_source_set(tex_path: Path, final_tables_dir: Path, external_summary_dir: Path) -> AuditRow:
    required = [
        tex_path,
        final_tables_dir / "split_accounting.csv",
        final_tables_dir / "session_order_pair_qc.csv",
        final_tables_dir / "table_adjacency_ablation.csv",
        final_tables_dir / "table_roi_token_controls.csv",
        final_tables_dir / "publication_paired_stats.csv",
        final_tables_dir / "table_external_visual_roi_smoke.csv",
        external_summary_dir / "external_visual_roi_all4_summary.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    return AuditRow(
        "reviewer-response source artifacts",
        ready(not missing),
        "all reviewer-response source artifacts exist" if not missing else "; ".join(missing),
    )


def audit_dataset_accounting(final_tables_dir: Path) -> AuditRow:
    rows = read_csv(final_tables_dir / "split_accounting.csv")
    problems: list[str] = []
    if len(rows) != 8:
        problems.append(f"expected 8 folds, found {len(rows)}")
    if len({row.get("test_subject", "") for row in rows}) != 8:
        problems.append("test subjects are not unique")
    for row in rows:
        fold = row.get("fold", "?")
        for split in ("train", "val", "test"):
            seq = int(float(row.get(f"{split}_seq", "nan")))
            pairs = int(float(row.get(f"{split}_pairs", "nan")))
            if pairs != seq * 6:
                problems.append(f"{fold} {split}: pairs={pairs}, expected {seq * 6}")
    return AuditRow(
        "dataset accounting response",
        ready(not problems),
        "8 folds; unique held-out subjects; pair counts equal strict T=3 sequences x 6" if not problems else "; ".join(problems[:6]),
    )


def audit_session_order_controls(tex: str, final_tables_dir: Path) -> AuditRow:
    qc_rows = read_csv(final_tables_dir / "session_order_pair_qc.csv")
    eval_rows = read_csv(final_tables_dir / "single_ref_matched_summary.csv")
    retrain_rows = read_csv(final_tables_dir / "single_ref_matched_allseed_summary.csv")
    qc_ok = bool(qc_rows) and all(row.get("problem_groups") == "0" and row.get("anchor_match") == "100%" for row in qc_rows)
    single_ref_ok = bool(eval_rows) and bool(retrain_rows)
    text_ok = has_all(tex, ("anchor-side session/order control", "Single-reference session-matched control", "retrained the gated BNT/ReGraph variant"))
    ok = qc_ok and single_ref_ok and text_ok
    return AuditRow(
        "session/order confound response",
        ready(ok),
        "anchor-side QC is exact and both eval-only/retrained single-reference controls are present"
        if ok
        else f"qc_ok={qc_ok}; single_ref_ok={single_ref_ok}; text_ok={text_ok}",
    )


def audit_adjacency_response(tex: str, final_tables_dir: Path) -> AuditRow:
    adj_rows = read_csv(final_tables_dir / "table_adjacency_ablation.csv")
    stats_rows = read_csv(final_tables_dir / "final_adjacency_ablation_tests.csv")
    text_ok = has_all(
        tex,
        (
            "explicit fixed adjacency matrix is not the main driver",
            "it does not come from explicit fixed adjacency",
            "A learned edge-bias variant is competitive but does not improve over the no-adjacency gated ROI Transformer",
        ),
    )
    rows_ok = has_rows(
        adj_rows,
        "model",
        ("ROI-MLP+CLIP", "No-adj gated ROI Transformer+CLIP", "Gated ReGraph/BNT+CLIP"),
    )
    stats_ok = any(row.get("comparison") == "main_noadj_gated_vs_gated_regraph" for row in stats_rows)
    ok = text_ok and rows_ok and stats_ok
    return AuditRow(
        "graph-adjacency limitation response",
        ready(ok),
        "no-adj, adjacency, and ROI-MLP rows plus main no-adj-vs-adj paired tests support non-adjacency framing"
        if ok
        else f"text_ok={text_ok}; rows_ok={rows_ok}; stats_ok={stats_ok}",
    )


def audit_roi_token_mechanism(tex: str, final_tables_dir: Path) -> AuditRow:
    rows = read_csv(final_tables_dir / "table_roi_token_controls.csv")
    rows_ok = has_rows(rows, "model", ("No-adj gated ROI-T", "Uniform gate", "Random fixed gate", "ROI-order shuffle"))
    text_ok = has_all(tex, ("ROI-order shuffling collapses performance", "uniform or random gates substantially reduce performance", "fixed ROI-token layout"))
    ok = rows_ok and text_ok
    return AuditRow(
        "ROI-token/gate mechanism response",
        ready(ok),
        "ROI-order shuffle and gate controls are present and described as mechanism evidence" if ok else f"rows_ok={rows_ok}; text_ok={text_ok}",
    )


def audit_implementation_response(tex: str) -> AuditRow:
    required = (
        "\\mathcal{L}_{\\mathrm{BCE}}",
        "\\mathcal{L}_{\\mathrm{repeat\\ InfoNCE}}",
        "\\mathcal{L}_{\\mathrm{CLIP}}",
        "L2-normalized projected CLIP image embeddings",
        "all other images in the minibatch are negatives",
        "BC^\\top",
        "\\tau_{\\mathrm{clip}}",
        "hidden dimension 64",
        "2 layers",
        "4 heads",
        "AdamW",
        "batch size 128",
        "validation AUROC",
    )
    ok = has_all(tex, required)
    missing = [fragment for fragment in required if fragment not in tex]
    return AuditRow(
        "math/implementation detail response",
        ready(ok),
        "loss definitions and architecture/training table fragments present" if ok else "; ".join(missing),
    )


def audit_statistical_response(final_tables_dir: Path) -> AuditRow:
    rows = read_csv(final_tables_dir / "publication_paired_stats.csv")
    required_pairs = (
        ("main_allfold", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP"),
        ("main_allfold", "Gated ReGraph/BNT+CLIP - Flat ReGraph+CLIP"),
        ("hard_negative", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP"),
        ("component_baselines", "Gated ReGraph/BNT+CLIP - MindLink-style subject-adversarial ROI-MLP"),
        ("single_ref_retrained", "No-adj gated ROI Transformer+CLIP - ROI-MLP+CLIP"),
    )
    present = {(row.get("setting", ""), row.get("comparison", "")) for row in rows}
    missing = [f"{setting}/{comparison}" for setting, comparison in required_pairs if (setting, comparison) not in present]
    numeric_ok = bool(rows) and all(row.get("n") and row.get("mean_diff") and row.get("bootstrap_ci_low") for row in rows)
    ok = not missing and numeric_ok
    return AuditRow(
        "statistical-reporting response",
        ready(ok),
        "paired tests with bootstrap CIs cover main, flat, hard-negative, component, and single-reference comparisons"
        if ok
        else f"missing={missing[:4]}; numeric_ok={numeric_ok}",
    )


def audit_component_baseline_response(tex: str, final_tables_dir: Path) -> AuditRow:
    rows = read_csv(final_tables_dir / "table_phase2_sota_graph_baselines.csv")
    text_ok = has_all(tex, ("task-matched component baselines", "not full image-reconstruction systems", "not full image-reconstruction system comparisons"))
    rows_ok = has_rows(rows, "model", ("MindEye2-style shared ROI mapper", "UMBRAE-style subject encoder", "MindLink-style subject-adversarial ROI-MLP"))
    ok = text_ok and rows_ok
    return AuditRow(
        "component-baseline framing response",
        ready(ok),
        "component baselines are framed as task-matched, not full-system SOTA comparisons" if ok else f"text_ok={text_ok}; rows_ok={rows_ok}",
    )


def audit_semantic_alignment_response(tex: str, final_tables_dir: Path) -> AuditRow:
    rows = read_csv(final_tables_dir / "table_heldout_image.csv")
    rows_ok = has_rows(rows, "model", ("Gated ReGraph/BNT+CLIP", "Gated random embedding"))
    text_ok = has_all(tex, ("pair discrimination", "image/brain retrieval", "real CLIP semantics", "random image embeddings remain competitive"))
    ok = rows_ok and text_ok
    return AuditRow(
        "semantic-alignment response",
        ready(ok),
        "held-out-image CLIP/random controls separate pair discrimination from image/brain retrieval"
        if ok
        else f"rows_ok={rows_ok}; text_ok={text_ok}",
    )


def audit_external_validation_response(tex: str, final_tables_dir: Path, external_summary_dir: Path) -> AuditRow:
    rows = read_csv(final_tables_dir / "table_external_visual_roi_smoke.csv")
    datasets = {row.get("model", "").split(" visual ROI")[0] for row in rows if " visual ROI" in row.get("model", "")}
    text_ok = has_all(
        tex,
        (
            "not full HCP-MMP external replications",
            "external feasibility evidence",
            "not a full external replication",
            "broader validation on independent trial-wise atlas-ROI beta maps remains needed",
        ),
    )
    summary_ok = (external_summary_dir / "external_visual_roi_all4_summary.md").exists()
    rows_ok = {"BOLD5000", "CNeuroMod", "THINGS-fMRI", "LAION-fMRI"}.issubset(datasets)
    ok = text_ok and summary_ok and rows_ok
    return AuditRow(
        "external-validation response",
        ready(ok),
        "four public visual-ROI smoke checks are present and explicitly limited as feasibility evidence"
        if ok
        else f"text_ok={text_ok}; summary_ok={summary_ok}; datasets={sorted(datasets)}",
    )


def audit_fold07_response(tex: str, final_tables_dir: Path) -> AuditRow:
    rows = read_csv(final_tables_dir / "fold_difficulty_qc.csv")
    fold07_ok = any(row.get("fold") == "fold_07" and row.get("test_subject") == "subj07" for row in rows)
    text_ok = has_all(tex, ("fold\\_07 remains the hardest", "not a simple sample-count artifact", "unresolved robustness case"))
    ok = fold07_ok and text_ok
    return AuditRow(
        "fold_07 robustness response",
        ready(ok),
        "fold_07 has a QC row and remains framed as an unresolved robustness case" if ok else f"fold07_ok={fold07_ok}; text_ok={text_ok}",
    )


def write_outputs(output_dir: Path, output_prefix: str, rows: list[AuditRow]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{output_prefix}.csv"
    md_path = output_dir / f"{output_prefix}.md"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item", "status", "evidence"], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({"item": row.item, "status": row.status, "evidence": row.evidence})
    counts = {status: sum(1 for row in rows if row.status == status) for status in sorted({row.status for row in rows})}
    lines = [
        "# Reviewer Response Readiness Audit",
        "",
        "This audit maps likely reviewer concerns to manuscript text and committed result artifacts.",
        "",
        f"Status counts: {counts}",
        "",
        "| Item | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row.item} | {row.status} | {row.evidence} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"), end="")


def main() -> int:
    args = parse_args()
    tex = read_text(args.tex)
    rows = [
        audit_source_set(args.tex, args.final_tables_dir, args.external_summary_dir),
        audit_dataset_accounting(args.final_tables_dir),
        audit_session_order_controls(tex, args.final_tables_dir),
        audit_adjacency_response(tex, args.final_tables_dir),
        audit_roi_token_mechanism(tex, args.final_tables_dir),
        audit_implementation_response(tex),
        audit_statistical_response(args.final_tables_dir),
        audit_component_baseline_response(tex, args.final_tables_dir),
        audit_semantic_alignment_response(tex, args.final_tables_dir),
        audit_external_validation_response(tex, args.final_tables_dir, args.external_summary_dir),
        audit_fold07_response(tex, args.final_tables_dir),
    ]
    write_outputs(args.output_dir, args.output_prefix, rows)
    return 0 if all(row.status == "ready" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
