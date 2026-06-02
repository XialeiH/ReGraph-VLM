#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TableSpec:
    label: str
    csv_name: str
    metrics: tuple[str, ...]
    raw_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuditRow:
    item: str
    status: str
    evidence: str


TABLE_SPECS = [
    TableSpec("tab:split_accounting", "split_accounting.csv", (), ("train_seq", "val_seq", "test_seq", "train_pairs", "val_pairs", "test_pairs", "test_imgs")),
    TableSpec("tab:session_order_pair_qc", "session_order_pair_qc.csv", (), ("pairs", "positive", "negative", "complete_groups", "problem_groups", "anchor_match")),
    TableSpec("tab:within_subject", "table_within_subject.csv", (), ("AUROC", "AUPRC", "R@5", "MRR")),
    TableSpec("tab:cross_subject_main", "table_allfold_final.csv", ("AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "brain_R@5")),
    TableSpec("tab:sota_baselines", "table_phase2_sota_graph_baselines.csv", ("AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "brain_R@5")),
    TableSpec("tab:graph_only", "table_graph_only.csv", ("AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "brain_R@5")),
    TableSpec("tab:adjacency_ablation", "table_adjacency_ablation.csv", ("AUROC", "AUPRC", "R@5", "MRR", "brain_R@5")),
    TableSpec("tab:roi_token_controls", "table_roi_token_controls.csv", ("AUROC", "AUPRC", "R@5", "MRR", "brain_R@5")),
    TableSpec("tab:adjacency_perturbation", "table_adjacency_perturbation.csv", ("AUROC", "AUPRC", "R@5", "MRR", "brain_R@5")),
    TableSpec("tab:edge_bias_followup", "table_edge_bias_followup.csv", ("AUROC", "AUPRC", "R@5", "brain_R@5")),
    TableSpec("tab:single_ref_matched", "single_ref_matched_summary.csv", ("AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "brain_R@5", "brain_MRR")),
    TableSpec("tab:single_ref_retrained", "single_ref_matched_allseed_summary.csv", ("AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "brain_R@5", "brain_MRR")),
    TableSpec("tab:heldout", "table_heldout_image.csv", ("AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "brain_R@5")),
    TableSpec("tab:hardneg", "table_hard_negative_allfold.csv", ("AUROC", "AUPRC", "R@5", "MRR", "image_R@5", "brain_R@5", "brain_MRR")),
    TableSpec("tab:lowshot", "table_lowshot_calibration.csv", ("image_R@5", "image_MRR"), ("n",)),
    TableSpec("tab:external_visual_roi_smoke", "table_external_visual_roi_smoke.csv", ("AUROC", "AUPRC", "R@5", "MRR"), ("n",)),
    TableSpec("tab:gate_confound", "table_gate_confound.csv", (), ("spearman", "partial_spearman", "n_roi")),
    TableSpec("tab:matched_deletion", "table_matched_deletion.csv", (), ("k", "AUROC_drop", "R@5_drop", "brain_R@5_drop")),
    TableSpec("tab:fold_difficulty", "fold_difficulty_qc.csv", (), ("test_seq", "repeat_corr", "raw_AUROC", "raw_gap", "model_AUROC", "brain_R@5")),
]

MANUAL_TABLE_LABELS = {
    "tab:implementation_details": "architecture/protocol table without numeric result artifact",
    "tab:laion_external_pairwise": "external LAION paired-test table audited against external summary artifact",
    "tab:top_gated_roi_names": "manual ROI-name interpretation table in neuroscience section",
}

CAPTION_REQUIREMENTS = {
    "tab:split_accounting": ("Sequence counts", "pair counts"),
    "tab:session_order_pair_qc": ("Session/order QC", "same anchor trial"),
    "tab:implementation_details": ("Implementation and training details",),
    "tab:within_subject": ("available smoke folds and seeds",),
    "tab:cross_subject_main": ("mean $\\pm$ std", "8 held-out-subject folds and 3 random seeds"),
    "tab:sota_baselines": ("mean $\\pm$ std", "not full image-reconstruction system comparisons"),
    "tab:graph_only": ("mean $\\pm$ std", "8 held-out-subject folds and 3 random seeds"),
    "tab:adjacency_ablation": ("mean $\\pm$ std", "fold$\\times$seed"),
    "tab:roi_token_controls": ("mean $\\pm$ std", "fold$\\times$seed"),
    "tab:adjacency_perturbation": ("mean $\\pm$ std", "available diagnostic runs"),
    "tab:edge_bias_followup": ("mean $\\pm$ std", "$n$ gives"),
    "tab:single_ref_matched": ("mean $\\pm$ std", "$n=24$"),
    "tab:single_ref_retrained": ("mean $\\pm$ std", "$n=24$"),
    "tab:heldout": ("mean $\\pm$ std", "8 folds $\\times$ 3 seeds"),
    "tab:hardneg": ("mean $\\pm$ std where available", "point summaries"),
    "tab:lowshot": ("mean $\\pm$ std", "$n$ fold$\\times$seed"),
    "tab:external_visual_roi_smoke": ("mean $\\pm$ std", "not full HCP-MMP 180-ROI external validations"),
    "tab:laion_external_pairwise": ("paired differences", "bootstrap 95\\% confidence intervals", "paired-test $p$-values"),
    "tab:gate_confound": ("before and after controlling",),
    "tab:matched_deletion": ("Drops are absolute performance decreases",),
    "tab:fold_difficulty": ("Fold-level robustness diagnostic", "computed from Pearson correlations"),
    "tab:top_gated_roi_names": ("Top gated ROIs", "approximate HCP-MMP"),
}

MODEL_ALIASES = {
    ("tab:cross_subject_main", "Gated ReGraph/BNT+CLIP"): "Gated ReGraph+CLIP",
    ("tab:heldout", "Gated ReGraph/BNT+CLIP"): "Gated ReGraph+CLIP",
    ("tab:hardneg", "Gated ReGraph/BNT+CLIP"): "Gated ReGraph+CLIP",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit that key may30.tex table numbers match committed CSV artifacts.")
    parser.add_argument("--tex", type=Path, default=Path("reports/neurips_report/may30.tex"))
    parser.add_argument(
        "--final-tables-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument("--external-summary-dir", type=Path, default=Path("external_validation/summary"))
    parser.add_argument("--output-prefix", default="manuscript_table_values_audit")
    return parser.parse_args()


def ready(ok: bool) -> str:
    return "ready" if ok else "incomplete"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_table_block(tex: str, label: str) -> str:
    label_marker = f"\\label{{{label}}}"
    idx = tex.find(label_marker)
    if idx < 0:
        return ""
    start = tex.rfind("\\begin{table", 0, idx)
    end = tex.find("\\end{table}", idx)
    if start < 0 or end < 0:
        return ""
    return tex[start : end + len("\\end{table}")]


def manuscript_table_labels(tex: str) -> set[str]:
    return set(re.findall(r"\\label\{(tab:[^}]+)\}", tex))


def audit_table_coverage(tex: str) -> AuditRow:
    manuscript_labels = manuscript_table_labels(tex)
    audited_labels = {spec.label for spec in TABLE_SPECS}
    classified_labels = audited_labels | set(MANUAL_TABLE_LABELS)
    missing = sorted(manuscript_labels - classified_labels)
    stale = sorted(classified_labels - manuscript_labels)
    if missing:
        return AuditRow("table audit coverage", "incomplete", "unclassified manuscript tables: " + ", ".join(missing))
    if stale:
        return AuditRow("table audit coverage", "incomplete", "classified labels not in manuscript: " + ", ".join(stale))
    evidence = f"{len(audited_labels)} artifact-backed tables, {len(MANUAL_TABLE_LABELS)} manual tables classified"
    return AuditRow("table audit coverage", "ready", evidence)


def table_caption(tex: str, label: str) -> str:
    block = find_table_block(tex, label)
    if not block:
        return ""
    match = re.search(r"\\caption\{(.*?)\}", block, flags=re.S)
    return " ".join(match.group(1).split()) if match else ""


def audit_caption_reporting(tex: str) -> AuditRow:
    manuscript_labels = manuscript_table_labels(tex)
    missing_specs = sorted(manuscript_labels - set(CAPTION_REQUIREMENTS))
    stale_specs = sorted(set(CAPTION_REQUIREMENTS) - manuscript_labels)
    if missing_specs:
        return AuditRow("table caption reporting", "incomplete", "missing caption requirements for: " + ", ".join(missing_specs))
    if stale_specs:
        return AuditRow("table caption reporting", "incomplete", "caption requirements for absent tables: " + ", ".join(stale_specs))

    missing_fragments: list[str] = []
    for label, fragments in CAPTION_REQUIREMENTS.items():
        caption = table_caption(tex, label)
        if not caption:
            missing_fragments.append(f"{label}: caption missing")
            continue
        for fragment in fragments:
            if fragment not in caption:
                missing_fragments.append(f"{label}: {fragment}")
    return AuditRow(
        "table caption reporting",
        ready(not missing_fragments),
        f"{len(CAPTION_REQUIREMENTS)} table captions declare reporting basis" if not missing_fragments else "; ".join(missing_fragments[:8]),
    )


def tex_number(value: str) -> str:
    return f"{float(value):.4f}"


def raw_fragment(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if value.endswith("%"):
        return value.replace("%", r"\%")
    try:
        number = float(value)
    except ValueError:
        return value.replace("%", r"\%")
    if re.fullmatch(r"-?\d+", value):
        return str(int(number))
    return f"{number:.4f}"


def expected_fragments(row: dict[str, str], spec: TableSpec) -> list[str]:
    fragments: list[str] = []
    for column in spec.raw_columns:
        fragment = raw_fragment(row.get(column, ""))
        if fragment:
            fragments.append(fragment)
    for metric in spec.metrics:
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        if mean_key in row:
            mean = row.get(mean_key, "")
            if mean == "":
                continue
            fragments.append(tex_number(mean))
            std = row.get(std_key, "")
            if std != "":
                fragments.append(tex_number(std))
            continue

        formatted = row.get(metric, "")
        for value in re.findall(r"\d+\.\d+", formatted):
            fragments.append(tex_number(value))
    return fragments


def row_label(row: dict[str, str]) -> str:
    for key in ("row_label", "model", "Model", "condition", "Condition", "setting", "Setting", "fold", "split"):
        value = row.get(key, "").strip()
        if value:
            return value
    return "row"


def display_label(spec_label: str, label: str) -> str:
    display = MODEL_ALIASES.get((spec_label, label), label)
    if "\\" in display or "$" in display:
        return display
    return display.replace("_", r"\_").replace("%", r"\%")


def find_row_block(table_block: str, label: str) -> str:
    start = 0
    while True:
        idx = table_block.find(label, start)
        if idx < 0:
            return ""
        row_start = table_block.rfind("\n", 0, idx) + 1
        row_end = table_block.find("\\\\", idx)
        if row_end < 0:
            row_end = table_block.find("\n", idx)
        if row_end < 0:
            row_end = min(len(table_block), idx + 500)
        else:
            row_end += 2
        row_block = table_block[row_start:row_end]
        if "&" in row_block and re.search(r"\d", row_block):
            return row_block
        start = idx + len(label)


def audit_spec(tex: str, final_tables_dir: Path, spec: TableSpec) -> AuditRow:
    block = find_table_block(tex, spec.label)
    if not block:
        return AuditRow(spec.label, "missing", "table block not found")
    csv_path = final_tables_dir / spec.csv_name
    rows = read_csv(csv_path)
    if not rows:
        return AuditRow(spec.label, "missing", f"{csv_path} missing or empty")
    missing: list[str] = []
    checked = 0
    for row in rows:
        model = row_label(row)
        manuscript_label = display_label(spec.label, model)
        row_block = find_row_block(block, manuscript_label)
        if not row_block:
            missing.append(f"{model}: row not found as {manuscript_label}")
            continue
        for fragment in expected_fragments(row, spec):
            checked += 1
            if fragment not in row_block:
                missing.append(f"{model}: {fragment}")
    if checked == 0:
        return AuditRow(spec.label, "incomplete", f"no numeric fragments extracted from {spec.csv_name}")
    return AuditRow(
        spec.label,
        ready(not missing),
        f"{checked} numeric fragments match rows in {spec.csv_name}" if not missing else "; ".join(missing[:10]),
    )


def integer(row: dict[str, str], column: str) -> int:
    return int(float(row[column]))


def audit_split_accounting_invariants(final_tables_dir: Path) -> AuditRow:
    rows = read_csv(final_tables_dir / "split_accounting.csv")
    if not rows:
        return AuditRow("split accounting invariants", "missing", "split_accounting.csv missing or empty")
    missing: list[str] = []
    test_subjects: set[str] = set()
    for row in rows:
        fold = row.get("fold", "fold")
        test_subject = row.get("test_subject", "")
        val_subject = row.get("val_subject", "")
        if test_subject:
            test_subjects.add(test_subject)
        if test_subject and val_subject and test_subject == val_subject:
            missing.append(f"{fold}: validation subject equals test subject")
        for split in ("train", "val", "test"):
            seq = integer(row, f"{split}_seq")
            pairs = integer(row, f"{split}_pairs")
            expected_pairs = seq * 6
            if pairs != expected_pairs:
                missing.append(f"{fold}: {split}_pairs={pairs}, expected {expected_pairs}")
        if integer(row, "test_imgs") != integer(row, "test_seq"):
            missing.append(f"{fold}: test_imgs != test_seq")
    if len(rows) != 8:
        missing.append(f"fold rows={len(rows)}, expected 8")
    if len(test_subjects) != 8:
        missing.append(f"unique test subjects={len(test_subjects)}, expected 8")
    return AuditRow(
        "split accounting invariants",
        ready(not missing),
        "8 folds; unique held-out subjects; pair counts equal strict T=3 sequences x 6" if not missing else "; ".join(missing[:8]),
    )


def audit_session_order_qc_invariants(final_tables_dir: Path) -> AuditRow:
    rows = read_csv(final_tables_dir / "session_order_pair_qc.csv")
    if not rows:
        return AuditRow("session/order QC invariants", "missing", "session_order_pair_qc.csv missing or empty")
    by_split = {row.get("split", ""): row for row in rows}
    missing: list[str] = []
    for split, row in by_split.items():
        pairs = integer(row, "pairs")
        positive = integer(row, "positive")
        negative = integer(row, "negative")
        complete_groups = integer(row, "complete_groups")
        problem_groups = integer(row, "problem_groups")
        if pairs != positive + negative:
            missing.append(f"{split}: pairs != positive + negative")
        if positive != negative:
            missing.append(f"{split}: positive != negative")
        if complete_groups != positive:
            missing.append(f"{split}: complete_groups != positive")
        if problem_groups != 0:
            missing.append(f"{split}: problem_groups={problem_groups}")
        if row.get("anchor_match") != "100%":
            missing.append(f"{split}: anchor_match={row.get('anchor_match')}")
    if {"Train", "Val", "Test", "All"} - set(by_split):
        missing.append("expected Train/Val/Test/All rows")
    if "All" in by_split:
        for column in ("pairs", "positive", "negative", "complete_groups", "problem_groups"):
            observed = integer(by_split["All"], column)
            expected = sum(integer(by_split[split], column) for split in ("Train", "Val", "Test") if split in by_split)
            if observed != expected:
                missing.append(f"All {column}={observed}, expected {expected}")
    return AuditRow(
        "session/order QC invariants",
        ready(not missing),
        "positive/negative counts are balanced, anchor matches are 100%, and All totals match splits" if not missing else "; ".join(missing[:8]),
    )


def audit_parameter_count_artifact(tex: str, final_tables_dir: Path) -> AuditRow:
    rows = read_csv(final_tables_dir / "model_parameter_counts.csv")
    if not rows:
        return AuditRow("tab:implementation_details parameter counts", "missing", "model_parameter_counts.csv missing or empty")
    block = find_table_block(tex, "tab:implementation_details")
    if not block:
        return AuditRow("tab:implementation_details parameter counts", "missing", "implementation-details table not found")
    checked = 0
    missing: list[str] = []
    expected_models = {
        "Final gated BNT/ReGraph+CLIP",
        "No-adj gated ROI Transformer+CLIP",
        "Graph-bias BNT/ReGraph+CLIP",
        "Learned edge-bias BNT+CLIP",
        "ROI-MLP+CLIP",
    }
    seen_models = {row.get("model", "") for row in rows}
    for model in sorted(expected_models - seen_models):
        missing.append(f"missing artifact row: {model}")
    for row in rows:
        raw = row.get("trainable_parameters", "")
        if not raw:
            missing.append(f"{row.get('model', '<missing model>')}: empty trainable_parameters")
            continue
        fragment = f"{int(float(raw)):,}"
        checked += 1
        if fragment not in block:
            missing.append(f"{row.get('model', '<missing model>')}: {fragment}")
        source = row.get("source", "")
        if "tab:implementation_details" not in source:
            missing.append(f"{row.get('model', '<missing model>')}: source missing implementation table label")
    return AuditRow(
        "tab:implementation_details parameter counts",
        ready(not missing and checked > 0),
        f"{checked} parameter-count fragments match model_parameter_counts.csv" if not missing and checked > 0 else "; ".join(missing[:8]),
    )


def audit_laion_pairwise_table(tex: str, external_summary_dir: Path) -> AuditRow:
    block = find_table_block(tex, "tab:laion_external_pairwise")
    if not block:
        return AuditRow("tab:laion_external_pairwise", "missing", "table block not found")
    rows = read_csv(external_summary_dir / "laion_fmri_visual_roi_pairwise_tests.csv")
    if not rows:
        return AuditRow("tab:laion_external_pairwise", "missing", "laion_fmri_visual_roi_pairwise_tests.csv missing or empty")
    metric_labels = {
        "test_AUROC": "AUROC",
        "test_AUPRC": "AUPRC",
        "test_R@5": "R@5",
        "test_MRR": "MRR",
    }
    missing: list[str] = []
    checked = 0
    seen = {row.get("metric", "") for row in rows}
    for metric in sorted(set(metric_labels) - seen):
        missing.append(f"missing artifact metric: {metric}")
    for row in rows:
        metric = row.get("metric", "")
        manuscript_label = metric_labels.get(metric)
        if manuscript_label is None:
            missing.append(f"unexpected artifact metric: {metric}")
            continue
        row_block = find_row_block(block, manuscript_label)
        if not row_block:
            missing.append(f"{metric}: row not found as {manuscript_label}")
            continue
        fragments = [
            raw_fragment(row.get("n", "")),
            tex_number(row.get("mean_diff", "")),
            tex_number(row.get("ci95_low", "")),
            tex_number(row.get("ci95_high", "")),
            tex_number(row.get("paired_p", "")),
        ]
        for fragment in fragments:
            checked += 1
            if fragment not in row_block:
                missing.append(f"{metric}: {fragment}")
    return AuditRow(
        "tab:laion_external_pairwise",
        ready(not missing and checked > 0),
        f"{checked} numeric fragments match laion_fmri_visual_roi_pairwise_tests.csv" if not missing and checked > 0 else "; ".join(missing[:8]),
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
        "# Manuscript Table Values Audit",
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
    tex = args.tex.read_text(encoding="utf-8", errors="replace")
    rows = [
        audit_table_coverage(tex),
        audit_caption_reporting(tex),
        audit_split_accounting_invariants(args.final_tables_dir),
        audit_session_order_qc_invariants(args.final_tables_dir),
        audit_parameter_count_artifact(tex, args.final_tables_dir),
        audit_laion_pairwise_table(tex, args.external_summary_dir),
        *[audit_spec(tex, args.final_tables_dir, spec) for spec in TABLE_SPECS],
    ]
    write_outputs(args.final_tables_dir, args.output_prefix, rows)
    return 0 if all(row.status == "ready" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
