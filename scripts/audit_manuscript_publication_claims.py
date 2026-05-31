#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class AuditRow:
    item: str
    status: str
    evidence: str


REQUIRED_LABELS = [
    "tab:split_accounting",
    "tab:session_order_pair_qc",
    "tab:implementation_details",
    "tab:cross_subject_main",
    "tab:sota_baselines",
    "tab:graph_only",
    "tab:adjacency_ablation",
    "tab:roi_token_controls",
    "tab:adjacency_perturbation",
    "tab:edge_bias_followup",
    "tab:single_ref_matched",
    "tab:single_ref_retrained",
    "tab:heldout",
    "tab:hardneg",
    "tab:lowshot",
    "tab:external_visual_roi_smoke",
    "tab:gate_confound",
    "tab:matched_deletion",
    "tab:fold_difficulty",
    "tab:top_gated_roi_names",
]

REQUIRED_RESULT_FILES = {
    "split_accounting.csv": 8,
    "session_order_pair_qc.csv": 1,
    "table_within_subject.csv": 1,
    "table_allfold_final.csv": 24,
    "table_hard_negative_allfold.csv": 24,
    "table_heldout_image.csv": 24,
    "table_phase2_sota_graph_baselines.csv": 100,
    "table_graph_only.csv": 48,
    "table_adjacency_ablation.csv": 1,
    "table_roi_token_controls.csv": 1,
    "table_adjacency_perturbation.csv": 1,
    "table_edge_bias_followup.csv": 1,
    "single_ref_matched_summary.csv": 72,
    "single_ref_matched_allseed_summary.csv": 72,
    "table_lowshot_calibration.csv": 120,
    "table_external_visual_roi_smoke.csv": 100,
    "table_gate_confound.csv": 3,
    "table_matched_deletion.csv": 6,
    "fold_difficulty_qc.csv": 8,
    "publication_paired_stats.csv": 1,
}

REQUIRED_STATS = [
    ("main_allfold", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP"),
    ("main_allfold", "Gated ReGraph/BNT+CLIP - Flat ReGraph+CLIP"),
    ("hard_negative", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP"),
    ("heldout_real_vs_random_available_raw", "Gated ReGraph/BNT+CLIP - Gated random embedding"),
    ("component_baselines", "Gated ReGraph/BNT+CLIP - MindLink-style subject-adversarial ROI-MLP"),
    ("single_ref_eval_existing", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP"),
    ("single_ref_eval_existing", "Gated ReGraph/BNT+CLIP - No-adj gated ROI Transformer+CLIP"),
    ("single_ref_retrained", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP"),
    ("single_ref_retrained", "Gated ReGraph/BNT+CLIP - No-adj gated ROI Transformer+CLIP"),
    ("single_ref_retrained", "No-adj gated ROI Transformer+CLIP - ROI-MLP+CLIP"),
]

STAT_NUMERIC_COLUMNS = ["n", "mean_diff", "std_diff", "bootstrap_ci_low", "bootstrap_ci_high", "paired_t_p"]
STAT_BASE_NUMERIC_COLUMNS = ["n", "mean_diff", "std_diff", "bootstrap_ci_low", "bootstrap_ci_high"]

def deanon_patterns() -> list[str]:
    literals = [
        "".join(("Xia", "lei")),
        ".".join(("xia" + "lei", "huang")),
        " ".join(("NYU", "Shanghai")),
        "".join(("xh", "2906")),
    ]
    return [re.escape(value) for value in literals]

OVERCLAIM_PATTERNS = [
    r"adjacency improves",
    r"fixed adjacency improves",
    r"graph adjacency is the source",
    r"adjacency is the source",
    r"driven by explicit fixed adjacency",
    r"explicit fixed adjacency improves",
]

LATEX_ENVIRONMENTS = ["table", "figure", "equation", "tabular", "tabularx"]

IMPLEMENTATION_DETAIL_REQUIREMENTS = {
    "BCE loss": ("\\mathcal{L}_{\\mathrm{BCE}}", "pair matching"),
    "repeat InfoNCE": ("\\mathcal{L}_{\\mathrm{repeat\\ InfoNCE}}", "other positive pairs in the minibatch as negatives", "L2-normalize"),
    "CLIP alignment": ("frozen CLIP image embeddings", "\\tau_{\\mathrm{clip}}", "\\lambda_{\\mathrm{clip}}=2.0"),
    "train-only adjacency": ("training data only", "absolute ROI-response correlations", "Validation and test responses are never used"),
    "brain encoder": ("hidden dimension 64", "2 layers", "4 heads", "dropout 0.3"),
    "projection/readout": ("gated-flat ROI-preserving readout", "final embedding dimension 128", "shared 128-d space"),
    "optimization": ("AdamW", "learning rate $10^{-3}$", "weight decay $10^{-4}$", "batch size 128", "gradient clipping at 5.0"),
    "training selection": ("Up to 80 epochs", "patience 12", "validation AUROC"),
    "evaluation protocol": ("8 held-out-subject folds", "seeds 11, 22, and 33"),
}

COMPONENT_BASELINE_REQUIRED_FRAGMENTS = (
    "Task-matched component baseline comparison",
    "task-matched brain-decoding component baselines",
    "not intended to reproduce full image-reconstruction pipelines",
    "These are not full image-reconstruction system comparisons",
)

COMPONENT_BASELINE_FORBIDDEN_PATTERNS = (
    r"\bSOTA-style\b",
    r"\bstate-of-the-art\b",
    r"outperforms\s+SOTA",
    r"outperforms\s+full\s+image-reconstruction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit manuscript/result consistency for the AAAI-style ReGraph-VLM submission.")
    parser.add_argument("--tex", type=Path, default=Path("reports/neurips_report/may30.tex"))
    parser.add_argument("--final-tables-dir", type=Path, default=Path("preproc_v0/repetition_familiarity/results/final_tables"))
    parser.add_argument("--output-dir", type=Path, default=Path("preproc_v0/repetition_familiarity/results/final_tables"))
    parser.add_argument("--output-prefix", default="manuscript_publication_claims_audit")
    parser.add_argument(
        "--manuscript-only",
        action="store_true",
        help="Check TeX-facing manuscript issues without requiring local copies of all result-table artifacts.",
    )
    return parser.parse_args()


def status(ok: bool) -> str:
    return "ready" if ok else "incomplete"


def count_env(text: str, env: str) -> tuple[int, int]:
    begin = len(re.findall(rf"\\begin\{{{re.escape(env)}\}}", text))
    end = len(re.findall(rf"\\end\{{{re.escape(env)}\}}", text))
    return begin, end


def is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    pos = index - 1
    while pos >= 0 and text[pos] == "\\":
        backslashes += 1
        pos -= 1
    return backslashes % 2 == 1


def strip_latex_comments(text: str) -> str:
    stripped_lines: list[str] = []
    for line in text.splitlines():
        for index, char in enumerate(line):
            if char == "%" and not is_escaped(line, index):
                line = line[:index]
                break
        stripped_lines.append(line)
    return "\n".join(stripped_lines)


def audit_group_braces(text: str) -> AuditRow:
    clean_text = strip_latex_comments(text)
    stack: list[int] = []
    for index, char in enumerate(clean_text):
        if char not in "{}" or is_escaped(clean_text, index):
            continue
        line = clean_text.count("\n", 0, index) + 1
        if char == "{":
            stack.append(line)
            continue
        if not stack:
            return AuditRow("TeX group brace balance", "incomplete", f"line {line}: unmatched closing brace")
        stack.pop()
    if stack:
        return AuditRow("TeX group brace balance", "incomplete", f"line {stack[-1]}: unmatched opening brace")
    return AuditRow("TeX group brace balance", "ready", "all unescaped group braces balanced")


def audit_math_dollar_balance(text: str) -> AuditRow:
    clean_text = strip_latex_comments(text)
    single_dollars = 0
    double_dollars = 0
    index = 0
    while index < len(clean_text):
        if clean_text[index] != "$" or is_escaped(clean_text, index):
            index += 1
            continue
        if index + 1 < len(clean_text) and clean_text[index + 1] == "$":
            double_dollars += 1
            index += 2
            continue
        single_dollars += 1
        index += 1
    if single_dollars % 2 != 0:
        return AuditRow("TeX math dollar balance", "incomplete", f"odd number of unescaped single-dollar delimiters: {single_dollars}")
    if double_dollars % 2 != 0:
        return AuditRow("TeX math dollar balance", "incomplete", f"odd number of unescaped double-dollar delimiters: {double_dollars}")
    return AuditRow("TeX math dollar balance", "ready", f"single={single_dollars}, double={double_dollars}")


def audit_tracked_figures(tex_path: Path, figure_paths: list[str]) -> AuditRow:
    return audit_tracked_paths(tex_path, figure_paths, "figure files tracked by Git")


def audit_tracked_paths(tex_path: Path, raw_paths: list[str], item: str) -> AuditRow:
    root_cmd = ["git", "-C", str(tex_path.parent), "rev-parse", "--show-toplevel"]
    root_run = subprocess.run(root_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if root_run.returncode != 0:
        return AuditRow(item, "ready", "git metadata unavailable; skipped outside Git checkout")

    git_root = Path(root_run.stdout.strip())
    tracked_run = subprocess.run(
        ["git", "-C", str(git_root), "ls-files"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if tracked_run.returncode != 0:
        return AuditRow(item, "incomplete", tracked_run.stderr.strip() or "git ls-files failed")

    tracked = set(tracked_run.stdout.splitlines())
    missing: list[str] = []
    for raw_path in raw_paths:
        absolute = (tex_path.parent / raw_path).resolve()
        try:
            relative = absolute.relative_to(git_root.resolve()).as_posix()
        except ValueError:
            missing.append(raw_path)
            continue
        if relative not in tracked:
            missing.append(relative)
    return AuditRow(
        item,
        status(not missing),
        f"{len(raw_paths)} checked, all tracked" if not missing else ", ".join(missing),
    )


def audit_environment_nesting(text: str) -> AuditRow:
    env_pattern = "|".join(re.escape(env) for env in LATEX_ENVIRONMENTS)
    token_re = re.compile(rf"\\(begin|end)\{{({env_pattern})\}}")
    stack: list[tuple[str, int]] = []
    for match in token_re.finditer(text):
        kind, env = match.group(1), match.group(2)
        line = text.count("\n", 0, match.start()) + 1
        if kind == "begin":
            stack.append((env, line))
            continue
        if not stack:
            return AuditRow("LaTeX environment nesting", "incomplete", f"line {line}: unmatched \\end{{{env}}}")
        open_env, open_line = stack.pop()
        if open_env != env:
            return AuditRow(
                "LaTeX environment nesting",
                "incomplete",
                f"line {line}: \\end{{{env}}} closes \\begin{{{open_env}}} from line {open_line}",
            )
    if stack:
        open_env, open_line = stack[-1]
        return AuditRow("LaTeX environment nesting", "incomplete", f"line {open_line}: unclosed \\begin{{{open_env}}}")
    return AuditRow("LaTeX environment nesting", "ready", f"{len(list(token_re.finditer(text)))} begin/end tokens nested correctly")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def citation_keys(text: str) -> list[str]:
    keys = []
    for match in re.findall(r"\\cite\w*(?:\[[^\]]*\]){0,2}\{([^}]+)\}", text):
        keys.extend(key.strip() for key in match.split(",") if key.strip())
    return sorted(set(keys))


def pattern_line_numbers(text: str, pattern: str) -> list[int]:
    return [text.count("\n", 0, match.start()) + 1 for match in re.finditer(pattern, text)]


def bibliography_paths(tex_path: Path, text: str) -> list[Path]:
    paths: list[Path] = []
    for match in re.findall(r"\\bibliography\{([^}]+)\}", text):
        for raw_name in match.split(","):
            name = raw_name.strip()
            if not name:
                continue
            path = Path(name)
            if path.suffix != ".bib":
                path = path.with_suffix(".bib")
            if not path.is_absolute():
                path = tex_path.parent / path
            paths.append(path)
    return paths


def local_style_paths(tex_path: Path, text: str) -> list[Path]:
    paths: list[Path] = []
    for match in re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", text):
        for raw_name in match.split(","):
            name = raw_name.strip()
            if not name:
                continue
            style_path = tex_path.parent / f"{name}.sty"
            if style_path.exists():
                paths.append(style_path)
    return paths


def relative_to_tex_dir(tex_path: Path, paths: list[Path]) -> list[str]:
    raw_paths: list[str] = []
    for path in paths:
        try:
            raw_paths.append(path.resolve().relative_to(tex_path.parent.resolve()).as_posix())
        except ValueError:
            raw_paths.append(str(path))
    return raw_paths


def figure_dependency_paths(text: str) -> list[str]:
    paths: list[str] = []
    paths.extend(re.findall(r"\\IfFileExists\{([^}]+)\}", text))
    paths.extend(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text))
    return list(dict.fromkeys(paths))


def implementation_detail_block(text: str) -> str:
    marker = "\\subsection{Experimental protocol and implementation details}"
    start = text.find(marker)
    if start < 0:
        return ""
    end = text.find("\\section{Main Results}", start)
    return text[start:end] if end >= 0 else text[start:]


def audit_implementation_details(text: str) -> AuditRow:
    block = implementation_detail_block(text)
    if not block:
        return AuditRow("implementation detail coverage", "missing", "implementation details subsection not found")
    missing: list[str] = []
    for item, fragments in IMPLEMENTATION_DETAIL_REQUIREMENTS.items():
        absent = [fragment for fragment in fragments if fragment not in block]
        if absent:
            missing.append(f"{item}: " + ", ".join(absent))
    return AuditRow(
        "implementation detail coverage",
        status(not missing),
        f"{len(IMPLEMENTATION_DETAIL_REQUIREMENTS)} reproducibility detail groups present" if not missing else "; ".join(missing[:6]),
    )


def audit_component_baseline_framing(text: str) -> AuditRow:
    missing = [fragment for fragment in COMPONENT_BASELINE_REQUIRED_FRAGMENTS if fragment not in text]
    forbidden = [
        pattern
        for pattern in COMPONENT_BASELINE_FORBIDDEN_PATTERNS
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]
    ok = not missing and not forbidden
    evidence = "task-matched component framing present; full-system/SOTA overclaims absent"
    if not ok:
        evidence = f"missing={missing or 'none'}; forbidden={forbidden or 'none'}"
    return AuditRow("component-baseline framing", status(ok), evidence)


def bib_keys(paths: list[Path]) -> tuple[set[str], list[Path]]:
    keys: set[str] = set()
    missing_paths: list[Path] = []
    for path in paths:
        if not path.exists():
            missing_paths.append(path)
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        keys.update(re.findall(r"@\w+\{([^,]+),", text))
    return keys, missing_paths


def audit_text(tex_path: Path) -> list[AuditRow]:
    if not tex_path.exists():
        return [AuditRow("manuscript exists", "missing", f"{tex_path} not found")]
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    rows: list[AuditRow] = [AuditRow("manuscript exists", "ready", f"{tex_path}: {len(text.splitlines())} lines")]

    rows.append(AuditRow("anonymous author block", status("Anonymous Author(s)" in text), "Anonymous Author(s) present" if "Anonymous Author(s)" in text else "anonymous author marker missing"))

    deanon_hits = []
    for pattern in deanon_patterns():
        deanon_hits.extend(re.findall(pattern, text))
    rows.append(AuditRow("deanonymizing strings", status(not deanon_hits), "none found" if not deanon_hits else ", ".join(sorted(set(deanon_hits)))))

    overclaim_hits = []
    lower = text.lower()
    for pattern in OVERCLAIM_PATTERNS:
        if re.search(pattern, lower):
            overclaim_hits.append(pattern)
    rows.append(AuditRow("fixed-adjacency overclaims", status(not overclaim_hits), "none found" if not overclaim_hits else ", ".join(overclaim_hits)))

    labels = re.findall(r"\\label\{([^}]+)\}", text)
    refs = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text)
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    missing_refs = sorted(set(refs) - set(labels))
    rows.append(AuditRow("duplicate labels", status(not duplicate_labels), "none" if not duplicate_labels else ", ".join(duplicate_labels)))
    rows.append(AuditRow("unresolved refs", status(not missing_refs), "none" if not missing_refs else ", ".join(missing_refs)))

    missing_required = sorted(set(REQUIRED_LABELS) - set(labels))
    rows.append(AuditRow("required publication labels", status(not missing_required), "all present" if not missing_required else ", ".join(missing_required)))
    rows.append(audit_implementation_details(text))
    rows.append(audit_component_baseline_framing(text))

    cite_keys = citation_keys(text)
    bib_paths = bibliography_paths(tex_path, text)
    defined_bib_keys, missing_bib_paths = bib_keys(bib_paths)
    missing_cites = sorted(set(cite_keys) - defined_bib_keys)
    if missing_bib_paths:
        cite_status = "missing"
        cite_evidence = "missing bibliography files: " + ", ".join(str(path) for path in missing_bib_paths)
    elif missing_cites:
        cite_status = "incomplete"
        cite_evidence = "missing citation keys: " + ", ".join(missing_cites)
    else:
        cite_status = "ready"
        cite_evidence = f"{len(cite_keys)} citation keys covered by {len(bib_paths)} bibliography file(s)"
    rows.append(AuditRow("citation bibliography coverage", cite_status, cite_evidence))

    support_files = relative_to_tex_dir(tex_path, [*bib_paths, *local_style_paths(tex_path, text)])
    rows.append(audit_tracked_paths(tex_path, support_files, "manuscript support files tracked by Git"))

    figure_paths = figure_dependency_paths(text)
    missing_figures = sorted(path for path in figure_paths if not (tex_path.parent / path).exists())
    rows.append(
        AuditRow(
            "figure file availability",
            status(not missing_figures),
            f"{len(figure_paths)} checked, all present" if not missing_figures else ", ".join(missing_figures),
        )
    )
    needupdate_lines = pattern_line_numbers(text, r"\\needupdate\{")
    rows.append(
        AuditRow(
            "unresolved manuscript placeholders",
            status(not needupdate_lines and not missing_figures),
            (
                f"no \\needupdate uses and all {len(figure_paths)} figure dependencies resolve to files"
                if not needupdate_lines and not missing_figures
                else f"\\needupdate lines={needupdate_lines or 'none'}; missing figures={missing_figures or 'none'}"
            ),
        )
    )
    rows.append(audit_tracked_figures(tex_path, figure_paths))

    rows.append(audit_group_braces(text))
    rows.append(audit_math_dollar_balance(text))
    for env in LATEX_ENVIRONMENTS:
        begin, end = count_env(text, env)
        rows.append(AuditRow(f"{env} environment balance", status(begin == end), f"begin={begin}, end={end}"))
    rows.append(audit_environment_nesting(text))

    return rows


def audit_result_files(final_tables_dir: Path) -> list[AuditRow]:
    rows: list[AuditRow] = []
    for name, min_rows in REQUIRED_RESULT_FILES.items():
        path = final_tables_dir / name
        if not path.exists():
            rows.append(AuditRow(f"result file: {name}", "missing", f"{path} not found"))
            continue
        df = read_csv(path)
        effective_n = len(df)
        if "n" in df.columns:
            values = pd.to_numeric(df["n"], errors="coerce").dropna()
            if not values.empty:
                effective_n = int(values.sum())
        rows.append(
            AuditRow(
                f"result file: {name}",
                status(effective_n >= min_rows),
                f"{len(df)} rows, support n={effective_n}, expected at least {min_rows}",
            )
        )
    return rows


def finite_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.map(lambda value: pd.notna(value) and math.isfinite(float(value)))


def exact_zero_diff_tie(sub: pd.DataFrame) -> pd.Series:
    mean_diff = pd.to_numeric(sub["mean_diff"], errors="coerce")
    std_diff = pd.to_numeric(sub["std_diff"], errors="coerce")
    return mean_diff.eq(0.0) & std_diff.eq(0.0)


def audit_stat_rows(sub: pd.DataFrame) -> tuple[bool, str]:
    missing_columns = [column for column in STAT_NUMERIC_COLUMNS if column not in sub.columns]
    if missing_columns:
        return False, "missing numeric columns: " + ", ".join(missing_columns)

    metric_count = sub["metric"].nunique() if "metric" in sub.columns else 0
    if metric_count < 3:
        return False, f"{len(sub)} metric rows, expected at least 3"

    invalid: list[str] = []
    for column in STAT_BASE_NUMERIC_COLUMNS:
        mask = finite_numeric(sub[column])
        if not mask.all():
            invalid.append(f"{column} invalid={int((~mask).sum())}")

    p_values = pd.to_numeric(sub["paired_t_p"], errors="coerce")
    p_finite = finite_numeric(sub["paired_t_p"])
    p_missing = ~p_finite
    zero_tie = exact_zero_diff_tie(sub)
    if (p_missing & ~zero_tie).any():
        invalid.append(f"paired_t_p invalid={int((p_missing & ~zero_tie).sum())}")

    n_values = pd.to_numeric(sub["n"], errors="coerce")
    if not n_values.gt(0).all():
        invalid.append("n must be positive")
    std_values = pd.to_numeric(sub["std_diff"], errors="coerce")
    if not std_values.ge(0).all():
        invalid.append("std_diff must be nonnegative")
    low_values = pd.to_numeric(sub["bootstrap_ci_low"], errors="coerce")
    high_values = pd.to_numeric(sub["bootstrap_ci_high"], errors="coerce")
    if not low_values.le(high_values).all():
        invalid.append("bootstrap CI low must be <= high")
    if not p_values[p_finite].between(0.0, 1.0).all():
        invalid.append("paired_t_p must be in [0, 1]")

    if invalid:
        return False, "; ".join(invalid)

    tie_count = int((p_missing & zero_tie).sum())
    tie_note = f", exact-zero tie rows={tie_count}" if tie_count else ""
    return (
        True,
        f"{len(sub)} rows/{metric_count} metrics; n={int(n_values.min())}-{int(n_values.max())}; numeric fields valid{tie_note}",
    )


def audit_publication_stats(final_tables_dir: Path) -> list[AuditRow]:
    path = final_tables_dir / "publication_paired_stats.csv"
    if not path.exists():
        return [AuditRow("publication paired stats coverage", "missing", f"{path} not found")]
    df = read_csv(path)
    rows: list[AuditRow] = []
    for setting, comparison in REQUIRED_STATS:
        required_columns = {"setting", "comparison", "metric", *STAT_NUMERIC_COLUMNS}
        if df.empty or not required_columns.issubset(df.columns):
            rows.append(AuditRow(f"paired stats: {setting} / {comparison}", "incomplete", "missing expected columns"))
            continue
        sub = df[(df["setting"] == setting) & (df["comparison"] == comparison)]
        ok, evidence = audit_stat_rows(sub)
        rows.append(
            AuditRow(
                f"paired stats: {setting} / {comparison}",
                status(ok),
                evidence,
            )
        )
    return rows


def write_outputs(output_dir: Path, output_prefix: str, rows: list[AuditRow]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{output_prefix}.csv"
    md_path = output_dir / f"{output_prefix}.md"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item", "status", "evidence"], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({"item": row.item, "status": row.status, "evidence": row.evidence})

    counts = pd.Series([row.status for row in rows]).value_counts().to_dict()
    lines = [
        "# Manuscript Publication Claims Audit",
        "",
        "This audit checks manuscript/result consistency for the publication-facing ReGraph-VLM story.",
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
    rows = [*audit_text(args.tex)]
    if not args.manuscript_only:
        rows.extend(audit_result_files(args.final_tables_dir))
        rows.extend(audit_publication_stats(args.final_tables_dir))
    write_outputs(args.output_dir, args.output_prefix, rows)


if __name__ == "__main__":
    main()
