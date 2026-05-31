#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuditRow:
    item: str
    status: str
    evidence: str


@dataclass(frozen=True)
class PairedClaim:
    item: str
    cue: str
    setting: str
    comparison: str
    metric: str
    check_diff: bool = True
    check_p: bool = True
    check_ci: bool = False
    p_digits: int | None = None


PAIRED_CLAIMS = [
    PairedClaim("main all-fold AUROC", "The main final result", "main_allfold", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP", "AUROC"),
    PairedClaim("main all-fold AUPRC", "The main final result", "main_allfold", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP", "AUPRC"),
    PairedClaim("main all-fold R@5", "The main final result", "main_allfold", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP", "R@5"),
    PairedClaim("main all-fold brain R@5", "The main final result", "main_allfold", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP", "brain_R@5"),
    PairedClaim("component baseline AUROC", "The component-baseline comparison", "component_baselines", "Gated ReGraph/BNT+CLIP - MindLink-style subject-adversarial ROI-MLP", "AUROC"),
    PairedClaim("component baseline R@5", "The component-baseline comparison", "component_baselines", "Gated ReGraph/BNT+CLIP - MindLink-style subject-adversarial ROI-MLP", "R@5"),
    PairedClaim("component baseline brain R@5", "The component-baseline comparison", "component_baselines", "Gated ReGraph/BNT+CLIP - MindLink-style subject-adversarial ROI-MLP", "brain_R@5"),
    PairedClaim("single-reference AUROC", "The stricter control preserves", "single_ref_eval_existing", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP", "AUROC"),
    PairedClaim("single-reference AUPRC", "The stricter control preserves", "single_ref_eval_existing", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP", "AUPRC"),
    PairedClaim("single-reference brain R@5", "The stricter control preserves", "single_ref_eval_existing", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP", "brain_R@5"),
    PairedClaim("single-reference no-adj AUROC p", "The stricter control preserves", "single_ref_eval_existing", "Gated ReGraph/BNT+CLIP - No-adj gated ROI Transformer+CLIP", "AUROC", check_diff=False, p_digits=3),
    PairedClaim("single-reference no-adj brain R@5 p", "The stricter control preserves", "single_ref_eval_existing", "Gated ReGraph/BNT+CLIP - No-adj gated ROI Transformer+CLIP", "brain_R@5", check_diff=False, p_digits=3),
    PairedClaim("single-reference retrained AUROC", "Paired fold$\\times$seed tests favor the retrained", "single_ref_retrained", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP", "AUROC", p_digits=4),
    PairedClaim("single-reference retrained AUPRC", "Paired fold$\\times$seed tests favor the retrained", "single_ref_retrained", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP", "AUPRC"),
    PairedClaim("single-reference retrained R@5", "Paired fold$\\times$seed tests favor the retrained", "single_ref_retrained", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP", "R@5"),
    PairedClaim("single-reference retrained image R@5", "Paired fold$\\times$seed tests favor the retrained", "single_ref_retrained", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP", "image_R@5", p_digits=4),
    PairedClaim("single-reference retrained brain R@5", "Paired fold$\\times$seed tests favor the retrained", "single_ref_retrained", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP", "brain_R@5"),
    PairedClaim("hard-negative AUROC CI", "Hard-negative evaluation makes", "hard_negative", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP", "AUROC", check_p=False, check_ci=True),
    PairedClaim("hard-negative R@5 CI", "Hard-negative evaluation makes", "hard_negative", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP", "R@5", check_p=False, check_ci=True),
    PairedClaim("hard-negative brain R@5 CI", "Hard-negative evaluation makes", "hard_negative", "Gated ReGraph/BNT+CLIP - ROI-MLP+CLIP", "brain_R@5", check_p=False, check_ci=True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit manuscript statistical claims against paired-test artifacts.")
    parser.add_argument("--tex", type=Path, default=Path("reports/neurips_report/may30.tex"))
    parser.add_argument(
        "--paired-stats",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables/publication_paired_stats.csv"),
    )
    parser.add_argument(
        "--laion-pairwise",
        type=Path,
        default=Path("external_validation/summary/laion_fmri_visual_roi_pairwise_tests.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument("--output-prefix", default="manuscript_stat_claims_audit")
    return parser.parse_args()


def ready(ok: bool) -> str:
    return "ready" if ok else "incomplete"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def paragraph_for_cue(text: str, cue: str) -> str:
    idx = text.find(cue)
    if idx < 0:
        return ""
    start = text.rfind("\n\n", 0, idx)
    end = text.find("\n\n", idx)
    if start < 0:
        start = 0
    else:
        start += 2
    if end < 0:
        end = len(text)
    return text[start:end]


def paired_row(rows: list[dict[str, str]], claim: PairedClaim) -> dict[str, str] | None:
    for row in rows:
        if row.get("setting") == claim.setting and row.get("comparison") == claim.comparison and row.get("metric") == claim.metric:
            return row
    return None


def sci_p(value: float) -> str:
    if value == 0:
        return "0.00\\times10^{0}"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / (10**exponent)
    return f"{mantissa:.2f}\\times10^{{{exponent}}}"


def p_fragment(value: float, digits: int | None = None) -> str:
    if digits is not None:
        return f"p={value:.{digits}f}"
    if value < 0.001:
        return f"p={sci_p(value)}"
    return f"p={value:.4f}"


def diff_fragment(value: float) -> str:
    return f"{value:+.4f}"


def ci_fragment(row: dict[str, str]) -> str:
    low = float(row["bootstrap_ci_low"])
    high = float(row["bootstrap_ci_high"])
    return f"[{low:.4f},{high:.4f}]"


def audit_paired_claim(tex: str, rows: list[dict[str, str]], claim: PairedClaim) -> AuditRow:
    row = paired_row(rows, claim)
    if row is None:
        return AuditRow(claim.item, "missing", f"{claim.setting} / {claim.comparison} / {claim.metric} not found")
    paragraph = paragraph_for_cue(tex, claim.cue)
    if not paragraph:
        return AuditRow(claim.item, "missing", f"paragraph cue not found: {claim.cue}")
    expected: list[str] = []
    if claim.check_diff:
        expected.append(diff_fragment(float(row["mean_diff"])))
    if claim.check_p:
        expected.append(p_fragment(float(row["paired_t_p"]), claim.p_digits))
    if claim.check_ci:
        expected.append(ci_fragment(row))
    missing = [fragment for fragment in expected if fragment not in paragraph]
    return AuditRow(
        claim.item,
        ready(not missing),
        "matched " + ", ".join(expected) if not missing else "missing " + ", ".join(missing),
    )


def audit_laion_claim(tex: str, rows: list[dict[str, str]]) -> list[AuditRow]:
    paragraph = paragraph_for_cue(tex, "All external smoke checks")
    if not paragraph:
        return [AuditRow("LAION external paragraph", "missing", "paragraph cue not found")]
    by_metric = {row.get("metric"): row for row in rows if row.get("comparison") == "Gated ROI Transformer - ROI-MLP"}
    out: list[AuditRow] = []
    r5 = by_metric.get("test_R@5")
    if r5 is None:
        out.append(AuditRow("LAION R@5 trend", "missing", "test_R@5 paired row not found"))
    else:
        expected = [diff_fragment(float(r5["mean_diff"])), p_fragment(float(r5["paired_p"]), 4)]
        missing = [fragment for fragment in expected if fragment not in paragraph]
        out.append(AuditRow("LAION R@5 trend", ready(not missing), "matched " + ", ".join(expected) if not missing else "missing " + ", ".join(missing)))
    auroc = by_metric.get("test_AUROC")
    auprc = by_metric.get("test_AUPRC")
    nonsig = auroc is not None and auprc is not None and float(auroc["paired_p"]) > 0.05 and float(auprc["paired_p"]) > 0.05
    phrase = "no significant AUROC or AUPRC difference" in paragraph
    out.append(
        AuditRow(
            "LAION nonsignificant AUROC/AUPRC wording",
            ready(nonsig and phrase),
            "AUROC/AUPRC p>0.05 and wording present" if nonsig and phrase else "missing p>0.05 support or wording",
        )
    )
    return out


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
        "# Manuscript Statistical Claims Audit",
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
    paired_rows = read_csv(args.paired_stats)
    laion_rows = read_csv(args.laion_pairwise)
    rows = [audit_paired_claim(tex, paired_rows, claim) for claim in PAIRED_CLAIMS]
    rows.extend(audit_laion_claim(tex, laion_rows))
    write_outputs(args.output_dir, args.output_prefix, rows)
    return 0 if all(row.status == "ready" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
