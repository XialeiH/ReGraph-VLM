#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a compact publication evidence manifest.")
    parser.add_argument("--tex", type=Path, default=Path("reports/neurips_report/may30.tex"))
    parser.add_argument(
        "--final-tables-dir",
        type=Path,
        default=Path("preproc_v0/repetition_familiarity/results/final_tables"),
    )
    parser.add_argument("--external-summary-dir", type=Path, default=Path("external_validation/summary"))
    parser.add_argument("--output", type=Path, default=Path("preproc_v0/repetition_familiarity/results/final_tables/publication_evidence_manifest.md"))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def status_counts(path: Path) -> str:
    rows = read_csv(path)
    if not rows:
        return "missing or empty"
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.get("status", "")] = counts.get(row.get("status", ""), 0) + 1
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def row_count(path: Path) -> str:
    rows = read_csv(path)
    if not rows:
        return "missing or empty"
    if "n" in rows[0]:
        values: list[int] = []
        for row in rows:
            raw = row.get("n", "")
            try:
                values.append(int(float(raw)))
            except ValueError:
                continue
        if values:
            return f"{len(rows)} rows; n sum={sum(values)}"
    return f"{len(rows)} rows"


def artifact(path: Path, note: str) -> str:
    state = "present" if path.exists() else "missing"
    return f"| `{path.as_posix()}` | {state} | {note} |"


def main() -> int:
    args = parse_args()
    final = args.final_tables_dir
    external = args.external_summary_dir

    lines = [
        "# Publication Evidence Manifest",
        "",
        "This manifest gives reviewers a single index from each major claim or concern to the committed manuscript/result artifact that supports it.",
        "",
        f"Active manuscript: `{args.tex.as_posix()}`",
        "",
        "## Main Evidence Map",
        "",
        "| Claim or concern | Primary manuscript location | Committed evidence artifact | Evidence summary |",
        "| --- | --- | --- | --- |",
        "| Main cross-subject result | Table `tab:cross_subject_main` | `table_allfold_final.csv` | 8 held-out-subject folds x 3 seeds; " + row_count(final / "table_allfold_final.csv") + " |",
        "| Dataset accounting and split construction | Tables `tab:split_accounting`, `tab:session_order_pair_qc`, `tab:fold_difficulty` | `DATASET_CARD.md`, `split_accounting.csv`, `session_order_pair_qc.csv`, `fold_difficulty_qc.csv` | fold-level sequence/pair counts, session/order pair QC, and unresolved fold_07 difficulty are summarized for reviewers |",
        "| Model scope and non-claims | Sec. `Models`, Sec. `Discussion`, Sec. `Limitations` | `MODEL_CARD.md`, `publication_docs_audit.csv` | fixed-order ROI-token Transformer-VLM scope, explicit adjacency non-claim, task-matched component-baseline framing, and non-clinical intended use |",
        "| Reviewer concern coverage | Response memo and preflight audit | `REVIEWER_RESPONSE.md`, `reviewer_response_readiness_audit.csv` | human-readable concern-to-artifact map plus machine-checkable readiness audit |",
        "| Explicit adjacency is not the source of the gain | Tables `tab:adjacency_ablation`, `tab:roi_token_controls`, `tab:adjacency_perturbation`, `tab:edge_bias_followup` | `table_adjacency_ablation.csv`, `table_roi_token_controls.csv`, `final_adjacency_ablation_tests.csv` | no-adj and adjacency variants are statistically tied; ROI-order/gate controls drive the interpretation |",
        "| Session/order confound control | Tables `tab:session_order_pair_qc`, `tab:single_ref_matched`, `tab:single_ref_retrained` | `session_order_pair_qc.csv`, `single_ref_matched_summary.csv`, `single_ref_matched_allseed_summary.csv` | exact anchor-side QC plus eval-only and retrained single-reference controls |",
        "| Implementation reproducibility | Table `tab:implementation_details` | `model_parameter_counts.csv`, `manuscript_publication_claims_audit.csv` | loss, normalization, adjacency construction, architecture, parameter counts, optimizer, folds, and seeds are audited |",
        "| Package reproducibility | `pyproject.toml`, `REPRODUCIBILITY.md` | `package_metadata_audit.csv` | project metadata, dependency extras, and packaged model modules are structurally audited |",
        "| Statistical reporting | Results text and statistical claims | `publication_paired_stats.csv`, `manuscript_stat_claims_audit.csv` | paired fold x seed tests with bootstrap CIs; " + row_count(final / "publication_paired_stats.csv") + " |",
        "| Component baseline framing | Table `tab:sota_baselines` | `table_phase2_sota_graph_baselines.csv` | task-matched component baselines, not full image-reconstruction system claims |",
        "| Semantic alignment control | Table `tab:heldout` | `table_heldout_image.csv` | separates pair discrimination from image/brain retrieval under real CLIP versus random embeddings |",
        "| External validation limits | Tables `tab:external_visual_roi_smoke`, `tab:laion_external_pairwise` | `table_external_visual_roi_smoke.csv`, `external_visual_roi_all4_summary.md`, `laion_fmri_visual_roi_pairwise_tests.csv`, `external_validation_consistency_audit.csv` | four public visual-ROI smoke checks plus LAION paired tests; explicitly not full HCP-MMP external replications |",
        "| Fold_07 robustness | Table `tab:fold_difficulty` | `fold_difficulty_qc.csv` | fold_07 is diagnosed as difficult but left as an unresolved robustness case |",
        "| Reviewer-response coverage | Preflight artifact | `reviewer_response_readiness_audit.csv` | " + status_counts(final / "reviewer_response_readiness_audit.csv") + " |",
        "| Double-blind code sharing | `ANONYMIZATION.md` | `scripts/make_anonymous_submission_bundle.py`, `scripts/verify_anonymous_bundle_manifest.py`, `scripts/smoke_test_anonymous_bundle_archive.py`, `anonymous_bundle_manifest.csv` | Git-history-free anonymous archive workflow plus verifiable per-file source checksums and hardened archive smoke test; do not submit public GitHub metadata |",
        "",
        "## Audit Artifacts",
        "",
        "| Artifact | Status | Purpose |",
        "| --- | --- | --- |",
        artifact(Path("DATASET_CARD.md"), "reviewer-facing dataset accounting, external-validation scope, and large-data policy"),
        artifact(Path("MODEL_CARD.md"), "reviewer-facing model scope, intended use, supported claims, non-claims, and limitations"),
        artifact(Path("REVIEWER_RESPONSE.md"), "human-readable map from likely reviewer concerns to committed manuscript/result evidence"),
        artifact(final / "aaai_publication_readiness_audit.csv", status_counts(final / "aaai_publication_readiness_audit.csv")),
        artifact(final / "publication_artifact_provenance_audit.csv", status_counts(final / "publication_artifact_provenance_audit.csv")),
        artifact(final / "manuscript_publication_claims_audit.csv", status_counts(final / "manuscript_publication_claims_audit.csv")),
        artifact(final / "publication_docs_audit.csv", status_counts(final / "publication_docs_audit.csv")),
        artifact(final / "publication_evidence_manifest_audit.csv", status_counts(final / "publication_evidence_manifest_audit.csv")),
        artifact(final / "package_metadata_audit.csv", status_counts(final / "package_metadata_audit.csv")),
        artifact(final / "anonymous_bundle_manifest.csv", "per-file source byte counts and SHA-256 checksums for verifier-backed anonymous bundle contents"),
        artifact(final / "reviewer_response_readiness_audit.csv", status_counts(final / "reviewer_response_readiness_audit.csv")),
        artifact(final / "manuscript_table_values_audit.csv", status_counts(final / "manuscript_table_values_audit.csv")),
        artifact(final / "manuscript_stat_claims_audit.csv", status_counts(final / "manuscript_stat_claims_audit.csv")),
        artifact(final / "external_validation_consistency_audit.csv", status_counts(final / "external_validation_consistency_audit.csv")),
        artifact(external / "external_visual_roi_all4_summary.md", "external smoke-validation summary"),
        artifact(external / "laion_fmri_visual_roi_pairwise_tests.csv", "LAION paired external visual-ROI tests"),
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
