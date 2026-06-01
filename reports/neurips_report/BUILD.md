# ReGraph-VLM NeurIPS-Style Report Build Notes

This directory contains the NeurIPS-style report source:

- `may30.tex`
- `references.bib`
- `neurips_2025.sty`
- `figures/*.pdf` and `figures/*.png`

Run the full preflight from the project root before compiling or submitting:

```bash
python3 scripts/run_publication_preflight.py
```

This regenerates lightweight publication tables, runs the AAAI artifact audit,
verifies publication artifact provenance, runs the full manuscript/result audit,
checks README/BUILD consistency, verifies key manuscript table values and
statistical claims against committed CSV artifacts, verifies reviewer-response
readiness, runs the manuscript-only audit, and reports whether a local TeX
compiler is available. The manuscript audit also enforces framing guardrails for
adjacency, task-matched component baselines, external smoke validation, fold_07
robustness, and implementation details. GitHub Actions
installs a TeX distribution with recommended/extra LaTeX packages and runs the
compile-required preflight on pushes to `main` and pull requests. CI uses
`--require-clean`, so tracked-file drift and newly generated untracked artifacts
both fail the build.

The reviewer-response readiness audit maps likely reviewer concerns to concrete
manuscript/result evidence: dataset accounting, session/order controls,
adjacency limitations, ROI-token/gate mechanism controls, implementation detail,
paired statistics, component-baseline framing, semantic-alignment controls,
external-validation caveats, and fold_07 robustness.

The preflight also writes a compact Publication Evidence Manifest:

```text
preproc_v0/repetition_familiarity/results/final_tables/publication_evidence_manifest.md
```

Use it as the reviewer-facing index from each major claim or caveat to the
committed manuscript table and result artifact that supports it.

For double-blind review, do not submit the public GitHub URL or a Git clone.
Use the Anonymous Submission Bundle workflow instead; see `ANONYMIZATION.md`.
The non-mutating bundle check is included in the publication preflight:

```bash
python3 scripts/make_anonymous_submission_bundle.py --dry-run
```

To build the archive for submission:

```bash
python3 scripts/make_anonymous_submission_bundle.py
```

The archive writer normalizes tar/gzip metadata and reports a SHA-256 checksum,
so repeated builds from the same committed inputs should be byte-stable.

If a TeX distribution is installed, compile through the same preflight:

```bash
python3 scripts/run_publication_preflight.py --compile
```

For a manuscript-only audit without regenerating result artifacts:

```bash
python3 scripts/audit_manuscript_publication_claims.py \
  --tex reports/neurips_report/may30.tex \
  --manuscript-only \
  --output-dir /tmp/regraph_report_preflight
```

The manuscript-only audit checks anonymity strings, fixed-adjacency overclaims,
framing guardrails for adjacency, component baselines, external validation,
fold_07, and implementation details, duplicate labels, unresolved refs, required
labels, citation coverage in `references.bib`, figure file availability, and
LaTeX environment balance.

Local note: this machine currently does not have `pdflatex`, `xelatex`, `lualatex`, `latexmk`, or `tectonic` installed, so the PDF was not compiled locally in this step.
