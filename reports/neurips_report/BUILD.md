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
statistical claims against committed CSV artifacts, runs the manuscript-only
audit, and reports whether a local TeX compiler is available. GitHub Actions
installs a TeX distribution with recommended/extra LaTeX packages and runs the
compile-required preflight on pushes to `main` and pull requests. CI uses
`--require-clean`, so tracked-file drift and newly generated untracked artifacts
both fail the build.

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

The manuscript-only audit checks anonymity strings, fixed-adjacency overclaims, duplicate labels,
unresolved refs, required labels, citation coverage in `references.bib`, figure
file availability, and LaTeX environment balance.

Local note: this machine currently does not have `pdflatex`, `xelatex`, `lualatex`, `latexmk`, or `tectonic` installed, so the PDF was not compiled locally in this step.
