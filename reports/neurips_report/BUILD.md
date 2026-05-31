# ReGraph-VLM NeurIPS-Style Report Build Notes

This directory contains the NeurIPS-style report source:

- `may30.tex`
- `references.bib`
- `neurips_2025.sty`
- `figures/*.pdf` and `figures/*.png`

Run the manuscript-only preflight audit from the project root before compiling:

```bash
python3 scripts/audit_manuscript_publication_claims.py \
  --tex reports/neurips_report/may30.tex \
  --manuscript-only \
  --output-dir /tmp/regraph_report_preflight
```

The audit checks anonymity strings, fixed-adjacency overclaims, duplicate labels,
unresolved refs, required labels, citation coverage in `references.bib`, figure
file availability, and LaTeX environment balance.

For the full publication artifact audit, run the same script without
`--manuscript-only` after syncing the final result tables into
`preproc_v0/repetition_familiarity/results/final_tables`.

To compile on a machine with a TeX distribution:

```bash
cd "/Users/xialeihuang/Desktop/NYU Shanghai courses/Machine Learning with Graphs/Final_Project/reports/neurips_report"
pdflatex may30.tex
bibtex may30
pdflatex may30.tex
pdflatex may30.tex
```

Local note: this machine currently does not have `pdflatex`, `xelatex`, `lualatex`, `latexmk`, or `tectonic` installed, so the PDF was not compiled locally in this step.
