# ReGraph-VLM NeurIPS-Style Report Build Notes

This directory contains the NeurIPS-style report source:

- `regraph_vlm_report.tex`
- `references.bib`
- `neurips_2025.sty`
- `figures/*.pdf`

To compile on a machine with a TeX distribution:

```bash
cd "/Users/xialeihuang/Desktop/NYU Shanghai courses/Machine Learning with Graphs/Final_Project/reports/neurips_report"
pdflatex regraph_vlm_report.tex
bibtex regraph_vlm_report
pdflatex regraph_vlm_report.tex
pdflatex regraph_vlm_report.tex
```

Local note: this machine currently does not have `pdflatex`, `xelatex`, `lualatex`, `latexmk`, or `tectonic` installed, so the PDF was not compiled locally in this step.
