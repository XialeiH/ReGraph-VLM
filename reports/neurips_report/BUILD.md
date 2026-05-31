# ReGraph-VLM NeurIPS-Style Report Build Notes

This directory contains the NeurIPS-style report source:

- `may30.tex`
- `references.bib`
- `neurips_2025.sty`
- `figures/*.pdf` and `figures/*.png`

To compile on a machine with a TeX distribution:

```bash
cd "/Users/xialeihuang/Desktop/NYU Shanghai courses/Machine Learning with Graphs/Final_Project/reports/neurips_report"
pdflatex may30.tex
bibtex may30
pdflatex may30.tex
pdflatex may30.tex
```

Local note: this machine currently does not have `pdflatex`, `xelatex`, `lualatex`, `latexmk`, or `tectonic` installed, so the PDF was not compiled locally in this step.
