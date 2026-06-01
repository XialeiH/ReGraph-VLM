# Reproducibility Notes

This repository tracks source code, manuscript files, lightweight result
summaries, and publication-facing audit artifacts. It does not track raw fMRI
data, beta volumes, checkpoints, generated `.pt` datasets, Slurm logs, or other
large intermediate files.

## Environments

For publication audits and manuscript checks:

```bash
python3 -m pip install pandas
python3 scripts/run_publication_preflight.py
```

The full CI path also needs a TeX distribution with `latexmk`,
`texlive-latex-base`, `texlive-latex-recommended`, `texlive-latex-extra`, and
`texlive-fonts-recommended`.

For model execution, parameter-count verification, and training/evaluation
scripts, use Python 3.9 or newer with:

```text
numpy
pandas
scipy
scikit-learn
torch
tqdm
matplotlib
h5py
nibabel
nilearn
requests
Pillow
```

The repository metadata exposes the same dependency tiers:

```bash
python3 -m pip install -e .
python3 -m pip install -e '.[publication]'
python3 -m pip install -e '.[neuro]'
python3 -m pip install -e '.[dev]'
```

CLIP feature export additionally requires either OpenAI CLIP or OpenCLIP:

```bash
python3 -m pip install git+https://github.com/openai/CLIP.git
# or
python3 -m pip install open_clip_torch
```

`torch-geometric` is needed only for the legacy bridge-generation and standard
graph-baseline code paths. The final ReGraph-VLM ROI-token transformer code in
`models/` does not require `torch-geometric`.

## Data Policy

Large fMRI datasets should be downloaded and processed directly on remote HPC
scratch storage, not onto a local laptop checkout. The repository contains
helper scripts for NSD preprocessing and public external-validation probes, but
the raw datasets and generated tensors are intentionally excluded from Git.
Protected public external-validation download and probe scripts call
`scripts/external_data_policy.py` so their output roots must resolve to remote
HPC scratch paths. The publication preflight runs
`scripts/audit_external_data_policy.py` as an external data policy audit.

Publication-facing external validation is limited to committed summary tables
under `external_validation/summary/`. These are smoke checks on public
visual-ROI summaries or derivatives, not full HCP-MMP 180-ROI external
replications.

For fold-level split counts, session/order QC, fold difficulty, external
validation scope, and large-data handling, see `DATASET_CARD.md`.
For model scope, intended use, supported claims, non-claims, and limitations,
see `MODEL_CARD.md`.
For the reviewer-response concern-to-evidence checklist, see
`REVIEWER_RESPONSE.md`.

## Verification Commands

Run the full publication preflight from the repository root:

```bash
python3 scripts/run_publication_preflight.py
# or
make preflight
```

The preflight includes a structural package metadata audit for `pyproject.toml`
so the project name, dependency extras, and packaged `models/` module stay
aligned with the publication code.

The Makefile target audit is implemented in
`scripts/audit_makefile_targets.py`. It verifies that reviewer-facing `make`
targets still call the canonical publication scripts for `may30.tex` and writes
`preproc_v0/repetition_familiarity/results/final_tables/makefile_targets_audit.csv`.
The CI workflow audit is implemented in `scripts/audit_ci_workflow.py`. It
verifies that the GitHub Actions publication workflow still runs the
compile-required, clean-worktree preflight path and writes
`preproc_v0/repetition_familiarity/results/final_tables/ci_workflow_audit.csv`.
The result artifact schema audit is implemented in
`scripts/audit_result_artifact_schemas.py`. It verifies required CSV columns,
numeric fields, minimum row counts, and source metadata for reviewer-facing
result artifacts, then writes
`preproc_v0/repetition_familiarity/results/final_tables/result_artifact_schema_audit.csv`.
The result value-range audit is implemented in
`scripts/audit_result_value_ranges.py`. It verifies metric ranges, p-values,
correlations, standard deviations, counts, and split/QC invariants, then writes
`preproc_v0/repetition_familiarity/results/final_tables/result_value_range_audit.csv`.

When PyTorch is installed, the preflight also checks that
`model_parameter_counts.csv` matches instantiated `ReGraphVLM` modules. The
same check can be run directly:

```bash
python3 scripts/verify_model_parameter_counts.py
# or
make parameter-counts
```

For double-blind submission packaging:

```bash
python3 scripts/make_anonymous_submission_bundle.py --dry-run
python3 scripts/make_anonymous_submission_bundle.py
# or
make bundle-check
make bundle
```

The dry-run checks the exact bundle contents, scans every included file at the
byte level for deanonymizing strings, and reports a deterministic SHA-256
checksum.
The full publication preflight additionally regenerates:

```text
preproc_v0/repetition_familiarity/results/final_tables/anonymous_bundle_manifest.csv
```

This manifest records each included source path, source byte count, and SHA-256
checksum for reviewer-side artifact checking. The manifest excludes its own row
to avoid self-referential checksum drift.
`--manifest-output` also supports external sidecar paths outside the repository;
the publication preflight tests both the in-bundle manifest path and a temporary
sidecar manifest path.
Verify the committed source tree or an extracted anonymous bundle with:

```bash
python3 scripts/verify_anonymous_bundle_manifest.py
# or
make bundle-verify
```

To test the full archive/extract/verify reviewer path:

```bash
python3 scripts/smoke_test_anonymous_bundle_archive.py
# or
make bundle-smoke
```

The smoke test rebuilds the archive twice and checks byte-identical output. It
also rejects path traversal, Git metadata, symlink/hardlink entries, non-regular
archive members, and archive files that are not accounted for by the manifest.
It runs an extracted anonymous bundle preflight once, using a recursion guard so
the nested smoke test does not call itself indefinitely. When a TeX tool is
available, the extracted-bundle preflight is compile-required.

## Result Provenance

The main reviewer-facing result index is:

```text
preproc_v0/repetition_familiarity/results/final_tables/publication_evidence_manifest.md
```

It maps the manuscript claims to committed CSV/Markdown artifacts, including
main all-fold results, adjacency ablations, session/order controls, external
smoke validation, fold-level diagnostics, paired statistics, and anonymous
bundle readiness.

The dataset-specific reviewer index is:

```text
DATASET_CARD.md
```

The model-specific reviewer index is:

```text
MODEL_CARD.md
```

The reviewer-response memo is:

```text
REVIEWER_RESPONSE.md
```
