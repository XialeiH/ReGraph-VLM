# Anonymous Submission Bundle

For double-blind review, do not submit the public GitHub URL or a Git clone.
GitHub repository metadata, workflow metadata, and commit history can reveal
author identity even when the manuscript source is anonymous.

Instead, build a Git-history-free anonymous submission bundle:

```bash
python3 scripts/make_anonymous_submission_bundle.py
```

The default output is:

```text
dist/regraph_vlm_anonymous_submission.tar.gz
```

The bundle is built from an allowlist of committed source, manuscript, figure,
and lightweight result artifacts. It excludes `.git`, checkpoints, raw data,
HPC scratch scripts, Slurm logs, and local generated artifacts. The script also
scans included text files for project-specific deanonymizing strings before
writing the archive.

For a non-mutating check, run:

```bash
python3 scripts/make_anonymous_submission_bundle.py --dry-run
```
