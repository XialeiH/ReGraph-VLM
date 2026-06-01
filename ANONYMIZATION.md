# Anonymous Submission Bundle

For double-blind review, do not submit the public GitHub URL or a Git clone.
Public GitHub repository metadata, issue/PR metadata, and commit history can
reveal author identity even when the manuscript source is anonymous.

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
scans every included file at the byte level for project-specific deanonymizing
strings before writing the archive, including PDFs and image metadata. Archive
metadata is normalized, so repeated builds from the same committed inputs are
byte-stable and report a SHA-256 checksum.

For a non-mutating check, run:

```bash
python3 scripts/make_anonymous_submission_bundle.py --dry-run
```

The dry run does not write the archive, but it computes the deterministic
archive checksum that the full build should report.

The publication preflight also writes a reviewer-facing per-file source
manifest:

```text
preproc_v0/repetition_familiarity/results/final_tables/anonymous_bundle_manifest.csv
```

The manifest lists each included source path, source byte count, and SHA-256
checksum. It excludes its own row so that regenerating the manifest does not
change its own expected checksum.
After extracting a submitted bundle, verify the files against the manifest:

```bash
python3 scripts/verify_anonymous_bundle_manifest.py
```

To smoke-test the full bundle workflow before submission:

```bash
python3 scripts/smoke_test_anonymous_bundle_archive.py
```

The smoke test rejects path traversal, Git metadata, symlink/hardlink entries,
non-regular archive members, and archive files that are not accounted for by
the manifest. It also runs an extracted anonymous bundle preflight once, using a
recursion guard so the nested smoke test does not call itself indefinitely. When
a TeX tool is available, the extracted-bundle preflight is compile-required.
