#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen, urlretrieve
from xml.etree import ElementTree as ET


S3_BUCKET = "https://natural-scenes-dataset.s3.amazonaws.com"
S3_OBJECT_BASE = "https://natural-scenes-dataset.s3-us-east-2.amazonaws.com"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

ESSENTIAL_KEYS = [
    "nsddata/experiments/nsd/nsd_expdesign.mat",
    "nsddata/experiments/nsd/nsd_stim_info_merged.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a minimal or incremental subset of the Natural Scenes Dataset."
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/nsd_raw"),
        help="Destination directory for downloaded files.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        type=int,
        default=list(range(1, 9)),
        help="Subject IDs to include for ROI and beta downloads.",
    )
    parser.add_argument(
        "--include-roi",
        action="store_true",
        help="Download func1pt8mm ROI files for the selected subjects.",
    )
    parser.add_argument(
        "--include-betas",
        action="store_true",
        help="Download selected beta sessions for the chosen subjects.",
    )
    parser.add_argument(
        "--sessions",
        nargs="*",
        type=int,
        default=[],
        help="Session IDs for beta downloads, for example --sessions 1 2 3.",
    )
    parser.add_argument(
        "--include-stimuli",
        action="store_true",
        help="Download nsd_stimuli.hdf5. This file is about 39.6 GB.",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only list files and sizes without downloading.",
    )
    return parser.parse_args()


def format_subject(subject: int) -> str:
    return f"subj{subject:02d}"


def format_session(session: int) -> str:
    return f"session{session:02d}"


def list_prefix(prefix: str) -> list[tuple[str, int]]:
    url = f"{S3_BUCKET}/?prefix={quote(prefix)}&max-keys=1000"
    with urlopen(url, timeout=60) as response:
        root = ET.fromstring(response.read())
    entries = []
    for item in root.findall("s3:Contents", S3_NS):
        key = item.find("s3:Key", S3_NS).text
        size = int(item.find("s3:Size", S3_NS).text)
        entries.append((key, size))
    return entries


def object_size(key: str) -> int:
    entries = list_prefix(key)
    for entry_key, size in entries:
        if entry_key == key:
            return size
    raise FileNotFoundError(f"Could not find object size for {key}")


def build_manifest(args: argparse.Namespace) -> list[tuple[str, int]]:
    manifest: list[tuple[str, int]] = []
    for key in ESSENTIAL_KEYS:
        manifest.append((key, object_size(key)))

    if args.include_roi:
        for subject in args.subjects:
            prefix = f"nsddata/ppdata/{format_subject(subject)}/func1pt8mm/roi/"
            manifest.extend(list_prefix(prefix))

    if args.include_betas:
        if not args.sessions:
            raise SystemExit("--include-betas requires at least one session via --sessions")
        for subject in args.subjects:
            subj = format_subject(subject)
            for session in args.sessions:
                sess = format_session(session)
                key = (
                    f"nsddata_betas/ppdata/{subj}/func1pt8mm/"
                    f"betas_fithrf_GLMdenoise_RR/betas_{sess}.nii.gz"
                )
                manifest.append((key, object_size(key)))

    if args.include_stimuli:
        key = "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
        manifest.append((key, object_size(key)))

    dedup: dict[str, int] = {}
    for key, size in manifest:
        dedup[key] = size
    return sorted(dedup.items())


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def download_file(dest_root: Path, key: str, size: int) -> None:
    out_path = dest_root / key
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size == size:
        print(f"skip {key} ({human_size(size)})")
        return
    url = f"{S3_OBJECT_BASE}/{quote(key)}"
    print(f"get  {key} -> {out_path} ({human_size(size)})")
    urlretrieve(url, out_path)


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args)
    total_size = sum(size for _, size in manifest)
    print(f"files: {len(manifest)}")
    print(f"total: {human_size(total_size)}")
    for key, size in manifest[:20]:
        print(f"  {human_size(size):>9}  {key}")
    if len(manifest) > 20:
        print(f"  ... {len(manifest) - 20} more files")

    if args.estimate_only:
        return 0

    for key, size in manifest:
        download_file(args.dest, key, size)

    print(f"done: {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
