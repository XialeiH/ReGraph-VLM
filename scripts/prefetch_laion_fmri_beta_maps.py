#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from external_data_policy import enforce_hpc_external_path


BASE_URL = "https://laion-fmri.s3.amazonaws.com"
DEFAULT_SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefetch LAION-fMRI single-trial beta maps in parallel.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--max-session", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def beta_key(subject: str, session: str) -> str:
    name = f"{subject}_{session}_task-images_space-T1w_stat-effect_desc-SingletrialBetas_statmap.nii.gz"
    return f"derivatives/glmsingle-tedana/{subject}/{session}/func/{name}"


def local_name_from_key(key: str) -> str:
    return key.replace("/", "__")


def s3_url(key: str) -> str:
    return f"{BASE_URL}/{urllib.parse.quote(key)}"


def download_one(root: Path, subject: str, session: str) -> tuple[str, str, str, int]:
    key = beta_key(subject, session)
    path = root / "downloads" / subject / local_name_from_key(key)
    if path.exists() and path.stat().st_size > 0:
        return subject, session, "exists", path.stat().st_size
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.prefetch.{os.getpid()}.tmp")
    try:
        with urllib.request.urlopen(s3_url(key), timeout=300) as response, tmp.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        if path.exists() and path.stat().st_size > 0:
            tmp.unlink(missing_ok=True)
            return subject, session, "race_exists", path.stat().st_size
        tmp.replace(path)
        return subject, session, "downloaded", path.stat().st_size
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        return subject, session, f"error:{type(exc).__name__}:{exc}", 0


def main() -> None:
    args = parse_args()
    args.root = enforce_hpc_external_path(args.root, "LAION-fMRI beta prefetch root")
    tasks = [
        (subject, f"ses-{session_id:02d}")
        for subject in args.subjects
        for session_id in range(1, args.max_session + 1)
    ]
    print(f"Prefetching {len(tasks)} LAION-fMRI beta maps with workers={args.workers}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, args.root, subject, session) for subject, session in tasks]
        for idx, future in enumerate(as_completed(futures), start=1):
            print(idx, future.result(), flush=True)


if __name__ == "__main__":
    main()
