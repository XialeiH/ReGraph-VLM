#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from external_data_policy import enforce_hpc_external_path


S3_LIST_URL = "https://laion-fmri.s3.amazonaws.com/"
XML_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


@dataclass(frozen=True)
class S3Object:
    key: str
    size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe public LAION-fMRI S3 metadata without downloading large arrays.")
    parser.add_argument("--output-dir", type=Path, default=Path("external_validation/laion_fmri_probe"))
    parser.add_argument("--subjects", nargs="+", default=["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"])
    parser.add_argument("--max-pages", type=int, default=20)
    return parser.parse_args()


def list_prefix(prefix: str, max_pages: int) -> list[S3Object]:
    objects: list[S3Object] = []
    token: str | None = None
    for _ in range(max_pages):
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        url = S3_LIST_URL + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=60) as response:
            xml_text = response.read()
        root = ET.fromstring(xml_text)
        for item in root.findall("s3:Contents", XML_NS):
            key = item.findtext("s3:Key", default="", namespaces=XML_NS)
            size = int(item.findtext("s3:Size", default="0", namespaces=XML_NS))
            if key:
                objects.append(S3Object(key=key, size=size))
        truncated = root.findtext("s3:IsTruncated", default="false", namespaces=XML_NS) == "true"
        token = root.findtext("s3:NextContinuationToken", default="", namespaces=XML_NS) or None
        if not truncated or not token:
            break
    return objects


def summarize_subject(subject: str, objects: list[S3Object]) -> dict[str, object]:
    sessions = sorted(set(re.findall(r"/(ses-[^/]+)/", "\n".join(obj.key for obj in objects))))
    trial_tsv = [obj for obj in objects if obj.key.endswith("_desc-SingletrialBetas_trials.tsv")]
    beta_statmaps = [obj for obj in objects if "_stat-effect_desc-SingletrialBetas_statmap.nii.gz" in obj.key]
    model_h5 = [obj for obj in objects if obj.key.endswith("_desc-GLMsingle_model.h5")]
    noise_maps = [obj for obj in objects if "_desc-Noiseceiling_statmap.nii.gz" in obj.key]
    total_size = sum(obj.size for obj in objects)
    beta_size = sum(obj.size for obj in beta_statmaps)
    return {
        "subject": subject,
        "n_objects_listed": len(objects),
        "n_sessions": len(sessions),
        "sessions": ",".join(sessions),
        "n_singletrial_tsv": len(trial_tsv),
        "n_singletrial_beta_statmaps": len(beta_statmaps),
        "n_glmsingle_model_h5": len(model_h5),
        "n_noiseceiling_maps": len(noise_maps),
        "listed_total_gb": total_size / 1e9,
        "singletrial_beta_statmaps_gb": beta_size / 1e9,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# LAION-fMRI Public S3 Probe",
        "",
        "This probe lists public CC0 LAION-fMRI metadata from the AWS S3 HTTPS endpoint. It does not download raw stimulus images or large beta arrays.",
        "",
        "| Subject | Sessions | Trial TSVs | Singletrial beta maps | Listed GB | Beta-map GB |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {subject} | {n_sessions} | {n_singletrial_tsv} | {n_singletrial_beta_statmaps} | {listed_total_gb:.3f} | {singletrial_beta_statmaps_gb:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: if all subjects expose trial TSVs and single-trial beta statmaps, LAION-fMRI is technically feasible as a future full external validation target. A real validation still requires ROI projection/extraction and stimulus/metadata alignment.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    enforce_hpc_external_path(args.output_dir, "LAION-fMRI S3 probe output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for subject in args.subjects:
        prefix = f"derivatives/glmsingle-tedana/{subject}/"
        objects = list_prefix(prefix, max_pages=args.max_pages)
        rows.append(summarize_subject(subject, objects))
    write_csv(args.output_dir / "laion_fmri_public_s3_probe.csv", rows)
    write_md(args.output_dir / "laion_fmri_public_s3_probe.md", rows)
    print((args.output_dir / "laion_fmri_public_s3_probe.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
