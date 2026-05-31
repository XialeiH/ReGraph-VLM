#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


BASE_URL = "https://laion-fmri.s3.amazonaws.com"
XML_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
FEATURE_NAMES = ["mean_beta", "std_beta", "q90_beta", "positive_fraction"]
DEFAULT_SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
ROI_RE = re.compile(r"/(?P<category>[^/]+)/[^/]+_space-T1w_res-1pt8_label-(?P<label>.+)_mask\.nii\.gz$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LAION-fMRI public visual-ROI scalar4 tensors.")
    parser.add_argument("--root", type=Path, default=Path("external_validation/laion_fmri"))
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("external_validation/laion_fmri_probe/trial_metadata/tsv"),
    )
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--max-session", type=int, default=10)
    parser.add_argument("--min-repeats", type=int, default=3)
    parser.add_argument("--max-labels", type=int, default=200)
    parser.add_argument("--max-rois", type=int, default=64)
    parser.add_argument("--top-quantile", type=float, default=0.90)
    parser.add_argument("--keep-downloads", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--selection-manifest", type=Path, default=None)
    return parser.parse_args()


def s3_url(key: str) -> str:
    return f"{BASE_URL}/{urllib.parse.quote(key)}"


def list_s3_prefix(prefix: str, max_pages: int = 20) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    token: str | None = None
    for _ in range(max_pages):
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        with urllib.request.urlopen(BASE_URL + "?" + urllib.parse.urlencode(params), timeout=90) as response:
            root = ET.fromstring(response.read())
        for item in root.findall("s3:Contents", XML_NS):
            key = item.findtext("s3:Key", default="", namespaces=XML_NS)
            size = int(item.findtext("s3:Size", default="0", namespaces=XML_NS))
            if key:
                out.append((key, size))
        truncated = root.findtext("s3:IsTruncated", default="false", namespaces=XML_NS) == "true"
        token = root.findtext("s3:NextContinuationToken", default="", namespaces=XML_NS) or None
        if not truncated or not token:
            break
    return out


def download(key: str, path: Path, local_only: bool = False) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    if local_only:
        raise FileNotFoundError(f"Required LAION-fMRI file is missing in local-only mode: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with urllib.request.urlopen(s3_url(key), timeout=300) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    tmp.replace(path)


def beta_key(subject: str, session: str) -> str:
    name = f"{subject}_{session}_task-images_space-T1w_stat-effect_desc-SingletrialBetas_statmap.nii.gz"
    return f"derivatives/glmsingle-tedana/{subject}/{session}/func/{name}"


def trials_key(subject: str, session: str) -> str:
    name = f"{subject}_{session}_task-images_desc-SingletrialBetas_trials.tsv"
    return f"derivatives/glmsingle-tedana/{subject}/{session}/func/{name}"


def local_name_from_key(key: str) -> str:
    return key.replace("/", "__")


def read_trial_rows(metadata_dir: Path, subject: str, max_session: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for session_id in range(1, max_session + 1):
        session = f"ses-{session_id:02d}"
        path = metadata_dir / subject / f"{subject}_{session}_trials.tsv"
        if not path.exists():
            raise FileNotFoundError(f"Missing LAION trial TSV metadata: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                rows.append(
                    {
                        "subject": subject,
                        "session": str(row["session"]),
                        "session_id": session_id,
                        "beta_index": int(row["beta_index"]),
                        "label": str(row["label"]),
                    }
                )
    return rows


def select_labels(
    rows_by_subject: dict[str, list[dict[str, object]]],
    subjects: list[str],
    min_repeats: int,
    max_labels: int,
) -> tuple[list[str], dict[str, dict[str, list[dict[str, object]]]]]:
    by_label: dict[str, dict[str, list[dict[str, object]]]] = {subject: defaultdict(list) for subject in subjects}
    for subject, rows in rows_by_subject.items():
        for row in rows:
            by_label[subject][str(row["label"])].append(row)
        for label in by_label[subject]:
            by_label[subject][label] = sorted(by_label[subject][label], key=lambda r: (int(r["session_id"]), int(r["beta_index"])))

    common = sorted(
        set.intersection(
            *[
                {label for label, occurrences in by_label[subject].items() if len(occurrences) >= min_repeats}
                for subject in subjects
            ]
        )
    )
    if max_labels > 0:
        common = common[:max_labels]
    selected = {
        subject: {label: by_label[subject][label][:min_repeats] for label in common}
        for subject in subjects
    }
    return common, selected


def common_roi_keys(subjects: list[str], max_rois: int) -> list[tuple[str, str, dict[str, str]]]:
    subject_maps: dict[str, dict[tuple[str, str], str]] = {}
    for subject in subjects:
        mapping: dict[tuple[str, str], str] = {}
        for key, _ in list_s3_prefix(f"derivatives/rois/{subject}/"):
            match = ROI_RE.search(key)
            if not match:
                continue
            category = match.group("category")
            label = match.group("label")
            mapping[(category, label)] = key
        subject_maps[subject] = mapping
    common = sorted(set.intersection(*(set(mapping) for mapping in subject_maps.values())))
    if not common:
        raise RuntimeError("No common LAION-fMRI T1w 1.8mm ROI masks found across subjects.")
    if max_rois > 0:
        common = common[:max_rois]
    return [(category, label, {subject: subject_maps[subject][(category, label)] for subject in subjects}) for category, label in common]


def scalar4(values: np.ndarray, top_quantile: float) -> np.ndarray:
    out = np.zeros((values.shape[0], len(FEATURE_NAMES)), dtype=np.float32)
    if values.shape[1] == 0:
        return out
    out[:, 0] = values.mean(axis=1, dtype=np.float64).astype(np.float32)
    out[:, 1] = values.std(axis=1, dtype=np.float64).astype(np.float32)
    out[:, 2] = np.quantile(values, top_quantile, axis=1).astype(np.float32)
    out[:, 3] = (values > 0).mean(axis=1, dtype=np.float64).astype(np.float32)
    return out


def load_masks(
    subject: str,
    roi_keys: list[tuple[str, str, dict[str, str]]],
    download_dir: Path,
    beta_shape: tuple[int, int, int],
    local_only: bool,
) -> tuple[list[str], list[np.ndarray], list[int]]:
    labels: list[str] = []
    masks: list[np.ndarray] = []
    counts: list[int] = []
    for category, label, key_by_subject in roi_keys:
        key = key_by_subject[subject]
        path = download_dir / local_name_from_key(key)
        download(key, path, local_only=local_only)
        arr = np.asanyarray(nib.load(str(path)).dataobj) > 0
        if arr.shape != beta_shape:
            raise ValueError(f"ROI mask shape mismatch for {subject} {category}:{label}: {arr.shape} vs {beta_shape}")
        labels.append(f"{category}:{label}")
        flat = np.flatnonzero(arr.reshape(-1))
        masks.append(flat)
        counts.append(int(flat.size))
    return labels, masks, counts


def export_subject(
    subject: str,
    labels: list[str],
    selected: dict[str, list[dict[str, object]]],
    roi_keys: list[tuple[str, str, dict[str, str]]],
    args: argparse.Namespace,
) -> dict[str, object]:
    out_dir = args.root / "visual_roi_scalar4_laion"
    download_dir = args.root / "downloads" / subject
    out_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    sessions = sorted({str(row["session"]) for label in labels for row in selected[label]})
    first_beta_path = download_dir / local_name_from_key(beta_key(subject, sessions[0]))
    download(beta_key(subject, sessions[0]), first_beta_path, local_only=args.local_only)
    first_img = nib.load(str(first_beta_path))
    beta_shape = tuple(int(v) for v in first_img.shape[:3])
    roi_labels, roi_indices, roi_counts = load_masks(subject, roi_keys, download_dir, beta_shape, args.local_only)

    n_trials = len(labels) * args.min_repeats
    x = np.zeros((n_trials, 180, len(FEATURE_NAMES)), dtype=np.float32)
    image_labels: list[str] = []
    repetitions: list[int] = []
    source_rows: list[dict[str, object]] = []
    position_by_key: dict[tuple[str, int], int] = {}
    row_idx = 0
    for label in labels:
        for rep, row in enumerate(selected[label], start=1):
            image_labels.append(label)
            repetitions.append(rep)
            position_by_key[(str(row["session"]), int(row["beta_index"]))] = row_idx
            source_rows.append({"row_index": row_idx, "label": label, "repetition": rep, **row})
            row_idx += 1

    for session in sessions:
        key = beta_key(subject, session)
        beta_path = download_dir / local_name_from_key(key)
        download(key, beta_path, local_only=args.local_only)
        img = nib.load(str(beta_path))
        if tuple(img.shape[:3]) != beta_shape:
            raise ValueError(f"Beta shape changed for {subject} {session}: {img.shape[:3]} vs {beta_shape}")
        selected_indices = sorted(
            int(row["beta_index"])
            for label in labels
            for row in selected[label]
            if str(row["session"]) == session
        )
        if not selected_indices:
            continue
        volume = np.asanyarray(img.dataobj[..., selected_indices], dtype=np.float32)
        if volume.ndim == 3:
            volume = volume[..., None]
        flat = volume.reshape(-1, len(selected_indices))
        out_positions = [position_by_key[(session, beta_idx)] for beta_idx in selected_indices]
        for node_idx, mask_idx in enumerate(roi_indices):
            values = flat[mask_idx, :].T
            x[out_positions, node_idx, :] = scalar4(values, args.top_quantile)

    node_labels = roi_labels + [f"pad_{idx:03d}" for idx in range(len(roi_labels) + 1, 181)]
    voxel_counts = np.zeros(180, dtype=np.int32)
    voxel_counts[: len(roi_counts)] = np.asarray(roi_counts, dtype=np.int32)
    payload = {
        "x": torch.from_numpy(x),
        "subject": subject,
        "image_label": image_labels,
        "repetition": torch.tensor(repetitions, dtype=torch.int16),
        "feature_names": FEATURE_NAMES,
        "node_set_name": "LAION-fMRI public visual ROI masks padded to 180 nodes",
        "node_labels": node_labels,
        "source_roi_labels": roi_labels,
        "voxel_counts": torch.from_numpy(voxel_counts),
        "source_rows": source_rows,
    }
    torch.save(payload, out_dir / f"{subject}_laion_visual_roi_scalar4.pt")
    with (out_dir / f"{subject}_laion_visual_roi_scalar4_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)
    qc = {
        "subject": subject,
        "n_trials": int(n_trials),
        "n_images": int(len(labels)),
        "n_sessions_downloaded": int(len(sessions)),
        "sessions": sessions,
        "feature_shape": list(x.shape),
        "n_rois": len(roi_labels),
        "nan_count": int(np.isnan(x).sum()),
        "inf_count": int(np.isinf(x).sum()),
        "min_roi_voxels": int(min(roi_counts)),
        "max_roi_voxels": int(max(roi_counts)),
    }
    (out_dir / f"{subject}_laion_visual_roi_scalar4_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    if not args.keep_downloads:
        for path in download_dir.glob("*SingletrialBetas_statmap.nii.gz"):
            path.unlink(missing_ok=True)
    print(json.dumps(qc), flush=True)
    return qc


def manifest_roi_keys(rows: list[dict[str, object]]) -> list[tuple[str, str, dict[str, str]]]:
    return [
        (
            str(row["category"]),
            str(row["label"]),
            {str(subject): str(key) for subject, key in row["keys"].items()},
        )
        for row in rows
    ]


def download_selection_files(
    subjects: list[str],
    selected: dict[str, dict[str, list[dict[str, object]]]],
    labels: list[str],
    roi_keys: list[tuple[str, str, dict[str, str]]],
    root: Path,
) -> dict[str, object]:
    downloads: dict[str, object] = {}
    for subject in subjects:
        download_dir = root / "downloads" / subject
        download_dir.mkdir(parents=True, exist_ok=True)
        mask_keys = [key_by_subject[subject] for _, _, key_by_subject in roi_keys]
        sessions = sorted({str(row["session"]) for label in labels for row in selected[subject][label]})
        beta_keys = [beta_key(subject, session) for session in sessions]
        for key in mask_keys + beta_keys:
            download(key, download_dir / local_name_from_key(key), local_only=False)
        downloads[subject] = {
            "n_masks": len(mask_keys),
            "n_beta_maps": len(beta_keys),
            "sessions": sessions,
        }
        print(json.dumps({"downloaded": subject, **downloads[subject]}), flush=True)
    return downloads


def main() -> None:
    args = parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    out_dir = args.root / "visual_roi_scalar4_laion"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.selection_manifest:
        manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        labels = [str(label) for label in manifest["labels"]]
        selected = {
            str(subject): {
                str(label): [dict(row) for row in rows]
                for label, rows in by_label.items()
            }
            for subject, by_label in manifest["selected"].items()
        }
        roi_keys = manifest_roi_keys(manifest["roi_keys"])
    else:
        rows_by_subject = {subject: read_trial_rows(args.metadata_dir, subject, args.max_session) for subject in args.subjects}
        labels, selected = select_labels(rows_by_subject, args.subjects, args.min_repeats, args.max_labels)
        if not labels:
            raise RuntimeError("No labels satisfy the requested LAION-fMRI repeat/session constraints.")
        roi_keys = common_roi_keys(args.subjects, args.max_rois)

    if args.download_only:
        downloads = download_selection_files(args.subjects, selected, labels, roi_keys, args.root)
        manifest = {
            "subjects": args.subjects,
            "max_session": args.max_session,
            "min_repeats": args.min_repeats,
            "n_labels": len(labels),
            "labels": labels,
            "selected": selected,
            "roi_keys": [
                {"category": category, "label": label, "keys": key_by_subject}
                for category, label, key_by_subject in roi_keys
            ],
            "downloads": downloads,
        }
        (out_dir / "laion_download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps({"download_manifest": str(out_dir / "laion_download_manifest.json")}, indent=2))
        return

    qcs = [export_subject(subject, labels, selected[subject], roi_keys, args) for subject in args.subjects]
    manifest = {
        "subjects": args.subjects,
        "max_session": args.max_session,
        "min_repeats": args.min_repeats,
        "n_labels": len(labels),
        "labels": labels,
        "n_rois": len(roi_keys),
        "roi_labels": [f"{category}:{label}" for category, label, _ in roi_keys],
        "qcs": qcs,
    }
    (out_dir / "laion_visual_roi_scalar4_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "n_labels": len(labels), "n_subjects": len(args.subjects)}, indent=2))


if __name__ == "__main__":
    main()
