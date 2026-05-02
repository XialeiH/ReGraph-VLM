#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATASET_ROOT = Path("preproc_v0/repetition_familiarity/datasets/scalar4_T3")
DEFAULT_OUTPUT_DIR = Path("preproc_v0/repetition_familiarity/vlm/clip_embeddings")
DEFAULT_STIM_INFO = Path("data/nsddata/experiments/nsd/nsd_stim_info_merged.csv")
DEFAULT_STIMULI_HDF5 = Path("data/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5")
COCO_URL = "https://images.cocodataset.org/{split}/{coco_id:012d}.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export frozen CLIP image embeddings for repetition/familiarity images.")
    parser.add_argument("--root", type=Path, required=True, help="v0_shared_unit root.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stim-info", type=Path, default=DEFAULT_STIM_INFO)
    parser.add_argument("--stimuli-hdf5", type=Path, default=DEFAULT_STIMULI_HDF5)
    parser.add_argument("--coco-cache-dir", type=Path, default=Path("preproc_v0/repetition_familiarity/vlm/coco_image_cache"))
    parser.add_argument("--source", choices=["auto", "nsd_hdf5", "coco"], default="auto")
    parser.add_argument("--folds", nargs="+", default=["fold_01", "fold_04"])
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--clip-model", default="ViT-B/32", help="OpenAI CLIP model name.")
    parser.add_argument("--backend", choices=["auto", "clip", "open_clip"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--download-coco", action="store_true", help="Download missing COCO source images if HDF5 is absent.")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_nsd_ids(dataset_root: Path, folds: list[str], splits: list[str]) -> list[int]:
    nsd_ids: set[int] = set()
    for fold in folds:
        meta_path = dataset_root / fold / "metadata_sequences.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata_sequences.csv: {meta_path}")
        df = pd.read_csv(meta_path)
        if "nsdId" not in df.columns:
            raise ValueError(f"{meta_path} has no nsdId column")
        nsd_ids.update(int(v) for v in df["nsdId"].dropna().astype(int).tolist())

        # Validate against pair metadata too, because pairs are the primary VLM target.
        pair_path = dataset_root / fold / "metadata_pairs.csv"
        if pair_path.exists():
            pair_df = pd.read_csv(pair_path)
            for col in ["nsdId_1", "nsdId_2"]:
                if col in pair_df.columns:
                    nsd_ids.update(int(v) for v in pair_df[col].dropna().astype(int).tolist())
    if not nsd_ids:
        raise ValueError(f"No nsdIds found under {dataset_root}")
    return sorted(nsd_ids)


def load_stim_info(path: Path, nsd_ids: list[int]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing NSD stimulus info CSV: {path}")
    df = pd.read_csv(path)
    needed = {"nsdId", "cocoId", "cocoSplit"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"Stimulus info is missing columns: {sorted(missing)}")
    sub = df[df["nsdId"].astype(int).isin(nsd_ids)].copy()
    found = set(int(v) for v in sub["nsdId"].tolist())
    missing_ids = sorted(set(nsd_ids).difference(found))
    if missing_ids:
        raise ValueError(f"Stimulus info missing {len(missing_ids)} nsdIds; first={missing_ids[:10]}")
    return sub.sort_values("nsdId").reset_index(drop=True)


def load_clip_backend(backend: str, model_name: str, device: torch.device):
    if backend in {"auto", "clip"}:
        try:
            import clip  # type: ignore

            model, preprocess = clip.load(model_name, device=device, jit=False)
            model.eval()

            def encode(batch: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    return model.encode_image(batch.to(device))

            return {
                "backend": "clip",
                "model": model,
                "preprocess": preprocess,
                "encode": encode,
                "resolved_model_name": model_name,
            }
        except Exception as exc:
            if backend == "clip":
                raise RuntimeError("Failed to load OpenAI clip backend") from exc

    if backend in {"auto", "open_clip"}:
        try:
            import open_clip  # type: ignore

            open_clip_name = model_name.replace("/", "-")
            model, _, preprocess = open_clip.create_model_and_transforms(open_clip_name, pretrained="openai")
            model = model.to(device)
            model.eval()

            def encode(batch: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    return model.encode_image(batch.to(device))

            return {
                "backend": "open_clip",
                "model": model,
                "preprocess": preprocess,
                "encode": encode,
                "resolved_model_name": open_clip_name,
            }
        except Exception as exc:
            raise RuntimeError(
                "No usable CLIP backend found. Install one of: "
                "`pip install git+https://github.com/openai/CLIP.git` or `pip install open_clip_torch`."
            ) from exc

    raise ValueError(f"Unsupported backend: {backend}")


class ImageSource:
    def __init__(
        self,
        root: Path,
        stim_info: pd.DataFrame,
        source: str,
        stimuli_hdf5: Path,
        coco_cache_dir: Path,
        download_coco: bool,
        timeout: float,
    ) -> None:
        self.root = root
        self.stim_info = stim_info.set_index("nsdId")
        self.source = source
        self.stimuli_hdf5 = stimuli_hdf5
        self.coco_cache_dir = coco_cache_dir
        self.download_coco = download_coco
        self.timeout = timeout
        self.h5: h5py.File | None = None
        self.h5_dataset_name: str | None = None

        if self.source == "auto":
            self.source = "nsd_hdf5" if self.stimuli_hdf5.exists() else "coco"

        if self.source == "nsd_hdf5":
            if not self.stimuli_hdf5.exists():
                raise FileNotFoundError(f"Missing NSD stimuli HDF5: {self.stimuli_hdf5}")
            self.h5 = h5py.File(self.stimuli_hdf5, "r")
            for name in ["imgBrick", "images", "stimuli"]:
                if name in self.h5:
                    self.h5_dataset_name = name
                    break
            if self.h5_dataset_name is None:
                keys = list(self.h5.keys())
                raise ValueError(f"Could not find image dataset in {self.stimuli_hdf5}; keys={keys}")
        elif self.source == "coco":
            self.coco_cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError(f"Unsupported source: {self.source}")

    def close(self) -> None:
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None

    def image_path_for(self, nsd_id: int) -> Path | None:
        if self.source != "coco":
            return None
        row = self.stim_info.loc[nsd_id]
        split = str(row["cocoSplit"])
        coco_id = int(row["cocoId"])
        return self.coco_cache_dir / split / f"{coco_id:012d}.jpg"

    def load(self, nsd_id: int) -> tuple[Image.Image, str]:
        if self.source == "nsd_hdf5":
            assert self.h5 is not None and self.h5_dataset_name is not None
            arr = self.h5[self.h5_dataset_name][nsd_id]
            if arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:
                arr = np.transpose(arr, (1, 2, 0))
            arr = np.asarray(arr)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr).convert("RGB"), f"hdf5:{self.stimuli_hdf5}:{self.h5_dataset_name}:{nsd_id}"

        row = self.stim_info.loc[nsd_id]
        split = str(row["cocoSplit"])
        coco_id = int(row["cocoId"])
        path = self.coco_cache_dir / split / f"{coco_id:012d}.jpg"
        if not path.exists():
            if not self.download_coco:
                raise FileNotFoundError(
                    f"Missing cached COCO image for nsdId={nsd_id}: {path}. "
                    "Re-run with --download-coco or provide --stimuli-hdf5."
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            url = COCO_URL.format(split=split, coco_id=coco_id)
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            Image.open(BytesIO(resp.content)).convert("RGB").save(path)
        return Image.open(path).convert("RGB"), str(path)


class ClipImageDataset(Dataset):
    def __init__(self, nsd_ids: list[int], image_source: ImageSource, preprocess: Any):
        self.nsd_ids = nsd_ids
        self.image_source = image_source
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.nsd_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        nsd_id = int(self.nsd_ids[idx])
        image, source_path = self.image_source.load(nsd_id)
        return {"nsd_id": nsd_id, "image": self.preprocess(image), "source_path": source_path}


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "nsd_id": torch.tensor([int(item["nsd_id"]) for item in batch], dtype=torch.int64),
        "image": torch.stack([item["image"] for item in batch]),
        "source_path": [str(item["source_path"]) for item in batch],
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    dataset_root = (root / args.dataset_root).resolve()
    output_dir = (root / args.output_dir).resolve()
    stim_info_path = (root / args.stim_info).resolve()
    stimuli_hdf5 = (root / args.stimuli_hdf5).resolve()
    coco_cache_dir = (root / args.coco_cache_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    nsd_ids = collect_nsd_ids(dataset_root, args.folds, args.splits)
    stim_info = load_stim_info(stim_info_path, nsd_ids)
    device = resolve_device(args.device)
    clip_bundle = load_clip_backend(args.backend, args.clip_model, device)
    image_source = ImageSource(
        root=root,
        stim_info=stim_info,
        source=args.source,
        stimuli_hdf5=stimuli_hdf5,
        coco_cache_dir=coco_cache_dir,
        download_coco=args.download_coco,
        timeout=args.timeout,
    )

    dataset = ClipImageDataset(nsd_ids, image_source, clip_bundle["preprocess"])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    embeddings: list[torch.Tensor] = []
    metadata_rows: list[dict[str, Any]] = []
    start = time.time()
    try:
        for batch_idx, batch in enumerate(loader):
            emb = clip_bundle["encode"](batch["image"])
            emb = F.normalize(emb.float().cpu(), dim=-1)
            embeddings.append(emb)
            for nsd_id, source_path in zip(batch["nsd_id"].tolist(), batch["source_path"]):
                row = stim_info[stim_info["nsdId"].astype(int) == int(nsd_id)].iloc[0]
                metadata_rows.append(
                    {
                        "nsdId": int(nsd_id),
                        "image_index_or_label": int(nsd_id),
                        "cocoId": int(row["cocoId"]),
                        "cocoSplit": str(row["cocoSplit"]),
                        "image_path": str(source_path),
                        "clip_model_name": str(args.clip_model),
                        "clip_backend": str(clip_bundle["backend"]),
                        "resolved_model_name": str(clip_bundle["resolved_model_name"]),
                        "embedding_dim": int(emb.shape[-1]),
                    }
                )
            done = sum(t.shape[0] for t in embeddings)
            elapsed = max(time.time() - start, 1e-6)
            print(
                f"[progress] {done}/{len(dataset)} ({done / len(dataset):.1%}) "
                f"elapsed={elapsed / 60:.1f}m rate={done / elapsed:.2f}/s",
                flush=True,
            )
    finally:
        image_source.close()

    clip_emb = torch.cat(embeddings, dim=0)
    nsd_id_tensor = torch.tensor(nsd_ids, dtype=torch.int64)
    norms = clip_emb.norm(dim=-1)
    nan_count = int(torch.isnan(clip_emb).sum().item())
    inf_count = int(torch.isinf(clip_emb).sum().item())
    missing_images = int(len(nsd_ids) - len(metadata_rows))

    torch_payload = {
        "clip_emb": clip_emb,
        "nsdId": nsd_id_tensor,
        "clip_model_name": args.clip_model,
        "clip_backend": clip_bundle["backend"],
        "resolved_model_name": clip_bundle["resolved_model_name"],
        "source": image_source.source,
        "normalized": True,
        "embedding_dim": int(clip_emb.shape[-1]),
    }
    torch.save(torch_payload, output_dir / "clip_image_embeddings.pt")
    np.save(output_dir / "clip_image_embeddings.npy", clip_emb.numpy())
    write_csv(
        output_dir / "clip_image_metadata.csv",
        metadata_rows,
        [
            "nsdId",
            "image_index_or_label",
            "cocoId",
            "cocoSplit",
            "image_path",
            "clip_model_name",
            "clip_backend",
            "resolved_model_name",
            "embedding_dim",
        ],
    )
    qc = {
        "num_images": int(clip_emb.shape[0]),
        "embedding_dim": int(clip_emb.shape[1]),
        "missing_images": missing_images,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "mean_norm": float(norms.mean().item()),
        "min_norm": float(norms.min().item()),
        "max_norm": float(norms.max().item()),
        "clip_model_name": str(args.clip_model),
        "clip_backend": str(clip_bundle["backend"]),
        "resolved_model_name": str(clip_bundle["resolved_model_name"]),
        "source": str(image_source.source),
        "stimuli_hdf5": str(stimuli_hdf5),
        "coco_cache_dir": str(coco_cache_dir),
        "output_dir": str(output_dir),
        "elapsed_seconds": float(time.time() - start),
        "status": "ok" if missing_images == 0 and nan_count == 0 and inf_count == 0 else "check",
    }
    (output_dir / "clip_embedding_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2), flush=True)


if __name__ == "__main__":
    main()
