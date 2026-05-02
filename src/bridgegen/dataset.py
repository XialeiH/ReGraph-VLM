"""Dataset loading helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data


def load_split(data_root: str | Path, split_name: str) -> list[Data]:
    return torch.load(Path(data_root) / f"{split_name}.pt", weights_only=False)


def load_corpus(data_root: str | Path) -> dict[str, list[Data]]:
    data_root = Path(data_root)
    corpus = {}
    for path in sorted(data_root.glob("*.pt")):
        corpus[path.stem] = torch.load(path, weights_only=False)
    return corpus


def filter_for_task(data_list: list[Data], task_name: str) -> list[Data]:
    if task_name != "redundant":
        return data_list
    return [data for data in data_list if data.meta["eligible_for_redundant_task"]]
