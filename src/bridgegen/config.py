"""Configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class NoiseLevel:
    intra_cell_rewire: float
    same_side_shortcut: float
    fake_bridge_distractor: float


@dataclass(frozen=True)
class SplitSpec:
    num_graphs: int
    cell_range: tuple[int, int]
    cell_size_range: tuple[int, int]


@dataclass(frozen=True)
class BridgeSuiteConfig:
    output_root: str
    seed: int
    feature_mode: str
    stability_subset_size: int
    stability_variants: int
    local_audit_max_topology_accuracy: float
    local_audit_max_attack_accuracy: float
    mean_degree_target: float
    mean_degree_tolerance: float
    max_clustering_gap: float
    max_graph_attempts: int
    noise_levels: dict[str, NoiseLevel]
    splits: dict[str, SplitSpec]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BridgeSuiteConfig":
        noise_levels = {
            key: NoiseLevel(**value) for key, value in payload["noise_levels"].items()
        }
        splits = {
            key: SplitSpec(
                num_graphs=value["num_graphs"],
                cell_range=tuple(value["cell_range"]),
                cell_size_range=tuple(value["cell_size_range"]),
            )
            for key, value in payload["splits"].items()
        }
        return cls(
            output_root=payload["output_root"],
            seed=int(payload["seed"]),
            feature_mode=payload.get("feature_mode", "constant"),
            stability_subset_size=int(payload.get("stability_subset_size", 128)),
            stability_variants=int(payload.get("stability_variants", 5)),
            local_audit_max_topology_accuracy=float(
                payload.get("local_audit_max_topology_accuracy", 0.35)
            ),
            local_audit_max_attack_accuracy=float(
                payload.get("local_audit_max_attack_accuracy", 0.60)
            ),
            mean_degree_target=float(payload.get("mean_degree_target", 5.5)),
            mean_degree_tolerance=float(payload.get("mean_degree_tolerance", 1.0)),
            max_clustering_gap=float(payload.get("max_clustering_gap", 0.05)),
            max_graph_attempts=int(payload.get("max_graph_attempts", 64)),
            noise_levels=noise_levels,
            splits=splits,
        )


def load_config(path: str | Path) -> BridgeSuiteConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return BridgeSuiteConfig.from_dict(payload)

