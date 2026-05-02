"""General utility helpers."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_connected(graph: nx.Graph, rng: np.random.Generator) -> nx.Graph:
    if nx.is_connected(graph):
        return graph
    components = [list(component) for component in nx.connected_components(graph)]
    for left, right in zip(components[:-1], components[1:]):
        graph.add_edge(int(rng.choice(left)), int(rng.choice(right)))
    return graph


def sample_partition(
    total: int,
    num_parts: int,
    minimum: int,
    rng: np.random.Generator,
) -> list[int]:
    if minimum * num_parts > total:
        raise ValueError("cannot partition total with requested minimum")
    sizes = [minimum] * num_parts
    remainder = total - sum(sizes)
    for _ in range(remainder):
        sizes[int(rng.integers(0, num_parts))] += 1
    rng.shuffle(sizes)
    return sizes


def nx_to_edge_index(graph: nx.Graph) -> torch.Tensor:
    edges = []
    for source, target in graph.edges():
        edges.append((source, target))
        edges.append((target, source))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def sorted_edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def graph_mean_degree(graph: nx.Graph) -> float:
    if graph.number_of_nodes() == 0:
        return 0.0
    return float(np.mean([degree for _, degree in graph.degree()]))


def graph_mean_clustering(graph: nx.Graph) -> float:
    if graph.number_of_nodes() == 0:
        return 0.0
    return float(np.mean(list(nx.clustering(graph).values())))


def choose_lexicographic_edge(edges: Iterable[tuple[int, int]]) -> tuple[int, int]:
    edge_list = sorted(sorted_edge(u, v) for u, v in edges)
    if not edge_list:
        raise ValueError("expected at least one edge")
    return edge_list[0]


def as_jsonable(payload: dict) -> dict:
    return json.loads(json.dumps(payload))


def save_json(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def harmonic_mean(values: list[float]) -> float:
    values = [value for value in values if value > 0]
    if not values:
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def count_four_cycles(graph: nx.Graph) -> int:
    total = 0
    nodes = list(graph.nodes())
    for idx, left in enumerate(nodes):
        for right in nodes[idx + 1 :]:
            common = len(list(nx.common_neighbors(graph, left, right)))
            if common >= 2:
                total += math.comb(common, 2)
    return total // 2

