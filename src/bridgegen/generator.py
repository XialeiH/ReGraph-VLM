"""Synthetic bridge-sensitive graph generator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm.auto import tqdm

from .config import BridgeSuiteConfig, NoiseLevel, SplitSpec
from .constants import (
    INDEX_TO_ROLE,
    PRIMITIVE_NAMES,
    ROLE_TO_INDEX,
    TEMPLATE_NAMES,
    TOPOLOGY_TO_INDEX,
)
from .utils import (
    as_jsonable,
    choose_lexicographic_edge,
    ensure_connected,
    graph_mean_clustering,
    graph_mean_degree,
    nx_to_edge_index,
    sample_partition,
    save_json,
    set_global_seed,
    sorted_edge,
)


@dataclass(frozen=True)
class GeneratedSample:
    data: Data
    fine_graph: nx.Graph
    coarse_graph: nx.Graph


class BridgeGraphGenerator:
    """Generates and serializes the `bridge_v1` corpus."""

    def __init__(self, config: BridgeSuiteConfig):
        self.config = config
        self.output_root = Path(config.output_root)

    def generate_master_corpus(self) -> dict[str, list[Data]]:
        set_global_seed(self.config.seed)
        corpus: dict[str, list[Data]] = {}
        for split_index, (split_name, split_spec) in enumerate(self.config.splits.items()):
            split_seed = self.config.seed + 1000 * split_index
            corpus[split_name] = self.generate_split(split_name, split_spec, split_seed)
        noise_suite = self.generate_noise_suites(corpus["id_test"])
        stability_suite = self.generate_stability_suite(corpus["id_test"])
        corpus.update(noise_suite)
        corpus["stability"] = stability_suite
        self.serialize_corpus(corpus)
        return corpus

    def generate_split(
        self, split_name: str, split_spec: SplitSpec, seed: int
    ) -> list[Data]:
        rng = np.random.default_rng(seed)
        schedule = self._balanced_template_schedule(split_spec.num_graphs, rng)
        data_list: list[Data] = []
        iterator = tqdm(
            enumerate(schedule),
            total=len(schedule),
            desc=f"generate:{split_name}",
        )
        for graph_index, template_name in iterator:
            data_list.append(
                self._generate_with_retries(
                    template_name=template_name,
                    split_name=split_name,
                    split_spec=split_spec,
                    graph_index=graph_index,
                    rng=rng,
                ).data
            )
        return data_list

    def generate_noise_suites(self, base_graphs: list[Data]) -> dict[str, list[Data]]:
        suites: dict[str, list[Data]] = {}
        for noise_name, noise_level in self.config.noise_levels.items():
            suites[f"noise_{noise_name}"] = []
            rng = np.random.default_rng(self.config.seed + hash(noise_name) % 2**16)
            iterator = tqdm(base_graphs, desc=f"noise:{noise_name}")
            for graph in iterator:
                suites[f"noise_{noise_name}"].append(
                    self._generate_noisy_variant(graph, noise_name, noise_level, rng)
                )
        return suites

    def generate_stability_suite(self, base_graphs: list[Data]) -> list[Data]:
        subset = base_graphs[: min(self.config.stability_subset_size, len(base_graphs))]
        suite: list[Data] = []
        for graph_index, graph in enumerate(tqdm(subset, desc="generate:stability")):
            for variant_index in range(self.config.stability_variants):
                rng = np.random.default_rng(
                    self.config.seed + 500_000 + graph_index * 101 + variant_index
                )
                noisy = self._generate_noisy_variant(
                    graph=graph,
                    noise_name=f"stability_{variant_index}",
                    noise_level=self.config.noise_levels["med"],
                    rng=rng,
                )
                noisy.meta["stability_group"] = int(graph.meta["graph_id"])
                noisy.meta["stability_variant"] = variant_index
                suite.append(noisy)
        return suite

    def serialize_corpus(self, corpus: dict[str, list[Data]]) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "output_root": str(self.output_root),
            "seed": self.config.seed,
            "feature_mode": self.config.feature_mode,
            "splits": {name: len(values) for name, values in corpus.items()},
            "noise_levels": {
                name: vars(level) for name, level in self.config.noise_levels.items()
            },
        }
        for split_name, data_list in corpus.items():
            torch.save(data_list, self.output_root / f"{split_name}.pt")
        save_json(manifest, self.output_root / "manifest.json")

    def _generate_with_retries(
        self,
        template_name: str,
        split_name: str,
        split_spec: SplitSpec,
        graph_index: int,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        for attempt in range(self.config.max_graph_attempts):
            sample = self._generate_single_sample(
                template_name=template_name,
                split_name=split_name,
                split_spec=split_spec,
                graph_index=graph_index,
                attempt=attempt,
                rng=rng,
            )
            if self._passes_graph_quality(sample.fine_graph):
                return sample
        raise RuntimeError(
            f"failed to generate valid sample after {self.config.max_graph_attempts} attempts"
        )

    def _generate_single_sample(
        self,
        template_name: str,
        split_name: str,
        split_spec: SplitSpec,
        graph_index: int,
        attempt: int,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        coarse_graph = self._build_coarse_graph(template_name, split_spec.cell_range, rng)
        cell_sizes = {
            cell_id: int(rng.integers(split_spec.cell_size_range[0], split_spec.cell_size_range[1] + 1))
            for cell_id in coarse_graph.nodes()
        }
        fine_graph, node_to_cell = self._expand_to_fine_graph(
            coarse_graph=coarse_graph,
            cell_sizes=cell_sizes,
            rng=rng,
        )
        annotations = self._compute_annotations(coarse_graph)
        data = self._to_pyg_data(
            fine_graph=fine_graph,
            coarse_graph=coarse_graph,
            node_to_cell=node_to_cell,
            annotations=annotations,
            split_name=split_name,
            graph_index=graph_index,
            attempt=attempt,
            cell_sizes=cell_sizes,
            noise_level="clean",
            rng=rng,
        )
        return GeneratedSample(data=data, fine_graph=fine_graph, coarse_graph=coarse_graph)

    def _build_coarse_graph(
        self,
        template_name: str,
        cell_range: tuple[int, int],
        rng: np.random.Generator,
    ) -> nx.Graph:
        min_required = {
            "barbell": 6,
            "bow_tie": 5,
            "multi_neck": 6,
            "redundant_bow_tie": 7,
        }[template_name]
        num_cells = max(
            int(rng.integers(cell_range[0], cell_range[1] + 1)),
            min_required,
        )
        graph = nx.Graph()
        for cell_id in range(num_cells):
            graph.add_node(cell_id)

        if template_name == "barbell":
            self._populate_barbell(graph, rng)
        elif template_name == "bow_tie":
            self._populate_bow_tie(graph, rng)
        elif template_name == "multi_neck":
            self._populate_multi_neck(graph, rng)
        elif template_name == "redundant_bow_tie":
            self._populate_redundant_bow_tie(graph, rng)
        else:
            raise ValueError(f"unknown template: {template_name}")

        graph.graph["template_name"] = template_name
        graph.graph["narrow_neck"] = template_name != "multi_neck"
        graph.graph["redundant"] = template_name == "redundant_bow_tie"
        return graph

    def _populate_barbell(self, graph: nx.Graph, rng: np.random.Generator) -> None:
        num_cells = graph.number_of_nodes()
        bridge_len = 2 if num_cells < 7 else int(rng.integers(2, 4))
        wing_budget = num_cells - bridge_len
        left_size = max(2, wing_budget // 2)
        right_size = max(2, wing_budget - left_size)
        if left_size + right_size + bridge_len != num_cells:
            right_size = num_cells - left_size - bridge_len

        left = list(range(left_size))
        bridge = list(range(left_size, left_size + bridge_len))
        right = list(range(left_size + bridge_len, num_cells))

        self._dense_connected_subgraph(graph, left, rng)
        self._dense_connected_subgraph(graph, right, rng)
        nx.add_path(graph, bridge)
        graph.add_edge(left[0], bridge[0])
        graph.add_edge(bridge[-1], right[0])

        for node in left:
            graph.nodes[node]["role"] = "wing"
            graph.nodes[node]["side"] = "left"
        for node in right:
            graph.nodes[node]["role"] = "wing"
            graph.nodes[node]["side"] = "right"
        bridge_center = bridge[len(bridge) // 2]
        for node in bridge:
            graph.nodes[node]["role"] = "bridge" if node == bridge_center else "transit"
            graph.nodes[node]["side"] = "center"

    def _populate_bow_tie(self, graph: nx.Graph, rng: np.random.Generator) -> None:
        num_cells = graph.number_of_nodes()
        center = 0
        left_size = max(2, (num_cells - 1) // 2)
        right_size = max(2, num_cells - 1 - left_size)
        left = list(range(1, 1 + left_size))
        right = list(range(1 + left_size, num_cells))

        self._dense_connected_subgraph(graph, left, rng)
        self._dense_connected_subgraph(graph, right, rng)
        graph.add_edge(center, left[0])
        graph.add_edge(center, right[0])

        graph.nodes[center]["role"] = "bridge"
        graph.nodes[center]["side"] = "center"
        for node in left:
            graph.nodes[node]["role"] = "wing"
            graph.nodes[node]["side"] = "left"
        for node in right:
            graph.nodes[node]["role"] = "wing"
            graph.nodes[node]["side"] = "right"

    def _populate_multi_neck(self, graph: nx.Graph, rng: np.random.Generator) -> None:
        num_cells = graph.number_of_nodes()
        left_size = max(2, (num_cells - 2) // 2)
        right_size = max(2, num_cells - 2 - left_size)
        left = list(range(left_size))
        right = list(range(left_size, left_size + right_size))
        bridges = list(range(left_size + right_size, num_cells))
        if len(bridges) < 2:
            bridges = [num_cells - 2, num_cells - 1]

        self._dense_connected_subgraph(graph, left, rng)
        self._dense_connected_subgraph(graph, right, rng)
        graph.add_edge(left[0], bridges[0])
        graph.add_edge(bridges[0], right[0])
        graph.add_edge(left[-1], bridges[1])
        graph.add_edge(bridges[1], right[-1])

        for node in left:
            graph.nodes[node]["role"] = "wing"
            graph.nodes[node]["side"] = "left"
        for node in right:
            graph.nodes[node]["role"] = "wing"
            graph.nodes[node]["side"] = "right"
        for node in bridges:
            graph.nodes[node]["role"] = "bridge"
            graph.nodes[node]["side"] = "center"

    def _populate_redundant_bow_tie(
        self,
        graph: nx.Graph,
        rng: np.random.Generator,
    ) -> None:
        num_cells = graph.number_of_nodes()
        center = 0
        alt_left = num_cells - 2
        alt_right = num_cells - 1
        wing_budget = num_cells - 3
        left_size = max(2, wing_budget // 2)
        right_size = max(2, wing_budget - left_size)
        left = list(range(1, 1 + left_size))
        right = list(range(1 + left_size, num_cells - 2))
        if not right:
            right = [num_cells - 3]
            left = list(range(1, num_cells - 3))

        self._dense_connected_subgraph(graph, left, rng)
        self._dense_connected_subgraph(graph, right, rng)
        graph.add_edge(center, left[0])
        graph.add_edge(center, right[0])
        graph.add_edge(left[-1], alt_left)
        graph.add_edge(alt_left, alt_right)
        graph.add_edge(alt_right, right[-1])

        graph.nodes[center]["role"] = "bridge"
        graph.nodes[center]["side"] = "center"
        for node in left:
            graph.nodes[node]["role"] = "wing"
            graph.nodes[node]["side"] = "left"
        for node in right:
            graph.nodes[node]["role"] = "wing"
            graph.nodes[node]["side"] = "right"
        graph.nodes[alt_left]["role"] = "distractor"
        graph.nodes[alt_left]["side"] = "left"
        graph.nodes[alt_right]["role"] = "distractor"
        graph.nodes[alt_right]["side"] = "right"

    def _dense_connected_subgraph(
        self,
        graph: nx.Graph,
        nodes: Iterable[int],
        rng: np.random.Generator,
    ) -> None:
        nodes = list(nodes)
        for idx in range(1, len(nodes)):
            graph.add_edge(nodes[idx - 1], nodes[idx])
        for left_idx, source in enumerate(nodes):
            for target in nodes[left_idx + 1 :]:
                if rng.random() < 0.55:
                    graph.add_edge(source, target)

    def _expand_to_fine_graph(
        self,
        coarse_graph: nx.Graph,
        cell_sizes: dict[int, int],
        rng: np.random.Generator,
    ) -> tuple[nx.Graph, dict[int, int]]:
        fine_graph = nx.Graph()
        node_to_cell: dict[int, int] = {}
        cell_to_global_nodes: dict[int, list[int]] = {}
        next_node_id = 0

        for coarse_node in coarse_graph.nodes():
            cell_graph = self._generate_cell_graph(
                num_nodes=cell_sizes[coarse_node],
                rng=rng,
            )
            mapping = {
                local_id: next_node_id + offset
                for offset, local_id in enumerate(cell_graph.nodes())
            }
            relabeled = nx.relabel_nodes(cell_graph, mapping)
            fine_graph = nx.compose(fine_graph, relabeled)
            global_nodes = sorted(relabeled.nodes())
            cell_to_global_nodes[coarse_node] = global_nodes
            for node in global_nodes:
                node_to_cell[node] = coarse_node
                fine_graph.nodes[node]["cell_id"] = coarse_node
                fine_graph.nodes[node]["cell_role"] = coarse_graph.nodes[coarse_node]["role"]
                fine_graph.nodes[node]["coarse_side"] = coarse_graph.nodes[coarse_node]["side"]
            next_node_id += len(global_nodes)

        critical_edges = {
            sorted_edge(source, target) for source, target in nx.bridges(coarse_graph)
        }
        for source, target in coarse_graph.edges():
            fine_edge_count = 1 if sorted_edge(source, target) in critical_edges else int(
                rng.integers(2, 4)
            )
            source_ports = self._sample_port_nodes(cell_to_global_nodes[source], fine_graph, fine_edge_count, rng)
            target_ports = self._sample_port_nodes(cell_to_global_nodes[target], fine_graph, fine_edge_count, rng)
            for edge_index in range(fine_edge_count):
                fine_graph.add_edge(
                    source_ports[edge_index],
                    target_ports[edge_index],
                    coarse_edge=sorted_edge(source, target),
                    inter_cell=True,
                )

        return fine_graph, node_to_cell

    def _sample_port_nodes(
        self,
        candidate_nodes: list[int],
        fine_graph: nx.Graph,
        count: int,
        rng: np.random.Generator,
    ) -> list[int]:
        ranked = sorted(
            candidate_nodes,
            key=lambda node_id: fine_graph.degree(node_id),
            reverse=True,
        )
        if len(ranked) >= count:
            return ranked[:count]
        extra = list(rng.choice(ranked, size=count - len(ranked), replace=True))
        return ranked + extra

    def _generate_cell_graph(self, num_nodes: int, rng: np.random.Generator) -> nx.Graph:
        primitive_count = int(min(rng.integers(2, 5), max(2, num_nodes // 3)))
        primitive_sizes = sample_partition(num_nodes, primitive_count, minimum=3, rng=rng)
        cell_graph = nx.Graph()
        next_node = 0
        primitive_roots: list[int] = []
        for primitive_size in primitive_sizes:
            primitive_name = PRIMITIVE_NAMES[int(rng.integers(0, len(PRIMITIVE_NAMES)))]
            primitive = self._generate_primitive_graph(primitive_name, primitive_size, rng)
            mapping = {
                local_id: next_node + offset
                for offset, local_id in enumerate(primitive.nodes())
            }
            relabeled = nx.relabel_nodes(primitive, mapping)
            primitive_nodes = sorted(relabeled.nodes())
            primitive_roots.append(primitive_nodes[0])
            cell_graph = nx.compose(cell_graph, relabeled)
            next_node += primitive_size

        for left, right in zip(primitive_roots[:-1], primitive_roots[1:]):
            cell_graph.add_edge(left, right)
            if rng.random() < 0.5:
                left_alt = int(rng.choice(list(nx.node_connected_component(cell_graph, left))))
                right_alt = int(rng.choice(list(nx.node_connected_component(cell_graph, right))))
                cell_graph.add_edge(left_alt, right_alt)

        cell_graph = ensure_connected(cell_graph, rng)
        self._densify_graph(
            cell_graph,
            target_mean_degree=max(3.5, self.config.mean_degree_target - 0.35),
            rng=rng,
        )
        return cell_graph

    def _generate_primitive_graph(
        self,
        primitive_name: str,
        num_nodes: int,
        rng: np.random.Generator,
    ) -> nx.Graph:
        if primitive_name == "er":
            probability = float(
                np.clip(self.config.mean_degree_target / max(num_nodes - 1, 1), 0.25, 0.70)
            )
            graph = nx.erdos_renyi_graph(num_nodes, probability, seed=int(rng.integers(1, 10**9)))
        elif primitive_name == "watts_strogatz":
            degree = min(num_nodes - 1, max(2, 2 * int(round(self.config.mean_degree_target / 2))))
            if degree % 2 == 1:
                degree -= 1
            graph = nx.watts_strogatz_graph(
                num_nodes,
                max(2, degree),
                float(rng.uniform(0.10, 0.30)),
                seed=int(rng.integers(1, 10**9)),
            )
        elif primitive_name == "random_regular":
            degree = min(num_nodes - 1, max(2, int(round(self.config.mean_degree_target))))
            if degree >= num_nodes:
                degree = num_nodes - 1
            if (degree * num_nodes) % 2 == 1:
                degree -= 1
            if degree < 2:
                graph = nx.cycle_graph(num_nodes)
            else:
                graph = nx.random_regular_graph(
                    degree,
                    num_nodes,
                    seed=int(rng.integers(1, 10**9)),
                )
        elif primitive_name == "motif_bundle":
            degree = 3 if num_nodes > 4 else 2
            if (degree * num_nodes) % 2 == 1:
                degree -= 1
            graph = nx.random_regular_graph(
                max(2, degree),
                num_nodes,
                seed=int(rng.integers(1, 10**9)),
            )
            for _ in range(max(1, num_nodes // 3)):
                triplet = list(rng.choice(list(graph.nodes()), size=min(3, num_nodes), replace=False))
                for idx in range(len(triplet)):
                    graph.add_edge(triplet[idx], triplet[(idx + 1) % len(triplet)])
        else:
            raise ValueError(f"unknown primitive: {primitive_name}")
        return ensure_connected(graph, rng)

    def _densify_graph(
        self,
        graph: nx.Graph,
        target_mean_degree: float,
        rng: np.random.Generator,
    ) -> None:
        if graph.number_of_nodes() < 2:
            return
        while graph_mean_degree(graph) < target_mean_degree:
            source = int(rng.integers(0, graph.number_of_nodes()))
            target = int(rng.integers(0, graph.number_of_nodes()))
            if source == target or graph.has_edge(source, target):
                continue
            graph.add_edge(source, target)

    def _compute_annotations(self, coarse_graph: nx.Graph) -> dict:
        articulation_nodes = set(nx.articulation_points(coarse_graph))
        critical_edges = {sorted_edge(u, v) for u, v in nx.bridges(coarse_graph)}
        edge_betweenness = nx.edge_betweenness_centrality(coarse_graph, normalized=True)
        max_betweenness = max(edge_betweenness.values())
        candidate_edges = [
            edge for edge, value in edge_betweenness.items() if value == max_betweenness
        ]
        attack_edge = choose_lexicographic_edge(candidate_edges)
        attacked = coarse_graph.copy()
        attacked.remove_edge(*attack_edge)
        return {
            "articulation_nodes": sorted(int(node) for node in articulation_nodes),
            "critical_edges": sorted(critical_edges),
            "attack_edge": attack_edge,
            "attack_disconnect": not nx.is_connected(attacked),
            "bridge_count": len(articulation_nodes),
        }

    def _to_pyg_data(
        self,
        fine_graph: nx.Graph,
        coarse_graph: nx.Graph,
        node_to_cell: dict[int, int],
        annotations: dict,
        split_name: str,
        graph_index: int,
        attempt: int,
        cell_sizes: dict[int, int],
        noise_level: str,
        rng: np.random.Generator,
    ) -> Data:
        num_nodes = fine_graph.number_of_nodes()
        edge_index = nx_to_edge_index(fine_graph)
        x = self._build_node_features(num_nodes, rng)
        cell_id = torch.tensor([node_to_cell[node] for node in range(num_nodes)], dtype=torch.long)
        cell_role = torch.tensor(
            [
                ROLE_TO_INDEX[fine_graph.nodes[node]["cell_role"]]
                for node in range(num_nodes)
            ],
            dtype=torch.long,
        )
        cell_side = [
            0 if fine_graph.nodes[node]["coarse_side"] == "left" else
            1 if fine_graph.nodes[node]["coarse_side"] == "right" else
            2
            for node in range(num_nodes)
        ]
        critical_cells = set(annotations["articulation_nodes"])
        critical_cell_mask = torch.tensor(
            [1 if node_to_cell[node] in critical_cells else 0 for node in range(num_nodes)],
            dtype=torch.float,
        )
        coarse_edge_index = nx_to_edge_index(coarse_graph)
        critical_edge_mask = self._build_coarse_critical_mask(
            coarse_graph,
            annotations["critical_edges"],
        )
        meta = as_jsonable(
            {
                "template_name": coarse_graph.graph["template_name"],
                "num_cells": coarse_graph.number_of_nodes(),
                "cell_size_stats": {
                    "min": int(min(cell_sizes.values())),
                    "max": int(max(cell_sizes.values())),
                    "mean": float(np.mean(list(cell_sizes.values()))),
                },
                "noise_level": noise_level,
                "split_name": split_name,
                "seed": int(self.config.seed),
                "graph_id": int(graph_index),
                "attempt": int(attempt),
                "generator_hparams": {
                    "mean_degree_target": self.config.mean_degree_target,
                    "mean_degree_tolerance": self.config.mean_degree_tolerance,
                    "max_clustering_gap": self.config.max_clustering_gap,
                    "feature_mode": self.config.feature_mode,
                },
                "base_graph_id": int(graph_index),
                "narrow_neck": bool(coarse_graph.graph["narrow_neck"]),
                "eligible_for_redundant_task": bool(coarse_graph.graph["narrow_neck"]),
            }
        )
        return Data(
            x=x,
            edge_index=edge_index,
            num_nodes=num_nodes,
            y_topology=torch.tensor([TOPOLOGY_TO_INDEX[coarse_graph.graph["template_name"]]], dtype=torch.long),
            y_bridge_count=torch.tensor([annotations["bridge_count"]], dtype=torch.float),
            y_attack_disconnect=torch.tensor([int(annotations["attack_disconnect"])], dtype=torch.float),
            y_redundant=torch.tensor([int(coarse_graph.graph["redundant"])], dtype=torch.float),
            cell_id=cell_id,
            cell_role=cell_role,
            cell_side=torch.tensor(cell_side, dtype=torch.long),
            coarse_edge_index=coarse_edge_index,
            critical_cell_mask=critical_cell_mask,
            critical_edge_mask=critical_edge_mask,
            meta=meta,
        )

    def _build_node_features(
        self,
        num_nodes: int,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        if self.config.feature_mode == "gaussian":
            return torch.tensor(
                rng.normal(size=(num_nodes, 8)),
                dtype=torch.float,
            )
        return torch.ones((num_nodes, 1), dtype=torch.float)

    def _build_coarse_critical_mask(
        self,
        coarse_graph: nx.Graph,
        critical_edges: list[tuple[int, int]],
    ) -> torch.Tensor:
        critical_set = {sorted_edge(u, v) for u, v in critical_edges}
        mask = []
        for source, target in coarse_graph.edges():
            is_critical = float(sorted_edge(source, target) in critical_set)
            mask.extend([is_critical, is_critical])
        return torch.tensor(mask, dtype=torch.float)

    def _passes_graph_quality(self, fine_graph: nx.Graph) -> bool:
        mean_degree = graph_mean_degree(fine_graph)
        if abs(mean_degree - self.config.mean_degree_target) > self.config.mean_degree_tolerance:
            return False
        return nx.is_connected(fine_graph)

    def _balanced_template_schedule(
        self,
        num_graphs: int,
        rng: np.random.Generator,
    ) -> list[str]:
        schedule = []
        base = num_graphs // len(TEMPLATE_NAMES)
        remainder = num_graphs % len(TEMPLATE_NAMES)
        for template in TEMPLATE_NAMES:
            schedule.extend([template] * base)
        if remainder:
            extra = list(TEMPLATE_NAMES[:remainder])
            rng.shuffle(extra)
            schedule.extend(extra)
        rng.shuffle(schedule)
        return schedule

    def _generate_noisy_variant(
        self,
        graph: Data,
        noise_name: str,
        noise_level: NoiseLevel,
        rng: np.random.Generator,
    ) -> Data:
        fine_graph = self._data_to_nx(graph)
        coarse_graph = self._coarse_from_data(graph)
        node_to_cell = {node_id: int(graph.cell_id[node_id]) for node_id in range(graph.num_nodes)}
        original_annotations = self._compute_annotations(coarse_graph)

        for _ in range(self.config.max_graph_attempts):
            perturbed = fine_graph.copy()
            self._apply_intra_cell_rewire(perturbed, node_to_cell, noise_level.intra_cell_rewire, rng)
            self._apply_same_side_shortcuts(perturbed, node_to_cell, graph, noise_level.same_side_shortcut, rng)
            self._apply_fake_bridge_distractors(
                perturbed,
                node_to_cell,
                graph,
                noise_level.fake_bridge_distractor,
                rng,
            )
            if not nx.is_connected(perturbed):
                continue
            noisy = Data(
                x=graph.x.clone(),
                edge_index=nx_to_edge_index(perturbed),
                num_nodes=graph.num_nodes,
                y_topology=graph.y_topology.clone(),
                y_bridge_count=graph.y_bridge_count.clone(),
                y_attack_disconnect=graph.y_attack_disconnect.clone(),
                y_redundant=graph.y_redundant.clone(),
                cell_id=graph.cell_id.clone(),
                cell_role=graph.cell_role.clone(),
                cell_side=graph.cell_side.clone(),
                coarse_edge_index=graph.coarse_edge_index.clone(),
                critical_cell_mask=graph.critical_cell_mask.clone(),
                critical_edge_mask=graph.critical_edge_mask.clone(),
                meta=as_jsonable({**graph.meta, "noise_level": noise_name}),
            )
            noisy.meta["base_graph_id"] = int(graph.meta["base_graph_id"])
            return noisy
        raise RuntimeError("failed to generate label-preserving noisy graph")

    def _apply_intra_cell_rewire(
        self,
        fine_graph: nx.Graph,
        node_to_cell: dict[int, int],
        rate: float,
        rng: np.random.Generator,
    ) -> None:
        cells = sorted(set(node_to_cell.values()))
        for cell_id in cells:
            nodes = [node for node, owner in node_to_cell.items() if owner == cell_id]
            subgraph = fine_graph.subgraph(nodes).copy()
            swaps = int(max(0, round(rate * subgraph.number_of_edges())))
            if swaps < 1 or subgraph.number_of_edges() < 4:
                continue
            try:
                nx.double_edge_swap(
                    subgraph,
                    nswap=swaps,
                    max_tries=10 * max(1, swaps),
                    seed=int(rng.integers(1, 10**9)),
                )
            except nx.NetworkXException:
                continue
            for edge in list(fine_graph.edges(nodes)):
                if edge[0] in nodes and edge[1] in nodes:
                    fine_graph.remove_edge(*edge)
            fine_graph.add_edges_from(subgraph.edges())

    def _apply_same_side_shortcuts(
        self,
        fine_graph: nx.Graph,
        node_to_cell: dict[int, int],
        graph: Data,
        rate: float,
        rng: np.random.Generator,
    ) -> None:
        if rate <= 0:
            return
        cell_sides = self._cell_sides_from_data(graph)
        coarse_graph = self._coarse_from_data(graph)
        side_to_cells: dict[str, list[int]] = {"left": [], "right": [], "center": []}
        for cell_id, side in cell_sides.items():
            side_to_cells.setdefault(side, []).append(cell_id)
        trials = int(max(1, round(rate * fine_graph.number_of_edges())))
        for _ in range(trials):
            side = "left" if rng.random() < 0.5 else "right"
            cells = side_to_cells.get(side, [])
            if len(cells) < 2:
                continue
            same_side_edges = [
                (source, target)
                for source, target in coarse_graph.edges()
                if cell_sides[source] == side and cell_sides[target] == side
            ]
            if not same_side_edges:
                continue
            source_cell, target_cell = same_side_edges[int(rng.integers(0, len(same_side_edges)))]
            source_nodes = [node for node, owner in node_to_cell.items() if owner == int(source_cell)]
            target_nodes = [node for node, owner in node_to_cell.items() if owner == int(target_cell)]
            fine_graph.add_edge(int(rng.choice(source_nodes)), int(rng.choice(target_nodes)))

    def _apply_fake_bridge_distractors(
        self,
        fine_graph: nx.Graph,
        node_to_cell: dict[int, int],
        graph: Data,
        rate: float,
        rng: np.random.Generator,
    ) -> None:
        if rate <= 0:
            return
        cell_roles = self._cell_roles_from_data(graph)
        distractors = [cell_id for cell_id, role in cell_roles.items() if role == "distractor"]
        wing_cells = [cell_id for cell_id, role in cell_roles.items() if role == "wing"]
        coarse_graph = self._coarse_from_data(graph)
        if not distractors or not wing_cells:
            return
        trials = int(max(1, round(rate * fine_graph.number_of_edges())))
        for _ in range(trials):
            source_cell = int(rng.choice(distractors))
            neighbors = [neighbor for neighbor in coarse_graph.neighbors(source_cell) if cell_roles[neighbor] == "wing"]
            if not neighbors:
                continue
            target_cell = int(rng.choice(neighbors))
            source_nodes = [node for node, owner in node_to_cell.items() if owner == source_cell]
            target_nodes = [node for node, owner in node_to_cell.items() if owner == target_cell]
            fine_graph.add_edge(int(rng.choice(source_nodes)), int(rng.choice(target_nodes)))

    def _data_to_nx(self, data: Data) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(data.num_nodes))
        edges = data.edge_index.t().tolist()
        for source, target in edges:
            if source <= target:
                graph.add_edge(int(source), int(target))
        return graph

    def _coarse_from_data(self, data: Data) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(sorted(set(int(cell) for cell in data.cell_id.tolist())))
        edges = data.coarse_edge_index.t().tolist()
        for source, target in edges:
            if source <= target:
                graph.add_edge(int(source), int(target))
        template_name = TEMPLATE_NAMES[int(data.y_topology.item())]
        graph.graph["template_name"] = template_name
        graph.graph["narrow_neck"] = bool(data.meta["narrow_neck"])
        graph.graph["redundant"] = bool(data.y_redundant.item())
        for cell_id in graph.nodes():
            graph.nodes[cell_id]["role"] = self._cell_roles_from_data(data)[cell_id]
            graph.nodes[cell_id]["side"] = self._cell_sides_from_data(data)[cell_id]
        return graph

    def _cell_roles_from_data(self, data: Data) -> dict[int, str]:
        roles = {}
        for node_id in range(data.num_nodes):
            cell_id = int(data.cell_id[node_id])
            role_index = int(data.cell_role[node_id])
            roles[cell_id] = INDEX_TO_ROLE[role_index]
        return roles

    def _cell_sides_from_data(self, data: Data) -> dict[int, str]:
        sides = {}
        for node_id in range(data.num_nodes):
            cell_id = int(data.cell_id[node_id])
            if cell_id in sides:
                continue
            side_id = int(data.cell_side[node_id])
            sides[cell_id] = {0: "left", 1: "right", 2: "center"}[side_id]
        return sides


def build_generator_from_config(path: str | Path) -> BridgeGraphGenerator:
    from .config import load_config

    return BridgeGraphGenerator(load_config(path))
