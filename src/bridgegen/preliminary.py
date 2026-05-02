"""Preliminary validation probes for the synthetic suite."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from .config import BridgeSuiteConfig
from .constants import INDEX_TO_TOPOLOGY
from .dataset import load_corpus
from .generator import BridgeGraphGenerator
from .utils import count_four_cycles, save_json


class PreliminarySuite:
    """Runs the generator-level probes before model training."""

    def __init__(self, config: BridgeSuiteConfig, data_root: str | Path | None = None):
        self.config = config
        self.data_root = Path(data_root or config.output_root)

    def run(self) -> dict:
        corpus = load_corpus(self.data_root)
        report = {
            "structural_validity": self._check_structural_validity(corpus),
            "annotation_consistency": self._check_annotation_consistency(corpus),
            "local_shortcut_audit": self._run_local_shortcut_audit(corpus),
            "split_sanity": self._run_split_sanity(corpus),
            "size_control_sanity": self._run_size_control_sanity(corpus),
        }
        report["all_passed"] = all(
            section["passed"] for section in report.values() if isinstance(section, dict)
        )
        save_json(report, self.data_root / "preliminary_report.json")
        return report

    def _check_structural_validity(self, corpus: dict) -> dict:
        failures = []
        for split_name, data_list in corpus.items():
            for graph_index, data in enumerate(data_list):
                graph = self._data_to_nx(data)
                if not nx.is_connected(graph):
                    failures.append(f"{split_name}:{graph_index}:disconnected")
                if any(graph.degree(node) == 0 for node in graph.nodes()):
                    failures.append(f"{split_name}:{graph_index}:isolated_node")
                cell_ids = sorted(set(int(cell) for cell in data.cell_id.tolist()))
                for cell_id in cell_ids:
                    nodes = [node for node in range(data.num_nodes) if int(data.cell_id[node]) == cell_id]
                    if nodes and not nx.is_connected(graph.subgraph(nodes)):
                        failures.append(f"{split_name}:{graph_index}:cell_{cell_id}_disconnected")
        return {"passed": not failures, "num_failures": len(failures), "examples": failures[:20]}

    def _check_annotation_consistency(self, corpus: dict) -> dict:
        generator = BridgeGraphGenerator(self.config)
        failures = []
        for split_name, data_list in corpus.items():
            for graph_index, data in enumerate(data_list):
                coarse = generator._coarse_from_data(data)
                annotations = generator._compute_annotations(coarse)
                expected_bridge_count = int(data.y_bridge_count.item())
                expected_disconnect = int(data.y_attack_disconnect.item())
                actual_critical = {
                    int(data.cell_id[node_id])
                    for node_id in range(data.num_nodes)
                    if int(data.critical_cell_mask[node_id].item()) == 1
                }
                if annotations["bridge_count"] != expected_bridge_count:
                    failures.append(f"{split_name}:{graph_index}:bridge_count")
                if int(annotations["attack_disconnect"]) != expected_disconnect:
                    failures.append(f"{split_name}:{graph_index}:attack_disconnect")
                if set(annotations["articulation_nodes"]) != actual_critical:
                    failures.append(f"{split_name}:{graph_index}:articulation_nodes")
        return {"passed": not failures, "num_failures": len(failures), "examples": failures[:20]}

    def _run_local_shortcut_audit(self, corpus: dict) -> dict:
        train_graphs = corpus["train"]
        val_graphs = corpus["val"]
        x_train = np.stack([self._local_descriptor(data) for data in train_graphs])
        y_train_topology = np.array([int(data.y_topology.item()) for data in train_graphs])
        y_train_attack = np.array([int(data.y_attack_disconnect.item()) for data in train_graphs])
        x_val = np.stack([self._local_descriptor(data) for data in val_graphs])
        y_val_topology = np.array([int(data.y_topology.item()) for data in val_graphs])
        y_val_attack = np.array([int(data.y_attack_disconnect.item()) for data in val_graphs])

        probes = {}
        for name, model in {
            "logreg": LogisticRegression(max_iter=2000),
            "rf": RandomForestClassifier(n_estimators=200, random_state=self.config.seed),
        }.items():
            model.fit(x_train, y_train_topology)
            topology_acc = accuracy_score(y_val_topology, model.predict(x_val))
            model.fit(x_train, y_train_attack)
            attack_acc = accuracy_score(y_val_attack, model.predict(x_val))
            probes[name] = {
                "topology_accuracy": float(topology_acc),
                "attack_accuracy": float(attack_acc),
            }

        passed = all(
            probe["topology_accuracy"] <= self.config.local_audit_max_topology_accuracy
            for probe in probes.values()
        )
        return {"passed": passed, "probes": probes}

    def _run_split_sanity(self, corpus: dict) -> dict:
        counts = {}
        node_count_ranges = defaultdict(lambda: [10**9, -1])
        train_nodes = []
        train_labels = []
        balanced_splits = {
            "train",
            "val",
            "id_test",
            "near_ood",
            "far_ood",
            "cell_size_ood",
            "noise_low",
            "noise_med",
            "noise_high",
        }
        for split_name, data_list in corpus.items():
            class_hist = defaultdict(int)
            nodes = []
            labels = []
            for data in data_list:
                label = INDEX_TO_TOPOLOGY[int(data.y_topology.item())]
                class_hist[label] += 1
                nodes.append(data.num_nodes)
                labels.append(int(data.y_topology.item()))
                node_count_ranges[label][0] = min(node_count_ranges[label][0], data.num_nodes)
                node_count_ranges[label][1] = max(node_count_ranges[label][1], data.num_nodes)
            counts[split_name] = dict(class_hist)
            if split_name == "train":
                train_nodes = nodes
                train_labels = labels

        num_node_accuracies = {}
        model = LogisticRegression(max_iter=1000)
        model.fit(np.array(train_nodes).reshape(-1, 1), np.array(train_labels))
        for split_name, data_list in corpus.items():
            nodes = np.array([data.num_nodes for data in data_list]).reshape(-1, 1)
            labels = np.array([int(data.y_topology.item()) for data in data_list])
            num_node_accuracies[split_name] = float(accuracy_score(labels, model.predict(nodes)))

        imbalance = {}
        for split_name in balanced_splits:
            hist = counts[split_name]
            values = list(hist.values())
            target = np.mean(values)
            imbalance[split_name] = float(max(abs(value - target) / target for value in values))

        passed = all(value <= 0.02 for value in imbalance.values()) and all(
            accuracy <= 0.35 for accuracy in num_node_accuracies.values()
        )
        return {
            "passed": passed,
            "class_histograms": counts,
            "relative_imbalance": imbalance,
            "num_node_only_accuracy": num_node_accuracies,
            "node_count_ranges": {key: value for key, value in node_count_ranges.items()},
        }

    def _run_size_control_sanity(self, corpus: dict) -> dict:
        train_num_cells = [int(graph.meta["num_cells"]) for graph in corpus["train"]]
        near_num_cells = [int(graph.meta["num_cells"]) for graph in corpus["near_ood"]]
        far_num_cells = [int(graph.meta["num_cells"]) for graph in corpus["far_ood"]]
        cell_size_ood_cells = [int(graph.meta["num_cells"]) for graph in corpus["cell_size_ood"]]
        train_size_means = [float(graph.meta["cell_size_stats"]["mean"]) for graph in corpus["train"]]
        cell_size_ood_size_means = [
            float(graph.meta["cell_size_stats"]["mean"]) for graph in corpus["cell_size_ood"]
        ]
        passed = (
            np.mean(near_num_cells) > np.mean(train_num_cells)
            and np.mean(far_num_cells) > np.mean(near_num_cells)
            and abs(np.mean(cell_size_ood_cells) - np.mean(train_num_cells)) < 0.75
            and np.mean(cell_size_ood_size_means) > np.mean(train_size_means)
        )
        return {
            "passed": bool(passed),
            "mean_num_cells": {
                "train": float(np.mean(train_num_cells)),
                "near_ood": float(np.mean(near_num_cells)),
                "far_ood": float(np.mean(far_num_cells)),
                "cell_size_ood": float(np.mean(cell_size_ood_cells)),
            },
            "mean_cell_sizes": {
                "train": float(np.mean(train_size_means)),
                "cell_size_ood": float(np.mean(cell_size_ood_size_means)),
            },
        }

    def _local_descriptor(self, data) -> np.ndarray:
        graph = self._data_to_nx(data)
        num_nodes = max(1, graph.number_of_nodes())
        degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
        clustering = np.array(list(nx.clustering(graph).values()), dtype=float)
        degree_hist, _ = np.histogram(degrees, bins=10, range=(0, 20), density=True)
        clustering_hist, _ = np.histogram(clustering, bins=10, range=(0.0, 1.0), density=True)
        triangle_density = float(sum(nx.triangles(graph).values()) / (3.0 * num_nodes))
        four_cycle_density = float(count_four_cycles(graph) / num_nodes)
        descriptor = np.concatenate(
            [
                degree_hist,
                clustering_hist,
                np.array(
                    [
                        triangle_density,
                        four_cycle_density,
                    ]
                ),
            ]
        )
        return descriptor

    def _data_to_nx(self, data) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(data.num_nodes))
        for source, target in data.edge_index.t().tolist():
            if source <= target:
                graph.add_edge(int(source), int(target))
        return graph
