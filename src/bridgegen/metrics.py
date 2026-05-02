"""Task and mechanism metrics without sklearn runtime dependencies."""

from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .utils import sorted_edge


def _contingency(labels_a: np.ndarray, labels_b: np.ndarray) -> np.ndarray:
    unique_a, inverse_a = np.unique(labels_a, return_inverse=True)
    unique_b, inverse_b = np.unique(labels_b, return_inverse=True)
    table = np.zeros((len(unique_a), len(unique_b)), dtype=np.int64)
    np.add.at(table, (inverse_a, inverse_b), 1)
    return table


def _entropy(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log(probs + 1e-12)).sum())


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(np.concatenate([y_true, y_pred]))
    scores = []
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def binary_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def roc_auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_rank_sum = ranks[y_true == 1].sum()
    auc = (pos_rank_sum - pos * (pos + 1) / 2) / (pos * neg)
    return float(auc)


def adjusted_rand_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    table = _contingency(labels_true, labels_pred)
    n = table.sum()
    sum_comb_c = np.sum(table * (table - 1) // 2)
    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)
    sum_comb_rows = np.sum(row_sums * (row_sums - 1) // 2)
    sum_comb_cols = np.sum(col_sums * (col_sums - 1) // 2)
    total_pairs = n * (n - 1) / 2
    expected = (sum_comb_rows * sum_comb_cols) / max(total_pairs, 1)
    max_index = 0.5 * (sum_comb_rows + sum_comb_cols)
    denominator = max_index - expected
    if denominator == 0:
        return 1.0
    return float((sum_comb_c - expected) / denominator)


def normalized_mutual_info_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    table = _contingency(labels_true, labels_pred).astype(np.float64)
    n = table.sum()
    row_sums = table.sum(axis=1, keepdims=True)
    col_sums = table.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / max(n, 1.0)
    mask = table > 0
    mutual_info = np.sum((table[mask] / n) * np.log(table[mask] / expected[mask]))
    entropy_true = _entropy(labels_true)
    entropy_pred = _entropy(labels_pred)
    denom = entropy_true + entropy_pred
    if denom == 0:
        return 1.0
    return float(2.0 * mutual_info / denom)


def simple_kmeans(embeddings: np.ndarray, num_clusters: int, max_iters: int = 50) -> np.ndarray:
    rng = np.random.default_rng(0)
    if len(embeddings) <= num_clusters:
        return np.arange(len(embeddings)) % num_clusters
    centers = embeddings[rng.choice(len(embeddings), size=num_clusters, replace=False)]
    labels = np.zeros(len(embeddings), dtype=np.int64)
    for _ in range(max_iters):
        distances = np.linalg.norm(embeddings[:, None, :] - centers[None, :, :], axis=-1)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster_id in range(num_clusters):
            members = embeddings[labels == cluster_id]
            if len(members) == 0:
                centers[cluster_id] = embeddings[rng.integers(0, len(embeddings))]
            else:
                centers[cluster_id] = members.mean(axis=0)
    return labels


def compute_task_metrics(task: str, logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    logits_np = logits.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    if task == "topology":
        pred = logits_np.argmax(axis=-1)
        return {
            "accuracy": accuracy_score(target_np, pred),
            "macro_f1": macro_f1_score(target_np, pred),
        }
    if task == "bridge_count":
        pred = logits_np.reshape(-1)
        return {"mae": float(np.mean(np.abs(pred - target_np.reshape(-1))))}
    scores = logits_np.reshape(-1)
    pred = (scores > 0).astype(int)
    return {
        "auroc": roc_auc_score(target_np.reshape(-1), scores),
        "f1": binary_f1_score(target_np.reshape(-1), pred),
    }


def assignment_from_embeddings(embeddings: torch.Tensor, num_clusters: int) -> np.ndarray:
    return simple_kmeans(embeddings.detach().cpu().numpy(), num_clusters=num_clusters)


def soft_assignment_to_labels(assignment: torch.Tensor) -> np.ndarray:
    return assignment.detach().cpu().numpy().argmax(axis=-1)


def align_partition_labels(pred_labels: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
    pred_ids = np.unique(pred_labels)
    true_ids = np.unique(true_labels)
    cost = np.zeros((len(pred_ids), len(true_ids)), dtype=np.int64)
    for i, pred_id in enumerate(pred_ids):
        for j, true_id in enumerate(true_ids):
            cost[i, j] = -np.sum((pred_labels == pred_id) & (true_labels == true_id))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {pred_ids[row]: true_ids[col] for row, col in zip(row_ind, col_ind)}
    return np.array([mapping.get(label, true_ids[0]) for label in pred_labels], dtype=np.int64)


def partition_iou(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    aligned = align_partition_labels(pred_labels, true_labels)
    intersection = np.sum(aligned == true_labels)
    union = len(true_labels) + len(aligned) - intersection
    return float(intersection / max(union, 1))


def cell_recovery_scores(pred_labels: np.ndarray, true_labels: np.ndarray) -> dict[str, float]:
    return {
        "ari": adjusted_rand_score(true_labels, pred_labels),
        "nmi": normalized_mutual_info_score(true_labels, pred_labels),
        "iou": partition_iou(pred_labels, true_labels),
    }


def coarse_graph_from_partition(edge_index: torch.Tensor, partition: np.ndarray) -> nx.Graph:
    graph = nx.Graph()
    for cluster_id in np.unique(partition):
        graph.add_node(int(cluster_id))
    for source, target in edge_index.t().tolist():
        cluster_source = int(partition[source])
        cluster_target = int(partition[target])
        if cluster_source != cluster_target:
            graph.add_edge(cluster_source, cluster_target)
    return graph


def skeleton_fidelity(
    pred_partition: np.ndarray,
    true_partition: np.ndarray,
    edge_index: torch.Tensor,
    coarse_edge_index: torch.Tensor,
) -> dict[str, float]:
    aligned = align_partition_labels(pred_partition, true_partition)
    pred_graph = coarse_graph_from_partition(edge_index, aligned)
    true_graph = nx.Graph()
    true_graph.add_nodes_from(np.unique(true_partition).tolist())
    for source, target in coarse_edge_index.t().tolist():
        if source <= target:
            true_graph.add_edge(int(source), int(target))

    pred_edges = {sorted_edge(u, v) for u, v in pred_graph.edges()}
    true_edges = {sorted_edge(u, v) for u, v in true_graph.edges()}
    true_positive = len(pred_edges & true_edges)
    precision = true_positive / max(len(pred_edges), 1)
    recall = true_positive / max(len(true_edges), 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    edit_distance = len(pred_edges ^ true_edges) + abs(
        pred_graph.number_of_nodes() - true_graph.number_of_nodes()
    )
    return {
        "coarse_precision": float(precision),
        "coarse_recall": float(recall),
        "coarse_f1": float(f1),
        "coarse_edit_distance": float(edit_distance),
    }


def mechanism_metrics_for_batch(outputs, batch) -> dict[str, float]:
    scores = defaultdict(list)
    for graph, embeddings, assignment in zip(
        batch.to_data_list(),
        outputs.node_embeddings,
        outputs.assignments or [None] * len(outputs.node_embeddings),
    ):
        true_labels = graph.cell_id.cpu().numpy()
        if assignment is None:
            pred_labels = assignment_from_embeddings(
                embeddings,
                num_clusters=int(graph.cell_id.max().item()) + 1,
            )
        else:
            pred_labels = soft_assignment_to_labels(assignment)
        recovery = cell_recovery_scores(pred_labels, true_labels)
        fidelity = skeleton_fidelity(pred_labels, true_labels, graph.edge_index, graph.coarse_edge_index)
        for key, value in {**recovery, **fidelity}.items():
            scores[key].append(value)
    return {key: float(np.mean(values)) for key, values in scores.items()}


def stability_from_partitions(partitions: list[np.ndarray]) -> float:
    if len(partitions) < 2:
        return 1.0
    scores = []
    for left in range(len(partitions)):
        for right in range(left + 1, len(partitions)):
            scores.append(normalized_mutual_info_score(partitions[left], partitions[right]))
    return float(np.mean(scores))
