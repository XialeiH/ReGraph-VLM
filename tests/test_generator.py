from pathlib import Path

from bridgegen.config import BridgeSuiteConfig
from bridgegen.generator import BridgeGraphGenerator


def test_small_generation(tmp_path: Path):
    config = BridgeSuiteConfig.from_dict(
        {
            "output_root": str(tmp_path / "bridge_v1"),
            "seed": 7,
            "feature_mode": "constant",
            "stability_subset_size": 4,
            "stability_variants": 2,
            "local_audit_max_topology_accuracy": 0.35,
            "local_audit_max_attack_accuracy": 0.60,
            "mean_degree_target": 5.5,
            "mean_degree_tolerance": 1.5,
            "max_clustering_gap": 0.05,
            "max_graph_attempts": 16,
            "noise_levels": {
                "low": {
                    "intra_cell_rewire": 0.05,
                    "same_side_shortcut": 0.01,
                    "fake_bridge_distractor": 0.00,
                },
                "med": {
                    "intra_cell_rewire": 0.10,
                    "same_side_shortcut": 0.02,
                    "fake_bridge_distractor": 0.01,
                },
                "high": {
                    "intra_cell_rewire": 0.20,
                    "same_side_shortcut": 0.05,
                    "fake_bridge_distractor": 0.02,
                },
            },
            "splits": {
                "train": {"num_graphs": 8, "cell_range": [6, 8], "cell_size_range": [8, 10]},
                "val": {"num_graphs": 4, "cell_range": [6, 8], "cell_size_range": [8, 10]},
                "id_test": {"num_graphs": 4, "cell_range": [6, 8], "cell_size_range": [8, 10]},
                "near_ood": {"num_graphs": 4, "cell_range": [9, 10], "cell_size_range": [8, 10]},
                "far_ood": {"num_graphs": 4, "cell_range": [11, 12], "cell_size_range": [8, 10]},
                "cell_size_ood": {"num_graphs": 4, "cell_range": [6, 8], "cell_size_range": [12, 14]},
            },
        }
    )
    generator = BridgeGraphGenerator(config)
    corpus = generator.generate_master_corpus()

    assert "train" in corpus
    assert len(corpus["train"]) == 8
    sample = corpus["train"][0]
    assert hasattr(sample, "edge_index")
    assert hasattr(sample, "cell_id")
    assert hasattr(sample, "coarse_edge_index")
    assert sample.num_nodes > 0

