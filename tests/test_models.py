from pathlib import Path

from torch_geometric.loader import DataLoader

from bridgegen.config import BridgeSuiteConfig
from bridgegen.generator import BridgeGraphGenerator
from bridgegen.models import create_model


def _tiny_config(tmp_path: Path) -> BridgeSuiteConfig:
    return BridgeSuiteConfig.from_dict(
        {
            "output_root": str(tmp_path / "bridge_v1"),
            "seed": 11,
            "feature_mode": "constant",
            "stability_subset_size": 2,
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


def test_all_models_forward(tmp_path: Path):
    config = _tiny_config(tmp_path)
    generator = BridgeGraphGenerator(config)
    corpus = generator.generate_master_corpus()
    batch = next(iter(DataLoader(corpus["train"][:2], batch_size=2, shuffle=False)))
    input_dim = int(batch.x.size(-1))

    for model_name in ["gin", "gatv2", "graphgps", "diffpool", "mincut", "ptr"]:
        model = create_model(model_name, input_dim=input_dim, hidden_dim=96, task="topology", num_layers=4)
        outputs = model(batch, return_aux=True)
        assert outputs.logits.size(0) == 2
        assert outputs.logits.size(1) == 4

