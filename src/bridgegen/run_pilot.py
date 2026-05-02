"""Pilot runner for topology classification."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .models import model_family
from .train import run_experiment
from .utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the topology pilot matrix.")
    parser.add_argument("--config", required=True, help="Path to pilot YAML config.")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for model_name in config["models"]:
        summary = run_experiment(
            data_root=config["data_root"],
            output_root=str(output_root / model_name),
            model_name=model_name,
            task=config["task"],
            seed=int(config["seed"]),
            train_limit=int(config["train_limit"]),
            val_limit=int(config["val_limit"]),
            eval_limits={
                "id_test": int(config["id_test_limit"]),
                "far_ood": int(config["far_ood_limit"]),
            },
            min_params=int(config["target_params"]["min"]),
            max_params=int(config["target_params"]["max"]),
            lr=float(config["trainer"]["lr"]),
            weight_decay=float(config["trainer"]["weight_decay"]),
            max_epochs=int(config["trainer"]["max_epochs"]),
            patience=int(config["trainer"]["patience"]),
            batch_size_flat=int(config["trainer"]["batch_size_flat"]),
            batch_size_hier=int(config["trainer"]["batch_size_hier"]),
        )
        summaries.append(summary)

    best_flat = max(
        (summary for summary in summaries if model_family(summary["model"]) == "direct"),
        key=lambda summary: summary["eval"]["id_test"]["accuracy"],
    )
    best_hier = max(
        (summary for summary in summaries if model_family(summary["model"]) != "direct"),
        key=lambda summary: summary["eval"]["far_ood"]["accuracy"],
    )
    gate = {
        "best_flat_id_accuracy": best_flat["eval"]["id_test"]["accuracy"],
        "best_hier_far_ood_accuracy": best_hier["eval"]["far_ood"]["accuracy"],
        "best_flat_far_ood_accuracy": best_flat["eval"]["far_ood"]["accuracy"],
    }
    gate["pilot_passed"] = (
        gate["best_flat_id_accuracy"] >= 0.60
        and gate["best_hier_far_ood_accuracy"] >= gate["best_flat_far_ood_accuracy"] + 0.05
    )
    payload = {"summaries": summaries, "gate": gate}
    save_json(payload, output_root / "pilot_summary.json")
    print(payload)


if __name__ == "__main__":
    main()

