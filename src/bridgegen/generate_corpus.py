"""CLI for corpus generation."""

from __future__ import annotations

import argparse

from .config import load_config
from .generator import BridgeGraphGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the bridge_v1 corpus.")
    parser.add_argument("--config", required=True, help="Path to bridge_v1 YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    generator = BridgeGraphGenerator(config)
    generator.generate_master_corpus()


if __name__ == "__main__":
    main()

