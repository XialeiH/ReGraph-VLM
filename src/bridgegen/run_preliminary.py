"""CLI for preliminary validation probes."""

from __future__ import annotations

import argparse

from .config import load_config
from .preliminary import PreliminarySuite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preliminary generator probes.")
    parser.add_argument("--config", required=True, help="Path to bridge_v1 YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    suite = PreliminarySuite(config)
    report = suite.run()
    print(report)


if __name__ == "__main__":
    main()

