#!/usr/bin/env bash
set -euo pipefail

python3 -m pytest tests/test_generator.py tests/test_models.py
python3 -m bridgegen.generate_corpus --config configs/smoke_bridge.yaml
python3 -m bridgegen.run_pilot --config configs/smoke_pilot.yaml
