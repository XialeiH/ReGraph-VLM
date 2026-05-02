#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/bridge_v1}"
OUTPUT_ROOT="${2:-results/full_matrix}"

for task in topology bridge_count attack_disconnect redundant; do
  for model in gin gatv2 graphgps diffpool mincut ptr ptr_sup; do
    python3 -m bridgegen.train \
      --data-root "${DATA_ROOT}" \
      --output-root "${OUTPUT_ROOT}/${task}/${model}" \
      --model "${model}" \
      --task "${task}" \
      --seed 11
  done
done

