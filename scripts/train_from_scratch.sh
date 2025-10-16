#!/usr/bin/env bash
set -euo pipefail

# Unique run directory (UTC ISO timestamp). Use +%s for pure Unix time if you prefer.
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="dataset0/runs/${RUN_ID}"

echo "→ Using run dir: ${RUN_DIR}"

torchrun --standalone --nproc_per_node=4 transformers/model/train3.py \
  --mode train \
  --windows-per-round 5 \
  --T-frames 200 \
  --epochs 12 \
  --run-dir "${RUN_DIR}" \
  --use-precomputed-embeddings \
  --log-every 100 \
  --data-root dataset0/ \
  --manifest dataset0/manifest.json \
  --save-every 2500 \
  --eval-every 300 \
  --batch-size 2 \
  --accum-steps 4
