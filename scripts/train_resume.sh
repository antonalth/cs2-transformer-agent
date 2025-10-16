#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <RUN_DIR> [extra-args...]"
  echo "Example: $0 dataset0/runs/20251015T030000Z --epochs 20 --eval-every 1000"
  exit 1
fi

RUN_DIR="$1"
shift || true

if [[ ! -d "$RUN_DIR" ]]; then
  echo "ERROR: run dir not found: $RUN_DIR"
  exit 1
fi

CKPT_DIR="$RUN_DIR/checkpoints"
if [[ ! -d "$CKPT_DIR" ]]; then
  echo "ERROR: checkpoints dir not found: $CKPT_DIR"
  exit 1
fi

# Pick resume checkpoint: last.ckpt → best.ckpt → latest last_step*.ckpt
RESUME=""
if [[ -f "$CKPT_DIR/last.ckpt" ]]; then
  RESUME="$CKPT_DIR/last.ckpt"
elif [[ -f "$CKPT_DIR/best.ckpt" ]]; then
  RESUME="$CKPT_DIR/best.ckpt"
else
  CAND=$(ls -1t "$CKPT_DIR"/last_step*.ckpt 2>/dev/null | head -n1 || true)
  if [[ -n "${CAND:-}" ]]; then
    RESUME="$CAND"
  fi
fi

if [[ -z "$RESUME" ]]; then
  echo "ERROR: no checkpoint found in $CKPT_DIR (looked for last.ckpt, best.ckpt, last_step*.ckpt)"
  exit 1
fi

# Number of processes (GPUs). Set NPROC env var to override.
if [[ -z "${NPROC:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC="$(nvidia-smi -L | wc -l | tr -d ' ')"
    [[ -z "$NPROC" || "$NPROC" -eq 0 ]] && NPROC=1
  else
    NPROC=1
  fi
fi

echo "→ Resuming from: $RESUME"
echo "→ Run dir      : $RUN_DIR"
echo "→ nproc        : $NPROC"
echo "→ Extra args   : ${*@Q}"

# NOTE: Pass your usual args (data roots, manifest, etc.) after RUN_DIR.
# Example extras: --data-root dataset0/ --manifest dataset0/manifest.json --eval-every 1000 --batch-size 2 --accum-steps 4
torchrun --standalone --nproc_per_node="$NPROC" transformers/model/train3.py \
  --mode train \
  --run-dir "$RUN_DIR" \
  --resume "$RESUME" \
  "$@"
