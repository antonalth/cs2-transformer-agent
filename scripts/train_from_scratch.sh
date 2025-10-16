#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-dataset0/runs/$(date -u +%Y%m%dT%H%M%SZ)}"
PORT="${PORT:-6006}"
NPROC="${NPROC:-4}"
LOG_FILE="${RUN_DIR}/train.log"

export OMP_NUM_THREADS=10

echo "→ Run dir : ${RUN_DIR}"
mkdir -p "${RUN_DIR}"

echo "→ Logging to ${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

# start TensorBoard in background (if not already on that port)
if ! lsof -i :"${PORT}" >/dev/null 2>&1; then
  tensorboard --logdir "${RUN_DIR}" --port "${PORT}" --bind_all >/dev/null 2>&1 &
  TB_PID=$!
  echo "→ TensorBoard: http://localhost:${PORT}  (pid ${TB_PID})"
else
  echo "→ TensorBoard already serving on port ${PORT}"
  TB_PID=""
fi

# ensure TB is stopped on Ctrl-C
cleanup() {
  if [[ -n "${TB_PID:-}" ]] && ps -p "${TB_PID}" >/dev/null 2>&1; then
    echo "→ Stopping TensorBoard (${TB_PID})"
    kill "${TB_PID}" || true
  fi
}
trap cleanup INT TERM

torchrun --standalone --nproc_per_node="${NPROC}" transformers/model/train3.py \
    --mode train \
    --windows-per-round 5 \
    --T-frames 192 \
    --epochs 12 \
    --run-dir "${RUN_DIR}" \
    --use-precomputed-embeddings \
    --dali-threads 10 \
    --log-every 100 \
    --data-root dataset0/ \
    --manifest dataset0/manifest.json \
    --save-every 2000 \
    --eval-every 300 \
    --lr 2.5e-4 \
    --min-lr 1e-5 \
    --batch-size 1 \
    --accum-steps 8

echo "✅ Training finished."
if [[ -n "${TB_PID:-}" ]] && ps -p "${TB_PID}" >/dev/null 2>&1; then
  echo "ℹ️  TensorBoard is still running at http://localhost:${PORT} (pid ${TB_PID})."
  echo "    Press Ctrl-C to stop it."
  # keep script alive so you can read the message; remove the next line if you don't want this behavior
  wait "${TB_PID}"
fi
