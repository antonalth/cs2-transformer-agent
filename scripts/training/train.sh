#!/usr/bin/env bash
set -euo pipefail

# Unified training launcher
# Usage:
#   bash train.sh RUN_DIR [--resume [CKPT]] [extra-args...]
#
# Env:
#   PORT   : TensorBoard port (default 6006)
#   NPROC  : (optional) desired processes; will be clamped to available GPUs

join_by() { local IFS="$1"; shift; echo "$*"; }

# ---------- Parse RUN_DIR ----------
if (( $# == 0 )) || [[ "${1:-}" == --* ]]; then
  echo "ERROR: First argument must be RUN_DIR." >&2
  exit 2
fi
RUN_DIR="$1"; shift
mkdir -p "$RUN_DIR"

# ---------- Parse flags ----------
RESUME_MODE="none"     # none | auto | path
RESUME_PATH=""
SANITIZED_ARGS=()

while (( $# > 0 )); do
  case "$1" in
    --resume)
      if (( $# >= 2 )) && [[ -n "${2:-}" && "${2:0:2}" != "--" ]]; then
        RESUME_MODE="path"; RESUME_PATH="$2"; shift 2
      else
        RESUME_MODE="auto"; shift 1
      fi
      ;;
    *)
      [[ -n "$1" ]] && SANITIZED_ARGS+=("$1"); shift 1
      ;;
  esac
done

# ---------- Resolve resume path if needed ----------
if [[ "$RESUME_MODE" == "auto" ]]; then
  CANDIDATE="${RUN_DIR%/}/checkpoints/last.ckpt"
  if [[ -f "$CANDIDATE" ]]; then
    RESUME_PATH="$CANDIDATE"
  else
    echo "WARN: --resume given without path but no last.ckpt at $CANDIDATE; continuing without resume."
    RESUME_MODE="none"
  fi
fi

# ---------- TensorBoard (best effort) ----------
PORT="${PORT:-6006}"
if command -v tensorboard >/dev/null 2>&1; then
  mkdir -p "$RUN_DIR/tb"
  (tensorboard --logdir "$RUN_DIR" --port "$PORT" --host 0.0.0.0 >/dev/null 2>&1 & echo $! >"$RUN_DIR/.tb.pid") || true
  TB_PID="$(cat "$RUN_DIR/.tb.pid" 2>/dev/null || true)"
  echo "→ TensorBoard: http://localhost:${PORT} ${TB_PID:+(pid ${TB_PID})}"
else
  echo "→ TensorBoard: not installed (skipping)"
fi

# ---------- GPU detection ----------
gpu_count() {
  # If CUDA_VISIBLE_DEVICES is set, count its entries (ignore empties and -1)
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    # Normalize commas and strip spaces
    local list="${CUDA_VISIBLE_DEVICES// /}"
    # Remove any leading/trailing commas
    list="${list#,}"; list="${list%,}"
    # Count non-empty tokens not equal to -1
    awk -v s="$list" '
      BEGIN{
        n=split(s, a, ",");
        cnt=0;
        for(i=1;i<=n;i++){ if(a[i] != "" && a[i] != "-1") cnt++ }
        print cnt
      }'
    return
  fi
  # Try nvidia-smi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local n
    n=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$n" =~ ^[0-9]+$ ]] && (( n > 0 )); then echo "$n"; return; fi
  fi
  # Try PyTorch
  if command -v python >/dev/null 2>&1; then
    python - <<'PY' 2>/dev/null || true
import os, sys
try:
    import torch
    n = torch.cuda.device_count()
    print(n if n is not None else 0)
except Exception:
    print(0)
PY
    return
  fi
  echo 0
}

VISIBLE_GPUS="$(gpu_count)"
# Desired processes (optional override), but clamp to GPU count if GPUs>0
REQ_NPROC="${NPROC:-}"
if (( VISIBLE_GPUS > 0 )); then
  if [[ -n "$REQ_NPROC" ]] && [[ "$REQ_NPROC" =~ ^[0-9]+$ ]] && (( REQ_NPROC > 0 )); then
    NPROC=$(( REQ_NPROC < VISIBLE_GPUS ? REQ_NPROC : VISIBLE_GPUS ))
  else
    NPROC="${VISIBLE_GPUS}"
  fi
else
  # No GPUs → force single process (CPU) to avoid CUDA ordinal errors
  NPROC=1
fi

echo "→ visible gpus : ${VISIBLE_GPUS}"
echo "→ nproc        : ${NPROC}"

# ---------- Load shared arguments ----------
COMMON_ARGS=()
if [[ -f "common_train_args.sh" ]]; then
  # shellcheck source=/dev/null
  source "common_train_args.sh"
elif [[ -f "scripts/training/common_train_args.sh" ]]; then
  # shellcheck source=/dev/null
  source "scripts/training/common_train_args.sh"
fi

# ---------- Logging ----------
if [[ "$RESUME_MODE" == "path" || "$RESUME_MODE" == "auto" ]]; then
  echo "→ resume       : ${RESUME_PATH}"
else
  echo "→ resume       : (none)"
fi
if (( ${#SANITIZED_ARGS[@]} )); then
  echo "→ extra args   : $(printf '%q ' "${SANITIZED_ARGS[@]}")"
else
  echo "→ extra args   : (none)"
fi

# ---------- Build command ----------
CMD=(torchrun --standalone --nproc_per_node="${NPROC}"
     transformers/model/train3.py
     --mode train
     --run-dir "${RUN_DIR}")

# Append common args
if (( ${#COMMON_ARGS[@]} )); then
  CMD+=("${COMMON_ARGS[@]}")
fi

# Append resume iff set
if [[ "$RESUME_MODE" != "none" && -n "${RESUME_PATH}" ]]; then
  CMD+=(--resume "${RESUME_PATH}")
fi

# Append sanitized extra args
if (( ${#SANITIZED_ARGS[@]} )); then
  for a in "${SANITIZED_ARGS[@]}"; do
    [[ -n "$a" ]] && CMD+=("$a")
  done
fi

# ---------- Show & launch ----------
echo "→ Launch: ${CMD[*]}"
"${CMD[@]}"

echo "✅ Training finished."
if [[ -n "${TB_PID:-}" ]] && ps -p "${TB_PID}" >/dev/null 2>&1; then
  echo "ℹ️  TensorBoard still running at http://localhost:${PORT} (pid ${TB_PID})."
  echo "    Press Ctrl-C to stop it."
  wait "${TB_PID}"
fi
