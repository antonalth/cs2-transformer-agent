#!/usr/bin/env bash
# Common training arguments for both from-scratch and resume runs.
COMMON_ARGS=(
  --windows-per-round 5
  --T-frames 128
  --epochs 12
  --use-precomputed-embeddings
  --dali-threads 10
  --log-every 100
  --data-root dataset0/
  --manifest dataset0/manifest.json
  --save-every 2000
  --eval-every 300
  --lr 2.5e-4
  --min-lr 1e-5
  --batch-size 1
  --accum-steps 8
  --lr-schedule cosine_restarts
  --warmup-updates 1500
  --cycle-updates 0
  --cycle-mult 2.0
  --balance-losses-after-updates 200
)
