# Start on Simulation server:
$    ./.venv/bin/python -m sim_harness.main --config sim_harness.toml

# Start on Training/Inference server:
$ docker exec -it cs2-model-training-dev-1 bash -lc '
    cd /workspace &&
    python -m inference.model3_runtime.main \
      --checkpoint checkpoints_fsdp/model3_ga8_perceiver_learnedpe_noenemypos_20260322/checkpoint_stepstep=9000.ckpt \
      --harness-url https://10.7.30.53:8080/ \
      --data-root dataset0 \
      --device cuda \
      --port 8080 \
      --insecure
  '
