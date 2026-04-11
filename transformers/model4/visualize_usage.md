docker compose exec dev pkill inspect_webapp.py
  docker exec cs2-model-training-dev-1 bash -lc "
    cd /workspace/transformers/model4 &&
    python inspect_webapp.py \
      --checkpoint /workspace/checkpoints_fsdp/model4_vitsplus768_bs8_ga1_20260404/checkpoint_stepstep=9800.ckpt \
      --data_root /workspace/dataset0 \
      --device cuda \
      --default_split val \
      --host 0.0.0.0 \
      --port 8080 
  "
