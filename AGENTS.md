# AGENTS.md

## Project: Transformer-based Counter-Strike 2 agentic model
- Goal: Train a model to control all 5 players on a single team in the game CS2
- Training data: 1k hours recorded from pro-teams playing at majors/prize events
- Model receives as input video frames from each player, audio samples
- Model predicts keyboard presses, mouse deltas, player positions, enemy positions, money, health...

### Repository structure
- Most important directory is transformers/model3/...
- dataset.py: harness to work with dataset (epochs etc)
- model_loss.py: defines loss functions for outputs
- model.py: central model definition
- config.py: contains central dataclass definitions for model, training etc.
- train.py: central training script, must be run INSIDE docker container
"python transformers/model3/train.py --data_root dataset0"

### Practical information
- Actual testing/training must happen inside of docker (if not up use 'docker compose up -d' or 'docker compose start'), for more check out the docker-compose.yml
- Make sure when you start for example the training script to work with the timeout command to not get stuck waiting for it to finish, and BEFORE running the command ALWAYS check if any remaining processes are left in the docker due to missing cleanup (happens with timeout), since they will cause errors if the training script is restarted (like no VRAM left, or port already occupied etc) - kill them and verify that they have been stopped
- If you try many many changes to fix a bug, and nothing works, consider periodically reverting back to the state before any changes were made, for example using git to avoid changing too many variables at once
- Always read the important files mentioned above to have the required context for approaching bugs or implementing features
- When asked to fix a bug, first reproduce the bug. Then iterate until the bug is fixed, then figure out what the actual fix was(the MINIMAL fix, removing uncessary intermediate steps), for example by re-running the training script
- Do NOT commit fixes, a human must oversee the results first.
- **Training Policy:** Any training restart that is not part of an immediate debugging loop (e.g., iterating to find errors) but affects a longer existing run MUST be confirmed by a human first.
- **Long-term Training:** Always start long-term training runs inside a new tmux session **on the host machine** (wrapping the docker execution command) so they can be resumed and viewed later, as can be done like so: 
`docker exec cs2-model-training-dev-1 pkill -9 -f "train.py"; sleep 5; tmux new-session -d -s train 'docker exec -it cs2-model-training-dev-1 python transformers/model3/train.py --data_root dataset0'`

### Training Loss Monitoring
- **WandB Monitor**: CLI tool to track training progress and losses.
  - **Run command**: `docker exec <container_id> python3 tools/wandb_monitor.py [flags]`

### Inference Visualization
- **Video Generation**: Generate a side-by-side GT vs Prediction video from a sharded FSDP checkpoint.
  - **Run command**: `docker exec <container_id> accelerate launch --num_processes 4 --use_fsdp --fsdp_version 2 --mixed_precision bf16 transformers/model2/visualize_inference.py --checkpoint <path_to_checkpoint> --output <filename>.mp4`