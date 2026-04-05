# Model4 Task

Goal: build a new single-player causal CS2 model that is simpler, faster, and easier to train than `model3`.

Target design:
- One player POV per sample, not all 5 players jointly.
- One causal token per frame, not 6 tokens per frame.
- Condition each frame on:
  - current visual/audio observation for that player
  - previous action embedding for that player
  - an `<SOS>` embedding at sequence start
- Previous action embedding should combine:
  - summed learned embeddings for active previous keyboard actions
  - learned embedding for previous `mouse_x` bin
  - learned embedding for previous `mouse_y` bin
- Training should be teacher-forced on previous actions.
- Inference should run autoregressively, feeding back the model's previous predicted actions.

Why:
- Greatly reduce token count and sequence cost.
- Allow much longer temporal context at the same transformer budget.
- Make action persistence and control continuity explicit.
- Simplify the modeling problem from 5-player joint prediction to 1-player control.

Dataset direction:
- Prefer true single-player windows over flattening dead/blank 5-player tails.
- Sample windows where the selected player is alive for the requested duration when possible.
- Keep the data format causal and compatible with autoregressive training.

Initial implementation plan:
1. Define new config and tensor shapes for single-player training.
2. Rewrite the model around 1 token per frame plus previous-action embeddings.
3. Adapt loss and Lightning training code to single-player outputs.
4. Implement a single-player dataset path.
5. Add smoke tests for forward, autoregressive inference, and one training step.

Non-goals for the first pass:
- Preserve full `model3` checkpoint compatibility.
- Preserve 5-player joint strategy modeling.
- Wire the full inference runtime/UI before training works.
