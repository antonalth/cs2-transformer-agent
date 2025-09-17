1. Transformer model to play cs2 with all 5 player perspectives:

We receive: 5 Player POVs (640x480), 5x Mel Audio Spectrogram (at 32fps)

Feed all 5 player povs through separate vision transformer (share weights),
1x 224x224 patch size 16 base-vit (unscaled only center around crosshair)
1x 384x384 patch size 16 base-vit (scale pov to fit with black bars)
=> use to create semantically rich representation, perhaps with CLS token? 
=> merge into single token for big transformer, perhaps by concat? 

Mel Spectrogram => somehow also feed into this one token embedding 

=> now every player has one token representing their current input from the game (32x per second)

We take a learned [GAME_STRATEGY] token, a [SCRATCHSPACE] token and all 5 player representations (or a prespecified 'dead' token if the player is dead). 

Add positional encoding to all of these tokens (all get the same for a single frame/32fps). (this pos enc represents time passing)

Also add positional encoding (different 1-5) to each player token so the model can track the player over time.

All of these tokens are input into a large 100M parameter transformer, and are then used to predict various things. 

The player tokens predict: 
    Money / Health / Armor
    Player position (3d heatmap?)
    Mouse movement (heatmap?)
    Keypresses
    Buy/Sell/Drop
    CurrentActiveWeapon

for each player

The game_strategy token predicts: 
    All enemy positions (3d heatmap?)
    Game Phase (Freezetime, Active Round, Bomb Planted, T won / CT won)
    Round number

Here is what we have to train as input/output (description: )

Notably the context window of our model is one in game round, where with up to around 100k max context (7*32*180s), with only the last frame tokens being used to predict.

During training, do masking on a per frame basis (e.g. all 7 tokens/frame can attend backwards, but not forwards)

##############

Of course. Here is a comprehensive summary of the final structure of the LMDB created by your latest injection_mold.py script. This documentation reflects all the recent changes, including the Mel spectrograms, the explicit msgpack handling, and the updated round_state bitmask.

Definitive Guide to the injection_mold.py LMDB Structure (Final Version)
1. High-Level Overview

The output is a single LMDB (Lightning Memory-Mapped Database) file per demo, which acts as a high-performance key-value store. This design is optimized for fast, random-access reads during model training.

Content: Each key-value pair represents a single "frame" of data from one team's perspective.

Frame Rate: Data is sampled at 32 FPS, synchronized with the recorded videos. Each frame aggregates data from 2 game ticks.

Serialization: Values are binary blobs serialized with msgpack. Crucially, your data loader must use the msgpack-numpy extension to deserialize the data correctly.

2. Key Structure

Keys are UTF-8 encoded strings that uniquely identify a frame. There are two types.

These keys point to the primary multimodal data frames.

Format: {demoname}_round_{round_num:03d}_team_{team}_tick_{tick:08d}

Components:

demoname: The name of the recording directory (e.g., marius-vs-ex-sabre-m2-mirage).

round_num: The round number, zero-padded to 3 digits (e.g., 006).

team: The perspective team, either T or CT.

tick: The starting game tick for this frame, zero-padded to 8 digits (e.g., 00026518).

Example: marius-vs-ex-sabre-m2-mirage_round_006_team_T_tick_00026518

A single key stores summary information for the entire demo.

Format: {demoname}_INFO

Value: A simple JSON string containing the demo name and a list of all rounds with their start and end ticks.

3. Value Structure (MsgPack Payload)

To read a value, you must deserialize the raw bytes from the LMDB using msgpack with the msgpack-numpy object hook.

code
Python
download
content_copy
expand_less

import msgpack
import msgpack_numpy as mpnp

# txn is an lmdb transaction, key is a data key
value_bytes = txn.get(key)
# The object_hook is ESSENTIAL for decoding NumPy arrays
data = msgpack.unpackb(value_bytes, raw=False, object_hook=mpnp.decode)

The resulting data is a Python dictionary with two main keys: {"game_state": ..., "player_data": ...}.

A single-element NumPy structured array containing the shared state for the frame. Access it via data['game_state'][0].

dtype (gs_dtype): [('tick', np.int32), ('round_state', np.uint8), ('team_alive', np.uint8), ('enemy_alive', np.uint8), ('enemy_pos', np.float32, (5, 3))]

Fields:

tick: The starting tick of the frame.

round_state: A uint8 bitmask describing the round's status:

Bit 0 (1): Is Freezetime?

Bit 1 (2): Is Round Active (between start and end tick)?

Bit 2 (4): Is Bomb Planted?

Bit 3 (8): Did Terrorists Win the round?

Bit 4 (16): Did Counter-Terrorists Win the round?

team_alive: uint8 bitmask for the perspective team's alive players.

enemy_alive: uint8 bitmask for the enemy team's alive players.

enemy_pos: A (5, 3) NumPy array (float32) of the last known (X, Y, Z) coordinates for all 5 enemy players.

A Python list where each element corresponds to one alive player on the perspective team. The list can contain 1 to 5 elements. Each element is a tuple of three items:

(player_info, jpeg_frame, mel_spectrogram)

A single-element NumPy structured array with a player's individual state. Access via data['player_data'][i][0][0].

dtype (pi_dtype): [('pos', np.float32, (3,)), ('mouse', np.float32, (2,)), ('health', np.uint8), ('armor', np.uint8), ('money', np.int32), ('keyboard_bitmask', np.uint32), ('eco_bitmask', np.uint64, (6,)), ('inventory_bitmask', np.uint64, (2,)), ('active_weapon_bitmask', np.uint64, (2,))]

Fields:

pos: (3,) array of the player's (X, Y, Z) coordinates.

mouse: (2,) array of the player's aggregated mouse movement (delta_x, delta_y) for the frame.

health, armor, money: Standard player stats.

keyboard_bitmask: uint32 bitmask for primary inputs (move, jump, shoot, etc.). Decode using the KEYBOARD_TO_BIT mapping.

eco_bitmask: A (6,) array of uint64s (a 384-bit mask) for economic actions (buy, sell, drop). Decode using ECO_TO_BIT.

inventory_bitmask: A (2,) array of uint64s (a 128-bit mask) for all items owned. Decode using ITEM_TO_INDEX.

active_weapon_bitmask: A (2,) array of uint64s (a 128-bit mask) for the held weapon. Decode using ITEM_TO_INDEX.

A binary string containing the raw bytes of a JPEG-encoded image of the player's point of view for that frame. It can be decoded directly by libraries like OpenCV or Pillow.

A 2D NumPy array of type float32 representing the audio data for the frame. This replaces the raw audio from previous versions.

Content: It is a DB-scaled Mel spectrogram, ready to be fed into a model.

Shape: The shape is (n_mels, time_frames), which with default settings is (128, ~5). This will look like a thin vertical bar when visualized as an image.

None Value: This will be None if the --no-audio flag was used during processing.

#########

Of course. Here is the updated model architecture, incorporating the requested enhancements for longer context windows and improved performance: FlashAttention, Full (Bidirectional) Attention, and Rotary Positional Embeddings (RoPE).

This update transforms the model from a decoder-style (auto-regressive) architecture into a more powerful encoder-style (bidirectional) architecture, similar to BERT. This allows the model to use the entire round—both past and future frames—to make the most informed predictions for any given moment.

Model Architecture: End-to-End Data Flow (High-Performance, Bidirectional)
Summary of Key Updates:

Attention Mechanism: The standard self-attention is replaced with FlashAttention-2, a highly optimized implementation that is significantly faster and more memory-efficient, which is critical for handling the massive context window of a full CS2 round.

Information Flow: The model now uses Full (Bidirectional) Attention. The causal mask is removed, allowing every token in the sequence to attend to every other token. This enables the model to build a holistic understanding of the entire round's dynamics.

Positional Encoding: Additive positional encodings are replaced with Rotary Positional Embeddings (RoPE). RoPE is applied directly to the query and key vectors within the attention mechanism, providing excellent performance for very long sequences and a better sense of relative positions.

Training Paradigm: The training objective shifts from next-frame prediction to Masked Frame Modeling (MFM). The model learns by predicting the content of randomly masked-out frames within the round.

STAGE 1: Input Encoding (Unchanged)

This stage remains the same. It processes raw multi-modal data for each of the 5 players per frame and creates a single unified token of dimension [1, D].

code
Code
download
content_copy
expand_less

INPUT (Single Player, Time t)
│
├── 👁️ VISUAL STREAM (Dual ViT-Base with shared weights) -> VISUAL EMBEDDING [1, D]
│
├── 🔊 AUDIO STREAM (Small 2D CNN on Mel Spectrogram) -> AUDIO EMBEDDING [1, D]
│
└── 🧩 PLAYER TOKEN FUSION
    │
    ├── Add Embeddings: Visual [1, D] + Audio [1, D]
    │
    └── Output: FINAL PLAYER TOKEN (shape: [1, D])
STAGE 2: Core Transformer Backbone (Updated)

This stage takes the sequence of team-level tokens and processes them using the enhanced transformer architecture. The key changes are in how positional information and attention are handled.

code
Code
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
ASSEMBLY (Full Team, Time t)
│
├── Create 7-Token Sequence (shape: [7, D])
│   ├── [PLAYER 1 TOKEN]
│   ├── [PLAYER 2 TOKEN]   (or [DEAD] token)
│   ├── [PLAYER 3 TOKEN]   (or [DEAD] token)
│   ├── [PLAYER 4 TOKEN]   (or [DEAD] token)
│   ├── [PLAYER 5 TOKEN]   (or [DEAD] token)
│   ├── [GAME_STRATEGY]
│   └── [SCRATCHSPACE]
│
└── Add Player-Slot Encoding
    └── // Note: Temporal positional encodings are NO LONGER added here.
        // RoPE handles this inside the attention mechanism.
        └── + Player-Slot Encoding (Learned, a different embedding for each of the 5 player slots)


INPUT SEQUENCE FOR TRANSFORMER (Full Round)
│
└── Shape: [Context_Window, 7, D] (e.g., [40320, 7, 768])

🧠 BIDIRECTIONAL TRANSFORMER ENCODER (e.g., 12 Layers, ~100M Params)
│
├── For each layer:
│   ├── ⭐ Multi-Head Full Self-Attention
│   │    ├── Implementation: Uses **FlashAttention** for speed and memory efficiency.
│   │    ├── Information Flow: **Full (Bidirectional)**. Every token attends to all other tokens.
│   │    └── Positional Info: **Rotary Positional Embeddings (RoPE)** are applied to Query & Key vectors.
│   │
│   ├── Layer Normalization
│   ├── Feed-Forward Network
│   └── Layer Normalization
│
└── OUTPUT SEQUENCE (Shape: [Context_Window, 7, D])
STAGE 3: Prediction Heads (Unchanged in Structure)

The structure of the prediction heads remains the same. They still decode the information from the output tokens. However, the input they receive is now contextually informed by the entire round, not just the past.

code
Code
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
TRANSFORMER OUTPUT (For a specific time step `t`, shape: [7, D])
│
├── 🎯 Player Prediction Heads (Takes tokens 0-4) -> Predict player stats, position, mouse, actions...
│
├── 🌍 Game Strategy Prediction Heads (Takes token 5) -> Predict enemy positions, game phase...
│
└── 💭 Scratchspace Token (Takes token 6) -> Discarded
New Training Paradigm: Masked Frame Modeling (MFM)

The switch to a bidirectional encoder necessitates a new training objective.

Input Preparation: The model is presented with the entire sequence of tokens for a round [Context_Window, 7, D].

Random Masking: Before feeding the sequence to the model, a certain percentage of frames (e.g., 15%) are randomly selected for masking.

Applying the Mask: For each selected frame t:

The 7 input tokens [t, :, :] are replaced with a special, learned [MASK] token of shape [1, D].

Prediction Goal: The model processes this partially masked sequence. Its objective is to predict the original, unmasked content only for the frames that were masked.

Loss Calculation:

The output tokens from the transformer corresponding to the masked positions are fed into the prediction heads (Stage 3).

The loss (e.g., MSE for stats, Cross-Entropy for heatmaps) is calculated by comparing the predictions to the ground-truth data of the original, unmasked frames.

Crucially, no loss is calculated for the unmasked frames.

This approach forces the model to learn a deep, contextual understanding of game flow, using surrounding events to infer what must be happening at a specific, unknown moment in time.



data/
    manifest.json
    db
    lmdb
    recordings
    demos

write split_manifest.py
    splits up all lmdb and recordings e.g. data/lmdb/gamenameXXX.lmdb and data/recordings/gamenameXXX (contains .mp4s and .wav)
    into a train test split (default 80 20)
    receives as cli parameter the path to the data/ folder, the seed for the deterministic shuffle (default 42)
    writes to manifest.json
    metadata:
        seed, creation date
    train:
        [
            gamenameXXX (omit lmdb since base for both lmdb/... and recordings/...)
            gamenameXXY
        ]
    test:
        [
            same
        ]



Strat for train.py

Here’s our DALI plan in a nutshell:

* **Fix clip length per epoch:** choose a constant window in **frames** (e.g., `T_frames = 512` ≈ 16s at 32 fps). We still get variety by randomizing **start** each sample.

* **Sample windows randomly (once per epoch):**

  * For every team-round, pick a random `start_tick`, then compute
    `start_frame = (start_tick - round_start_tick) // ticks_per_frame`,
    `end_frame = start_frame + T_frames`.

* **Write five aligned file lists (per epoch):**

  * Create **5 `.txt` files** (one per POV).
  * **Line k** in each file = the same sample’s POV:
    `<abs/path/to/player_i.mp4>  0  <start_frame>  <end_frame>`
  * All files have the same number of lines; order defines the epoch order.

* **Build one DALI pipeline for the epoch:**

  * For each POV use `fn.readers.video_resize(..., file_list_frame_num=True, sequence_length=T_frames, step=T_frames, random_shuffle=False, pad_sequences=True, read_ahead=True, additional_decode_surfaces=8, resize_y=H, resize_x=W)`.
  * Follow with `fn.crop_mirror_normalize(..., output_layout="FCHW", mean=…, std=…, output_dtype=FLOAT16)` to get AMP-friendly `[B, T, C, H, W]` CUDA tensors.

* **Iterate batches without rebuilding:**

  * Each step: call `pipe.run()` (or `DALIGenericIterator`) to get 5 tensors, stack to `[B, T, 5, C, H, W]`.
  * Fetch **mel/alive/targets** from LMDB in the **same sample order**, move to GPU, and run the model.

* **Handle early deaths:** `pad_sequences=True` zero-pads tails if a POV ends early (or use experimental reader with `pad_mode="edge"` to repeat last frame).

* **Rebuild only at epoch boundaries:**

  * Delete temp file lists, re-sample new random windows, rewrite the 5 files, and rebuild the pipeline.

* **Extras:** For multi-GPU, pass `num_shards/shard_id` to each reader (or pre-split lists). If you need multiple lengths (15–30s), either **bucket by T** (one pipeline per bucket) or set T to max and **pad shorter** clips.

