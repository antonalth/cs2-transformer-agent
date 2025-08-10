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