# first collect all demofiles in single directory called 
    data/demos/demo_name.dem
# extract all data using extract.py
    data/info/demo_name.db
# record each item in the RECORDING table into
    data/recording/demo_name/video.mp4
    data/recording/demo_name/audio.wav
# package into lmdbs
    data/packaged/info.db
    data/packaged/200_GB.lmdb

# write efficient python to unpack

...
# train model


Orchestrator.py: 
this was recording/record2.py, now comes demo_extract/extract.py. Both of these should work in conjunction to first take in the demofile and extract the relevant data into an sql document, and then later the record2.py should record each demofile. 

Task: Write a harness python script that is used to orchestrate this data collection process. It receives the parameters --demodir path/to/demofiles --datadir path/to/dbfiles --recdir path/for/videofolders --workers X (default 2) --override 0,1,2 --no_data_gen

First, the script goes through the demofiles directory, and for each demofile checks if it finds a corresponding .db file in the datadir. If no_data_gen is set, only later queue demos where sqls ALREADY exist (dont spawn extract.py). For any missing db files after this step (if not in nodatagen) run extract.py with the relevant parameters to generate the missing db files in X worker threads to speed up things. Once all queued db files have been created and all threads are finished, now is time to record the video.

Then, iterate through all db files and corresponding demofiles and run the record2.py script accordingly, passing along the override setting (default 0) and the required folder paths etc. If parts of the demos have already been recorded, record2.py script will manage. Here we can also utilize workers: the /list endpoint on the localhost server returns a json list with the ids offered, for example with just one: ["1"]. These are the --id parameters for the record files, and simultaneously dictate how many workers we can spawn here. If there is only one entry, only one worker etc.. only one instance of record2 can be running for each id! (note: if /list does not have any running or fails to respond, exit w error)

Add enough information output to show what is happening in each phase, and preface any outputs from the spawned workers with information about what worker is outputting this info.

CTRL+C should gracefully pass on the sigint to each worker and then exit clean.

injection_mold.py
Information for each player at each tick:
    tick, isdead, inbuyzone, , Keyboard Input, Buy Input, MouseX, MouseY, Position, Armor, Health, Money
    General: tick, isdead, inbuyzone
    Movement, Stance keys
    BuySellDrop
    MouseX, MouseY,

    format: tick_int, mouse_x_int, mouse_y_int, position_ints, armor_int, health_int, money_int
    bitmask for: isdead, inbuyzone, movement, stance, buyselldrop
Gamestate:
    state: freezetime, round, bomb_planted, wonround, lostround 
    enemy positions

Task:
script receives: --recdir path/to/recordings/of/demofile --dbfile path/to/sqldb (--overwrite) --outlmdb path/to/outlmdb

This script helps prepare the video and player input for model training into the lmdb format. 
Then, check the sqldb in the RECORDING table if all recordings are recoded e.g boolean is set (also add --overridesql flag to ignore problems here)
Compare and check if for each row in the recording table we have the right .mp4 and .wav file in the recordings folder, if missing exit w error
Check if outlmdb exists, and if overwrite is set (delete lmdb first) or exit w error. 

We are looking at this from a cs2 game perspective. Each game has rounds, and each round has one team perspective (t_team or ct_team), with each perspective having 5 player_perspectives.
For each round:
    for each team (t or ct):
        Find all video files (and the ticks they correspond to from the recording table)
        Find all corresponding player inputs etc from the player table in the db.
            This means all of the things listed above e.g . inbuyzone, keyboardinput, buysellinput, mousex, mousey, position, armor, health, money for each tick
        ALIGN for each video of a player perspective the individual video frames and the player input based on the player names and the tick data (tick data in both tables)
            Often times there are two ticks per frame (since tickrate 64, recordingrate 32) -> merge function that combines two player input ticks into one -> PLACEHOLDER just take the first tick, disregard second
            If alignment causes problems, since we have too many or too little video frames for example (not an exact 2:1 match), throw warning with info about how much is missing or too much.
        For each aligned tick (with 5 aligned player perspectives) should now have something like this:
            for each frame and corresponding two ticks, we have:
                game_state = [round_state: bool freezetime, round, bomb_planted, wonround, lostround; team_alive 5bools; enemy_alive 5 bools; enemy_positions = x,y,z floats *5
                    => one big numpy dtype: tick_int, round_state*5(bool),team_alive*5(bool),enemy_alive*5(bool),enemy_pos_x_y_z*5(floats)
                game_state (numpy dtype)
                    tick: np.int32, 1 
                    round_state: np.uint8, 1 #5 bools: freezetime, inround, bomb_planted, wonround, lostround
                    team_alive: np.uint8, 1 #5 bools, one for each teamplayer
                    enemy_alive: np.uint8, 1 #5 bools, one for each enemy
                    enemy_pos: np.float32, 5x3 (dead players just have all 0, and we know since enemy_alive is 0 for position)
                player_input (one for each player pov) (numpy dtype): 
                    pos: np.float32, 3
                    mouse: np.float32, 2
                    armor: np.uint8, 1
                    health: np.uint8, 1
                    money: np.int32, 1
                    keyboard: np.uint8, ENOUGH_TO_STORE_BITMASK
                    inbuyzone+buysell: same as keyboard
                    inventory: same (multiple can be true)
                    active_weapon: same (one-hot)
                tick_data = 
                    msgpack
                    [
                        game_state, 
                        (player_input1, jpeg1, audio1)
                        , ... # if player dead, no entry here but we can figure out which is which based on team_alive mask, right?
                        (player_input5, jpeg5, audio5)
                    ]
                
                each of these tick_data is stored in the lmdb db with the key format:
                    demoname_round_XXX_team_[ct/t]_tick_XXXXXXX

at first, initialize the lmdb at the arg path with default size 20GB, expand by 5GB if close to limit (<200 MB left)
Important: CTRL+C behavior -> if game is not completely finished and all entries are done, delete lmdb file so we dont have inconsistent state
Additionally, after all ticks are processed and added, with key demoname_INFO:
    json{
        demoname string,
        rounds: [
                [round#, starttick, endtick]
        ]
    }
  ###########
  
Of course. By synthesizing your detailed notes with the context and schemas from extract.py, annotate_many.py, and record2.py, I have crafted a comprehensive task description for the new script, injection_mold.py.

Task Description: injection_mold.py - ML Data Preprocessing and Packaging
1. High-Level Objective

injection_mold.py is the final and most critical script in the data preparation pipeline. It acts as an "injection mold" for our machine learning data, taking all the structured and unstructured data generated by previous steps (extract.py, record2.py) and packaging it into a single, highly-optimized, and training-ready LMDB (Lightning Memory-Mapped Database).

The script will process an entire demo file's worth of data, iterating through each valid round and team perspective (T and CT). For every synchronized moment in the game (represented by a video frame), it will package the global game state, along with the individual inputs and visual/auditory data for each of the five alive players on a team, into a single, atomic database entry.

2. Core Functionality & Workflow

Initialization and Validation:

Parse command-line arguments: --recdir, --dbfile, --outlmdb, --overwrite, and --overridesql.

Verify that the input database (--dbfile) and recordings directory (--recdir) exist.

Check if the output LMDB path (--outlmdb) already exists. If it does, exit with an error unless --overwrite is specified, in which case the existing LMDB will be safely deleted before starting.

Initialize a new LMDB database at the output path with a large initial map size (e.g., 20 GB).

Data Manifest Verification:

Connect to the input SQLite database (--dbfile).

Query the RECORDING table to get a complete manifest of all expected player recordings for the demo.

For each entry in the RECORDING table:

Check if the is_recorded flag is True. If not, and the --overridesql flag is not set, exit with an error explaining which recording is missing its flag.

Using the naming scheme from record2.py ({round:02d}_{team}_{playername}_{starttick}_{stoptick}), construct the expected base filename.

Verify that both the .mp4 and .wav files for this entry exist in the --recdir. If any file is missing, exit with a descriptive error.

Data Processing Loop:

Iterate through each round number and then each team (T and CT) present in the verified manifest.

For each (round, team) combination:

Fetch Data: Load all relevant player data for the 5 players on the current team from the player table for the round's duration. Concurrently, load the opponent team's data to derive their positions and alive status. Load the round-level data (e.g., bomb plant status) from the rounds table.

Align Data: Open the 5 video (.mp4) and 5 audio (.wav) streams for the team. Align the 32 FPS video frames with the 64 Ticks Per Second game data. Since there are two game ticks for every video frame, the data from both ticks must be merged into a single player_input structure representative of that frame.

Merge Rule (Initial): For each video frame, use the player input data from the first of the two corresponding game ticks. Log a warning if the alignment is not a perfect 2:1 ratio (e.g., due to dropped frames).

Create LMDB Entries: For each aligned video frame:

Construct a game_state numpy object.

Construct a player_input numpy object for each of the 5 players.

Package the frame's data (JPEG-encoded video frame, audio segment, game_state, and all living players' player_input) using msgpack.

Write the packaged data to the LMDB with a uniquely formatted key.

Finalization:

After processing all rounds, add a final _INFO key to the LMDB. The value will be a JSON object summarizing the database contents (e.g., demo name, list of processed rounds with their start/end ticks).

Safely close the LMDB environment.

3. Command-Line Arguments
Generated bash
python injection_mold.py \
    --recdir /path/to/pov_recordings/ \
    --dbfile /path/to/extracted_data.db \
    --outlmdb /path/to/output.lmdb \
    [--overwrite] \
    [--overridesql]


--recdir (Required): Path to the directory containing the .mp4 video and .wav audio recordings.

--dbfile (Required): Path to the SQLite database file generated by extract.py.

--outlmdb (Required): Path where the output LMDB database will be created.

--overwrite (Optional): If specified, the script will delete and recreate the LMDB file if it already exists.

--overridesql (Optional): If specified, the script will ignore is_recorded=False flags in the RECORDING table and proceed, assuming the files exist.

4. Data Structures for LMDB

Key Format: demoname_round_XX_team_Y_tick_ZZZZZZZ

demoname: Base name of the original demo file (e.g., match_123).

XX: The round number, zero-padded (e.g., 01, 12).

Y: The team perspective (T or CT).

ZZZZZZZ: The game tick corresponding to the start of the frame.

Value Format (Msgpacked List):

Generated python
[
    game_state,  # A single structured NumPy array
    (player_input_1, jpeg_bytes_1, audio_bytes_1),
    (player_input_2, jpeg_bytes_2, audio_bytes_2),
    # ... up to 5 players.
    # The list of player tuples only contains entries for players who are
    # alive at that tick, determined by the `team_alive_mask`.
]
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END

A. game_state Numpy Dtype:
A single structured numpy array representing the shared state of the round.

Generated python
# Sourced from `rounds` and `player` tables in the DB.
game_state_dtype = np.dtype([
    ('tick', np.int32),
    # Bitmask: 0=in_freezetime, 1=in_round, 2=bomb_planted, 3=round_won, 4=round_lost
    ('round_state_flags', np.uint8),
    # Bitmask: 5 bits, one for each teammate. 1=alive, 0=dead.
    ('team_alive_mask', np.uint8),
    # Bitmask: 5 bits, one for each enemy. 1=alive, 0=dead.
    ('enemy_alive_mask', np.uint8),
    # Positions of the 5 enemies. (0,0,0) for dead enemies.
    ('enemy_positions', np.float32, (5, 3))
])
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END

B. player_input Numpy Dtype:
A structured numpy array for each player, representing their personal state and inputs.

Generated python
# Sourced from a row in the `player` table in the DB.
player_input_dtype = np.dtype([
    ('position', np.float32, 3),          # From position_x, position_y, position_z
    ('mouse_delta', np.float32, 2),       # From mouse_x, mouse_y
    ('health', np.uint8),
    ('armor', np.uint8),
    ('money', np.int32),
    # Bitmask for all keyboard inputs (movement, actions, buy/sell/drop)
    # Merges `keyboard_input` and `buy_sell_input` columns.
    ('input_flags', np.uint64),
    # Integer ID representing the active weapon category (e.g., 0=None, 1=Rifle, 2=Pistol, etc.)
    ('active_weapon_id', np.uint8),
    # Bitmask for key inventory items (e.g., defuse kit, bomb, grenades)
    ('inventory_flags', np.uint32),
    # Boolean flag: 1 if in buyzone, 0 otherwise.
    ('is_in_buyzone', bool)
])
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END
5. Robustness and Error Handling

LMDB Resizing: The script must monitor the LMDB's remaining capacity. If free space drops below a threshold (e.g., 200 MB), the map size must be increased automatically (e.g., by 5 GB) to prevent write failures.

Interrupt Handling (CTRL+C): The script must gracefully handle KeyboardInterrupt. If the full process is interrupted and does not complete successfully, the partially written LMDB file at --outlmdb must be deleted to prevent data corruption and ensure atomic completion. This can be achieved using a try...finally block around the main processing loop.