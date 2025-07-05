Your task is to write annotate_many.py

parameters of script --sql input.db --data recording_folder --round 1 --team ct --out video.mp4

Information about input.db
    CREATE TABLE player (
        tick INTEGER,
        steamid INTEGER,
        playername TEXT,
        position_x REAL,
        position_y REAL,
        position_z REAL,
        inventory TEXT,
        active_weapon TEXT,
        health INTEGER,
        armor INTEGER,
        money INTEGER,
        keyboard_input TEXT,
        mouse_x REAL,
        mouse_y REAL,
        is_in_buyzone INTEGER,
        buy_sell_input TEXT,
        PRIMARY KEY (tick, steamid)
    )
     CREATE TABLE rounds (
        round INTEGER PRIMARY KEY,
        starttick INTEGER,
        freezetime_endtick INTEGER,
        endtick INTEGER,
        t_team TEXT,
        ct_team TEXT
    )
    CREATE TABLE RECORDING (
        roundnumber        INTEGER,
        starttick          INTEGER,
        stoptick           INTEGER,
        team               TEXT,
        playername         TEXT,
        is_recorded        BOOLEAN,
        recording_filepath TEXT,
        PRIMARY KEY (starttick, stoptick, playername)
    );
Steps:
- Extract from table RECORDING * where roundnumber = (passed in args), team = (passed in args)
    - if no rows, print notification and exit
    - if rows, check for all entries is_recorded = true, otherwise exit
    - rebuild file paths according to format from the db data, check for existance in recording_folder, if not found, exit
            f"{round_num:02d}_{team}_{player_name}_{start_tick}_{stop_tick}" #video name format .mp4 in av1
- query all db information from table player between (and including) start and stop tick given, once for EACH player entry from the earlier RECORDING query. 
- For each row from the RECORDING table and the corresponding tick data from the player table
    Fully Stringify each tick (will later add to each video frame)
    Fill out any missing ticks (check that we have continous data for every number between starttick and stoptick) with the text "NO_TICK_INFO"
    Align video frames and this tick data from the END going back forwards.
    The video is at 32frames/second, while the demo ticks ran at 64ticks/s
    For each frame starting from the back, add both matched tick strings from the db (add into each frame using cv2 or similar)
        In practice it makes sense to align from the back, and then write from front to back (but make sure to align from the back first so it fits perfectly! (also take below considerations into account))
    Do this until we reach the start tick.
    Any frames before this start tick should be labeled "BEFORE_START" (frames left over with no corresponding tick data)
    If we dont manage to use all tick rows from the db (some are left over at the front), print a warning with the amount of ticks left over.

- once we have overlayed all 5 player videos (povs), we need to merge all 5 videos (3 up top, 2 at bottom in a 3x2 structur) in one large higher res video.
- align the videos based on the calculated ticks from the end. Note: some videos are shorter than others since the players die and the recording stops. 
    So the step at the front where we accurately align each video frame with two ticks from the end of the video is essential to align all frames and povs later.
    Write out compiled video to --out outfile.mp4

 ====== 
 
Information:
new_filename_base = f"{round_num:02d}_{team}_{player_name}_{start_tick}_{stop_tick}" #video name format .mp4

Information about 
    CREATE TABLE player (
        tick INTEGER,
        steamid INTEGER,
        playername TEXT,
        position_x REAL,
        position_y REAL,
        position_z REAL,
        inventory TEXT,
        active_weapon TEXT,
        health INTEGER,
        armor INTEGER,
        money INTEGER,
        keyboard_input TEXT,
        mouse_x REAL,
        mouse_y REAL,
        is_in_buyzone INTEGER,
        buy_sell_input TEXT,
        PRIMARY KEY (tick, steamid)
    )
Requirements:
parameters: --sql merged.db --input video_in_format_from_above.mp4 --out outfile.mp4
Steps:
- Extract from video filename: round number, player name, start tick, stop tick
- Open sqlite merged.db
- query all db information from table player between (and including) start and stop tick with playername given.
- Turn each tick into a well formatted string (we will add onto video)
- Fill out any missing ticks in the data from the db (check that we have continous data for every number between starttick and stoptick) with the text "NO_TICK_INFO"
- Align video frames and this tick data from the end
    The video is at 32frames/second, while the demo ticks ran at 64ticks/s
    For each frame starting from the back, add both matched tick strings from the db (add into each frame using cv2 or similar)
        In practice it makes sense to align from the back, and then write from front to back (but make sure to align from the back first so it fits perfectly! (also take below considerations into account))
    Do this until we reach the start tick.
    Any frames before this start tick should be labeled "BEFORE_START" (frames left over with no corresponding tick data)
    If we dont manage to use all tick rows from the db (some are left over at the front), print a warning with the amount of ticks left over.
    Write out modified video to --out outfile.mp4