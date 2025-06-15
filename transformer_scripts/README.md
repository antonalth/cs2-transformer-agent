Plan for extracting training data:

folder structure:
    data/
        transformer/
            staging/
                demo00001_name/
                    info.json #tick info about every round (start, stop); streamid:name mapping
                    video_pov_segments.db #see below
                    user_input.db #per tick: tick, steamid (together primary key), W, A, S, D, buy..., Attack, Zoom, mouse_dX, mouse_dY (correct? if Attack + Rifle?)
                    rounds/
                        1/
                            roundinfo.json #fromtick - totick
                            pov_STEAMID.mp4
                            audio_STEAMID.wav
                            pov_STEAMID/ (debug, not actually on disk?)
                                frame_0000X.jpeg #quality 80?
                                frame_X.spectrogram #64 mel bins
                                frame_X.userinput
                        ...
            compact/
                demo00002_name/
                    info.json
                    round_X_t.lmdb
                        -internally store as frame:pov -> sizes,jpeg,spectrogram,inputdata,other
                        -there are 5 povs per round. some stop early
                    round_X_ct.lmdb
TODOs:
- Parse demo, extract all player inputs/tick; round start - round stop; player recording timeframes (round#, starttick, endtick, isdeath?)
    - separate tables: playerinput, info for player pov recording (round#, starttick ,endtick, isdeath, (video_path, audio_path), (team)), separate table auxiliary information (money, roundcount)
- Record based on sql table, put at path,
- All 5 team: Video -> frames, Audio snippets -> spectrogram, input to 64 bit?, aux? ==> LMDB format per round


Required Algorithms:
- Segment round into player spectate segments (round start - death/round end): round#, team, steamid, playername, starttick, endtick, isdeath
    -> output video__pov_segments.db //leave column is_recorded = false
- Record Rounds from video_segments.db , set is_recorded=true
- Extract player information to user_input.db #for entire game
    - WASD, Space, Jump, CTRL
    - dX, dY, lClick, rClick
    - usePrimary, useSecondary, useKnife, useFlash, useHE, useMolo, useSmoke
    - actions: (defuse, plant, open door)
    - Reloading switch primary/secondary/knife,buy/drop/buydrop/switch weapons, actions (plant, defusing, open door)
    = rewrite self: https://github.com/LaihoE/demoparser/tree/main/documentation/python
    -  player attr: player_name, player_steamid, team_num, is_alive, active_weapon, weapon_purchases_this_round?, is_defusing, buttons?, pitch, yaw, balance?
    - usercmd: usercmd_mouse_dx, usercmd_mouse_dy? + m_sensitivity usercmd_weapon_select? usercmd_input_histr
        -> get ticks with no recoil, and then correlate usercmd_mouse_dx,dy with dAng -> get Sensitivity (average over many ticks!)
        -> then

TF notes:
 - feed in frame, audio, etc + last keyboard input.

Step 1: Extract player input + buy commands for each tick from demo
    -> what storage format for quick access? sqlite?
    -> separate script to extract per demo/per player per tick
Step 2: For each round (each team):
    -> figure out tick for round_start, round_end, player_death?
    -> Record each player pov until death
    -> demo_goto..., mirv record, mirv_cmd on tick death/endround to stop recording
    -> store tickStart, tickEnd etc in json with recording, bool round_end or deathtime? 
        -> when training and death, stop head inference since player dead
        -> classifier based on death audio inference time
        -> for each frame of video, interpolate player input from n frames and save in json
    -> name demoname_round_X_pov_Y_tickStart_tickEnd.mp4

    Group for round:
        -> merge jsons, now we have {
            demofile:""
            round:""
            team:""
            players{
                {
                    pov_file:"file.mp4"
                    audio_file:"audio.wav"
                    playername:""
                    deathtime:"" or -1
                    inputcommands: [(W,A,S,D,dX,dY, buyX, buyY)]
                }
                ,{}
            }
        }
        -> additionally get all player positions with t/ct label and normed to 0-1 (add 2048, divide by 4096)
        -> when training use Hungarian loss
    Convert Round Video+Audio+Input+Ext -> LMDB round
        Round has n frames


For every Round:
    For every Player:
        For each 1/20s

task: find item events/tick where a player action = buys/sells/drops an item. Use the demoparser zip to figure out how to extract information required. Add a --sqlout flag and write found info to sqlite db
sqlite table RAREACTIONS should contain: tick, steamid, playername, action, item

things to consider: 
- Player death DOES NOT count as a drop
- Picking up a weapon with 'e' (resulting in a drop) does not count as a drop, but dropping an item with 'g' and autopickup DOES count as drop
- Buying a weapon, which results in the drop of the old weapon DOES NOT count as drop (we drop the old weapon but never press the 'g' key)
    -> create BUY, item entry
- Grenade throw DOES NOT count (but dropping a grenade to a teammate for example is possible and does count)
    - we need to check for recent related grenade throw events to differentiate, here we should use AWPY, not demoparser since they provide a high level way of checking grenade throws. Use search to check for options. 
- Selling a weapon again DOES NOT count (since the player never actually drops e.g 'g')
    - check if after weapon/item dissapears money has gone up again -> create a SELL, item entry
- Buying a weapon and instant dropping DOES count e.g player presses control while buying it
    - create BUY entry and DROP entry (at relevant ticks)
- If the player is in the buy_zone and can still buy/sell (beginning round freeze time and after round begins until the zone closes), add an entry to a second table BUYZONE an entry with tick, steamid, playername, 

Brainstorm first, dont write any code yet but collect all relevant details needed for later implementation.

//also need buy, sell , is_buy_time?/freeze_time?
//conditional output head

//check rounds.py for freezetime

(exercise-cv) antonalthoff@Antons-MacBook-Pro-4 transformer_scripts % python extractstruct.py data_marius-vs-ex-sabre-m2-mirage
Database: mouse.db
  Table 'MOUSE': tick (INTEGER), player_name (TEXT), x (REAL), y (REAL)
    First row: 2, crickeyyy, 0.0, 0.0

Database: keyboard_location.db
  Table 'inputs': tick (INTEGER), steamid (INTEGER), playername (TEXT), keyboard_input (TEXT), inventory (TEXT), x (REAL), y (REAL), z (REAL), active_weapon (TEXT), health (INTEGER), armor (INTEGER), money (INTEGER)
    First row: 1, 76561198019527985, homeboyz, IN_FORWARD,SWITCH_2, ["knife_t", "Glock-18", "C4 Explosive", "Smoke Grenade", "Flashbang"], 1135.60205078125, -57.1380615234375, -164.5367431640625, Glock-18, 100, 0, 100

Database: rounds.db
  Table 'ROUNDS': round (INTEGER), starttick (INTEGER), freezetime_endtick (INTEGER), endtick (INTEGER), t_team (TEXT), ct_team (TEXT)
    First row: 1, 1, None, 2789, [], []

Database: buy_sell_drop.db
  Table 'RAREACTIONS': tick (INTEGER), steamid (TEXT), playername (TEXT), action (TEXT), item (TEXT)
    First row: 3163, 76561199041006873, hodix, BUY, Kevlar & Helmet
  Table 'BUYZONE': tick (INTEGER), steamid (TEXT), playername (TEXT)
    First row: 1, 76561198019527985, homeboyz

\\\\ NEW Unified TABLE:

TABLE player 
    tick int
    steamid int
    playername text
    position = x,y,z (from keyboard_location)
    inventory
    active_weapon
    health 
    armor 
    money 
    keyboard_input (from keyboard_location) + if at tick DROP action from RAREACTIONS, add DROP_itemname to list
    mouse_input -> x,y from mouse.db
    is_in_buyzone boolean (if entry in BUYZONE)
    buy_sell_input (from buyselldrop where action = BUY or SELL) text , if at tick BUY or SELL action from RAREACTIONS, add BUY/SELL_itemname to list
TABLE rounds
    /keep as before.
    
Note: only keep player_entries if the player is alive, e.g. if health != for player at tick. //write a python script that reads in all of these db files and merges them into the larger merged.db file according to spec. note that these files might be in a dir and we need to specifiy the dirpath to the .db files. also put merged.db in the same dir. Also if for a certain tick there is no entry in keyboard_location inputs table, no entry in the new player table (ignore content from other dbs if they have content for that player at that tick). Finally, also drop any entries in the player table where the tick is not between any starttick and endtick in the ROUNDS table.


RECORD.py

//cli type deal open conn with mirv,
//command record starttick endtick playername path 

# mirv_streams record startMovieWav 1
# mirv_streams record name "C:\PATH"
# mirv_streams record screen enabled 1
# mirv_streams record screen settings afxFfmpegYuv420P
# mirv_streams record fps 30
...setup viewmodel settings...

# demo_gototick 1234
# demo_spectate playername

# mirv_streams record start
# demo_resume
...
# mirv_cmd addAtTick 5678 "mirv_streams record end"
# mirv_cmd addAtTick 5679 "demo_pause"