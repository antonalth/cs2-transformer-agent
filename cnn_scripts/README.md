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