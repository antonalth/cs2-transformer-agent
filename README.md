Plan for extracting training data:

Step 1: Extract player input + buy commands for each tick from demo
    -> what storage format for quick access? sqlite?
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