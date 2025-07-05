parameters:
--sql input.db --demofile path/to/file.dem

first, do setup:
    restart cs2 with hlae:

        LOG.info("restarting cs2+hlae")
        subprocess.Popen(["python", "s1a_restart_all.py"], cwd="../../cnn_scripts", creationflags=subprocess.CREATE_NEW_CONSOLE)
        time.sleep(15)
    start console connection with cs2, and load demo to playback:

        import from libs.mirv_client import connect as mirv_connect
        conn = mirv_connect()

        full_path = str(demo_file.resolve())
        conn.sendCommand(f'playdemo "{full_path}"')
        #sleep 25s
        #send command (like above)
            demo_pause

        #create folder with name recordings/TEMP_RECORD if not exists, if exists delete contents

        #send commands:(wait 1s between each one)
            mirv_streams record startMovieWav 1
            mirv_streams record name "C:\..\currentdir\TEMP_RECORD"
            mirv_streams record screen enabled 1
            mirv_streams record screen settings afxDefault
            exec ffmpeg.cfg
            n1
            mirv_streams record fps 32
            cl_drawhud 0; demoui; demoui; cl_drawhud_force_radar -1; spec_mode 0
            spec_show_xray 0; sv_cheats 1; cl_hide_avatar_images 1;

            
for each entry with starttick, stoptick, roundnumber, playername, team in the input.db RECORDING table
    #send commands (wait 2s between each one)
        mirv_cmd clear
        demo_gototick starttick #wait 4s after this one
        spec_player playername
        mirv_cmd addAtTick stoptick "mirv_streams record end; demo_pause"
        mirv_streams record start; demo_resume

    #poll at 1s intervals until there is no process called "ffmpeg.exe" anymore. (+3s wait-time)
    
    #if not exists create folder named recordings/recording_DEMOFILENAME
    #rename the mp4 file in temp_record to roundnumber_team_starttick_stoptick_playername.mp4 (same with the coressponding .wav) and move to the folder created earlier
    #update the corresponding sql entry to set is_recorded to TRUE, recording_filepath to the corresponding mp4 path

#cleanup (or when CTRL+C is pressed)
    #send commands:
        demo_pause; mirv_streams record end;
    
    #wait 2s
    #remove folder TEMP_RECORD
    #close sqlite


    
Additional Info:
    log to stdout most steps you take, add a --debug to print even more.
database format:
 (format:
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

