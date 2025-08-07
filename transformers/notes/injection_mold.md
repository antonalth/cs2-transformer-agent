What we have:
    Video and audio recordings of each round, for each player
    A database with the schema:  

        player (
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
        keyCREATEboard_input TEXT, 
        mouse_x REAL, mouse_y REAL,
        is_in_buyzone INTEGER, 
        buy_sell_input TEXT, 
        PRIMARY KEY (tick, steamid))

        rounds (round INTEGER PRIMARY KEY, 
                starttick INTEGER, 
                freezetime_endtick INTEGER, 
                endtick INTEGER,
                 win_tick INTEGER,
                  win_team TEXT, 
                  bomb_planted_tick INTEGER, 
                  t_team TEXT, 
                  ct_team TEXT)
        
        RECORDING (
            roundnumber        INTEGER,
            starttick          INTEGER,
            stoptick           INTEGER,
            team               TEXT,
            playername         TEXT,
            is_recorded        BOOLEAN,
            recording_filepath TEXT,
            PRIMARY KEY (starttick, stoptick, playername)
        )

        This database is filled by extract.py in reference to a demofile passed to it, and the videos are then later created by record2.py.
        These python files are attached should you need to reference the exact implementation details.

        The final database contains three tables: player, rounds, and RECORDING.

Information about the Database: 

Table 1: player
This table contains detailed, per-tick information about each player's state.
Column	Data Type	Description
tick	INTEGER	The specific in-game tick number for this row of data. A tick is a single simulation step on the server.
steamid	INTEGER	The player's unique 64-bit Steam ID.
playername	TEXT	The player's in-game name at that specific tick.
position_x, _y, _z	REAL	The player's precise 3D coordinates in the game world.
inventory	TEXT	A JSON-formatted list of every item the player is carrying at that tick. The names are the official, capitalized names from the game.
active_weapon	TEXT	The official, capitalized name of the weapon the player is currently holding.
health	INTEGER	The player's health, typically ranging from 1 to 100.
armor	INTEGER	The player's armor value.
money	INTEGER	The amount of in-game money the player possesses at that tick.
keyboard_input	TEXT	A comma-separated string of all actions detected for the player at that tick. This is a combination of direct key presses and inferred actions. See the "keyboard_input Field Enumeration" section below for a full list.
mouse_x, _y	REAL	The calculated, sensitivity-independent horizontal (x) and vertical (y) mouse movement for that tick. This represents raw mouse input without recoil.
is_in_buyzone	INTEGER	A boolean flag indicating if the player is in a buy zone. 1 means they are in the buy zone, 0 means they are not.
buy_sell_input	TEXT	A comma-separated string of any buy or sell actions that were inferred at that tick. See the "buy_sell_input Field Enumeration" section below for a full list.
Table 2: rounds
This table contains summary information for each round of the match.
Column	Data Type	Description
round	INTEGER	The round number (e.g., 1, 2, 15). This is the primary key.
starttick	INTEGER	The tick number when the round officially began.
freezetime_endtick	INTEGER	The tick number when the pre-round freeze time ended and players could move freely.
endtick	INTEGER	The tick number when the round concluded.
win_tick	INTEGER	The specific tick on which the round's winner was determined.
win_team	TEXT	The winning team, represented as either "T" or "CT".
bomb_planted_tick	INTEGER	The tick on which the bomb was planted. If the bomb was not planted in the round, this value is -1.
t_team	TEXT	A JSON-formatted list of lists. Each inner list contains a Terrorist player's name and their death tick ([playername, deathtick]). If the player survived, the death tick is -1.
ct_team	TEXT	A JSON-formatted list of lists, structured identically to t_team but for the Counter-Terrorist players.
Table 3: RECORDING
This table is generated in the final step to identify segments of the demo that are suitable for creating video recordings, based on strict validation criteria (e.g., 5v5, no early deaths).
Column	Data Type	Description
roundnumber	INTEGER	The round number this recording candidate belongs to.
starttick	INTEGER	The recommended starting tick for the recording (usually the round start).
stoptick	INTEGER	The recommended ending tick for the recording (either the player's death or the end of the round).
team	TEXT	The team of the player to be recorded ("T" or "CT").
playername	TEXT	The name of the player this recording candidate is for.
is_recorded	BOOLEAN	A flag, initially False, that can be used by other tools to track whether this segment has been recorded.
recording_filepath	TEXT	A placeholder, initially None, for the file path of the created video.
Enumeration of Field Values
Official Weapon & Item Names (inventory and active_weapon fields)
These are the capitalized names that come directly from the game data.
Rifles: AK-47, M4A4, M4A1-S, Galil AR, FAMAS, AUG, SG 553, AWP, SSG 08, G3SG1, SCAR-20
Pistols: Glock-18, USP-S, P250, P2000, Dual Berettas, Five-SeveN, Tec-9, CZ75-Auto, R8 Revolver, Desert Eagle (Note: the script maps this as "deagle" for actions, but the official name may appear)
SMGs: MP9, MAC-10, MP7, MP5-SD, UMP-45, P90, PP-Bizon
Heavy: Nova, XM1014, MAG-7, Sawed-Off, M249, Negev
Knives: knife, knife_ct, knife_t, Bayonet, Flip Knife, Gut Knife, Karambit, M9 Bayonet, Huntsman Knife, Falchion Knife, Bowie Knife, Butterfly Knife, Shadow Daggers, Ursus Knife, Navaja Knife, Stiletto Knife, Talon Knife, Classic Knife, Paracord Knife, Survival Knife, Nomad Knife, Skeleton Knife
Grenades: High Explosive Grenade, Flashbang, Smoke Grenade, Molotov, Incendiary Grenade, Decoy Grenade
Gear & Other: C4 Explosive, Defuse Kit, Zeus x27, Kevlar Vest, Helmet (Note: script infers actions as vest and vesthelm)
keyboard_input Field Enumeration
This field is a comma-separated string containing any of the following:
Direct Inputs (from KEY_MAPPING): IN_ATTACK, IN_JUMP, IN_DUCK, IN_FORWARD, IN_BACK, IN_USE, IN_CANCEL, IN_TURNLEFT, IN_TURNRIGHT, IN_MOVELEFT, IN_MOVERIGHT, IN_ATTACK2, IN_RELOAD, IN_ALT1, IN_ALT2, IN_SPEED, IN_WALK, IN_ZOOM, IN_WEAPON1, IN_WEAPON2, IN_BULLRUSH, IN_GRENADE1, IN_GRENADE2, IN_ATTACK3, IN_SCORE, IN_INSPECT
Inferred Weapon Switches (from WEAPON_CATEGORIES): These are added when the player changes to a weapon in a new category.
SWITCH_1: Primary Weapons (Rifles, SMGs, Heavy)
SWITCH_2: Secondary Weapons (Pistols)
SWITCH_3: Melee/Special (Knife, Zeus)
SWITCH_4: Grenades
SWITCH_5: C4 / Defuse Kit
Inferred Drop Actions: Formatted as DROP_itemname. The itemname is one of the lowercase names from the ITEM_ID_MAP, such as:
DROP_ak47, DROP_awp, DROP_deagle, DROP_flashbang, DROP_c4, etc.
buy_sell_input Field Enumeration
This field is a comma-separated string containing buy/sell actions inferred at a specific tick, formatted as ACTION_itemname.
Actions: BUY or SELL
Item Names (from ITEM_ID_MAP): deagle, elite, fiveseven, glock, ak47, aug, awp, famas, g3sg1, galilar, m249, m4a1, mac10, p90, mp5sd, ump45, xm1014, bizon, mag7, negev, sawedoff, tec9, p2000, mp7, mp9, nova, p250, scar20, sg556, ssg08, knife, flashbang, hegrenade, smokegrenade, molotov, decoy, incgrenade, c4, knife_t, m4a1_silencer, usp_silencer, cz75a, revolver, defuser, vest, vesthelm.

Here is a sample from an .db
sqlite> select * from player limit 10;
tick|steamid|playername|position_x|position_y|position_z|inventory|active_weapon|health|armor|money|keyboard_input|mouse_x|mouse_y|is_in_buyzone|buy_sell_input
361|76561197981355847|BTN|1296.0|32.0|-167.868774414063|["Stiletto Knife", "Glock-18"]|Glock-18|100|0|800||0.0|0.0|1|
362|76561197981355847|BTN|1296.0|32.0|-167.868774414063|["Stiletto Knife", "Glock-18"]|Glock-18|100|0|800||||0|
363|76561197981355847|BTN|1296.0|32.0|-167.96875|["Stiletto Knife", "Glock-18"]|Glock-18|100|0|800||0.0|0.0|1|
364|76561197981355847|BTN|1296.0|32.0|-167.96875|["Stiletto Knife", "Glock-18"]|Glock-18|100|0|800||0.0|0.0|1|
365|76561197981355847|BTN|1296.0|32.0|-167.96875|["Stiletto Knife", "Glock-18"]|Glock-18|100|0|800||0.0|0.0|1|
366|76561197981355847|BTN|1296.0|32.0|-167.96875|["Stiletto Knife", "Glock-18"]|Glock-18|100|0|800||0.0|0.0|1|
367|76561197981355847|BTN|1296.0|32.0|-167.96875|["Stiletto Knife", "Glock-18"]|Glock-18|100|0|800||0.0|0.0|1|
368|76561197981355847|BTN|1296.0|32.0|-167.96875|["Stiletto Knife", "Glock-18"]|Glock-18|100|0|800||0.0|0.0|1|
369|76561197981355847|BTN|1296.0|32.0|-167.96875|["Stiletto Knife", "Glock-18"]|Glock-18|100|0|800||0.0|0.0|1|
370|76561197981355847|BTN|1296.0|32.0|-167.96875|["Stiletto Knife", "Glock-18"]|Glock-18|100|0|800||0.0|0.0|1|
sqlite> select * from rounds limit 10;
round|starttick|freezetime_endtick|endtick|win_tick|win_team|bomb_planted_tick|t_team|ct_team
1|361|1641|9449|9001|ct|-1|[["BTN", 8481], ["s0und", 3464], ["lauNX", -1], ["xellow", 4881], ["ragga", 3373]]|[["TRAVIS", -1], ["BELCHONOKK", 8268], ["Qikert", 8899], ["Jame", 3029], ["nota", -1]]
2|9449|10729|13999|13551|ct|-1|[["s0und", 12905], ["xellow", 13210], ["lauNX", 12614], ["BTN", 13551], ["ragga", 13073]]|[["BELCHONOKK", -1], ["TRAVIS", -1], ["Qikert", -1], ["nota", -1], ["Jame", -1]]
3|13999|15279|17753|17305|ct|-1|[["xellow", 16970], ["BTN", 17240], ["ragga", 17305], ["s0und", 16713], ["lauNX", 16898]]|[["TRAVIS", -1], ["Qikert", -1], ["BELCHONOKK", -1], ["Jame", -1], ["nota", 16894]]
4|17753|19033|26486|26038|ct|-1|[["ragga", 26038], ["s0und", 22292], ["xellow", 24535], ["BTN", 21982], ["lauNX", 23259]]|[["BELCHONOKK", 23015], ["nota", -1], ["Qikert", 24885], ["Jame", -1], ["TRAVIS", 20733]]
5|26486|27766|35827|35379|t|32755|[["BTN", -1], ["s0und", -1], ["ragga", -1], ["xellow", -1], ["lauNX", -1]]|[["Jame", 29411], ["TRAVIS", -1], ["BELCHONOKK", -1], ["Qikert", -1], ["nota", 28868]]
6|35827|37107|45631|45183|ct|43135|[["lauNX", 42325], ["ragga", 44158], ["xellow", 42639], ["BTN", 41186], ["s0und", 44768]]|[["TRAVIS", 42663], ["Qikert", 42483], ["nota", 42829], ["BELCHONOKK", -1], ["Jame", -1]]
7|45631|46911|53814|53366|t|50742|[["s0und", -1], ["ragga", 49855], ["xellow", -1], ["BTN", 49671], ["lauNX", -1]]|[["TRAVIS", 49357], ["Jame", 49823], ["BELCHONOKK", -1], ["Qikert", 49891], ["nota", -1]]
8|53814|55094|63724|63276|ct|61678|[["lauNX", 59072], ["BTN", 60038], ["ragga", 62377], ["xellow", 61925], ["s0und", 59155]]|[["Qikert", 60575], ["Jame", -1], ["BELCHONOKK", 58984], ["nota", -1], ["TRAVIS", -1]]
9|63724|67257|72435|71987|ct|-1|[["ragga", 71092], ["xellow", 71987], ["BTN", 70933], ["lauNX", 68857], ["s0und", 71110]]|[["TRAVIS", -1], ["nota", -1], ["Qikert", 71586], ["Jame", -1], ["BELCHONOKK", -1]]
10|72435|73715|82469|82021|t|80252|[["lauNX", -1], ["s0und", 75723], ["BTN", 79221], ["xellow", 81622], ["ragga", -1]]|[["Qikert", 80411], ["BELCHONOKK", 79224], ["Jame", 82021], ["TRAVIS", 81900], ["nota", 79204]]
sqlite> select * from RECORDING limit 10;
roundnumber|starttick|stoptick|team|playername|is_recorded|recording_filepath
1|361|8481|T|BTN|0|
1|361|3464|T|s0und|0|
1|361|9449|T|lauNX|0|
1|361|4881|T|xellow|0|
1|361|3373|T|ragga|0|
1|361|9449|CT|TRAVIS|0|
1|361|8268|CT|BELCHONOKK|0|
1|361|8899|CT|Qikert|0|
1|361|3029|CT|Jame|0|
1|361|9449|CT|nota|0|

Goal: 
    We want to compact all of this data into lmdb files for Transformer training. To do this we create "injection_mold.py".

    We receive --recdir path/to/recordings/of/demofile, where the mp4 and wav files exist
        new_filename_base = f"{round_num:02d}_{team}_{player_name}_{start_tick}_{stop_tick}"
    
    We receive -dbfile path/to/sqldb

    We receive --outlmdb path/to/outlmdbfile


This script helps prepare the video and player input for model training into the lmdb format. 
Then, check the sqldb in the RECORDING table if all recordings are recoded e.g boolean is set (also add --overridesql flag to ignore problems here)
Compare and check if for each row in the recording table we have the right .mp4 and .wav file in the recordings folder, if missing exit w error
Check if outlmdb exists, and if --overwrite is set delete lmdb first or exit w error if flag is not set.

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
                    demoname_round_XXX_team_[ct/t]_tick_XXXXXXX -> so essentially we store tick data for each team on a per tick basis.

Additionally, after all ticks are processed and added, with key demoname_INFO:
    json{
        demoname string,
        rounds: [
                [round#, starttick, endtick]
        ]
    }

at first, initialize the lmdb at the arg path with default size 20GB, expand by 5GB if close to limit (<200 MB left)
Important: CTRL+C behavior -> if game is not completely finished and all entries are done, delete lmdb file so we dont have inconsistent state

To understand the recording to database alignment step, annotate_many.py is attached. This is a testing script that generates an annotated video with the tick data to understand the alignment, just to get some information.

Your task now is to outline a detailed game plan e.g. a checklist of all things in order, planning out the script before we actually write it.