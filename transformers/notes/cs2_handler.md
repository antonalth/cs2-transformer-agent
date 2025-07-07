# on startup:
//ini file with max 10 clients
//kill all in the sandboxes
//spawn websocket server, listen for commands
//parameter --count X #create sandbox ini

listen on port 30000 for incoming network traffic

//ini file with max 10 clients
//parameter --num 1-10 (default 3)
//spawn all cs2 clients, run some relevant (sbx names gameX aswell)
    note which ones were spawned
//wait for all connect
//wait for application commands
    -list: return all cs2game ids available
    -send: send command to game with id specified
    -done: check if ffmpeg finished in id (but filter by sandbox)
//on ctrl+c -> stop all sandbox processes that were spawned


Task: attached is an example server script that is used to bridge between a game client and other applications. a single game client joins and among other things, connecting applications can send console commands to the cs2 game client that is connected. We want to expand on this. Modify the code so that multiple game clients can join at the original port, which are then enumerated somehow (all game clients join through the exact same websocker url), it does not matter in which order they are consistently enumerated, just that later the applications can uniquely address one exactly with the assigned identifier. The script should also open up an http server on port 8080 that has 2 endpoints: /list -> returns list of available identifiers to use later,
/run?id=...&cmd=URLENCODED_COMMAND_TO_RUN, which then sends the decoded command in the json format to the unique game client (enumerated above). Remove any features in this code that are not directly needed for these two endpoints. Original Example:

client_manager.py

--num 3 (1-10;default 3) --delay (default 60)
Sandbox path: 'C:\Program Files\Sandboxie-Plus\Start.exe'

Task: write a python script that runs the game counter strike 2 num times using sandboxie to start different clients. 

To begin, kill all processes inside of all sandboxes:

'...Start.exe' /terminate_all

The first step is to start steam inside of each sandbox using the command: (X represents the number from 1 to 10, once for each num starting at 1)

'...Start.exe' /hide_window /box:gameX steam://open

After starting all of the steam clients, wait delay seconds

Start the node server to listen and spawn the http server (run broker/broker.py)

then, for each X of the num game, run

'..Start.exe' /box:gameX ../start_cs2.bat #must be relative to this broker.py script fileloc
wait 15seconds -> needed to get the id order right e.g. id=1 -> sandbox game1

CTRL+C terminates node serve + ALL sandbox items:

'Start.exe' /terminate_all
