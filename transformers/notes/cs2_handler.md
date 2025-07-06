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
