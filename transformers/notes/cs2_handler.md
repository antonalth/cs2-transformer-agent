# CS2-Handler --spawn_many
script that manages cs2 processes
start of script: wipe all sandbox contents
load sandbox ini -> create steam sandbox

## always: check which ones are running (sandboxie)
ports starting at 30000, ...
check what pids they are using
check what ports are used

always

## starting cs2 (--spawn)
- give first free port unused
- generate unique index_mts in handler_files/portX.mts
- spawn unique node process in sandbox (modify script to accept portX)
- return port used (if none free -1)

## kill process: --kill PORT
- terminate all proc

## killall --killall
- terminate all sandboxes
## getpids --getpids PORT

# modify scripts to have different conn ports
# modify node server to run at different ip
# modify python lib to connect with port specifcation

# idea: cs2_handler spawns http server, listen on 127.0.0.1:8080,
30000,30001,30002,30003,30004,30005...
# stopping script kills all instances
# request types:
start_instance 
