The goal is to create a harness that can be used to actually run inference on the model found in the transformers directory. (model3/)

We have 5 users and a server (steam1, steam2... steam5, server) user. You can run commands as each of these users using the runas command (installed, script here aswell)

eg: $sudo runas steam1 gamescope --backend headless -- steam -tenfoot #start steam in big picture mode
or: $sudo runas steam1 gamescope --backend headless -- steam steam://rungameid/730 #starts cs2 as the user

notably we can receive frames/input keyboard+mouse into the gamescope. 
an example script is found in the inference/ dir here: 
$ python3 vidserver.py   --node-id 42 --eis-socket /run/user/$(id -u)/gamescope-0-ei   --video-port 5500   --input-port 5501  --fps 5
#client:
python3 vidclient.py 10.7.30.53 --video-port 5500 --input-port 5501

the node id can be found using wpctl status -n, hopefully with a better method later

The goal: We want to be able to run a script as the unknown user or root on this machine that 
A) spawns a controlling webserver at port 8080 which we can use to interface and control the simulation environment
B) spawns 5 instances of cs2 using the 5 steam users. (open in tmux sessions)
C) capture all 5 gamescope windows at 30fps, and open the respective input channels. 
D) able to restart each game instance individually from the webserver 
E) (later feature) start cs2 server aswell (tmux)
F) open ports for sending video to the inference server, and for receiving control inputs. (also write corresponding client that can be used with our model files)
G) A recording mode where the output frames and received keyboard+mouse inputs are displayed (triggered via the webserver)
    Recording should be similar to the output of the visualize_inference.py script eg 6 composite tiles with overlayed predictions)
H) The webserver should have an inspection pane where it displays this composite of the 6 tiles with received inputs for the frame (if any) - refresh rate for pane should be configurable since the controlling browser has limited bandwith to the webserver eg. also be able to set rate at 1fps, perhaps also use somewhat efficient coding)
J) A similar feature where from the webserver you can control one of the 5 clients eg send mouse+kb (frame display rate configurable to limit bandwith - not that this should not impact the actual 30hz capture rate)
K) the actual commands used to run start the clients should be configurable via a config file
L) the performance of this script should be somewhat optimized eg use gpu/avx if possible (priority is getting it working though)

