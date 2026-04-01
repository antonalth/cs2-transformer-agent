The task is to write the code that runs the model in inference mode and forwards the control decisions to the simulation harness running in another vm on this server.
The code for said simulation harness can be found in inference/ - including helper files for interacting with said server.

The inference part of this application should run the model in autoregressive mode, such that each new 5-frame pov is passed in, while previous inputs are kept via a kv-cache.
The control outputs of the model should be passed back into to the simulation server via the provided functions. 
To control the inference process, we also want another webserver on this machine that shows the current frames we have received, and overlayed the control decisions (similar to visualize_inference.py) - at a controllable refresh rate to limit bandwidth (check out the ui for the sim harness).

The webserver allows:

0) A connect/disconnect button to connect to the sim harness (with ip addr entry), should be specifiable via cli arg.
1) A view tab similar to the one used in the sim harness that shows all 5 povs with overlayed the model outputs (see visualize_inference for good idea), selectable fps)
A) Start/Pause button to start/pause inference (keeping kv-cache on pause, allowing resume) 
B) information on size of kv-cache, metrics on inference speed etc, settings on the size of the window of kv cache kept (default no drop, otherwise rolling window)
C) A reset button that clears all kv-cache context and pauses the inference process if currently running.
D) A start/stop record feature that records the inference process similar to visualize_inference.py in realtime - good default naming and save under videos/
E) A separate tab that allows us to select one of the five players and view (perhaps with graphs) all of the raw logits of each output head(eg before threshold etc) - available when the inference process is paused to inspect

Files should be created in the inference/ folder when necessary, under a suitable subfolder.
