#!/bin/bash
tmux new-session -d -s jupyter 'jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'

