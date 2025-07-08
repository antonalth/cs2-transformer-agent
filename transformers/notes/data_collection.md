# first collect all demofiles in single directory called 
    data/demos/demo_name.dem
# extract all data using extract.py
    data/info/demo_name.db
# record each item in the RECORDING table into
    data/recording/demo_name/video.mp4
    data/recording/demo_name/audio.wav
# package into lmdbs
    data/packaged/info.db
    data/packaged/200_GB.lmdb

# write efficient python to unpack

...
# train model


Orchestrator.py: 
this was recording/record2.py, now comes demo_extract/extract.py. Both of these should work in conjunction to first take in the demofile and extract the relevant data into an sql document, and then later the record2.py should record each demofile. 

Task: Write a harness python script that is used to orchestrate this data collection process. It receives the parameters --demodir path/to/demofiles --datadir path/to/dbfiles --recdir path/for/videofolders --workers X (default 2) --override 0,1,2 --no_data_gen

First, the script goes through the demofiles directory, and for each demofile checks if it finds a corresponding .db file in the datadir. If no_data_gen is set, only later queue demos where sqls ALREADY exist (dont spawn extract.py). For any missing db files after this step (if not in nodatagen) run extract.py with the relevant parameters to generate the missing db files in X worker threads to speed up things. Once all queued db files have been created and all threads are finished, now is time to record the video.

Then, iterate through all db files and corresponding demofiles and run the record2.py script accordingly, passing along the override setting (default 0) and the required folder paths etc. If parts of the demos have already been recorded, record2.py script will manage. Here we can also utilize workers: the /list endpoint on the localhost server returns a json list with the ids offered, for example with just one: ["1"]. These are the --id parameters for the record files, and simultaneously dictate how many workers we can spawn here. If there is only one entry, only one worker etc.. only one instance of record2 can be running for each id! (note: if /list does not have any running or fails to respond, exit w error)

Add enough information output to show what is happening in each phase, and preface any outputs from the spawned workers with information about what worker is outputting this info.

CTRL+C should gracefully pass on the sigint to each worker and then exit clean.