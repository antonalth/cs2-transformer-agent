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


this was recording/record2.py, now comes demo_extract/extract.py. Both of these should work in conjunction to first take in the demofile and extract the relevant data into an sql document, and then later the record2.py should record each demofile. 

Task: Write a harness python script that is used to orchestrate this data collection process. It receives the parameters --demodir path/to/demofiles --datadir path/to/dbfiles --recdir path/for/videofolders --workers X (default 2) --override_video 0,1,2 --no_data_gen

First, the script goes through the demofiles directory, and for each demofile checks if it finds a corresponding .db file in the datadir. If no_data_gen is set, only later queue demos where sqls ALREADY exist (dont spawn extract.py). For any missing db files after this step (if not in nodatagen) run extract.py with the relevant parameters to generate the missing db files in X worker threads to speed up things. Once all queued db files have been created and all threads are finished, now is time to record the video.

First, iterate through all db files and check if there are still entries in the RECORDING table that have not been recorded for each DB file. (these need to be queued). 