#!/bin/bash
# Script to extract stereo and mono .wav files from the .tgz switchboard file
# Usage: Download the swb1_LDC97S62.tgz file from the LDC website...
# ... Create a folder called "raw_data", put the .tgz file in it.
# ... Edit RAW_DATA_DIR accordingly, and run "sh get_wavs.sh".
# ... Requires about 80 GB in total. Takes about 15 mins to run.
source ./paths.sh
cd $RAW_DATA_DIR 
mkdir $STEREO_DIR 
mkdir $MONO_DIR 
tar -xvzf ./swb1_LDC97S62.tgz

for fold in swb1_d*; do
	for f in $fold/data/*.sph; do
		fname=$(basename -- $f)
        fname_s=sw${fname:3:4}
        fname_l=sw${fname:3}
        echo $fname
        echo $fname_s
		# stereoname="$STEREO_DIR/${fname%.*}.wav"
		# mononame="$MONO_DIR/${fname%.*}"
		stereoname="$STEREO_DIR/$fname_s.wav"
		mononame="$MONO_DIR/$fname_s"
		sox -t sph "$f" -b 16 -t wav $stereoname
		sox $stereoname $mononame.A.wav 'remix' 1
		sox $stereoname $mononame.B.wav 'remix' 2
	done;
done;
