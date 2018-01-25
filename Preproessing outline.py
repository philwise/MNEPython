# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 22:04:01 2017

Preprocessing of raw dataset

import raw from file path
    read info
    display raw?

from rawdata get triggers (ST002?)

Bads and artefact correction
    bad channels from excel (or already preprocessed)
    in Range of triggers (2s before 3s after)
        EOG correction
        EKG correction
        muscle artefacts?

cut data into raw segments
    raw cut 1.5s before and 2.5s after trigger
    (what speed were hand/foot tactile?)
write into data block (check what format is used here)

    THIS MAY BE POST-PROCESSING
possibly write second/third data block with filter (2-35Hz 20-80Hz)
    depending on effort
compile FFT file block

on averaged dataset: already have average data from BESA, remember to pull off laptop

@author: Philipp Wise
"""

import mne
print ("importing mne documentation")
from mne import find_events

data_path = 'C:\Users\Philipp Wise\mne_data\MEGAnalysis'
workdir = data_path + '\\170314m1'
raw_fname = workdir + '\post_h_2J_pwraw.fif'
raw = mne.io.read_raw_fif (raw_fname)
# raw.info['bads']

raw.plot() #plot raw file in console, hash if not needed.

events = mne.find_events(raw, 'STI 006', min_duration=0.02)
print('Found %s events, first five:' % len(events))
print(events[:5])
#lasevents = mne.find_events(raw, 'STI 006') #STI 006 is laser trigger

