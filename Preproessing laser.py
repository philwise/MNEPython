# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 22:04:01 2017

Bads and artefact correction
    bad channels from excel (or already preprocessed)
    in Range of triggers (2s before until 3s after)
        EOG correction
        EKG correction
        muscle artefacts?

@author: Philipp Wise
"""

#Import packages
import os.path as op
import numpy as np

import mne
print ("importing mne documentation")
from mne import find_events


#declare file path
data_path = 'C:\Users\Philipp Wise\mne_data\MEGAnalysis'
workdir = data_path + '\\170511m1'
raw_fname = workdir + '\prae_fu_2J_sa.fif'
raw = mne.io.read_raw_fif (raw_fname, preload=True) #import raw file
# raw.info['bads']
raw.filter(1.5, 40)
#raw.plot() #plot raw file in console, hash if not needed.


#declare triggers and find events
#STI: 006 is laser trigger, 001 is hand trigger, 003 is foot trigger
events_laser = find_events(raw, 'STI 006', min_duration=0.02)
print('Found %s events, first five:' % len(events_laser))
print(events_laser[:5])

#mne.viz.plot_events(events_laser)

#declare epochs
tmin, tmax = -0.2, 1.2
event_id = {'Pewpew':5}
baseline = (None, 0.0)
epochs = mne.Epochs(raw, events=events_laser, event_id=event_id, tmin=tmin,
                    tmax=tmax, preload=True)
#epochs.plot(block=True)

#Average epochs
picks = mne.pick_types(epochs.info, meg=True, eeg=True)
evoked_laser = epochs['Pewpew'].average(picks=picks)
evoked_laser.plot() #for plotting in console
