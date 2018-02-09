# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 22:04:01 2017

To-Do
Bads and artefact correction
    bad channels from excel (or already preprocessed)
    in Range of triggers (2s before 3s after)
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
workdir = data_path + '\\170405m2'
raw_fname = workdir + '\praecaps_tac_ha_fu_sm.fif'
raw = mne.io.read_raw_fif (raw_fname, preload=True) #import raw file
#raw.info['bads']
raw.filter(1.6, 90)
raw.notch_filter(50)
#raw.plot() #plot raw file in console, hash if not needed.


#declare triggers and find events
#STI: 006 is laser trigger, 001 is hand trigger, 003 is foot trigger
events_hand = find_events(raw, 'STI 001', min_duration=0.02)
print('Found %s events, first five:' % len(events_hand))
print(events_hand[:5])

events_hand[:,2] += 1
print('Adding 1 to amplitutde value')

events_foot = find_events(raw, 'STI 002', min_duration=0.02)
print('Found %s events, first five:' % len(events_foot))
print(events_foot[:5])

#summarize events into one list
allevents = np.concatenate((events_hand,events_foot))
print('displaying all events')
#mne.viz.plot_events(allevents)

#declare epochs
tmin, tmax = -0.2, 0.8
event_id = {'Foot':5 , 'Hand':6}
baseline = (None, 0.0)

epochs = mne.Epochs(raw, events=allevents, event_id=event_id, tmin=tmin,
                    tmax=tmax, preload=True)
#epochs.plot(block=True)

#Average epochs
picks = mne.pick_types(epochs.info, meg=True, eeg=True)
epochs.drop_bad()
evoked_hand = epochs['Hand'].average(picks=picks)
evoked_foot = epochs['Foot'].average(picks=picks)
evoked_hand.plot() #for plotting in console
evoked_foot.plot()
