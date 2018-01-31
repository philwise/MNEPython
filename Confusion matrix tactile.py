# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 22:04:01 2017

Preprocessing of raw dataset

import raw from file path
    read info
    display raw?

from rawdata get triggers (ST006)

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

#Import packages
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

import mne
print ("importing mne documentation")
from mne import find_events
from mne.preprocessing import Xdawn
from mne.decoding import Vectorizer
from mne.viz import tight_layout

#declare file path
data_path = 'C:\Users\Philipp Wise\mne_data\MEGAnalysis'
workdir = data_path + '\\170314m1'
raw_fname = workdir + '\post_tachafu_pw.fif'
raw = mne.io.read_raw_fif (raw_fname, preload=True) #import raw file
# raw.info['bads']
raw.filter(1.5, 40)
#raw.plot() #plot raw file in console, hash if not needed.


#declare triggers and find events
#STI: 006 is laser trigger, 001 is hand trigger, 003 is foot trigger
events_hand = mne.find_events(raw, 'STI 001', min_duration=0.02)
print('Found %s events, first five:' % len(events_hand))
print(events_hand[:5])

events_hand[:,2] += 1
print('Adding 1 to amplitutde value')

events_foot = mne.find_events(raw, 'STI 003', min_duration=0.02)
print('Found %s events, first five:' % len(events_foot))
print(events_foot[:5])

#summarize events into one list
allevents = np.concatenate((events_hand,events_foot))
print('displaying all events')
mne.viz.plot_events(allevents)

#declare epochs
tmin, tmax = -0.2, 0.5
event_id = {'Foot':5 , 'Hand':6}
baseline = (None, 0.0)
epochs = mne.Epochs(raw, events=allevents, event_id=event_id, tmin=tmin,
                    tmax=tmax, preload=True)
#epochs.plot(block=True)

#Average epochs
picks = mne.pick_types(epochs.info, meg=True, eeg=True)
evoked_hand = epochs['Hand'].average(picks=picks)
evoked_foot = epochs['Foot'].average(picks=picks)
evoked_hand.plot() #for plotting in console
evoked_foot.plot()

"""
# Create classification pipeline
clf = make_pipeline(Xdawn(n_components=3),
                    Vectorizer(),
                    MinMaxScaler(),
                    LogisticRegression(penalty='l1'))

# Get the labels
labels = epochs.events[:, -1]

# Cross validator
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
print('everything still good')
# Do cross-validation
preds = np.empty(len(labels))
for train, test in cv.split(epochs, labels):
    clf.fit(epochs[train], labels[train])
    preds[test] = clf.predict(epochs[test])

# Classification report
target_names = ['Hand', 'Foot']
report = classification_report(labels, preds, target_names=target_names)
print(report)

# Normalized confusion matrix
cm = confusion_matrix(labels, preds)
cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

# Plot confusion matrix
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Normalized Confusion matrix')
plt.colorbar()
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45)
plt.yticks(tick_marks, target_names)
tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
"""