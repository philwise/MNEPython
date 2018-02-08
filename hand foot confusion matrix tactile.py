# -*- coding: utf-8 -*-
"""
Split off on Thur Feb 01 12:30 2018

Steps:
    1. Import packages, import data
    2. Add frequency filters
    3. Find events from STI triggers, declare epochs
    4. Create classification pipeline
    5. Start matrix calculation, print matrix
    
To-do:
    improve confusion matrix results
    why is MEG data not good for model?
    Filter freq vital to learning algorithm

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
workdir = data_path + '\\170720m1'
raw_fname = workdir + '\prae_tac_ha_fu_kr.fif'
raw = mne.io.read_raw_fif (raw_fname, preload=True) #import raw file
# raw.info['bads']

raw.filter(10,65) #filter for physiological freqs
raw.notch_filter(50) #filter DC offset
#raw.plot() #plot raw file in console, hash if not needed.

#declare triggers and find events
#STI: 006 is laser trigger, 001 is hand trigger, 003 (or 002?) is foot trigger
events_hand = mne.find_events(raw, 'STI 001', min_duration=0.02)
print('Found %s events, first five:' % len(events_hand))
print(events_hand[:5])

events_hand[:,2] += 1
print('Adding 1 to amplitutde value')

events_foot = mne.find_events(raw, 'STI 002', min_duration=0.02)
print('Found %s events, first five:' % len(events_foot))
print(events_foot[:5])

#summarize events into one list
allevents = np.concatenate((events_hand,events_foot))

#declare epochs
tmin, tmax = -0.2, 0.4
event_id = {'Foot':5 , 'Hand':6}
baseline = (None, 0.0)
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

epochs = mne.Epochs(raw, events=allevents, event_id=event_id, tmin=tmin,
                    tmax=tmax, proj=False, picks=picks, baseline=None, preload=True,
                verbose=False)

# Create classification pipeline
print ('beginning matrix calculation')
clf = make_pipeline(Xdawn(n_components=3),
                    Vectorizer(),
                    MinMaxScaler(),
                    LogisticRegression(penalty='l1'))

# Get the labels
labels = epochs.events[:, -1]

# Cross validator
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
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
print('plotting matrix')
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
