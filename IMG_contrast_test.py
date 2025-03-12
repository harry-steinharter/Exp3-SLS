from psychopy.hardware import keyboard
from psychopy import core, visual, data, event#, gui #psychopy.gui doesn't work in lab
from psychopy.tools.filetools import fromFile, toFile
import numpy as np
import random
import itertools
import math
import os
import re
import pickle
import pandas as pd
import glob

os.chdir('/Users/harrysteinharter/Documents/MSc/Timo Internship/Pilot_V4_mandoh')
import otherFunctions as OF # Has to go after changing directory bc of it's location


#### Set up window ####
screenDim = np.array([2560, 1600])
myWin = visual.Window(monitor = 'myMacbook', fullscr = False, colorSpace = 'rgb', color = (0,0,0), bpc = (10,10,10), depthBits=10, units = 'deg')

#### Establish participant number ####
nSub = int(OF.ParticipantInput(myWin))
thisFile = glob.glob(os.path.join(os.getcwd(),'data',f"*{nSub}*"))
if nSub == 999:
    firstSession = True
else:
    firstSession = False

#### Stuff to determine condition and participant number ####
conds = {0: ['C','TL_BR'], 1: ['C','TR_BL'], 2: ['BC','TL_BR'], 3: ['BC','TR_BL']}
if firstSession:
    #### Set up condition pseudorandomizer ####
    nSub = OF.SubNumber('Participant.txt')
    trainShape, trainLocations = conds[nSub%4][0], conds[nSub%4][1]
    thisSession = 1

    #### Create data file ####
    fullFile = os.path.join(os.getcwd(),'data',f"Subject_{nSub}_data")
    dataFile = open(fullFile+'.csv', 'w')
    dataFile.write('participant,session,train_Shape,train_Locations,trial,this_Shape,this_Location,image,intensity,response,RT\n')
else:
    #### Load a previous file and get values ####
    fullFile = thisFile[0]
    df = pd.read_csv(fullFile)
    trainShape, trainLocations = df['train_Shape'].iloc[0], df['train_Locations'].iloc[0]
    thisSession = df['session'].max() + 1
    dataFile = open(fullFile, 'a')

#### Load images ####
condA, condB = trainShape+'_'+trainLocations[0:2], trainShape+'_'+trainLocations[3:5]
condNull = 'STN' # Mandoh's blank images start w/ STN
if thisSession == 3:
    stimuli = OF.loadImages(myWin,ALL=True)
else:
    stimuli = OF.loadImages(myWin,condA,condB)
labels = list(stimuli.keys())
images = list(stimuli.values())
# Session 3 has 4 times the number of images, so when training each image is duplicated 4 times
# The dict has to be split before shuffling
if thisSession != 3:
    labels *= 4
    images *= 4
    box = list(zip(labels,images))
    random.shuffle(box)
    labels, images = zip(*box)
else:
    box = list(zip(labels,images))
    random.shuffle(box)
    labels, images = zip(*box)
    stimuli = dict(zip(labels,images))

box = np.array(box)
nullStim = OF.loadNull(myWin,n = len(stimuli)//4)

#### Exp variables ####
nReal = len(images)
nNull = len(nullStim)
maxTrials = nReal + nNull
T_response = float('inf') # float('inf') or 0.5
T_intertrial = .5
T_stim = .25
nUp = 1
nDown = 1
nReversals = None # 1 or None?

#### Staircases ####
if thisSession == 3:
    C_TL_BR = {k: v for k, v in stimuli.items() if (k.startswith('C_TL') or k.startswith('C_BR'))}
    C_TR_BL = {k: v for k, v in stimuli.items() if (k.startswith('C_TR') or k.startswith('C_BL'))}
    BC_TL_BR = {k: v for k, v in stimuli.items() if (k.startswith('BC_TL') or k.startswith('BC_BR'))}
    BC_TR_BL = {k: v for k, v in stimuli.items() if (k.startswith('BC_TR') or k.startswith('BC_Bl'))}

if thisSession == 3:
    conditions = [
        {"label":"Null",          'startVal': 1, "nReversals":nReversals, "stepType":'log', "stepSizes":0.1, "nUp":nUp, "nDown":nDown, "nTrials":nNull, "minVal":0, "maxVal":1},
        {"label":"C_TL",          'startVal': 1, "nReversals":nReversals, "stepType":'log', "stepSizes":0.1, "nUp":nUp, "nDown":nDown, "nTrials":nReal, "minVal":0, "maxVal":1},
        {"label":"BC_TL",         'startVal': 1, "nReversals":nReversals, "stepType":'log', "stepSizes":0.1, "nUp":nUp, "nDown":nDown, "nTrials":nReal, "minVal":0, "maxVal":1},
        {"label":"C_TR",          'startVal': 1, "nReversals":nReversals, "stepType":'log', "stepSizes":0.1, "nUp":nUp, "nDown":nDown, "nTrials":nReal, "minVal":0, "maxVal":1},
        {"label":"BC_TR",         'startVal': 1, "nReversals":nReversals, "stepType":'log', "stepSizes":0.1, "nUp":nUp, "nDown":nDown, "nTrials":nReal, "minVal":0, "maxVal":1},
        {"label":"C_BL",          'startVal': 1, "nReversals":nReversals, "stepType":'log', "stepSizes":0.1, "nUp":nUp, "nDown":nDown, "nTrials":nReal, "minVal":0, "maxVal":1},
        {"label":"BC_BL",         'startVal': 1, "nReversals":nReversals, "stepType":'log', "stepSizes":0.1, "nUp":nUp, "nDown":nDown, "nTrials":nReal, "minVal":0, "maxVal":1},
        {"label":"C_BR",          'startVal': 1, "nReversals":nReversals, "stepType":'log', "stepSizes":0.1, "nUp":nUp, "nDown":nDown, "nTrials":nReal, "minVal":0, "maxVal":1},
        {"label":"BC_BR",         'startVal': 1, "nReversals":nReversals, "stepType":'log', "stepSizes":0.1, "nUp":nUp, "nDown":nDown, "nTrials":nReal, "minVal":0, "maxVal":1},
    ]
else:
    conditions = [
        {"label":"Null",          'startVal': 1, "nReversals":nReversals, "stepType":'log', "stepSizes":0.1, "nUp":nUp, "nDown":nDown, "nTrials":nNull, "minVal":0, "maxVal":1},
        {"label":condA,           'startVal': 1, "nReversals":nReversals, "stepType":'log', "stepSizes":0.1, "nUp":nUp, "nDown":nDown, "nTrials":nReal, "minVal":0, "maxVal":1},
        {"label":condB,           'startVal': 1, "nReversals":nReversals, "stepType":'log', "stepSizes":0.1, "nUp":nUp, "nDown":nDown, "nTrials":nReal, "minVal":0, "maxVal":1},
        ]

stairs = data.MultiStairHandler(stairType="simple",conditions=conditions, nTrials=maxTrials,method="random") 

#### Experiment Here ####
def experiment():
    
    trialClock = core.Clock()
    message = visual.TextStim(win = myWin, text = f'Press any key', color = 'black')
    fixation = visual.GratingStim(win=myWin, color=1, colorSpace='rgb',tex=None, mask='cross', size=0.2, pos = [0,0])
    blank = visual.TextStim(win = myWin, text = "You shouldn't see this", color = (0,0,0))
    OF.drawOrder(message,myWin)
    event.waitKeys()
    
    for trial in range(maxTrials):
        stairs.currentStaircase = stairs.runningStaircases[OF.nullRandomizer(thisSession)]
            # the staircases and conditions attributes have the same indexing
            # runningStaircases is the same as staircases but takes into accoount nReversals and nTrials
        thisCondition = stairs.currentStaircase.condition
        thisIntensity = stairs.currentStaircase.intensity
        
        thisShape = thisCondition['label'][:-3]
        thisLocation = thisCondition['label'][-2:]
        if thisCondition['label'] == 'Null':
            thisShape, thisLocation = 'NA','NA'
        thisLabel, thisImage = OF.imageChoice(thisCondition['label'], box)
        
        # Stimulus presentation
        OF.drawOrder(fixation,myWin)
        core.wait(T_intertrial)
        OF.drawOrder([thisImage,fixation],myWin)
        core.wait(T_stim)
        OF.drawOrder(blank,myWin)
        trialClock.reset()
        keys = event.waitKeys(keyList = ['q','escape','right','left'], maxWait = T_response)
        rt = trialClock.getTime()
        for key in keys:
            if key in ['q','escape']:
                event.clearEvents()
                core.quit()
            elif key == 'right':
                response = 1
            else:
                response = 0
        dataFile.write(f"{nSub},{thisSession},{trainShape},{trainLocations},{trial},{thisShape},{thisLocation},{thisLabel},{thisIntensity},{response},{rt}\n")
experiment()
event.clearEvents()
core.quit()
