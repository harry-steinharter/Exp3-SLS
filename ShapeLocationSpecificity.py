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
small = True # Logical indicating if we are running the experiment with a lot of images

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
    dataFile.write('participant,session,train_Shape,train_Locations,block,trial,this_Shape,this_Location,image,intensity,response,RT\n')
else:
    #### Load a previous file and get values ####
    fullFile = thisFile[0]
    df = pd.read_csv(fullFile)
    trainShape, trainLocations = df['train_Shape'].iloc[0], df['train_Locations'].iloc[0]
    thisSession = df['session'].max() + 1
    dataFile = open(fullFile, 'a')

thisSession = 3 ######## Remember to delete!! 
#### Load images ####
condA, condB = trainShape+'_'+trainLocations[0:2], trainShape+'_'+trainLocations[3:5]
if thisSession == 3:
    stimuli = OF.loadImages(myWin,ALL=True,small = small)
else:
    stimuli = OF.loadImages(myWin,condA,condB,small = small)
labels = list(stimuli.keys())
images = list(stimuli.values())
stim_array = np.array(list(zip(labels,images)))

if small:
    nullStim = OF.loadNull(myWin,n = len(stim_array),small = small) # len(x)//4 when I have enough images
else:
    nullStim = OF.loadNull(myWin,n = len(stim_array)//4,small = small) # len(x)//4 when I have enough images
nullLab = list(nullStim.keys())
nullIma = list(nullStim.values())
null_stim_array = np.array(list(zip(nullLab,nullIma)))

#### Exp variables ####
if small:
    nReal = 8
    nNull = 2
else:
    nReal = 100
    nNull = nReal // 4
nBlocks = 4                                                                            # Number of blocks
maxTrials = (nReal + nNull) * nBlocks                                                  # Number of trials per staircase before it can close
breakTrials = np.linspace(0,maxTrials,nBlocks+1, dtype='int')[1:-1]                     # Trial numbers where a break happens
T_response = float('inf')                                                               # float('inf') or 0.5 Time for them to respond (s)
T_intertrial = .5                                                                       # Intertrial time (s)
T_stim = .25                                                                            # Stimulus presentation time (s)
T_break = 5                                                                             # Inter-block required waiting time (s)
T_block_max = 600                                                                       # Max time a block is allowed to go on for (s)
nUp = 1                                                                                 # Required consecutive correct responses in a staircase for difficulty to increase
nDown = 1                                                                               # Required consecutive incorrect responses in a staircase for difficulty to decrease
nReversals = None                                                                       # 1 or None? Number of reversals before a staircase can end (always >0)
keylist = ['q','escape','right','left']
maxIntensity = 1                                                                        # Max and starting intensity of the staircases
iOdds = 0.1                                                                             # The odds of the intensity being randomly high/low to prevent unwanted learning

#### Staircases ####
conditionsThree = [
    {"label":"Null",          'startVal': maxIntensity, "nReversals":nReversals, "stepType":'log', "stepSizes":0.1, "nUp":nUp, "nDown":nDown, "nTrials":nNull*nBlocks, "minVal":0, "maxVal":maxIntensity},
    {"label":"C_TL",          'startVal': maxIntensity, "nReversals":nReversals, "stepType":'log', "stepSizes":0.1, "nUp":nUp, "nDown":nDown, "nTrials":nReal*nBlocks, "minVal":0, "maxVal":maxIntensity},
    {"label":"BC_TL",         'startVal': maxIntensity, "nReversals":nReversals, "stepType":'log', "stepSizes":0.1, "nUp":nUp, "nDown":nDown, "nTrials":nReal*nBlocks, "minVal":0, "maxVal":maxIntensity},
    {"label":"C_TR",          'startVal': maxIntensity, "nReversals":nReversals, "stepType":'log', "stepSizes":0.1, "nUp":nUp, "nDown":nDown, "nTrials":nReal*nBlocks, "minVal":0, "maxVal":maxIntensity},
    {"label":"BC_TR",         'startVal': maxIntensity, "nReversals":nReversals, "stepType":'log', "stepSizes":0.1, "nUp":nUp, "nDown":nDown, "nTrials":nReal*nBlocks, "minVal":0, "maxVal":maxIntensity},
    {"label":"C_BL",          'startVal': maxIntensity, "nReversals":nReversals, "stepType":'log', "stepSizes":0.1, "nUp":nUp, "nDown":nDown, "nTrials":nReal*nBlocks, "minVal":0, "maxVal":maxIntensity},
    {"label":"BC_BL",         'startVal': maxIntensity, "nReversals":nReversals, "stepType":'log', "stepSizes":0.1, "nUp":nUp, "nDown":nDown, "nTrials":nReal*nBlocks, "minVal":0, "maxVal":maxIntensity},
    {"label":"C_BR",          'startVal': maxIntensity, "nReversals":nReversals, "stepType":'log', "stepSizes":0.1, "nUp":nUp, "nDown":nDown, "nTrials":nReal*nBlocks, "minVal":0, "maxVal":maxIntensity},
    {"label":"BC_BR",         'startVal': maxIntensity, "nReversals":nReversals, "stepType":'log', "stepSizes":0.1, "nUp":nUp, "nDown":nDown, "nTrials":nReal*nBlocks, "minVal":0, "maxVal":maxIntensity},
]
conditionsOneTwo = [
    {"label":"Null",          'startVal': maxIntensity, "nReversals":nReversals, "stepType":'log', "stepSizes":0.1, "nUp":nUp, "nDown":nDown, "nTrials":nNull*nBlocks, "minVal":0, "maxVal":maxIntensity},
    {"label":condA,           'startVal': maxIntensity, "nReversals":nReversals, "stepType":'log', "stepSizes":0.1, "nUp":nUp, "nDown":nDown, "nTrials":nReal*nBlocks, "minVal":0, "maxVal":maxIntensity},
    {"label":condB,           'startVal': maxIntensity, "nReversals":nReversals, "stepType":'log', "stepSizes":0.1, "nUp":nUp, "nDown":nDown, "nTrials":nReal*nBlocks, "minVal":0, "maxVal":maxIntensity},
    ]
triplets = np.array((('Null','C_TL','C_BR'),('Null','C_TR','C_BL'),('Null','BC_TL','BC_BR'),('Null','BC_TR','BC_BL')))
random.shuffle(triplets)

#### Experiment Here ####
def training():
    nTrials = 20
    nEach = nTrials//2
    trainStim = np.vstack((null_stim_array[0:nEach],stim_array[0:nEach]))
    T_wait = 1
    resp_m = visual.TextStim(myWin, text = 'Respond now. LEFT for NO. RIGHT for YES.', color = 'black')
    fixation = visual.GratingStim(win=myWin, color=1, colorSpace='rgb',tex=None, mask='cross', size=0.2, pos = [0,0])
    
    for trial in range(nTrials):
        l,i = random.choice(trainStim)
#        shape_debug_m = visual.TextStim(myWin, text = f'{l}', color = 'green')
        if l.startswith('Null'):
            null = True
        else:
            null = False
        OF.drawOrder(fixation,myWin)
        core.wait(.5)
        fixation.color = 'white'
        OF.drawOrder((i,fixation),myWin)
        core.wait(T_wait)
        OF.drawOrder(resp_m,myWin)
        keys = event.waitKeys(keyList = keylist)
        for key in keys:
            if key == 'right':
                if null:
                    fixation.color = 'red'
                else:
                    fixation.color = 'green'
            if key == 'left':
                if null:
                    fixation.color = 'green',
                else:
                    fixation.color = 'red'
            elif key in ['q','escape']:
                core.quit()

def sessionOneTwo():
    
    stairs = data.MultiStairHandler(stairType="simple",conditions=conditionsOneTwo, nTrials=maxTrials,method="random") 
    
    trialClock = core.Clock()
    blockClock = core.Clock()
    thisBlock = 0
    
    intro_m = visual.TextStim(win = myWin, text = f'You are looking for the {trainShape} SHAPE in the {trainLocations} locations. Press any key to continue.', color = 'black')
    break_m = visual.TextStim(win = myWin, text = f'Take a short break for at least {T_break} seconds. You can stretch, go on your phone, get some water, just relax :)', color = 'black')
    continue_m = visual.TextStim(win = myWin, text = f'It has been {T_break} seconds. You can keep resting or you can continue if you are ready. The next block will be exactly the same as the previous one. Press any key to continue.', color = 'black')
    blank = visual.TextStim(win = myWin, text = "You shouldn't see this", color = (0,0,0))
    fixation = visual.GratingStim(win=myWin, color=1, colorSpace='rgb',tex=None, mask='cross', size=0.2, pos = [0,0])
    
    OF.drawOrder(intro_m,myWin)
    event.waitKeys()
    OF.countdown(myWin)
    
    for trial in range(maxTrials):
        if (trial in breakTrials) or (blockClock.getTime() >= T_block_max):
            thisBlock += 1
            OF.drawOrder(break_m,myWin)
            core.wait(T_break)
            OF.drawOrder(continue_m,myWin)
            event.waitKeys()
            OF.countdown(myWin)
            
        stairs.currentStaircase = stairs.staircases[OF.nullRandomizer(thisSession)]
            # the `staircases` and `conditions` attributes have the same indexing
            # `runningStaircases` is the same as staircases but takes into accoount nReversals and nTrials
        thisLabel = stairs.currentStaircase.condition['label']
        thisIntensity = stairs.currentStaircase.intensity
        stairs.currentStaircase.intensities.append(thisIntensity)
        if random.random() >=  iOdds:
            if thisLabel == 'Null':
                thisIntensity = 0.05
            else:
                thisIntensity = maxIntensity
        # Done to prevent learning the wrong thing, but don't want to actually affect the staircase
        
        if thisLabel == 'Null':
            thisShape, thisLocation = 'NA','NA'
            thisImageID, thisImage = OF.imageChoice(thisLabel, null_stim_array)
        else:
            thisShape, thisLocation = thisLabel[:-3], thisLabel[-2:]
            thisImageID, thisImage = OF.imageChoice(thisLabel, stim_array)
        thisImage.contrast = thisIntensity
        # Stimulus presentation
        OF.drawOrder(fixation,myWin)
        core.wait(T_intertrial)
        fixation.color = 'white'
        OF.drawOrder([thisImage,fixation],myWin)
        core.wait(T_stim)
        OF.drawOrder(blank,myWin)
        trialClock.reset()
        keys = event.waitKeys(keyList = keylist, maxWait = T_response)
        
        # Record response
        if keys:
            for key in keys:
                if key in ['q','escape']:
                    event.clearEvents()
                    core.quit()
                elif key == 'right':
                    if thisLabel == 'Null':
                        response = 0
                        fixation.color = 'red'
                    else:
                        response = 1
                        fixation.color = 'green'
                elif key == 'left':
                    if thisLabel == 'Null':
                        response = 1
                        fixation.color = 'green'
                    else:
                        response = 0
                        fixation.color  = 'red'
                else:
                    print(f"Error: Only {keylist} keys should be possible")
                    event.clearEvents()
                    core.quit()
            rt = trialClock.getTime()
        else:
            response = 0
            rt = T_response
        stairs.currentStaircase.addResponse(response)
        dataFile.write(f"{nSub},{thisSession},{trainShape},{trainLocations},{thisBlock+1},{trial+1},{thisShape},{thisLocation},{thisImageID},{thisIntensity},{response},{rt}\n")

def sessionThree():
    
    trialClock = core.Clock()
    blockClock = core.Clock()
    thisBlock = 0
    
    intro_m = visual.TextStim(win = myWin, text = f"""You are looking for the {triplets[thisBlock,1][0:-3]} SHAPE in the {triplets[thisBlock,1][-2:]+'_'+triplets[thisBlock,2][-2:]} LOCATION.
    Press any key to continue.""", color = 'black')
    break_m = visual.TextStim(win = myWin, text = f'Take a short break for at least {T_break} seconds. You can stretch, go on your phone, get some water, just relax :)', color = 'black')
    continue_m = visual.TextStim(win = myWin, text = f"""It has been {T_break} seconds. You can keep resting or you can continue if you are ready.
    The next block will show the {triplets[thisBlock,1][0:-3]} SHAPE in the {triplets[thisBlock,1][-2:]+'_'+triplets[thisBlock,2][-2:]} LOCATION.
    Press any key to continue.""", color = 'black')
    blank = visual.TextStim(win = myWin, text = "You shouldn't see this", color = (0,0,0))
    fixation = visual.GratingStim(win=myWin, color=1, colorSpace='rgb',tex=None, mask='cross', size=0.2, pos = [0,0])
    
    OF.drawOrder(intro_m,myWin)
    event.waitKeys()
    OF.countdown(myWin)
    
    for trial in range(maxTrials):
        if trial == 0:
            # Set up the first staircase
            thisGroup = triplets[thisBlock]
            blockConditions = [c for c in conditionsThree if (c['label'] in thisGroup)]
            stairs = data.MultiStairHandler(stairType="simple",conditions=blockConditions, nTrials=maxTrials,method="random")
        if (trial in breakTrials) or (blockClock.getTime() >= T_block_max):
            # Change the staircases
            thisGroup = triplets[thisBlock]
            blockConditions = [c for c in conditionsThree if (c['label'] in thisGroup)]
            stairs = data.MultiStairHandler(stairType="simple",conditions=blockConditions, nTrials=maxTrials,method="random")
            # Display messages
            thisBlock += 1
            OF.drawOrder(break_m,myWin)
            core.wait(T_break)
            OF.drawOrder(continue_m,myWin)
            event.waitKeys()
            OF.countdown(myWin)
        #### The actual experimental loop ####
        stairs.currentStaircase = stairs.staircases[OF.nullRandomizer(thisSession)]
            # the `staircases` and `conditions` attributes have the same indexing
            # `runningStaircases` is the same as staircases but takes into accoount nReversals and nTrials
            
        thisLabel = stairs.currentStaircase.condition['label']
        thisIntensity = stairs.currentStaircase.intensity
        stairs.currentStaircase.intensities.append(thisIntensity)
#        if random.random() >= iOdds:
#            if thisLabel == 'Null':
#                thisIntensity = 0.05
#            else:
#                thisIntensity = maxIntensity
        # Done to prevent learning the wrong thing, but don't want to actually affect the staircase
        
        if thisLabel == 'Null':
            thisShape, thisLocation = 'NA','NA'
            thisImageID, thisImage = OF.imageChoice(thisLabel, null_stim_array)
        else:
            thisShape, thisLocation = thisLabel[:-3], thisLabel[-2:]
            thisImageID, thisImage = OF.imageChoice(thisLabel, stim_array)
        thisImage.contrast = thisIntensity
        # Stimulus presentation
        OF.drawOrder(fixation,myWin)
        core.wait(T_intertrial)
        fixation.color = 'white'
        OF.drawOrder([thisImage,fixation],myWin)
        core.wait(T_stim)
        OF.drawOrder(blank,myWin)
        trialClock.reset()
        keys = event.waitKeys(keyList = keylist, maxWait = T_response)
        
        # Record response
        if keys:
            for key in keys:
                if key in ['q','escape']:
                    event.clearEvents()
                    core.quit()
                elif key == 'right':
                    if thisLabel == 'Null':
                        response = 0
                        fixation.color = 'red'
                    else:
                        response = 1
                        fixation.color = 'green'
                elif key == 'left':
                    if thisLabel == 'Null':
                        response = 1
                        fixation.color = 'green'
                    else:
                        response = 0
                        fixation.color  = 'red'
                else:
                    print(f"Error: Only {keylist} keys should be possible")
                    event.clearEvents()
                    core.quit()
            rt = trialClock.getTime()
        else:
            response = 0
            rt = T_response
        stairs.currentStaircase.addResponse(response)
        dataFile.write(f"{nSub},{thisSession},{trainShape},{trainLocations},{thisBlock+1},{trial+1},{thisShape},{thisLocation},{thisImageID},{thisIntensity},{response},{rt}\n")


hello = visual.TextStim(myWin, color='black',text = """Welcome! This first part is training. 
An image will appear on screen. 
Respond with RIGHT if the hidden shape is present.
Respond with LEFT if the hidden shape is not present.
Press any key to begin.""")
intro_m = visual.TextStim(win = myWin, text = f'You are looking for the {trainShape} SHAPE in the {trainLocations} locations. Press any key to continue.', color = 'black')
nextPart = visual.TextStim(myWin, color = 'black',text = """Nice job! The next portion will be the real experiment. There will be a few differences. 
The images will appear much faster. 
The text telling you to respond will not be present.
The response window will be much shorter.""")
goodbye = visual.TextStim(myWin, color='black',text = """It's over! I hope it wasn't too bad :)""")


if thisSession == 3:
    sessionThree()
else:
    experiment()

OF.drawOrder(goodbye,myWin)
core.wait(5)

event.clearEvents()
core.quit()
