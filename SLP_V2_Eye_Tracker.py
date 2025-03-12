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
import pylink

os.chdir('/Users/visionlab/Documents/Wouter/Exp3-SLS-main')
#os.chdir('/Users/harrysteinharter/Documents/MSc/Timo Internship/Pilot_V4_mandoh')
import otherFunctions as OF # Has to go after changing directory bc of it's location
small = True # Logical indicating if we are running the experiment with a lot of images
pilotThree = True # indicating if we are only doing session 3

#### Set up window ####
screenDim = np.array([2560, 1600])
myWin = visual.Window(monitor = 'Flanders', fullscr = True, colorSpace = 'rgb', color = (0,0,0), bpc = (10,10,10), depthBits=10, units = 'deg')

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

if pilotThree:
    thisSession = 3 ######## Remember to delete!!

fileName = str(nSub)+str(thisSession)+'.edf'
outPath = "./output_eye/"
localFile = outPath + fileName

#### Load images ####
condA, condB = trainShape+'_'+trainLocations[0:2], trainShape+'_'+trainLocations[3:5]
if thisSession == 3:
    stimuli = OF.loadImages(myWin,ALL=True,small = small)
else:
    stimuli = OF.loadImages(myWin,condA,condB,small = small)
labels = list(stimuli.keys())
images = list(stimuli.values())
stim_array = np.array(list(zip(labels,images)))
if thisSession == 3:
    C_TL_BR_array = np.array([c for c in stim_array if c[0].startswith(('C_TL', 'C_BR'))])
    C_TR_BL_array = np.array([c for c in stim_array if c[0].startswith(('C_TR', 'C_BL'))])
    BC_TL_BR_array = np.array([c for c in stim_array if c[0].startswith(('BC_TL', 'BC_BR'))])
    BC_TR_BL_array = np.array([c for c in stim_array if c[0].startswith(('BC_TR', 'BC_BL'))])
    blockConditions = [C_TL_BR_array,C_TR_BL_array,BC_TL_BR_array,BC_TR_BL_array]
    random.shuffle(blockConditions)

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
breakTrials = np.linspace(0,maxTrials,nBlocks+1, dtype='int')[1:-1]                    # Trial numbers where a break happens
T_response = .5#float('inf')                                                               # float('inf') or 0.5 Time for them to respond (s)
T_intertrial = .5                                                                       # Intertrial time (s)
T_stim = .25                                                                            # Stimulus presentation time (s)
T_break = 5                                                                             # Inter-block required waiting time (s)
T_block_max = 600                                                                       # Max time a block is allowed to go on for (s)
nUp = 1                                                                                 # The number of ‘incorrect’ (or 0) responses before the staircase level increases.
nDown = 3                                                                               # The number of ‘correct’ (or 1) responses before the staircase level decreases.
nReversals = None                                                                       # 1 or None? Number of reversals before a staircase can end (always >0)
keylist = ['q','escape','right','left']
maxIntensity = 1                                                                        # Max and starting intensity of the staircases
nullOdds = 0.2                                                                          # The odds of getting a null trial with no shape

### Eyetracker info ###
dummy = False
if dummy == True:
    tracker = pylink.EyeLink(None)
else:
    tracker = pylink.EyeLink("100.1.1.2:255.255.255.0")  # our eyelink is at 100.1.1.2:255.255.255.0. default is 100.1.1.1:...

tracker.openDataFile(fileName)
tracker.sendCommand("screen_pixel_coords = 0 0 1919 1079")

pylink.openGraphics()
tracker.doTrackerSetup()
pylink.closeGraphics()

time = core.Clock()
time.reset()

#### Experiment Here ####
def closeTracker(tracker,fileName,localFile):
    tracker.closeDataFile()
    tracker.receiveDataFile(fileName, localFile) # Takes closed data file from 'src' on host PC and copies it to 'dest' at Stimulus PC
    tracker.close()

def trainingOneTwo():
    whatDo = visual.TextStim(myWin, color = 'black', text = f'This is training. You are looking for the {trainShape} SHAPE in the {trainLocations} locations. Press any key to continue.')
    nTrials = 4
    nEach = nTrials//2
    trainStim = np.vstack((null_stim_array[0:nEach],stim_array[0:nEach]))
    T_wait = 1
    resp_m = visual.TextStim(myWin, text = 'Respond now. LEFT for NO. RIGHT for YES.', color = 'black')
    fixation = visual.GratingStim(win=myWin, color=1, colorSpace='rgb',tex=None, mask='cross', size=0.2, pos = [0,0])
    
    OF.drawOrder(whatDo,myWin)
    event.waitKeys()
    OF.countdown(myWin)
    
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

def trainingThree(a,b,c):
    whatDo = visual.TextStim(myWin, color = 'black', text = f'This is training. You are looking for the {b} SHAPE in the {c} locations. Press any key to continue.')
    nTrials = 4
    nEach = nTrials//2
    trainStim = np.vstack((null_stim_array[0:nEach],a[0:nEach]))
    T_wait = 1
    resp_m = visual.TextStim(myWin, text = 'Respond now. LEFT for NO. RIGHT for YES.', color = 'black')
    fixation = visual.GratingStim(win=myWin, color=1, colorSpace='rgb',tex=None, mask='cross', size=0.2, pos = [0,0])
    
    OF.drawOrder(whatDo,myWin)
    event.waitKeys()
    OF.countdown(myWin)
    
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
    
    stairs = data.StairHandler(
                startVal = maxIntensity,
                nReversals =  nReversals, 
                nTrials = float('inf'),
                stepSizes = 0.1, stepType = 'log',
                nUp = nUp, nDown = nDown,
                minVal = 0, maxVal = maxIntensity
                )                                   
    
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
        # Determine if it will be a Null or Real trial
        if random.random() <= nullOdds:
            thisLabel, thisImage = random.choice(null_stim_array)
            thisShape, thisLocation = 'NA','NA'
        else:
            thisLabel, thisImage = random.choice(stim_array)
            thisShape, thisLocation = OF.textExtract(thisLabel)
        thisIntensity = stairs.intensity
        stairs.intensities.append(thisIntensity)
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
                    if thisShape == 'NA':
                        response = 0
                        fixation.color = 'red'
                    else:
                        response = 1
                        fixation.color = 'green'
                elif key == 'left':
                    if thisShape == 'NA':
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
        stairs.addResponse(response)
        dataFile.write(f"{nSub},{thisSession},{trainShape},{trainLocations},{thisBlock+1},{trial+1},{thisShape},{thisLocation},{thisLabel},{thisIntensity},{response},{rt}\n")

def sessionThree():
    
    trialClock = core.Clock()
    blockClock = core.Clock()
    thisBlock = 0
    blockShape, blockLocation = 'ABCD', 'WXYZ'
    intro_m = visual.TextStim(win = myWin, text = f"""This is the experiment now.
    You are looking for the {blockShape} SHAPE in the {blockLocation} LOCATION.
    Press any key to continue.""", color = 'black')
    break_m = visual.TextStim(win = myWin, text = f'Take a short break for at least {T_break} seconds. You can stretch, go on your phone, get some water, just relax :)', color = 'black')
    continue_m = visual.TextStim(win = myWin, text = f"""This is not training.
    You can keep resting or you can continue if you are ready.
    The next block will show the {blockShape} SHAPE in the {blockLocation} LOCATION.
    Press any key to continue.""", color = 'black')
    trainEnd = visual.TextStim(myWin, color = 'black', text = 'Training is over. Press any key to continue.')
    blank = visual.TextStim(win = myWin, text = "You shouldn't see this", color = (0,0,0))
    fixation = visual.GratingStim(win=myWin, color=1, colorSpace='rgb',tex=None, mask='cross', size=0.2, pos = [0,0])
    diode = visual.GratingStim(win=myWin, color=-1, colorSpace='rgb',tex=None, mask='circle', units = 'pix', size=80, pos = [-780,-440],autoDraw=True)
    
    tracker.startRecording(1,1,1,1)
    for trial in range(maxTrials):
        if trial == 0:
            # Set up the first staircase
            thisBlock_array = blockConditions[thisBlock]
            stairs = data.StairHandler(
                startVal = maxIntensity,
                nReversals =  nReversals, 
                nTrials = float('inf'),
                stepSizes = 0.1, stepType = 'log',
                nUp = nUp, nDown = nDown,
                minVal = 0, maxVal = maxIntensity
                )
            blockShape, blockLocation = OF.textExtract(thisBlock_array[0,0])
            if blockLocation == 'TL':
                blockLocation = 'TL & BR'
            else:
                blockLocation = 'TR & BL'
            intro_m.text = f"This is not training. You are looking for the {blockShape} SHAPE in the {blockLocation} LOCATION. Press any key to continue."
            # do training
            trainingThree(thisBlock_array,blockShape,blockLocation)
            OF.drawOrder(trainEnd,myWin)
            event.waitKeys()
            # end training,  show messages
            OF.drawOrder(intro_m,myWin)
            event.waitKeys()
            OF.countdown(myWin)
        if (trial in breakTrials) or (blockClock.getTime() >= T_block_max):
            # Change the staircases
            thisBlock += 1
            thisBlock_array = blockConditions[thisBlock]
            stairs = data.StairHandler(
                startVal = maxIntensity,
                nReversals =  nReversals, 
                nTrials = float('inf'),
                stepSizes = 0.1, stepType = 'log',
                nUp = nUp, nDown = nDown,
                minVal = 0, maxVal = maxIntensity
                )
            # Display messages
            blockShape, blockLocation = OF.textExtract(thisBlock_array[0,0])
            if blockLocation == 'TL':
                blockLocation = 'TL & BR'
            else:
                blockLocation = 'TR & BL'
            continue_m.text = f"""This is not training.
            You can keep resting or you can continue if you are ready.
            The next block will show the {blockShape} SHAPE in the {blockLocation} LOCATION.
            Press any key to continue."""
            OF.drawOrder(break_m,myWin)
            core.wait(T_break)
            # do training
            trainingThree(thisBlock_array,blockShape,blockLocation)
            OF.drawOrder(trainEnd,myWin)
            event.waitKeys()
            OF.drawOrder(continue_m,myWin)
            event.waitKeys()
            OF.countdown(myWin)
        #### The actual experimental loop ####
        tracker.sendMessage(f"TRIAL_START {trial}")
        if random.random() <= nullOdds:
            thisLabel, thisImage = random.choice(null_stim_array)
            thisShape, thisLocation = 'NA','NA'
        else:
            thisLabel, thisImage = random.choice(thisBlock_array)
            thisShape, thisLocation = OF.textExtract(thisLabel)
        thisIntensity = stairs.intensity
        thisImage.contrast = thisIntensity
        stairs.intensities.append(thisIntensity)
        # Stimulus presentation
        OF.drawOrder(fixation,myWin)
        core.wait(T_intertrial)
        fixation.color = 'white'
        diode.color *= -1
        OF.drawOrder([thisImage,fixation],myWin)
        core.wait(T_stim)
        diode.color *= -1
        OF.drawOrder(blank,myWin)
        trialClock.reset()
        tracker.sendMessage(f"TRIAL_END {trial}")
#        tracker.stopRecording()
        keys = event.waitKeys(keyList = keylist, maxWait = T_response)
        
        # Record response
        if keys:
            for key in keys:
                if key in ['q','escape']:
                    event.clearEvents()
                    core.quit()
                elif key == 'right':
                    if thisShape == 'NA':
                        response = 0
                        fixation.color = 'red'
                    else:
                        response = 1
                        fixation.color = 'green'
                elif key == 'left':
                    if thisShape == 'NA':
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
        stairs.addResponse(response)
        dataFile.write(f"{nSub},{thisSession},{trainShape},{trainLocations},{thisBlock+1},{trial+1},{thisShape},{thisLocation},{thisLabel},{thisIntensity},{response},{rt}\n")
    closeTracker(tracker,fileName,localFile)


hello = visual.TextStim(myWin, color='black',text = """Welcome! This first part is training. 
An image will appear on screen. 
Respond with RIGHT if the hidden shape is present.
Respond with LEFT if the hidden shape is not present.
Press any key to begin.""")
goodbye = visual.TextStim(myWin, color='black',text = """It's over! I hope it wasn't too bad :)""")
trainEnd = visual.TextStim(myWin, color = 'black', text = 'Training is over. Press any key to continue.')

OF.drawOrder(hello,myWin)
event.waitKeys()
if thisSession == 3:
    sessionThree()
else:
    trainingOneTwo()
    OF.drawOrder(trainEnd,myWin)
    event.waitKeys()
    sessionOneTwo()

OF.drawOrder(goodbye,myWin)
core.wait(5)

event.clearEvents()
core.quit()
