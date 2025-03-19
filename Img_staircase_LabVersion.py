# %%
from psychopy.hardware import keyboard
from psychopy import core, visual, data, event#, gui #psychopy.gui doesn't work in lab
from psychopy.tools.filetools import fromFile, toFile
import numpy as np
import random
import math
import os
import re
import pandas as pd
import pylink
#import importlib
os.chdir('../img_staircase')
import otherFunctions as OF # Has to go after changing directory bc of it's location
#importlib.reload(OF) # Allows me to edit OF w/out restartig VSCode
# %%
#### Set up information ####
myWin = visual.Window(monitor = 'Flanders', fullscr = True, colorSpace = 'rgb', color = (0,0,0), bpc = (10,10,10), depthBits=10, units = 'deg')
nSub = OF.SubNumber("SubNumStaircase.txt")
# Create data file
cd = os.getcwd()
fullFile = os.path.join(cd,'Img_staircase_data',str(nSub)+'.csv')
dataFile = open(fullFile, 'w')
dataFile.write('participant,block,trial,trial_true,this_Shape,this_Location,image,intensity,response,RT\n')

small = True # Logical for piloting

#### Tracker ####
dummy = False
tracker = pylink.EyeLink(None) if dummy else pylink.EyeLink("100.1.1.2:255.255.255.0")  #our eyelink is at 100.1.1.2:255.255.255.0. default is 100.1.1.1:...
edfFile = str(nSub)+'.edf'
edfOutPath = "./Img_staircase_edf/"
edfLocalFile = edfOutPath + edfFile
tracker.openDataFile(edfFile)
tracker.sendCommand("screen_pixel_coords = 0 0 1919 1079")
def closeTracker(tracker,fileName,localFile):
    tracker.closeDataFile()
    tracker.receiveDataFile(fileName, localFile) # Takes closed data file from 'src' on host PC and copies it to 'dest' at Stimulus PC
    tracker.close()

pylink.openGraphics()
tracker.doTrackerSetup()
pylink.closeGraphics()
# %%
#### Load Images ####
stimuli = OF.loadImages(myWin, ALL=True,small=small)
if small:
    null = OF.loadNull(myWin,n=len(stimuli),small=small)
else:
    null = OF.loadNull(myWin,n=len(stimuli)//4,small=small)

#  Now seperate  images into different lists for easier indexing later
stimuli_L = np.zeros(shape=(len(stimuli),2),dtype='O')
n = 0
for i in stimuli:
    key = i.name
    stimuli_L[n] = (key,i)
    n += 1
C_TL_BR_array = np.array([c for c in stimuli_L if c[0].startswith(('C_TL', 'C_BR'))])
C_TR_BL_array = np.array([c for c in stimuli_L if c[0].startswith(('C_TR', 'C_BL'))])
BC_TL_BR_array = np.array([c for c in stimuli_L if c[0].startswith(('BC_TL', 'BC_BR'))])
BC_TR_BL_array = np.array([c for c in stimuli_L if c[0].startswith(('BC_TR', 'BC_BL'))])
null1,null2,null3,null4 = np.split(null,4)
# %%
#### Experimental Variables ####
nReal = 8 if small else 100
nNull = 2 if small else nReal//4
nBlocks = 4 # Number of blocks
maxTrials = (nReal + nNull) * nBlocks   # Number of trials per staircase before it can close
breakTrials = np.linspace(0,maxTrials,nBlocks+1, dtype='int')[:-1]  # Trial numbers where a break happens
T_response = 2   # float('inf') or 0.5 Time for them to respond (s)
T_intertrial = .5   # Intertrial time (s)
T_stim = .25    # Stimulus presentation time (s)
T_break = 5 # Inter-block required waiting time (s)
T_block_max = 600   # Max time a block is allowed to go on for (s)
nUp = 1 # The number of ‘incorrect’ (or 0) responses before the staircase level increases.
nDown = 3   # The number of ‘correct’ (or 1) responses before the staircase level decreases.
nReversals = None   # 1 or None? Number of reversals before a staircase can end (always >0)
keylist = ['q','escape','right','left'] # Allowed keyboard input during response window
maxIntensity = 0.2 # Max and starting intensity of the staircases
stepSize = .1
nullOdds = 0.2  # The odds of getting a null trial with no shape
nTraining = 4 # Total number of trials in training
T_train_stim = 1.5 # Duration of stimulus presentation during training
# %%
#### Define messages and additional stimuli ####
hello_m = visual.TextStim(myWin, color='black', text = f"""
Welcome to the experiment!
There will be 4 blocks. Each block will last less than {math.ceil(T_block_max/60)} minutes.
During each block you will look for a spcified shape in two locations.
Before each block you will be told what the shape is, and where to look for it.
You will also do a short practice session before each block.
                          """)
goodbye_m = visual.TextStim(myWin, color='black', text = """
Thank you for participating! 
You're finally done! :)
This window will close automatically.
                              """)
break_m = visual.TextStim(myWin, color = 'black', text = f"""
That block is over.  Take a {T_break} second break.
Go on your phone, get some water, stretch. Just relax for a bit.""")
continue_m = visual.TextStim(myWin, color = 'black',text=f"""
It has been {T_break} seconds. You may keep resting, or you can continue.
The next portion of the experiment will be a training block. 
Press any key to continue.
The next screen will contain more instructions.
                              """)
train_end_m = visual.TextStim(myWin, color = 'black',text=None, wrapWidth=1.8, units='norm')
train_begin_m = visual.TextStim(myWin, text = None, color='black', wrapWidth=1.8, units='norm')
train_resp_m = visual.TextStim(myWin, text = 'Respond now. LEFT for NO. RIGHT for YES.', color = 'black')
block_begin_m = visual.TextStim(myWin, text = None,color='black')
fixation = visual.GratingStim(win=myWin, color=1, colorSpace='rgb',tex=None, mask='cross', size=0.2, pos = [0,0])
blank = visual.TextStim(myWin, text = "You shouldn't see this.", colorSpace=myWin.colorSpace,color=myWin.color)
diode = visual.GratingStim(myWin,color=-1, colorSpace = 'rgb',tex=None,mask='circle',units='pix',size=80,pos=[-780,-440],autoDraw=True)

# %%
#### Define the training block ####
def training(stim,small):
    # `stim` = Should be an array containing images and name
    # `small` = Logical indicating if we are in a piloting mode w/ small image sets
    nEach = nTraining//2
    labs = stim[:,0]
    images = stim[:,1]

    # Set up the stimuli
    if small:
        images = np.random.choice(images,nEach,True)
        nullTrain = np.random.choice(null,nEach,True)
    else:
        images = np.random.choice(images,nEach,False)
        nullTrain = np.random.choice(null,nEach,False)
    stimTrain = np.hstack((images,nullTrain))
    np.random.shuffle(stimTrain)

    # Set up the text
    shape, locations = OF.textExtract(labs[0])
    if locations in ['TL','BR']:
        locations = 'TL & BR'
    elif locations in ['TR','BL']:
        locations = 'TR & BL'
    else:
        locations = 'ERROR'
    train_begin_m.text = f"""
    This is training. Wait for the experimenter, or knock on the door to to get their attention.
    You are looking for the {shape} SHAPE in the {locations} LOCATIONS.
    Press any key to begin the training session.
    """

    # Display the text
    OF.drawOrder(train_begin_m,myWin)
    event.waitKeys()
    OF.countdown(myWin)

    # The training loop
    for image in stimTrain:
        thisLab = image.name
        OF.drawOrder(fixation,myWin)
        core.wait(T_intertrial)
        fixation.color = 'white'
        OF.drawOrder([image,fixation],myWin)
        core.wait(T_train_stim)
        OF.drawOrder(train_resp_m,myWin)
        allKeys = event.waitKeys(keyList=keylist,maxWait=float('inf'))
        if allKeys:
            for key in allKeys:
                if key == 'right':
                    if thisLab.startswith('Null'):
                        resp = 0 # wrong
                    else:
                        resp = 1
                elif key == 'left':
                    if thisLab.startswith('Null'):
                        resp = 1 # right
                    else:
                        resp = 0
                else:
                    print('Wrong button, or on purpose?')
                    myWin.close()
                    core.quit()
        OF.giveFeedback(fixation,resp)
    return(shape,locations)

# %%
#### Define the Experiment ####
def experiment():
    thisBlock = -1
    blockConditions = [C_TL_BR_array,C_TR_BL_array,BC_TL_BR_array,BC_TR_BL_array]
    blockNullConds = [null1,null2,null3,null4]
    random.shuffle(blockConditions)
    random.shuffle(blockNullConds)
    trialClock = core.Clock()
    blockClock = core.Clock()
    ix = 0

    OF.drawOrder(hello_m,myWin)
    event.waitKeys()
    
    for i in range(maxTrials):
        if (i in breakTrials) or (blockClock.getTime() >= T_block_max):
            blockClock.reset()
            # Display message
            if i != 0: # Don't do this on the first breakTrials
                OF.drawOrder(break_m,myWin)
                core.wait(T_break)
                OF.drawOrder(continue_m,myWin)
                event.waitKeys()
            # Create new staircase
            stairs = data.StairHandler(
                startVal=maxIntensity,
                maxVal=maxIntensity,
                minVal=0,
                nReversals=None,
                nTrials=maxTrials//nBlocks,
                stepSizes=stepSize,
                stepType='log',
                nUp=nUp,
                nDown=nDown,
                applyInitialRule=True
            )
            # Get new stimuli
            thisBlock += 1
            blockStimuli = blockConditions[thisBlock]
            blockNulls = blockNullConds[thisBlock]
            np.random.shuffle(blockStimuli)
            np.random.shuffle(blockNulls)
            # Do training
            shape,locations = training(blockStimuli,small)
            train_end_m.text = f"""
            That training period is over. The next section will be the experiment.
            You will have less time to respond, the stimuli will appear for less time, and you will not receive feedback if you provided the correct response.
            The text telling you when to respond will not appear, but you still respond once the image has disappeared.
            You are still looking for the {shape} SHAPE in the {locations} LOCATIONS.
            If you have any questions, ask the experimenter now. Press any key to start the next block.
            """
            OF.drawOrder(train_end_m,myWin)
            event.waitKeys()
            OF.countdown(myWin)
            fixation.color = 'white' # Reset from the last training() trial
        # Outside of breakTrials
        # Eye tracker
        tracker.startRecording(1,1,1,1)
        core.wait(.1)
        # Establish this trial's image
        thisI = stairs.intensity
        stairs.intensities.append(thisI)
        if random.random() <= nullOdds:
            thisImage = random.choice(blockNulls)
            thisLabel = thisImage.name
            thisShape, thisLocation = 'NA','NA'
        else:
            index = random.randint(0,len(blockStimuli)-1)
            thisLabel, thisImage = blockStimuli[index]
            thisShape, thisLocation = OF.textExtract(thisLabel)
        # Eye tracker
        tracker.sendMessage(f"TRIAL_START {i+ix+1}_{thisShape}_{thisLocation}")
        # Set image contrast
        thisImage.contrast = thisI
        # Display stimuli
        diode.color *= -1
        OF.drawOrder(fixation,myWin)
        core.wait(T_intertrial)
        OF.drawOrder([thisImage,fixation],myWin)
        core.wait(T_stim)
        diode.color *= -1
        OF.drawOrder(blank,myWin)
        # eye tracker
        tracker.sendMessage(f"TRIAL_END {i+ix+1}_{thisShape}_{thisLocation}")
        tracker.sendMessage(f"IMAGE ../test_outputs_new/{thisImage.name}.png")
        trialClock.reset()
        tracker.stopRecording()
        # Get response
        keys = event.waitKeys(T_response,keylist)
        rt = trialClock.getTime()
        T_left = T_response-rt
        core.wait(T_left)
        if keys:
            for key in keys:
                if key == 'right':
                    if thisLabel.startswith('Null'):
                        resp = 0
                    else:
                        resp = 1
                elif key == 'left':
                    if thisLabel.startswith('Null'):
                        resp = 1
                    else:
                        resp = 0
                else:
                    print('Ended on Purpose?')
                    dataFile.close()
                    closeTracker(tracker,edfFile,edfLocalFile)
                    myWin.close()
                    core.quit()
            stairs.addResponse(resp) # We do not alter the staircase if they don't respond
        else: # They did not press anything
            i -= 1 # Decrement the trial counter by  1
            ix += 1 # Increment the fake trial counter
            resp = 'NA'
            rt = 'NA'
        dataFile.write(f"{nSub},{thisBlock+1},{i+1},{i+ix+1},{thisShape},{thisLocation},{thisLabel},{thisI},{resp},{rt}\n")
    OF.drawOrder(goodbye_m,myWin)
    core.wait(2)
    dataFile.close()
    closeTracker(tracker,edfFile,edfLocalFile)
    print("Success!")
    myWin.close()
    core.quit()
    return
experiment()
# %%
