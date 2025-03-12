import os
from psychopy.visual import ImageStim, TextStim, Rect
from psychopy.event import waitKeys
from psychopy.core import CountdownTimer
import re
import random
import math
import numpy as np


def SubNumber(filename):
    with open(filename, 'r', encoding='utf-8-sig') as file:
        content = int(file.read().strip())

    content_int = int(content)
    new_content = (content_int + 1)
    
    with open(filename, 'w') as file:
        file.write(str(new_content))
    return new_content

def drawOrder(stimuli, win):
    if not isinstance(stimuli, (list,tuple)):
        stimuli.draw()
    elif isinstance(stimuli, (list,tuple)):
        for stimulus in stimuli:
            stimulus.draw()
    win.flip()
    return

def loadingBar(myWin, c, first:bool,w=1.8,h=.2):
    """
    c: completeness. Express it as a float. So 50% done would be 0.5
    w, h: width and height
    """
    if first:
        f = 'first'
    else:
        f = 'second'
    ed = ((2-w)/2)-1 # edge position. `norm` units range -1:1
    barBox = Rect(myWin,units='norm',pos = (0,-.7),size=(w,h),lineColor='black',lineColorSpace='rgb')
    barFill = Rect(myWin,units='norm',pos = (ed+(.5*w*c),-.7),size=(w*c,h),fillColor = 'black', fillColorSpace='rgb')
    textFill = TextStim(myWin, text = f"Loading {f} image set... {int(c*100)}%", color = 'white', pos=(0,-.7),units='norm', height = h*.75, wrapWidth = w)
    drawOrder([barBox,barFill,textFill],myWin)

def loadImages(myWin,condA=None,condB=None,ALL=False,small = True):
    if small:
        imgPath = os.path.join(os.getcwd(),'test_outputs')
    else:
        imgPath = os.path.join(os.getcwd(),'test_outputs_new')
    file_list = os.listdir(imgPath)
    if not ALL:
        file_list = [file for file in file_list if (file.startswith(condA) or file.startswith(condB))]  # Keep only files from the current condition
    if all:
        file_list = [file for file in file_list if file.endswith('.png')] # remove invisible files like `.DS_Store` and folder like `Null`
    
    stimuli = {}
    for file in file_list:
        key = os.path.splitext(file)[0]
        stimuli[key] = ImageStim(win = myWin, image = os.path.join(imgPath,file), contrast = 1.0, size = (14.25,14.25), units = 'deg', name = key) # Size from Mandoh's paper
            # visual.ImageStim.contrast ranges from -1 to 1. < 0 is inverted colors so we will not use that.
        #print(f"Stimulus {key} created.")
        i = file_list.index(file)
        l = len(file_list)
        loadingBar(myWin,c=(i+1)/l,first=True)
    return(stimuli)

def loadNull(myWin,n,small=True):
    if small:
        imgPath = os.path.join(os.getcwd(),'test_outputs','Null')
    else:
        imgPath = os.path.join(os.getcwd(),'test_outputs_new','Null')
    file_list = os.listdir(imgPath)
    file_list = [file for file in file_list if file.endswith('.png')] # remove invisible files like `.DS_Store`
    file_list = random.sample(file_list, k=n)
    
    stimuli = {}
    for file in file_list:
        key = 'Null'+os.path.splitext(file)[0][-2:]
        print(key)
        stimuli[key] = ImageStim(win = myWin, image = os.path.join(imgPath,file), contrast = 1.0, size = (14.25,14.25), units = 'deg', name=key) # Size from Mandoh's paper
            # visual.ImageStim.contrast ranges from -1 to 1. < 0 is inverted colors so we will not use that.
        #print(f"Stimulus {key} created.")
        i = file_list.index(file)
        l = len(file_list)
        loadingBar(myWin,c=(i+1)/l,first=False)
    return(stimuli)

def ParticipantInput(myWin):
    done = False
    text_input = "999"
    text = TextStim(win = myWin, text = f"Participant Number: {text_input}", color = 'black')
    while not done:
        text.draw()
        myWin.flip()
        
        keys = waitKeys()
        for key in keys:
            if key == 'return':
                print(f"Final Input: {text_input}")
                done = True
            elif key == 'backspace':
                text_input = text_input[:-1]
            elif key == 'escape':
                myWin.close()
                core.quit()
            else:
                text_input += key
            text.text = f"Participant Number: {text_input}"
    return(text_input)

def textExtract(text):
    pattern = r"^(C|BC)_(BL|BR|TL|TR)_\d+$"
    match = re.match(pattern, text)
    if match:
        shape, position = match.groups()  # Extracted values
    return(shape,position)

def nullRandomizer(session_n,nullOdds = 0.2):
    i = random.random()
    if i <= nullOdds:
        condIndex = 0
        return(condIndex)
    else:
        condIndex = random.randint(1,2)
        return(condIndex)

def imageChoice(condition_label, stimuli_array):
    s = random.choice(stimuli_array[np.char.find(stimuli_array[:,0].astype('str'), condition_label) >= 0])
    l,i = s[0],s[1]
    return(l,i)

def countdown(win):
    t = CountdownTimer(int(3))
    while t.getTime() >= 0:
        cd = TextStim(win, color = 'black', text = f"Experiment will continue in {math.ceil(t.getTime())}...")
        drawOrder(cd,win)

