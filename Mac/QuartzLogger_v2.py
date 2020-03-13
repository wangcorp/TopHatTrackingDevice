# -*- coding: utf-8 -*-
#
# Author: Ruijia Wang <w.ruijia@gmail.com>
"""
    MouseLogger for stimulation timestamp
"""

import os
import Quartz
import datetime
import subprocess

from AppKit import NSEvent
from os.path import expanduser

### User parameters ###
# Input path containing .csv files
home = expanduser("~")
output_path = os.path.join(home,'Desktop/Ruijia/Output')

# Name of subdirectory of interest (without extention)
file_name = 'Result'

### Script variables and switches ###
break_counter = 0
break_th = 3

cmd_bool = False
cmd_first = True

start_time = datetime.datetime.now()
time_0 = 0
time_1 = 0
delay = 0

event_list = []
event_ROI = []
event_YES = []
event_NO = []

ROI_points = [(0,0), (1920,1080)]
ROIx_min = 0
ROIx_max = 0
ROIy_min = 0
ROIy_max = 0

YES_points = [(2300,1200), (2500,1400)]
YESx_min = 0
YESx_max = 0
YESy_min = 0
YESy_max = 0

NO_points = [(1800,1200), (2000,1400)]
NOx_min = 0
NOx_max = 0
NOy_min = 0
NOy_max = 0

### Functions ###
def keyboardTapCallback(proxy, type_, event, refcon):
    global break_counter, cmd_first, time_0, time_1, delay
    # Get Keyboard event
    keyEvent = NSEvent.eventWithCGEvent_(event)
    if keyEvent is None:
        pass
    else:
        key =  keyEvent.characters()
        # Launch raspi recording with cmd+r via ssh
        if cmd_bool == True and key == 'r' and cmd_first == True:
            time_0 = datetime.datetime.now()
            home = expanduser("~")
            ssh_key_path = os.path.join(home,'.ssh/id_rsa')
            HOST='pi@raspberrypi.local'
            COMMAND='export DISPLAY=:0.0 && python3 /home/pi/Desktop/optoRecord_v2.py'
            ssh = subprocess.Popen(["ssh", "-i", ssh_key_path, "%s" % HOST, COMMAND],
                                    shell=False,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            time_1 = datetime.datetime.now()

            # Calculate delay of ssh
            delay = time_1 - time_0
            print('Acquisition started.')

            # Prevent a second cmd+r
            cmd_first = False
        break_counter = 0

def mouseClickCallback(proxy, type_, event, refcon):
    global break_counter
    # Get mouse click event
    mouseEvent = NSEvent.mouseLocation()
    if mouseEvent is None:
        pass
    else:
        # Save coordinates and time of clicking when cmd+r has been pressed
        if cmd_first == False:
            x = mouseEvent.x
            y = mouseEvent.y
            time = datetime.datetime.now()
            dt = time - time_0
            info = [time, x, y]
            event_list.append(info)
            if ROIx_min<=x<=ROIx_max and ROIy_min<=y<=ROIy_max:
                # Save second copy of clicks within ROI
                event_ROI.append(info)
                print('ROI', info)
            elif YESx_min<=x<=YESx_max and YESy_min<=y<=YESy_max:
                event_YES.append(info)
                print('YES', info)
            elif NOx_min<=x<=NOx_max and NOy_min<=y<=NOy_max:
                event_NO.append(info)
                print('NO', info)
        break_counter = 0

def modifierTapCallback(proxy, type_, event, refcon):
    global break_counter, cmd_bool
    # Get special keyboard event (modifiers)
    modifierEvent = NSEvent.eventWithCGEvent_(event)
    if modifierEvent is None:
        pass
    else:
        flag = modifierEvent.modifierFlags()
        # On press the flag is different from zero
        if flag != 0:
            # Stop the acquisition when ctrl is pressed 'break_th' times
            if flag == 262145: #ctrl
                break_counter = break_counter + 1
                if break_counter >= break_th:
                    raise Exception('BRUT FORCE STOP')
            else:
                break_counter = 0
            if flag == 1048584: #cmd
                cmd_bool = True
        else:
            # On release, the flag is equal to zero
            cmd_bool = False

def setRange(points_2D):
    x = (points_2D[0][0], points_2D[1][0])
    y = (points_2D[0][1], points_2D[1][1])

    x_min = min(x)
    x_max = max(x)
    y_min = min(y)
    y_max = max(y)

    return x_min, x_max, y_min, y_max

def writeEvent(data, data_name, output_path, data_dir):
    output_file_path = os.path.join(output_path, data_dir)

    with open(output_file_path +'_'+data_name+'.txt', 'w') as f:
        for event in data:
            f.write("%s,%f,%f\n" % (event[0], event[1], event[2]))
    f.close()

def writeDelay(output_path, data_dir, delay):
    output_file_path = os.path.join(output_path, data_dir)

    with open(output_file_path + '_delay.txt', 'w') as f:
        f.write(str(delay.total_seconds()))
    f.close()

### Main function ###
# Create the event masks
tap = Quartz.CGEventTapCreate(
                              Quartz.kCGSessionEventTap,
                              Quartz.kCGHeadInsertEventTap,
                              Quartz.kCGEventTapOptionListenOnly,
                              Quartz.CGEventMaskBit(Quartz.kCGEventKeyUp),
                              keyboardTapCallback,
                              None
                             )

click = Quartz.CGEventTapCreate(
                              Quartz.kCGSessionEventTap,
                              Quartz.kCGHeadInsertEventTap,
                              Quartz.kCGEventTapOptionListenOnly,
                              Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseUp),
                              mouseClickCallback,
                              None
                             )

mod = Quartz.CGEventTapCreate(
                              Quartz.kCGSessionEventTap,
                              Quartz.kCGHeadInsertEventTap,
                              Quartz.kCGEventTapOptionListenOnly,
                              Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged),
                              modifierTapCallback,
                              None
                             )

# Add the mask to the run
runLoopSource_tap = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
Quartz.CFRunLoopAddSource(
                          Quartz.CFRunLoopGetCurrent(),
                          runLoopSource_tap,
                          Quartz.kCFRunLoopDefaultMode
                         )

runLoopSource_click = Quartz.CFMachPortCreateRunLoopSource(None, click, 0)
Quartz.CFRunLoopAddSource(
                          Quartz.CFRunLoopGetCurrent(),
                          runLoopSource_click,
                          Quartz.kCFRunLoopDefaultMode
                         )

runLoopSource_mod = Quartz.CFMachPortCreateRunLoopSource(None, mod, 0)
Quartz.CFRunLoopAddSource(
                          Quartz.CFRunLoopGetCurrent(),
                          runLoopSource_mod,
                          Quartz.kCFRunLoopDefaultMode
                         )

Quartz.CGEventTapEnable(tap, True)
Quartz.CGEventTapEnable(click, True)
Quartz.CGEventTapEnable(mod, True)

# Main loop
# Set different regions of interest
ROIx_min, ROIx_max, ROIy_min, ROIy_max = setRange(ROI_points)
YESx_min, YESx_max, YESy_min, YESy_max = setRange(YES_points)
NOx_min, NOx_max, NOy_min, NOy_max = setRange(NO_points)

# Wait for acquisition start
print('Press CMD+R to start the acquisition.')
print('To end the recording at any time, press CTRL %s times.' % break_th)

try:
    Quartz.CFRunLoopRun()

except Exception as e:
    # Send stop command to raspi
    home = expanduser("~")
    ssh_key_path = os.path.join(home,'.ssh/id_rsa')
    HOST='pi@raspberrypi.local'
    COMMAND="export DISPLAY=:0.0 && xdotool key Escape"
    ssh = subprocess.Popen(["ssh", "-i", ssh_key_path, "%s" % HOST, COMMAND],
                            shell=False,
                        	stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    print('Acquisition terminated.')

    # Save data
    data_dir = file_name + '_' + start_time.strftime("%Y-%m-%d-%H-%M-%S")
    output_path = os.path.join(output_path,data_dir)
    os.mkdir(output_path)

    writeEvent(event_list, 'raw', output_path, data_dir)
    writeEvent(event_ROI, 'ROI', output_path, data_dir)
    writeEvent(event_YES, 'YES', output_path, data_dir)
    writeEvent(event_NO, 'NO', output_path, data_dir)
    writeDelay(output_path, data_dir, delay)
