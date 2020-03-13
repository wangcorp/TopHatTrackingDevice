# Imports
import io
import os
import csv
import sys
import time
import smbus
import signal
import datetime
import picamera
import subprocess
import numpy as np

from pynput import keyboard

# Global variables
start_time = datetime.datetime.now()

break_bool = True
preview_bool = True

resolution = (1296,730)
framerate = 42
#resolution = (1920,1080)
#framerate = 30

output_path = '/home/pi/Desktop/'
file_name = start_time.strftime("%Y-%m-%d-%H-%M-%S")
output_directory = os.path.join(output_path,'Result_'+file_name)
os.mkdir(output_directory)

time_file = open(os.path.join(output_directory,'timestamp_'+file_name+'.txt'),'a+')

class customOutput(object):
    def __init__(self, camera, filename):
        self.camera = camera
        self._file = io.open(filename, 'wb')
        
    def write(self, buf):
        t_frame = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        time_info = [self.camera.frame.timestamp, t_frame]
        time_file.write("%s\n" % time_info)
        return self._file.write(buf)

    def flush(self):
        self._file.flush()
        
    def close(self):
        self._file.close()

def on_press(key):
    global break_bool, preview_bool
    if key == keyboard.Key.esc:
        break_bool = False
    if key == keyboard.Key.ctrl_l:
        preview_bool = not preview_bool
        if preview_bool == True:
            camera.start_preview()
        else:
            camera.stop_preview()

# Setting up keyboard listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Setup recording
video_path = os.path.join(output_directory,('video_'+file_name+'.h264'))
mp4_path = os.path.join(output_directory,('video_'+file_name+'.mp4'))
camera = picamera.PiCamera()
camera.resolution = resolution
camera.framerate = framerate
output = customOutput(camera, video_path)

# Start and warmup camera
camera.start_preview()
#time.sleep(2)

# Start recording
print("Acquisition start.")
camera.start_recording(output,format='h264')

while break_bool:
    try:
        pass
    except Exception as e:
        print(e, '>>>>>>>>>>>>>>>>')

camera.stop_recording()
if preview_bool == True:
    camera.stop_preview()
output.close()
camera.close()
listener.stop()
time_file.close()

sys.exit('Acquisition terminated.')