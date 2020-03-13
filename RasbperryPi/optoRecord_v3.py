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

from mpu6050 import mpu6050
from pynput import keyboard

# Global variables
start_time = datetime.datetime.now()

break_bool = True
preview_bool = True

resolution = (1296,730)
framerate = 42
#resolution = (1920,1080)
#framerate = 30
#resolution = (680,480)
#framerate = 90

output_path = '/home/pi/Desktop/'
file_name = start_time.strftime("%Y-%m-%d-%H-%M-%S")
output_directory = os.path.join(output_path,'Result_'+file_name)
os.mkdir(output_directory)

time_file = open(os.path.join(output_directory,'timestamp_'+file_name+'.txt'),'a+')

# Class and functions
class mpu6050(mpu6050):
    def read_i2c_word_fixed(self):
        fourteen_bytes = self.bus.read_i2c_block_data(0x68, 0x3b, 14)

        his, los = fourteen_bytes[0::2], fourteen_bytes[1::2]

        values = []
        for hi, lo in zip(his, los):
            value  = (hi << 8) + lo
            if value >= 0x8000:
                value = -((65535 - value) + 1)
            values.append(value)
        return values
    
    def get_accel_data_fixed(self, g=False):
        raw_values = self.read_i2c_word_fixed()
        
        x = raw_values[0]
        y = raw_values[1]
        z = raw_values[2]

        accel_scale_modifier = None
        accel_range = self.read_accel_range(True)

        if accel_range == self.ACCEL_RANGE_2G:
            accel_scale_modifier = self.ACCEL_SCALE_MODIFIER_2G
        elif accel_range == self.ACCEL_RANGE_4G:
            accel_scale_modifier = self.ACCEL_SCALE_MODIFIER_4G
        elif accel_range == self.ACCEL_RANGE_8G:
            accel_scale_modifier = self.ACCEL_SCALE_MODIFIER_8G
        elif accel_range == self.ACCEL_RANGE_16G:
            accel_scale_modifier = self.ACCEL_SCALE_MODIFIER_16G
        else:
            print("Unkown range - accel_scale_modifier set to self.ACCEL_SCALE_MODIFIER_2G")
            accel_scale_modifier = self.ACCEL_SCALE_MODIFIER_2G

        x = x / accel_scale_modifier
        y = y / accel_scale_modifier
        z = z / accel_scale_modifier

        if g is True:
            return {'x': x, 'y': y, 'z': z}
        elif g is False:
            x = x * self.GRAVITIY_MS2
            y = y * self.GRAVITIY_MS2
            z = z * self.GRAVITIY_MS2
            return {'x': x, 'y': y, 'z': z}

    def get_gyro_data_fixed(self):
        raw_values = self.read_i2c_word_fixed()
        
        x = raw_values[4]
        y = raw_values[5]
        z = raw_values[6]

        gyro_scale_modifier = None
        gyro_range = self.read_gyro_range(True)

        if gyro_range == self.GYRO_RANGE_250DEG:
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_250DEG
        elif gyro_range == self.GYRO_RANGE_500DEG:
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_500DEG
        elif gyro_range == self.GYRO_RANGE_1000DEG:
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_1000DEG
        elif gyro_range == self.GYRO_RANGE_2000DEG:
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_2000DEG
        else:
            print("Unkown range - gyro_scale_modifier set to self.GYRO_SCALE_MODIFIER_250DEG")
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_250DEG

        x = x / gyro_scale_modifier
        y = y / gyro_scale_modifier
        z = z / gyro_scale_modifier

        return {'x': x, 'y': y, 'z': z}

    def get_all_data_fixed(self, g=False):
        raw_values = self.read_i2c_word_fixed()
        
        ax = raw_values[0]
        ay = raw_values[1]
        az = raw_values[2]

        accel_scale_modifier = None
        accel_range = self.read_accel_range(True)

        if accel_range == self.ACCEL_RANGE_2G:
            accel_scale_modifier = self.ACCEL_SCALE_MODIFIER_2G
        elif accel_range == self.ACCEL_RANGE_4G:
            accel_scale_modifier = self.ACCEL_SCALE_MODIFIER_4G
        elif accel_range == self.ACCEL_RANGE_8G:
            accel_scale_modifier = self.ACCEL_SCALE_MODIFIER_8G
        elif accel_range == self.ACCEL_RANGE_16G:
            accel_scale_modifier = self.ACCEL_SCALE_MODIFIER_16G
        else:
            print("Unkown range - accel_scale_modifier set to self.ACCEL_SCALE_MODIFIER_2G")
            accel_scale_modifier = self.ACCEL_SCALE_MODIFIER_2G

        ax = ax / accel_scale_modifier
        ay = ay / accel_scale_modifier
        az = az / accel_scale_modifier

        if g is True:
            a = {'x': ax, 'y': ay, 'z': az}
        elif g is False:
            ax = ax * self.GRAVITIY_MS2
            ay = ay * self.GRAVITIY_MS2
            az = az * self.GRAVITIY_MS2
            a = {'x': ax, 'y': ay, 'z': az}
        
        gx = raw_values[4]
        gy = raw_values[5]
        gz = raw_values[6]

        gyro_scale_modifier = None
        gyro_range = self.read_gyro_range(True)

        if gyro_range == self.GYRO_RANGE_250DEG:
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_250DEG
        elif gyro_range == self.GYRO_RANGE_500DEG:
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_500DEG
        elif gyro_range == self.GYRO_RANGE_1000DEG:
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_1000DEG
        elif gyro_range == self.GYRO_RANGE_2000DEG:
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_2000DEG
        else:
            print("Unkown range - gyro_scale_modifier set to self.GYRO_SCALE_MODIFIER_250DEG")
            gyro_scale_modifier = self.GYRO_SCALE_MODIFIER_250DEG

        gx = gx / gyro_scale_modifier
        gy = gy / gyro_scale_modifier
        gz = gz / gyro_scale_modifier
        g = {'x': gx, 'y': gy, 'z': gz}
        
        return a, g

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

# Initialize sensor
sensor = mpu6050(0x68)

# Setting up keyboard listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Setup data acquisition
accel_file = open(os.path.join(output_directory,'accel_'+file_name+'.txt'),'a+')
gyro_file = open(os.path.join(output_directory,'gyro_'+file_name+'.txt'),'a+')

keys_mpu = ['x','y','z','datetime']

a_writer = csv.DictWriter(accel_file, fieldnames=keys_mpu)
a_writer.writeheader()

g_writer = csv.DictWriter(gyro_file, fieldnames=keys_mpu)
g_writer.writeheader()

# Setup recording
video_path = os.path.join(output_directory,('video_'+file_name+'.h264'))
mp4_path = os.path.join(output_directory,('video_'+file_name+'.mp4'))
camera = picamera.PiCamera()
camera.resolution = resolution
camera.framerate = framerate
output = customOutput(camera, video_path)

# Start and warmup camera
camera.start_preview()
#time.sleep(1)

# Start recording
print("Acquisition start.")
camera.start_recording(output,format='h264')

while break_bool:
    try:
        pass
        #time.sleep(1/framerate)
        loop_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        a_data, g_data = sensor.get_all_data_fixed(g=True)
        if a_data is None:
            pass
        else:
            a_data['datetime'] = loop_time
            a_writer.writerow(a_data)
        if g_data is None:
            pass
        else:
            g_data['datetime'] = loop_time
            g_writer.writerow(g_data)
        
    except Exception as e:
        print(e, '>>>>>>>>>>>>>>>>')

camera.stop_recording()
if preview_bool == True:
    camera.stop_preview()
output.close()
camera.close()
listener.stop()
accel_file.close()
gyro_file.close()
time_file.close()

sys.exit('Acquisition terminated.')

#cmd = 'MP4Box -add {0}:fps={1} -new {2}'.format(video_path, framerate, mp4_path)
#subprocess.check_call([cmd],shell=True)
#sys.exit("MP4 file created.")