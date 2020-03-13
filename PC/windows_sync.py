# -*- coding: utf-8 -*-
#
# Author: Ruijia Wang <w.ruijia@gmail.com>
### Imports ###
import os
import sys
import random
import datetime
import paramiko

from pynput import mouse
from pynput import keyboard

### User parameters ###
# Server info
server = '192.168.43.3'
username = 'pi'
password = 'helab1'
# Connect to rasbperry pi
print('\nCONNECTION TO SERVER.')
try:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(server, username=username, password=password)
    print('\nCONNECTION ESTABLISHED.')
    print('------------------------')
except:
    print("\nCONNECTION FAILED. WE'LL GET THEM NEXT TIME.")
    print('------------------------')

# Script variables and switches ###
counter = 0
start_bool = False
click_bool = False
ctrl_th = 3

start_time = datetime.datetime.now()

### Functions ###
def on_press(key):
    global counter,start_bool
    # Start mouse listener when CTRL is pressed ctrl_th times
    if key == keyboard.Key.ctrl_l and start_bool == False:
        counter = counter + 1
        if counter == ctrl_th:
            start_bool = True
            counter = 0
            print('\nMOUSE ENGAGED.')
    elif key == keyboard.Key.ctrl_l and start_bool == True:
        counter = counter + 1
        if counter == ctrl_th:
            # Listener terminated
            return False
    else:
        counter = 0

def on_click(x,y,button,pressed):
    global click_bool, start_time
    # Start recording if condition
    if start_bool == True and click_bool == False:
        try:
            cmd = 'export DISPLAY=:0.0 && python3 /home/pi/Desktop/optoRecord_v3.py'
            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(cmd)
            start_time = datetime.datetime.now()
            click_bool = True
        except:
            print('\nCONNECTION WAS NOT ESTABLISHED.')

### Main function ###
def main():
    # Print info for user
    print('\nPress %s times CTRL to start the synchronization script.' % ctrl_th)
    print('\nYour next left click will start the camera recording.')
    print('\nTo end the recording at any time, press CTRL %s times again.' % ctrl_th)

    # Creating listener
    mouse_listener = mouse.Listener(on_click=on_click)
    mouse_listener.start()

    with keyboard.Listener(on_press=on_press) as keyboard_listener:
        keyboard_listener.join()

    mouse_listener.stop()

    # Send stop command to raspi
    try:
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command('export DISPLAY=:0.0 && xdotool key Escape')
        ssh.close()
    except:
        print('\nErrrr... Nothing to close here.')

    # Save time of click
    suffixe = str(int(1000*random.random())) #avoid overwrite
    path = os.path.abspath(os.path.dirname(sys.argv[0]))
    time_str = start_time.strftime("%Y-%m-%d-%H:%M:%S.%f")
    file_name = 'start_time_' + suffixe + '.txt'
    time_file = open(os.path.join(path,file_name),'w')
    time_file.write(time_str)
    time_file.close()

if __name__ == '__main__':
    ### Call main() function
    main()
