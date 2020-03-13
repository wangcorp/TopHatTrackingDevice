# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 17:28:53 2020

@author: Ruijia Wang <w.ruijia@gmail.com>
"""
import os
import csv
import cv2

import numpy as np
import pandas as pd

from datetime import datetime

# Global variables
ROIxmin = 0
ROIymin = 0

release_bool = []

# Functions definition
def processEvent(timestamp, event):
    '''
        Create a dataframe with stimulation info for each frame of the video
    '''
    # Create dataframe backbone
    event_idx = timestamp.copy()
    event_idx['Stimulation'] = False
    event_idx['Frequency'] = np.nan
    event_idx['Direction'] = np.nan
    
    # Fill dataframe
    for idx, row in event.iterrows():
        if row['Freq'] != 0:
            event_idx.loc[(event_idx['Time'] >= row['Start_Time']) &
                          (event_idx['Time'] <= row['End_Time']),
                          ['Stimulation','Frequency','Direction']] = [True,row['Freq'],row['Direction']]
    
    return event_idx

def updateROI(info):
    '''
        Update coordinate of past ROI to adjust global frame
    '''
    global ROIxmin, ROIymin
    
    ROIxmin = float(info['ROI_x_min'])
    ROIymin = float(info['ROI_y_min'])

def createOutputFull(cap, save_path):
    '''
        Create video output to save frames
    '''
    # Get frame info for video capture
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = 42

    # Save video
    video_name = 'full_corrected.avi'
    out = cv2.VideoWriter(os.path.join(save_path,video_name),
                          cv2.VideoWriter_fourcc('M','J','P','G'),
                          fps, (frame_width,frame_height))
    return out

def createOutputEpoch(cap, epoch, save_path):
    '''
        Create list of video output based on epoch
    '''
    # Get frame info for video capture
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = 42
    
    # Create list of output
    out_epoch = []

    for idx, row in epoch.iterrows():
        epoch_suffix = ('%.3f' % float(row['Freq']))
        video_name = 'video_'+epoch_suffix+'.avi'
        video_path = os.path.join(save_path,'Epoch_'+epoch_suffix)
        out_epoch.append(
            cv2.VideoWriter(os.path.join(video_path,video_name),
                          cv2.VideoWriter_fourcc('M','J','P','G'),
                          fps, (frame_width,frame_height)))
    return out_epoch

def setReleaseBool(epoch):
    '''
        Create release bool logic array
    '''
    global release_bool
    
    # Array of bool to control release of frame capture
    release_bool = np.zeros((len(epoch),1),dtype=bool)
    
def preProcessing(frame):
    '''
        Rotation and gray-scale of the frame
    '''
    # Rotation of the frame
    frame = cv2.flip(frame, 0)
    
    # Gray Grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Back-conversion to color for further overlay
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    return frame

def createEllipse(data):
    '''
        Create and adjust the ellipse coordinates to the uncropped window
    '''
    temp = [[0,0],[0,0],0]
    temp[0] = (data[1]+ROIxmin,data[2]+ROIymin)
    temp[1] = (data[3]*2,data[4]*2) #semi-axis stored
    temp[2] = data[5]
    ellipse = tuple(temp)
    
    return ellipse

def drawCR(frame, data):
    '''
        Draw crosshair for CR tracking
    '''
    # Draw crosshair in global frame
    frame = cv2.line(frame,
                    (int(data[6]+ROIxmin-3),
                     int(data[7]+ROIymin)),
                    (int(data[6]+ROIxmin+3),
                     int(data[7]+ROIymin)),
                     (255,0,0),1)
    
    frame = cv2.line(frame,
                    (int(data[6]+ROIxmin),
                     int(data[7]+ROIymin-3)),
                    (int(data[6]+ROIxmin),
                     int(data[7]+ROIymin+3)),
                     (255,0,0),1)

    return frame

def drawEvent(frame, event, data):
    '''
        Draw event info on the video
    '''
    # Draw timestamp
    dt = datetime.utcfromtimestamp(data[0]).strftime('%M:%S.%f')[:8]
    cv2.putText(frame,dt,(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
    
    # Draw stimulation time
    if event[1] == False:
        text1_1 = 'Stimulation'
        text1_2 = 'OFF'
        cv2.putText(frame,text1_1,(1000,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
        cv2.putText(frame,text1_2,(1190,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
        
    if event[1] == True:
        text1_1 = 'Stimulation'
        text1_2 = 'ON'
        text2_1 = 'Frequency:'
        text2_2 = ('%.3f' % float(event[2]))
        
        if event[3] == 1:
            text3_1 = 'Direction:'
            text3_2 = 'CW'
            color = (0,255,0)
        else:
            text3_1 = 'Direction:'
            text3_2 = 'CCW'
            color = (255,0,0)
            
        cv2.putText(frame,text1_1,(1000,40),cv2.FONT_HERSHEY_SIMPLEX,1,color,2,cv2.LINE_AA)
        cv2.putText(frame,text1_2,(1190,40),cv2.FONT_HERSHEY_SIMPLEX,1,color,2,cv2.LINE_AA)
                
        cv2.putText(frame,text2_1,(1000,90),cv2.FONT_HERSHEY_SIMPLEX,1,color,2,cv2.LINE_AA)
        cv2.putText(frame,text2_2,(1190,90),cv2.FONT_HERSHEY_SIMPLEX,1,color,2,cv2.LINE_AA)
        
        cv2.putText(frame,text3_1,(1000,140),cv2.FONT_HERSHEY_SIMPLEX,1,color,2,cv2.LINE_AA)
        cv2.putText(frame,text3_2,(1190,140),cv2.FONT_HERSHEY_SIMPLEX,1,color,2,cv2.LINE_AA)
    return frame

def manageEpochWriter(frame, frame_time, epoch, out_epoch):
    '''
        Manage start and end of epoch video writing
    '''
    global release_bool
    
    # Iterate over epoch to write corresponding frame into corresponding epoch
    for idx, row in epoch.iterrows():
        if (frame_time >= row['Start_Time']) & (frame_time <= row['End_Time']):
            out_epoch[idx].write(frame)
        if (frame_time > row['End_Time']) & (release_bool[idx] == False):
            out_epoch[idx].release
            release_bool[idx] = True
            
def epochVideo(video_path, result_path, info, timestamp, epoch, event_dt, CRP):
    '''
        Create snippset of the video for each stimulation epoch (with overlay)
    '''
     # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Create video feed acquisition
    out_full = createOutputFull(cap, result_path)
    out_epoch = createOutputEpoch(cap, epoch, result_path)
    
    # Iterate over frames
    timestamp_length = len(timestamp)
    frame_idx = 0
    while(cap.isOpened()):
        ret, frame = cap.read()

        # Stop the loop if all frames have been played
        if not ret or frame_idx == timestamp_length:
            # if frame_idx == 0:
            #     print('\nVideo file empty.')
            # print('\nEpoching finished.')
            break
        
        # Light processing of the frame
        frame = preProcessing(frame)

        # Extraction of ellipse and CR data at given frame
        # [Time, xc, yc, a, b, theta, x, y]
        data_idx = CRP.loc[frame_idx].values.tolist()
        
        # Extraction of event at given frame
        # [Time, Stimulation, Frequency, Direction]
        event_idx = event_dt.loc[frame_idx].values.tolist()

        # Extraction of time at given frame
        frame_time = round(timestamp.iloc[frame_idx]['Time'],6)
        
        # Create  and draw ellipse and correct for global frame
        ellipse = createEllipse(data_idx)
        cv2.ellipse(frame,ellipse,(0,0,255),1)
        
        # Draw CR crosshair
        frame = drawCR(frame, data_idx)
        
        # Draw stimulation trigger and info
        frame = drawEvent(frame,event_idx, data_idx)
        
        # Show frame
        # cv2.imshow('ddd',frame)
        
        # Write frames in full video
        out_full.write(frame)
        
        # Write frames in epoch videos
        manageEpochWriter(frame, frame_time, epoch, out_epoch)
        
        # Update frame idx for next iteration
        frame_idx = frame_idx + 1
        
        # Press Esc to exit the window
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            print('Tracking manually terminated.')
            break
    
    cap.release
    cv2.destroyAllWindows()
    out_full.release
    return frame_idx

def processHalf(main_path, info_path, video_path, data_path, result_path):
    '''
        Processing pipeling for low and high protocol
    '''
    # Import video info
    with open(os.path.join(data_path,'Info.csv'),mode='r') as f:
        reader = csv.reader(f)
        info = {row[0]:row[1] for row in reader}

    # Import timestamps
    timestamp = pd.read_csv(os.path.join(info_path,'timestamp.csv')).drop(columns=['chip_time'])
    timestamp = timestamp.rename(columns={'sys_time': 'Time'})
    
    # Import events
    event = pd.read_csv(os.path.join(info_path,'event.csv'))
    
    # Import epoch
    epoch = pd.read_csv(os.path.join(result_path,'epoch.csv'))
    setReleaseBool(epoch)

    # Import corrected pupil and CR data
    CRP_large = pd.read_csv(os.path.join(result_path,'CRP.csv'))
    CRP = CRP_large[['Time','xc','yc','a','b','theta','x','y']].copy()
    
    # Create event info list per frame
    event_dt = processEvent(timestamp, event)
    
    # Update past ROI coordinate
    updateROI(info)
    
    # Generate overlay and video epoch
    frame_idx = epochVideo(video_path, result_path, info, timestamp, epoch, event_dt, CRP)
    if frame_idx == 0:
        print('\nVideo file empty.')

# Main function
def main(main_path, info_path, video_path, data_path, result_path):        
    # Process low protocol
    #processHalf(main_path['low'], info_path['low'], video_path['low'], data_path['low'],  result_path['low'])
    
    # Process high protocol
    processHalf(main_path['high'], info_path['high'], video_path['high'], data_path['high'], result_path['high'])
    print('\nEpoching finished.')
    
# Code initialization 
if __name__ == '__main__':
    # Parameters
    main_dir = r'C:\Users\HeLab\Documents\Ruijia\Project\EyeTracking\Data\v2\WT4'
    run_name = 'run1'
    low_main_path = os.path.join(main_dir,run_name, 'low')
    high_main_path = os.path.join(main_dir,run_name, 'high')
    main_path = {'low' : low_main_path,
                 'high' : high_main_path}
    
    info_dir = 'OUT_INFO'
    low_info_path = os.path.join(low_main_path, info_dir)
    high_info_path = os.path.join(high_main_path, info_dir) 
    info_path = {'low' : low_info_path,
                 'high' : high_info_path}
    
    video_dir = 'IN_VIDEO'     
    video_name = 'video.h264'
    low_video_path = os.path.join(low_main_path, video_dir, video_name)
    high_video_path = os.path.join(high_main_path,video_dir,video_name)
    video_path = {'low' : low_video_path,
                  'high' : high_video_path}
    
    data_dir = 'OUT_VIDEO'
    low_data_path = os.path.join(low_main_path, data_dir)
    high_data_path = os.path.join(high_main_path, data_dir) 
    data_path = {'low' : low_data_path,
                 'high' : high_data_path}    
    
    output_dir = 'RESULT'
    low_output_path = os.path.join(low_main_path, output_dir)
    high_output_path = os.path.join(high_main_path, output_dir) 
    result_path = {'low' : low_output_path,
                   'high' : high_output_path}   
    
    ### Call main() function
    main(main_path, info_path, video_path, data_path, result_path)