# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:30:54 2020

@author: Ruijia Wang <w.ruijia@gmail.com>
"""

import os
import pandas as pd

from datetime import datetime, timedelta

# Functions definition
def importTimestamp(info_path, csv_name):
    '''
        Import acquisition time of the frames
    '''
    filename = os.path.join(info_path,csv_name)
    t_df = pd.read_csv(filename, sep=',', header=None)
    t_df.columns = ['chip_time', 'sys_time']
    t_df['chip_time'] = t_df['chip_time'].str[1:]
    t_df['sys_time'] = t_df['sys_time'].str[2:-2]
    t_df = t_df.drop(t_df.index[0]).reset_index(drop=True)
    t_df.loc[0,'chip_time'] = "0"
    t_df = t_df[~t_df['chip_time'].str.contains('None')].reset_index(drop=True)
    return t_df   

def setTime(df, start_time, time_layout, delay):
    '''
        Set all times according to the start time
    '''
    df_temp = df.copy()
    time_0 = 0
    
    for idx, row in df_temp.iterrows():
        time = datetime.strptime(row['sys_time'], time_layout)
        time_delta = time - start_time
        time_delta = timedelta(seconds=(time_delta.seconds-delay),
                               microseconds=time_delta.microseconds)
        
        if idx == 0:
            time_0 = time_delta
            
        row['sys_time'] = time_delta.total_seconds()
        row['chip_time'] = int(row['chip_time']) + time_0.microseconds + time_0.seconds*1000000
        
    return df_temp

def createTimestamp(info_path,delay):
    '''
        Adjusts video timestamp according to stimulation protocol
    '''
    # Get start time
    file = open(os.path.join(info_path,'start_time.txt'),'r')
    start_time = file.read()
    start_time = datetime.strptime(start_time,"%Y-%m-%d-%H:%M:%S.%f")
    file.close()
    
    # Load and preocess timestamp
    timestamp = importTimestamp(info_path,'timestamp.txt')
    timestamp = setTime(timestamp,start_time,'%Y-%m-%d-%H-%M-%S.%f',delay)

    return timestamp

def importAccel(info_path):
    '''
        Import and adjust timestamps for accelerometer data
    '''
    # Get start time
    file = open(os.path.join(info_path,'start_time.txt'),'r')
    start_time = file.read()
    start_time = datetime.strptime(start_time,"%Y-%m-%d-%H:%M:%S.%f")
    file.close()
    
    # Get data
    df = pd.read_csv(os.path.join(info_path,'accel.txt'),sep=',')
    df_temp = df.copy()
    timestamp = []
    
    for idx, row in df_temp.iterrows():
        # Calculate elapsed time from start
        time = datetime.strptime(row['datetime'], "%Y-%m-%d-%H-%M-%S.%f")
        time_delta = time - start_time
        time_delta = timedelta(seconds=time_delta.seconds,
                               microseconds=time_delta.microseconds)
        
        # Correct timestamp
        timestamp.append(time_delta.total_seconds())
        
    # Rename and reorder columns
    df_temp['Time'] = timestamp
    df_temp = df_temp.drop(['datetime'],axis=1)
    df_temp = df_temp.rename(columns={'x': 'ax', 'y': 'ay', 'z': 'az'})
    return df_temp
    
def importGyro(info_path):
    '''
        Import and adjust timestamps for gyroscope data
    '''
    # Get start time
    file = open(os.path.join(info_path,'start_time.txt'),'r')
    start_time = file.read()
    start_time = datetime.strptime(start_time,"%Y-%m-%d-%H:%M:%S.%f")
    file.close()
    
    # Get data
    df = pd.read_csv(os.path.join(info_path,'gyro.txt'),sep=',')
    df_temp = df.copy()
    timestamp = []
    
    for idx, row in df_temp.iterrows():
        # Calculate elapsed time from start
        time = datetime.strptime(row['datetime'], "%Y-%m-%d-%H-%M-%S.%f")
        time_delta = time - start_time
        time_delta = timedelta(seconds=time_delta.seconds,
                               microseconds=time_delta.microseconds)
        
        # Correct timestamp
        timestamp.append(time_delta.total_seconds())
    
    # Rename and reorder columns
    df_temp['Time'] = timestamp
    df_temp = df_temp.drop(['datetime'],axis=1)
    df_temp = df_temp.rename(columns={'x': 'gx', 'y': 'gy', 'z': 'gz'})
    return df_temp

def importData(main_path,info_path,stim,delay):
    '''
        Import stimulation protocol and create timestamps
    '''
    # Create output directory if not existing
    output_path = os.path.join(main_path,'OUT_INFO')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # Copy corresponding stimulation protocol
    stim.to_csv(os.path.join(output_path,'event.csv'),index=None,header=True)
    
    # Create timestamps
    timestamp = createTimestamp(info_path,delay)
    timestamp.to_csv(os.path.join(output_path,'timestamp.csv'),index=None,header=True)
    
    # Import acceleration data
    accel = importAccel(info_path)
    accel.to_csv(os.path.join(output_path,'accel.csv'),index=None,header=True)
    
    # Import gyroscope data
    gyro = importGyro(info_path)
    gyro.to_csv(os.path.join(output_path,'gyro.csv'),index=None,header=True)
    
# Main function
def main(main_path, info_path, stim_dir,delay):
    # Load stimulation parameters
    low_stim_path = os.path.join(stim_dir,'low_event.csv')
    low_stim = pd.read_csv(low_stim_path)
    
    high_stim_path = os.path.join(stim_dir,'high_event.csv')
    high_stim = pd.read_csv(high_stim_path)
    
    # Import low stim
    importData(main_path['low'],info_path['low'],low_stim,delay)

    # Import high stim
    importData(main_path['high'],info_path['high'],high_stim,delay)

# Code initialization 
if __name__ == '__main__':
    # Paths
    main_dir = r'C:\Users\HeLab\Documents\Ruijia\Project\EyeTracking\Data\v2\WT4'
    run_name = 'run1'
    low_main_path = os.path.join(main_dir,run_name, 'low')
    high_main_path = os.path.join(main_dir,run_name, 'high')
    main_path = {'low' : low_main_path,
                 'high' : high_main_path}
    
    info_dir = 'IN_INFO'
    low_info_path = os.path.join(low_main_path, info_dir)
    high_info_path = os.path.join(high_main_path, info_dir) 
    info_path = {'low' : low_info_path,
                 'high' : high_info_path}
    
    stim_dir = r'C:\Users\HeLab\Documents\Ruijia\Project\EyeTracking\Data\Stim_Param'
    
    delay = 1.0 # Modify if there is a known delay between computer and rapsberry pi
    
    ### Call main() function
    main(main_path, info_path, stim_dir, delay)