# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:40:52 2019

@author: Ruijia Wang <w.ruijia@gmail.com>
"""

import os
import datetime
import pandas as pd

def importEvent(csv_file, data_dir):
    '''
        Import the time of stimulation (time of clicks)
    '''
    filename = os.path.join(data_dir,csv_file)
    e_df = pd.read_csv(filename, sep=',', header=None)
    e_df.columns = ['datetime','x','y']
    return e_df

def importLabel(csv_file, data_dir):
    '''
        Import stimulation parameters and time
    '''
    filename = os.path.join(data_dir,csv_file)
    l_df = pd.read_csv(filename, sep='\t', lineterminator='\r')
    return l_df

def importTimestamp(csv_file, data_dir):
    '''
        Import acquisition time of the frames
    '''
    filename = os.path.join(data_dir,csv_file)
    t_df = pd.read_csv(filename, sep=',', header=None)
    t_df.columns = ['chip_time', 'sys_time']
    t_df['chip_time'] = t_df['chip_time'].str[1:]
    t_df['sys_time'] = t_df['sys_time'].str[2:-2]
    t_df = t_df.drop(t_df.index[0]).reset_index(drop=True)
    t_df.loc[0,'chip_time'] = "0"
    t_df = t_df[~t_df['chip_time'].str.contains('None')].reset_index(drop=True)
    return t_df

def setTime(df, col_name, time_0, time_layout):
    '''
        Set all times to zero according to the first frame
    '''
    temp_df = pd.DataFrame(columns=[col_name])
    for idx, row in df.iterrows():
        time = datetime.datetime.strptime(row[col_name], time_layout)
        time_delta = time - time_0
        time_delta = datetime.timedelta(seconds=time_delta.seconds,
                                        microseconds=time_delta.microseconds)
        temp_df = temp_df.append({col_name: time_delta.total_seconds()},ignore_index=True)
    df[col_name] = temp_df
    return df
    
def corrTime(label_df, YES_df, NO_df):
    '''
       Correct the time of the label based on exact time click     
    ''' 
    yes_idx, no_idx = 0, 0
    for idx, row in label_df.iterrows():
        if row['Correct'] == 0:
            label_df.loc[idx,'Time'] = NO_df.loc[no_idx,'datetime']
            no_idx = no_idx + 1
        else:
            label_df.loc[idx,'Time'] = YES_df.loc[yes_idx,'datetime']
            yes_idx = yes_idx + 1
    return label_df

def createLabel(event_df, label_df):
    '''
        Create the label for stimulation
    '''
    df = pd.DataFrame(columns=['Freq','Contrast','Direction','Correct'])
    temp_df = pd.DataFrame(columns=['End_Time'])
    for idx, row in event_df.iterrows():
        try:
            label_idx = label_df[label_df['Time'].gt(row['datetime'])].index[0]
        except:
            label_idx = None
        if label_idx is None:
            df = df.append(pd.Series(), ignore_index=True)
            temp_df = temp_df.append(pd.Series(), ignore_index=True)
        else:
            df = df.append(label_df.loc[label_idx,['Freq','Contrast','Direction','Correct']])
            temp_df = temp_df.append([{'End_Time': label_df.loc[label_idx,'Time']}])
    df.reset_index(drop=True, inplace=True)
    temp_df.reset_index(drop=True, inplace=True)
    df = pd.concat([temp_df,df],axis=1)
    return df
    
def setDeltaStim(event_df):
    '''
        Create time for start and end of stimulation
    '''
    for idx, row in event_df.iterrows():
        if idx + 1 < len(event_df.index):
            if row['datetime'] + 5 < row['End_Time']:
                if row['datetime'] + 5 < event_df.loc[idx+1,'datetime']:
                    event_df.loc[idx,'End_Time'] = row['datetime'] + 5
                else:
                    event_df.loc[idx,'End_Time'] = event_df.loc[idx+1,'datetime'] - 0.15
            else:
                if row['datetime'] + 5 > event_df.loc[idx+1,'datetime']:
                    event_df.loc[idx,'End_Time'] = event_df.loc[idx+1,'datetime'] - 0.15
        else:
            if row['datetime'] + 5 < row['End_Time']:
                event_df.loc[idx,'End_Time'] = row['datetime'] + 5
    return event_df

def main(run_name, main_path, data_path):
    # Import data
    ROI_df = importEvent('ROI.txt', data_path)
    YES_df = importEvent('YES.txt', data_path)
    NO_df = importEvent('NO.txt', data_path)
    label_df = importLabel('log_clean.txt', data_path)
    timestamp_df = importTimestamp('timestamp.txt', data_path)
    
    # Save first timestamp of head-mounted device video
    time_0 = datetime.datetime.strptime(timestamp_df.loc[0,'sys_time'],'%Y-%m-%d-%H-%M-%S.%f')

    # Format dataframes timestamps to seconds after time_0
    timestamp_df = setTime(timestamp_df,'sys_time',time_0,'%Y-%m-%d-%H-%M-%S.%f')
    ROI_df = setTime(ROI_df,'datetime',time_0,'%Y-%m-%d %H:%M:%S.%f')
    YES_df = setTime(YES_df,'datetime',time_0,'%Y-%m-%d %H:%M:%S.%f')
    NO_df = setTime(NO_df,'datetime',time_0,'%Y-%m-%d %H:%M:%S.%f')
    label_df = setTime(label_df,'Time',time_0,'%I:%M:%S %p')
    
    label_df = corrTime(label_df, YES_df, NO_df)
    
    # Assign label to stimulation
    event_label_df = createLabel(ROI_df, label_df)
    event_label_df = pd.concat([ROI_df,event_label_df],axis=1).dropna()
    
    # Create stimulation time interval
    event_label_df = setDeltaStim(event_label_df)

    # Save processed data
    output_path = os.path.join(main_path,'OUT_INFO')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    event_label_df.to_csv(os.path.join(output_path,'event.csv'), index = None, header=True)
    timestamp_df.to_csv(os.path.join(output_path,'timestamp.csv'), index = None, header=True)

# Code initialization
if __name__ == '__main__':
    # Paths
    run_name = '20191104'
    main_path = r'C:\Users\HeLab\Documents\Ruijia\Project\EyeTracking\Data\Batch_11_04_19\intact_20191104-OKR'
    
    data_dir = 'IN_INFO'
    data_path = os.path.join(main_path, data_dir)
    
    ### Call main() function
    main(run_name, main_path, data_path)