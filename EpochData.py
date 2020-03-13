# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:28:02 2019

@author: Ruijia Wang <w.ruijia@gmail.com>
"""
import os
import pandas as pd

# Functions definition
def getEpoch(df, result_path):
    '''
        Extract start and end time of each main event (epoch) 
    '''
    epoch = []
    mini_epoch = {
        'Start_Time': 0,
        'End_Time': 0,
        'Freq': 0,
        'Direction': 0
        }    
    
    for idx, row in df.iterrows():
        if idx == 0:
            mini_epoch['Start_Time'] = row['Time']
            mini_epoch['Freq'] = row['Freq']
            mini_epoch['Direction'] = row['Direction']
        else:
            if idx != len(df)-1:
                if row['Freq'] != df.loc[idx-1,'Freq'] or row['Direction'] != df.loc[idx-1,'Direction']:
                   mini_epoch['End_Time'] = df.loc[idx-1,'End_Time']
                   epoch.append(mini_epoch)
                   mini_epoch = {
                       'Start_Time': row['Time'],
                       'End_Time': 0,
                       'Freq': row['Freq'],
                       'Direction': row['Direction']
                       }
            else:
                if row['Freq'] != df.loc[idx-1,'Freq'] or row['Direction'] != df.loc[idx-1,'Direction']:
                   mini_epoch['End_Time'] = df.loc[idx-1,'End_Time']
                   epoch.append(mini_epoch)
                   mini_epoch = {
                       'Start_Time': row['Time'],
                       'End_Time': row['End_Time'],
                       'Freq': row['Freq'],
                       'Direction': row['Direction']
                       }
                   epoch.append(mini_epoch)
                else:
                    mini_epoch['End_Time'] = row['End_Time']
                    epoch.append(mini_epoch)

    epoch_df = pd.DataFrame(epoch)
    epoch_df.to_csv(os.path.join(result_path,'epoch.csv'),index=False)
    
    return epoch_df

def extractEpoch(epoch_info, event, CRP, result_path):
    '''
        Create snipps of data around each epoch
    '''
    # Save relevant event in the epoch
    freq_check = []
    
    for idx, row in epoch_info.iterrows():
        CRP_epoch = CRP[(CRP['Time'] >= row['Start_Time']-1) & (CRP['Time'] <= row['End_Time']+1)]
        event_epoch = event[(event['Time'] >= row['Start_Time']) & (event['Time'] <= row['End_Time'])]
        
        # Create a folder containing epoch data
        freq_check.append((row['Freq'],row['Direction']))    
        tail = '_' + str(freq_check.count((row['Freq'],row['Direction'])))
            
        epoch_suffix = ('%.3f' % float(row['Freq']))+'_'+str(int(row['Direction']>0)) + tail
        epoch_name = 'Epoch_' + epoch_suffix
        epoch_dir = os.path.join(result_path, epoch_name)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        
        # Write epoch data
        CRP_file = os.path.join(epoch_dir,('CRP_'+epoch_suffix+'.csv'))
        CRP_epoch.to_csv(CRP_file,index=False)
    
        event_file = os.path.join(epoch_dir,('event_'+epoch_suffix+'.csv'))
        event_epoch.to_csv(event_file,index=False)

# Main function
def main(main_path, data_path, video_path, result_path):
    # Import events
    event = pd.read_csv(os.path.join(data_path,'event.csv'))
    event = event.rename(columns={'datetime': 'Time'})
    
    # Import processed data
    CRP = pd.read_csv(os.path.join(result_path, 'CRP.csv'))

    # Get main event interval times
    epoch_info = getEpoch(event, result_path) 
    
    # Extract epoch data
    extractEpoch(epoch_info, event, CRP, result_path)
    print('\nEpoch extracted.')
    
# Code initialization 
if __name__ == '__main__':
    # Parameters
    main_path = r'C:\Users\HeLab\Documents\Ruijia\Project\EyeTracking\Data\Batch_11_04_19\966-1339395-OKR'
    
    data_dir = 'OUT_INFO'
    data_path = os.path.join(main_path, data_dir)
    
    video_dir = 'OUT_VIDEO'
    video_file = 'RUN2_2020-01-17-13-46'
    video_path = os.path.join(main_path, video_dir, video_file)
    
    result_dir = 'RESULT'
    result_path = os.path.join(main_path, result_dir, video_file)
    
    ### Call main() function
    main(main_path, data_path, video_path, result_path)