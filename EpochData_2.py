# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 17:37:26 2020

@author: Ruijia Wang <w.ruijia@gmail.com>
"""
import os
import pandas as pd

# Functions definition
def getEpoch(df, output_path):
    '''
        Extract start and end time of each main event (epoch) 
    '''
    epoch = []
    mini_epoch = {
        'Start_Time': 0,
        'End_Time': 0,
        'Freq': 0,
        }    
    
    for idx, row in df.iterrows():
        if idx == 0:
            mini_epoch['Start_Time'] = row['Start_Time']
            mini_epoch['Freq'] = df.loc[idx+1,'Freq']
        else:
            if row['Freq'] == 0:
                mini_epoch['End_Time'] = df.loc[idx-1,'End_Time']
                epoch.append(mini_epoch)
                mini_epoch = {
                    'Start_Time': row['Start_Time'],
                    'End_Time': 0,
                    'Freq': df.loc[idx+1,'Freq'],
                    }
            elif idx == len(df)-1:
                mini_epoch['End_Time'] = row['End_Time']
                epoch.append(mini_epoch)

    epoch_df = pd.DataFrame(epoch)
    epoch_df.to_csv(os.path.join(output_path,'epoch.csv'),index=False)
    
    return epoch_df

def extractEpoch(event, epoch, CRP, mpu, output_path):
    '''
        Create snipps of data around each epoch
    '''
    # Save mini epoch data
    for idx, row in epoch.iterrows():
        CRP_epoch = CRP[(CRP['Time'] >= row['Start_Time']) & (CRP['Time'] <= row['End_Time'])]
        event_epoch = event[(event['Start_Time'] >= row['Start_Time']) & (event['Start_Time'] < row['End_Time'])]
        mpu_epoch = mpu[(mpu['Time'] >= row['Start_Time']) & (mpu['Time'] <= row['End_Time'])]
        
        # Create epoch folder
        epoch_suffixe = ('%.3f' % float(row['Freq'])) 
        epoch_name = 'Epoch_' + epoch_suffixe
        epoch_dir = os.path.join(output_path, epoch_name)
        
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        
        # Save epoch data
        # Write epoch data
        CRP_file = os.path.join(epoch_dir,('CRP_'+epoch_suffixe+'.csv'))
        CRP_epoch.to_csv(CRP_file,index=False)
    
        event_file = os.path.join(epoch_dir,('event_'+epoch_suffixe+'.csv'))
        event_epoch.to_csv(event_file,index=False)
        
        mpu_file = os.path.join(epoch_dir,('mpu_'+epoch_suffixe+'.csv'))
        mpu_epoch.to_csv(mpu_file,index=False)

def extractMiniEpoch(event, CRP, mpu, output_path):
    '''
        Create snipps of data around each epoch
    '''
    # Save processed mini event in the epoch
    freq_check = []
    
    # Save data in epoch 
    for idx, row in event.iterrows():
        CRP_epoch = CRP[(CRP['Time'] >= row['Start_Time']) & (CRP['Time'] <= row['End_Time'])]
        event_epoch = event[(event['Start_Time'] >= row['Start_Time']) & (event['Start_Time'] < row['End_Time'])]
        mpu_epoch = mpu[(mpu['Time'] >= row['Start_Time']) & (mpu['Time'] <= row['End_Time'])]
        
        # Create epoch folder
        freq_check.append((row['Freq'],row['Direction']))
        tail = '_' + str(freq_check.count((row['Freq'],row['Direction'])))
            
        epoch_suffix = ('%.3f' % float(row['Freq']))+'_'+str(row['Direction']) + tail
        epoch_name = 'Epoch_' + epoch_suffix
        epoch_dir = os.path.join(output_path, epoch_name)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)
        
        # Write epoch data
        CRP_file = os.path.join(epoch_dir,('CRP_'+epoch_suffix+'.csv'))
        CRP_epoch.to_csv(CRP_file,index=False)
    
        event_file = os.path.join(epoch_dir,('event_'+epoch_suffix+'.csv'))
        event_epoch.to_csv(event_file,index=False)
        
        mpu_file = os.path.join(epoch_dir,('mpu_'+epoch_suffix+'.csv'))
        mpu_epoch.to_csv(mpu_file,index=False)

def processHalf(main_path, info_path, video_path, output_path):
    '''
        Processing pipeling for low and high protocol
    '''
    # Import events
    event = pd.read_csv(os.path.join(info_path,'event.csv'))
    
    # Import processed data
    CRP = pd.read_csv(os.path.join(output_path, 'CRP.csv'))
    mpu = pd.read_csv(os.path.join(output_path, 'mpu.csv'))
    
    # Get main event interval times
    epoch = getEpoch(event, output_path) 
    
    # Extract epoch data
    extractEpoch(event, epoch, CRP, mpu, output_path)

    # for filename in os.listdir(output_path):
    #     if filename.startswith('Epoch_'):
    #         suffixe = filename.split('_')[1]
    #         mini_event = pd.read_csv(os.path.join(output_path,filename,'event_'+suffixe+'.csv'))
    #         mini_CRP = pd.read_csv(os.path.join(output_path,filename,'CRP_'+suffixe+'.csv'))
    #         mini_mpu = pd.read_csv(os.path.join(output_path,filename,'mpu_'+suffixe+'.csv'))
    #         mini_path = os.path.join(output_path,'Epoch_'+suffixe)
            
    #         # Extract mini epoch
    #         extractMiniEpoch(mini_event, mini_CRP,mini_mpu, mini_path)
                
# Main function
def main(main_path, info_path, video_path, output_path):
    # Process low protocol
    processHalf(main_path['low'], info_path['low'], video_path['low'], output_path['low'])
    
    # Process high protocol
    processHalf(main_path['high'], info_path['high'], video_path['high'], output_path['high'])
    
    print('\nEpoch extracted.')
    
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
    
    video_dir = 'OUT_VIDEO'
    low_video_path = os.path.join(low_main_path, video_dir)
    high_video_path = os.path.join(high_main_path, video_dir) 
    video_path = {'low' : low_video_path,
                 'high' : high_video_path}    
    
    output_dir = 'RESULT'
    low_output_path = os.path.join(low_main_path, output_dir)
    high_output_path = os.path.join(high_main_path, output_dir) 
    output_path = {'low' : low_output_path,
                 'high' : high_output_path}   
    
    ### Call main() function
    main(main_path, info_path, video_path, output_path)