# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:29:41 2020

@author: Ruijia Wang <w.ruijia@gmail.com>
"""
import os
import csv
import json
import numpy as np
import pandas as pd

# Functions definition
def findOutliers(df, col1, col2, std_coeff, fill=False):
    '''
        Correct and interpolate outliers
    ''' 
    # Convert null entries to numpy nan
    temp_df = df.loc[:, df.columns!='Time'].copy().replace(0,np.nan)
    
    # Get relative pupil coordinates stat
    mean_x = np.nanmean(temp_df[col1])
    mean_y = np.nanmean(temp_df[col2])
    
    std_x = np.nanstd(temp_df[col1])
    std_y = np.nanstd(temp_df[col2])

    # Select outliers
    outliers_x = df[abs(temp_df[col1] - mean_x) > std_coeff*std_x].index.to_list()
    outliers_y = df[abs(temp_df[col2] - mean_y) > std_coeff*std_y].index.to_list()
    
    outliers = np.unique(outliers_x + outliers_y)
    
    # Interpolate outliers
    temp_df.loc[outliers] = np.nan
    
    if fill == True:
        temp_df = temp_df.interpolate()
    
    df_out = df.loc[:,df.columns=='Time'].copy()
    df_out = pd.concat([df_out,temp_df],axis=1)
    return df_out

def cleanDataframe(df):
    '''
        Interpolate 0 values and clean df head and tail
    '''
    temp_df = df.loc[:, df.columns!='Time'].copy()
    temp_df = temp_df.replace(0,np.nan).interpolate()
    head = temp_df.first_valid_index()
    tail = temp_df.last_valid_index()
    
    if head != 0:
        top_temp = temp_df.head(int(np.floor(len(temp_df)/2))).copy()
        idx, _ = np.where(pd.isnull(top_temp))
        for i in np.unique(idx):
            top_temp.loc[i] = temp_df.loc[head]
        temp_df.update(top_temp)
    
    if tail < len(df)-1:
        tail_temp = temp_df.tail(int(np.floor(len(temp_df)/2))).copy()
        idx, _ = np.where(pd.isnull(tail_temp))
        for i in np.unique(idx):
            tail_temp.loc[i] = temp_df.loc[tail]
        temp_df.update(tail_temp)
        
    df_out = df.copy()
    df_out.update(temp_df)
    return df_out

def adjustEyeCenter(pupil, CR, info, cal):
    '''
        Get relative position of pupil to CR
    '''
    df_rel = pd.concat([pupil, CR[['x','y']]],axis=1)

    CR_x = df_rel.loc[cal['frame_idx'],'x']
    CR_y = df_rel.loc[cal['frame_idx'],'y']
    
    d_x = (cal['C_x'] - int(info['ROI_x_min'])) - CR_x
    d_y = (cal['C_y'] - int(info['ROI_y_min'])) - CR_y
    
    # df_rel['cx'] = (df_rel['xc'] - df_rel['x'])
    # df_rel['cy'] = (df_rel['yc'] - df_rel['y'])
    
    df_rel['eye_center_x'] = (df_rel['x'] + d_x)
    df_rel['eye_center_y'] = (df_rel['y'] + d_y)
    
    return df_rel

def checkSign(point,center,coord_type):
    '''
        Return sign of distance based on type of coodinate
    '''
    # Handle x case
    if coord_type == 'x':
        if point[0] < center[0]:
            sign = -1
            return sign
        else:
            sign = 1
            return sign
    if coord_type == 'y':
        # Inversion of y axis
        if point[1] < center[1]:
            sign = 1
            return sign
        else:
            sign = -1
            return sign

def adjustCenter(df, info, cal):
    '''
        Get relative position of pupil to CR
    '''
    df_temp = df.copy()
    # Find distance of pupil center to each of the eye axis (projection)
    # Position x is the distance to the y axis
    pupil_x_proj = []
    
    P_x1_adj = cal['Perp_x1'] - int(info['ROI_x_min'])
    P_y1_adj = cal['Perp_y1'] - int(info['ROI_y_min'])
    P1 = (P_x1_adj,P_y1_adj)
    P1 = np.asarray(P1)
    
    P_x2_adj = cal['Perp_x2'] - int(info['ROI_x_min'])
    P_y2_adj = cal['Perp_y2'] - int(info['ROI_y_min'])
    P2 = (P_x2_adj,P_y2_adj)
    P2 = np.asarray(P2)
    
    # Calculate eye radius (average radius of adult mouse = 1.6mm)
    radius = np.linalg.norm(P1-P2)/2
    scale = 1.6/radius
    
    for idx, row in df.iterrows():
        # Calculate projection distance
        P3 = (row['xc'],row['yc'])
        P3 = np.asarray(P3)
        d = np.linalg.norm(np.cross(P2-P1,P1-P3))/np.linalg.norm(P2-P1)
        
        # Check sign
        eye_center = [row['eye_center_x'],row['eye_center_y']]        
        sign = checkSign(P3,eye_center,'x')
        
        # Append distance
        pupil_x_proj.append(sign*d*scale)
    
    df_temp['pupil_x_proj'] = pupil_x_proj
    
    # Position y is the distance to the x axis
    pupil_y_proj = []
    
    x1_adj = cal['ROI_x1'] - int(info['ROI_x_min'])
    y1_adj = cal['ROI_y1'] - int(info['ROI_y_min'])
    P1 = (x1_adj,y1_adj)
    P1 = np.asarray(P1)
    
    x2_adj = cal['ROI_x2'] - int(info['ROI_x_min'])
    y2_adj = cal['ROI_y2'] - int(info['ROI_y_min'])
    P2 = (x2_adj,y2_adj)
    P2 = np.asarray(P2)
    
    for idx, row in df.iterrows():
        P3 = (row['xc'],row['yc'])
        P3 = np.asarray(P3)
        d = np.linalg.norm(np.cross(P2-P1,P1-P3))/np.linalg.norm(P2-P1)
        
        # Check sign
        eye_center = [row['eye_center_x'],row['eye_center_y']]        
        sign = checkSign(P3,eye_center,'y')
        
        pupil_y_proj.append(sign*d*scale)
    
    df_temp['pupil_y_proj'] = pupil_y_proj
    
    return df_temp
    
def applyWindow(df):
    '''
        Apply moving window average on the data (low-pass filter)
    '''
    temp_df = df[['xc','yc','a','b','x','y','eye_center_x','eye_center_y','pupil_x_proj','pupil_y_proj']].copy()
    temp_df = temp_df.rolling(5, center=True, min_periods=1).mean()
    
    df_out = df.copy()
    df_out.update(temp_df)
    return df_out
 
def corrSize(df,info,cal):
    '''
        Correct pupil size to mm
    '''    
    # Get eye contour points
    P_x1_adj = cal['Perp_x1'] - int(info['ROI_x_min'])
    P_y1_adj = cal['Perp_y1'] - int(info['ROI_y_min'])
    P1 = (P_x1_adj,P_y1_adj)
    P1 = np.asarray(P1)
    
    P_x2_adj = cal['Perp_x2'] - int(info['ROI_x_min'])
    P_y2_adj = cal['Perp_y2'] - int(info['ROI_y_min'])
    P2 = (P_x2_adj,P_y2_adj)
    P2 = np.asarray(P2)
    
    # Calculate eye radius (average radius of adult mouse = 1.6mm)
    radius = np.linalg.norm(P1-P2)/2
    scale = 1.6/radius
    
    # Change pupil size
    df_temp = df.copy()
    df_temp['a_corr'] = df_temp['a']
    df_temp['b_corr'] = df_temp['b']
    
    df_temp['a_corr'] *= scale
    df_temp['b_corr'] *= scale
    
    return df_temp
    
def getDistance(df):
    '''
        Calculate distance of pupil center to eye center
    '''    
    temp_df = df[['pupil_x_proj','pupil_y_proj']].copy()
    temp_df['pupil_dist'] = np.sqrt(np.power(temp_df['pupil_x_proj'],2)+np.power(temp_df['pupil_y_proj'],2))
    
    temp_df = temp_df.drop(['pupil_x_proj','pupil_y_proj'],axis=1)
    df_out = pd.concat([df,temp_df],axis=1)
    return df_out
    
def getAngle(df,cal,info):
    temp_df = df.copy()

    # Calculate eye radius
    x1_adj = cal['ROI_x1'] - int(info['ROI_x_min'])
    y1_adj = cal['ROI_y1'] - int(info['ROI_y_min'])
    P1 = (x1_adj,y1_adj)
    P1 = np.asarray(P1)
    
    x2_adj = cal['ROI_x2'] - int(info['ROI_x_min'])
    y2_adj = cal['ROI_y2'] - int(info['ROI_y_min'])
    P2 = (x2_adj,y2_adj)
    P2 = np.asarray(P2)
    
    radius = np.linalg.norm(P1-P2)/2
    scale = 1.6/radius
    radius = radius* scale
    
    # Get angle
    pupil_x_angle = []
    pupil_y_angle = []
    pupil_d_angle = []
    
    for idx, row in temp_df.iterrows():
        angle_x = np.arctan(row['pupil_x_proj']/radius)
        angle_y = np.arctan(row['pupil_y_proj']/radius)
        angle_d = np.arctan(row['pupil_dist']/radius)
        
        pupil_x_angle.append(angle_x*(360/(2*np.pi)))
        pupil_y_angle.append(angle_y*(360/(2*np.pi)))
        pupil_d_angle.append(angle_d*(360/(2*np.pi)))
        
    temp_df['pupil_x_angle'] = pupil_x_angle
    temp_df['pupil_y_angle'] = pupil_y_angle
    temp_df['pupil_d_angle'] = pupil_d_angle
    
    return temp_df
        
def getVelocity(df):
    '''
        Calculate velocity in x, y and total velocity of the pupil
        relatively to the CR center
    '''
    # Compute change of time and position
    temp_df = df[['Time','pupil_x_proj','pupil_y_proj','pupil_x_angle','pupil_y_angle','pupil_d_angle']].copy()
    temp_df = temp_df.diff().fillna(0)
    
    # Calculate linear speed in x and y [mm/s]
    temp_df['pupil_x_v'] = temp_df['pupil_x_proj']/temp_df['Time']
    temp_df['pupil_y_v'] = temp_df['pupil_y_proj']/temp_df['Time']

    # Calculate resultant speed
    temp_df['pupil_v'] = temp_df['pupil_y_v']/np.sin(np.arctan(temp_df['pupil_y_v']/temp_df['pupil_x_v']))
    temp_df = temp_df.fillna(0)
    
    # Calculate angular speed in x 
    temp_df['angle_x_v'] = temp_df['pupil_x_angle']/temp_df['Time']
    temp_df['angle_y_v'] = temp_df['pupil_y_angle']/temp_df['Time']
    temp_df['angle_d'] = temp_df['pupil_d_angle']/temp_df['Time']
    
    # Clean out
    temp_df = temp_df.drop(['Time','pupil_x_proj','pupil_y_proj','pupil_x_angle','pupil_y_angle','pupil_d_angle'],axis=1).fillna(0)
    
    df_out = pd.concat([df,temp_df],axis=1)
    return df_out

# Main function
def main(main_path, data_path, video_path, output_directory):
    # Import timestamps
    timestamp = pd.read_csv(os.path.join(data_path,'timestamp.csv')).drop(columns=['chip_time'])
    timestamp = timestamp.rename(columns={'sys_time': 'Time'})
    
    # Import eye tracking    
    CR_raw = pd.read_csv(os.path.join(video_path,'CR_raw.csv'))
    CR_raw = pd.concat([timestamp, CR_raw],axis=1)
    
    pupil_raw = pd.read_csv(os.path.join(video_path,'Pupil_raw.csv'))
    pupil_raw = pd.concat([timestamp, pupil_raw],axis=1)

    # Import calibration dict
    with open(os.path.join(main_path,'calibration.txt')) as file:
        cal = json.load(file)
        
    # Import video info
    with open(os.path.join(video_path,'Info.csv'),mode='r') as f:
        reader = csv.reader(f)
        info = {row[0]:row[1] for row in reader}
        
    # Remove outliers
    CR_clean = findOutliers(CR_raw,'x','y', 2)
    pupil_clean = findOutliers(pupil_raw,'xc','yc', 4)
    
    # Correct missing values
    CR_fill = cleanDataframe(CR_clean)
    pupil_fill = cleanDataframe(pupil_clean)
    
    # Get center of eye relative to CR
    CRP = adjustEyeCenter(pupil_fill, CR_fill, info, cal)

    # Get relative position of pupil to eye center coordinate
    CRP_proj = adjustCenter(CRP, info, cal)

    # Correct outliers based on relative position top eye axis
    CRP_clean = findOutliers(CRP_proj,'pupil_x_proj','pupil_y_proj',4, fill=True)   

    # Denoising
    CRP_smooth = applyWindow(CRP_clean)

    # Correct pupil size unit
    CRP_size = corrSize(CRP_smooth, info, cal)

    # Get distance data to eye center
    CRP_dist = getDistance(CRP_size)

    # Get angle conversion
    CRP_angle = getAngle(CRP_dist,cal,info)

    # Get acceleration data
    CRP_velocity = getVelocity(CRP_angle)
   
    # Drop oversampled frames with no timestamp (if necessary)
    CRP_velocity.dropna(inplace=True)
    
    # Save processed Data
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    output_file = os.path.join(output_directory, 'CRP.csv')
    CRP_velocity.to_csv(output_file,index=False)
    print('\nData processed.')
    
# Code initialization 
if __name__ == '__main__':
    # Parameters
    main_path = r'C:\Users\HeLab\Documents\Ruijia\Project\EyeTracking\Data\v1\Batch\Batch_11_26_19\995-1339396-OKR'
    
    data_dir = 'OUT_INFO'
    data_path = os.path.join(main_path, data_dir)
    
    video_dir = 'OUT_VIDEO'
    video_file = 'RUN2_2020-01-20-11-07'
    video_path = os.path.join(main_path, video_dir, video_file)
    
    output_dir = 'RESULT'
    output_directory = os.path.join(main_path, output_dir, video_file)
    
    ### Call main() function
    main(main_path, data_path, video_path, output_directory)