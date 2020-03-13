# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 16:28:43 2020

@author: Ruijia Wang <w.ruijia@gmail.com>
"""
import os
import pandas as pd

# Functions definition
def createCalibration(main_path):
    '''
        Process data for head tracking
    '''
    # Import accel data
    accel = None
    gyro = None
    
    for filename in os.listdir(main_path):
        if filename.startswith('accel_'):
            if accel is None:
                accel = pd.read_csv(os.path.join(main_path,filename)).drop(columns=['datetime'])
            else:
                temp = pd.read_csv(os.path.join(main_path,filename)).drop(columns=['datetime'])
                accel = pd.concat([accel,temp]).reset_index(drop=True)
        elif filename.startswith('gyro_'):
            if gyro is None:
                gyro = pd.read_csv(os.path.join(main_path,filename)).drop(columns=['datetime'])
            else:
                temp = pd.read_csv(os.path.join(main_path,filename)).drop(columns=['datetime'])
                gyro = pd.concat([gyro,temp]).reset_index(drop=True) 
    
    accel_mean = accel.mean(axis=0).to_frame().T
    accel_path = os.path.join(main_path,'cal_accel.csv')
    accel_mean.to_csv(accel_path,index=False)
    
    gyro_mean = gyro.mean(axis=0).to_frame().T
    gyro_path = os.path.join(main_path,'cal_gyro.csv')
    gyro_mean.to_csv(gyro_path,index=False)
    
# Main function
def main(main_path):
    # Create calibration file
    createCalibration(main_path)
    
    print('\nCalibration files created.')
    
# Code initialization 
if __name__ == '__main__':
    # Parameters
    main_path = r'C:\Users\HeLab\Documents\Ruijia\Project\EyeTracking\Data\v2\Surgery_Calibration'
    
    ### Call main() function
    main(main_path)