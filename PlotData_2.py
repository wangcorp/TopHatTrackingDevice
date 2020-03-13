# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:10:20 2020

@author: Ruijia Wang <w.ruijia@gmail.com>
"""
import os
import time
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

# Global Variables
boundary = {
    'aMin': 0, 'aMax': 0,
    'bMin': 0, 'bMax': 0,
    'posxMin': 0, 'posxMax': 0,
    'posyMin': 0, 'posyMax': 0,
    'posdMin': 0, 'posdMax': 0,
    'aposxMin': 0, 'aposxMax': 0,
    'aposyMin': 0, 'aposyMax': 0,
    'aposdMin': 0, 'aposdMax': 0,    
    'velxMin': 0, 'velxMax': 0,
    'velyMin': 0, 'velyMax': 0,
    'veldMin': 0, 'veldMax': 0,
    'avelxMin': 0, 'avelxMax': 0,
    'avelyMin': 0, 'avelyMax': 0,
    'aveldMin': 0, 'aveldMax': 0,
    'axMin': 0, 'axMax': 0,
    'ayMin': 0, 'ayMin': 0,
    'azMin': 0, 'azMax': 0,
    'gxMin': 0, 'gxMax': 0,
    'gyMin': 0, 'gyMax': 0,
    'gzMin': 0, 'gzMax': 0
    }

# Functions definition
def getBoundary(path):
    '''
        Return min and max of the dataset
    '''
    global boundary
    
    CRP = pd.read_csv(os.path.join(path,'CRP.csv'))
    mpu = pd.read_csv(os.path.join(path,'mpu.csv'))
    
    boundary['aMax'] = np.max(CRP['a_corr'])

    boundary['bMin'] = np.min(CRP['b_corr'])
    boundary['bMax'] = np.max(CRP['b_corr'])
    
    boundary['posxMin'] = np.min(CRP['pupil_x_proj'])
    boundary['posxMax'] = np.max(CRP['pupil_x_proj'])
    
    boundary['posyMin'] = np.min(CRP['pupil_y_proj'])
    boundary['posyMax'] = np.max(CRP['pupil_y_proj'])
    
    boundary['posdMin'] = np.min(CRP['pupil_dist'])
    boundary['posdMax'] = np.max(CRP['pupil_dist'])
    
    boundary['velxMin'] = np.min(CRP['pupil_x_v'])
    boundary['velxMax'] = np.max(CRP['pupil_x_v'])
    
    boundary['velyMin'] = np.min(CRP['pupil_y_v'])
    boundary['velyMax'] = np.max(CRP['pupil_y_v'])
    
    boundary['veldMin'] = np.min(CRP['pupil_v'])
    boundary['veldMax'] = np.max(CRP['pupil_v'])
    
    boundary['aposxMin'] = np.min(CRP['pupil_x_angle'])
    boundary['aposxMax'] = np.max(CRP['pupil_x_angle'])
    
    boundary['aposyMin'] = np.min(CRP['pupil_y_angle'])
    boundary['aposyMax'] = np.max(CRP['pupil_y_angle'])
    
    boundary['aposdMin'] = np.min(CRP['pupil_d_angle'])
    boundary['aposdMax'] = np.max(CRP['pupil_d_angle'])
    
    boundary['avelxMin'] = np.min(CRP['angle_x_v'])
    boundary['avelxMax'] = np.max(CRP['angle_x_v'])
    
    boundary['avelyMin'] = np.min(CRP['angle_y_v'])
    boundary['avelyMax'] = np.max(CRP['angle_y_v'])
    
    boundary['aveldMin'] = np.min(CRP['angle_d'])
    boundary['aveldMax'] = np.max(CRP['angle_d'])
    
    boundary['axMin'] = np.min(mpu['ax'])
    boundary['axMaxn'] = np.max(mpu['ax'])
    
    boundary['ayMin'] = np.min(mpu['ay'])
    boundary['ayMax'] = np.max(mpu['ay'])
    
    boundary['azMin'] = np.min(mpu['az'])
    boundary['azMax'] = np.max(mpu['az'])
    
    boundary['gxMin'] = np.min(mpu['gx'])
    boundary['gxMax'] = np.max(mpu['gx'])
    
    boundary['gyMin'] = np.min(mpu['gy'])
    boundary['gyMax'] = np.max(mpu['gy'])
    
    boundary['gzMin'] = np.min(mpu['gz'])
    boundary['gzMax'] = np.max(mpu['gz'])
    
def loadInfo(path, filename):
    '''
        Retrieve epoch info and associated data
    '''
    # Get epoch info
    freq = filename.split('_')[1]
    
    #load data
    filename_path = os.path.join(path,filename)
    
    event_name = 'event_'+freq+'.csv'
    event = pd.read_csv(os.path.join(filename_path,event_name))
    
    data_name = 'CRP_'+freq+'.csv'
    data = pd.read_csv(os.path.join(filename_path,data_name))
    
    mpu_name = 'mpu_'+freq+'.csv'
    mpu = pd.read_csv(os.path.join(filename_path,mpu_name))

    return event, data, mpu, freq, filename_path

def plotPosition(event, CRP, freq, path):
    '''
        Plot position of the pupil center
    '''
    # Get epoch title
    title = 'Pupil position - Freq: '+freq+'c/d'
    
    # Plot pupil position
    fig, (ax1, ax2) = plt.subplots(2,1,sharex=True)
    
    CRP.plot(ax=ax1,x='Time', y=['pupil_x_proj'],label=['x'],color='dodgerblue',title=title)
    CRP.plot(ax=ax2,x='Time', y=['pupil_y_proj'],label=['y'],color='orangered')
    
    # Set axis limits and label
    ax1.set_ylim(1.1*min(boundary['posxMin'],boundary['posyMin']),1.1*max(boundary['posxMax'],boundary['posyMax']))
    ax2.set_ylim(1.1*min(boundary['posxMin'],boundary['posyMin']),1.1*max(boundary['posxMax'],boundary['posyMax']))
    ax1.set(ylabel='x [mm]')
    ax2.set(ylabel='y [mm]')
    
    # Set timestamp formatting
    formatter = matplotlib.ticker.FuncFormatter(lambda s, x: time.strftime('%M:%S', time.gmtime(s)))
    ax1.xaxis.set_major_formatter(formatter)
    
    # Draw stimulus time
    for idx, row in event.iterrows():
        if row['Direction'] == -1:
            ax1.axvspan(row['Start_Time'], row['End_Time'], facecolor='r', alpha=0.2)
            ax2.axvspan(row['Start_Time'], row['End_Time'], facecolor='r', alpha=0.2)
        elif row['Direction'] == 1:
            ax1.axvspan(row['Start_Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
            ax2.axvspan(row['Start_Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
            
    # Save file
    file_name = 'Position_'+freq
    plt.savefig(os.path.join(path,file_name+'.pdf'), format='pdf')
    plt.savefig(os.path.join(path,file_name+'.png'), format='png')
    plt.close()
    
def plot2DAngle(event, CRP, freq, path):
    '''
        Plot x/y position of the pupil center
    '''
    # Get epoch title
    title = 'Pupil angle - Freq: '+freq
    
    # Plot pupil position
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    
    CRP.plot(ax=ax1, x='Time', y=['pupil_x_angle'],label=[r'$\theta_x$'],color='dodgerblue',title=title)
    CRP.plot(ax=ax2, x='Time', y=['pupil_y_angle'],label=[r'$\theta_y$'],color='orangered')
    
    # Set axis limits and label
    ax1.set_ylim(1.1*min(boundary['aposxMin'],boundary['aposyMin']),1.1*max(boundary['aposxMax'],boundary['aposyMax']))
    ax2.set_ylim(1.1*min(boundary['aposxMin'],boundary['aposyMin']),1.1*max(boundary['aposxMax'],boundary['aposyMax']))
    ax1.set(ylabel=r'$\theta_x$ [deg]')
    ax2.set(ylabel=r'$\theta_y$ [deg]')
    
    # Set timestamp formatting
    formatter = matplotlib.ticker.FuncFormatter(lambda s, x: time.strftime('%M:%S', time.gmtime(s)))
    ax1.xaxis.set_major_formatter(formatter)
    
    # Draw stimulus time
    for idx, row in event.iterrows():
        if row['Direction'] == -1:
            ax1.axvspan(row['Start_Time'], row['End_Time'], facecolor='r', alpha=0.2)
            ax2.axvspan(row['Start_Time'], row['End_Time'], facecolor='r', alpha=0.2)
        elif row['Direction'] == 1:
            ax1.axvspan(row['Start_Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
            ax2.axvspan(row['Start_Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
            
    # Save file
    file_name = 'Angle2D_'+freq
    plt.savefig(os.path.join(path,file_name+'.pdf'), format='pdf')
    plt.savefig(os.path.join(path,file_name+'.png'), format='png')
    plt.close()
    
def plotDistance(event, CRP, freq, path):
    '''
        Plot distance of pupil center to CR
    '''
    # Get epoch title
    title = 'Pupil distance to eye center- Freq: '+freq+'c/d'
    
    # Plot pupil distance
    ax = CRP.plot(x='Time', y='pupil_dist',label='d',title=title)
    
    # Set axis limits and label
    ax.set_ylim(0.9*np.min(boundary['posdMin']),1.1*np.max(boundary['posdMax']))
    plt.ylabel('d [mm]')
    
    # Set timestamp formatting
    formatter = matplotlib.ticker.FuncFormatter(lambda s, x: time.strftime('%M:%S', time.gmtime(s)))
    ax.xaxis.set_major_formatter(formatter)
    
    # Draw stimulus time
    for idx, row in event.iterrows():
        if row['Direction'] == -1:
            ax.axvspan(row['Start_Time'], row['End_Time'], facecolor='r', alpha=0.2)
        elif row['Direction'] == 1:
            ax.axvspan(row['Start_Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
            
    # Save file
    file_name = 'Distance_'+freq
    plt.savefig(os.path.join(path,file_name+'.pdf'), format='pdf')
    plt.savefig(os.path.join(path,file_name+'.png'), format='png')
    plt.close()
    
def plot1DAngle(event, CRP, freq, path):
    '''
        Plot distance of pupil center to CR
    '''
    # Get epoch title
    title = 'Pupil total angle - Freq: '+freq
    
    # Plot pupil distance
    ax = CRP.plot(x='Time', y='pupil_d_angle',label=r'$\theta$',title=title)
    
    # Set axis limits and label
    ax.set_ylim(0.9*np.min(boundary['aposdMin']),1.1*np.max(boundary['aposdMax']))
    plt.ylabel(r'$\theta$ [deg]')
    
    # Set timestamp formatting
    formatter = matplotlib.ticker.FuncFormatter(lambda s, x: time.strftime('%M:%S', time.gmtime(s)))
    ax.xaxis.set_major_formatter(formatter)
    
    # Draw stimulus time
    for idx, row in event.iterrows():
        if row['Direction'] == -1:
            ax.axvspan(row['Start_Time'], row['End_Time'], facecolor='r', alpha=0.2)
        elif row['Direction'] == 1:
            ax.axvspan(row['Start_Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
    
    # Save file
    file_name = 'Angle1D_'+freq
    plt.savefig(os.path.join(path,file_name+'.pdf'), format='pdf')
    plt.savefig(os.path.join(path,file_name+'.png'), format='png')
    plt.close()
    
def plot2DVel(event, CRP, freq, path):
    '''
        Plot x/y velocity of the pupil center
    '''
    # Get epoch title
    title = 'Pupil velocity - Freq: '+freq+'c/d'
    
    # Plot pupil position
    fig, (ax1, ax2) = plt.subplots(2,1,sharex=True)
    
    CRP.plot(ax=ax1,x='Time', y=['pupil_x_v'],label=['$v_x$'],color='dodgerblue',title=title)
    CRP.plot(ax=ax2,x='Time', y=['pupil_y_v'],label=['$v_y$'],color='orangered')
    
    # Set axis limits and label
    ax1.set_ylim(1.1*min(boundary['velxMin'],boundary['velyMin']),1.1*max(boundary['velxMax'],boundary['velyMax']))
    ax2.set_ylim(1.1*min(boundary['velxMin'],boundary['velyMin']),1.1*max(boundary['velxMax'],boundary['velyMax']))
    
    ax1.set(ylabel='$v_x$ [mm/s]')
    ax2.set(ylabel='$v_y$ [mm/s]')
    
    # Set timestamp formatting
    formatter = matplotlib.ticker.FuncFormatter(lambda s, x: time.strftime('%M:%S', time.gmtime(s)))
    ax1.xaxis.set_major_formatter(formatter)
    
    # Draw stimulus time
    for idx, row in event.iterrows():
        if row['Direction'] == -1:
            ax1.axvspan(row['Start_Time'], row['End_Time'], facecolor='r', alpha=0.2)
            ax2.axvspan(row['Start_Time'], row['End_Time'], facecolor='r', alpha=0.2)
        elif row['Direction'] == 1:
            ax1.axvspan(row['Start_Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
            ax2.axvspan(row['Start_Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
    
    # Save file
    file_name = 'Velocity2D_'+freq
    plt.savefig(os.path.join(path,file_name+'.pdf'), format='pdf')
    plt.savefig(os.path.join(path,file_name+'.png'), format='png')
    plt.close()
    
def plot2DAVel(event, CRP, freq, path):
    '''
        Plot x/y velocity of the pupil center
    '''
    # Get epoch title
    title = 'Pupil angular velocity - Freq: '+freq
    
    # Plot pupil position
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    
    CRP.plot(ax=ax1, x='Time', y=['angle_x_v'],label=[r'$\dot{\theta}_x$'],color='dodgerblue',title=title)
    CRP.plot(ax=ax2, x='Time', y=['angle_y_v'],label=[r'$\dot{\theta}_y$'],color='orangered')
    
    # Set axis limits and label
    ax1.set_ylim(1.1*min(boundary['avelxMin'],boundary['avelyMin']),1.1*max(boundary['avelxMax'],boundary['avelyMax']))
    ax2.set_ylim(1.1*min(boundary['avelxMin'],boundary['avelyMin']),1.1*max(boundary['avelxMax'],boundary['avelyMax']))
    
    ax1.set(ylabel=r'$\dot{\theta}_x$ [deg/s]')
    ax2.set(ylabel=r'$\dot{\theta}_y$ [deg/s]')
    
    # Set timestamp formatting
    formatter = matplotlib.ticker.FuncFormatter(lambda s, x: time.strftime('%M:%S', time.gmtime(s)))
    ax1.xaxis.set_major_formatter(formatter)
    
    # Draw stimulus time
    for idx, row in event.iterrows():
        if row['Direction'] == -1:
            ax1.axvspan(row['Start_Time'], row['End_Time'], facecolor='r', alpha=0.2)
            ax2.axvspan(row['Start_Time'], row['End_Time'], facecolor='r', alpha=0.2)
        elif row['Direction'] == 1:
            ax1.axvspan(row['Start_Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
            ax2.axvspan(row['Start_Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
    
    # Save file
    file_name = 'AVelocity2D_'+freq
    plt.savefig(os.path.join(path,file_name+'.pdf'), format='pdf')
    plt.savefig(os.path.join(path,file_name+'.png'), format='png')
    plt.close()
    
def plot1DVel(event, CRP, freq, path):
    '''
        Plot resultant velocity of the pupil center
    '''
    # Get epoch title
    title = 'Pupil total velocity - Freq: '+freq+'c/d'
    
    # Plot pupil position
    ax = CRP.plot(x='Time', y='pupil_v',label='v',title=title)
    
    # Set axis limits and label
    ax.set_ylim(1.1*np.min(boundary['veldMin']),1.1*np.max(boundary['veldMax']))
    plt.ylabel('v [mm/s]')
    
    # Set timestamp formatting
    formatter = matplotlib.ticker.FuncFormatter(lambda s, x: time.strftime('%M:%S', time.gmtime(s)))
    ax.xaxis.set_major_formatter(formatter)
    
    # Draw stimulus time
    for idx, row in event.iterrows():
        if row['Direction'] == -1:
            ax.axvspan(row['Start_Time'], row['End_Time'], facecolor='r', alpha=0.2)
        elif row['Direction'] == 1:
            ax.axvspan(row['Start_Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
    
    # Save file
    file_name = 'Velocity1D_'+freq
    plt.savefig(os.path.join(path,file_name+'.pdf'), format='pdf')
    plt.savefig(os.path.join(path,file_name+'.png'), format='png')
    plt.close()
    
def plot1DAVel(event, CRP, freq, path):
    '''
        Plot resultant velocity of the pupil center
    '''
    # Get epoch title
    title = 'Pupil total angular velocity - Freq: '+freq
    
    # Plot pupil position
    ax = CRP.plot(x='Time', y='angle_d',label=r'$\dot{\theta}$',title=title)
    
    # Set axis limits and label
    ax.set_ylim(1.1*np.min(boundary['aveldMin']),1.1*np.max(boundary['aveldMax']))
    plt.ylabel(r'$\dot{\theta}$ [deg/s]')
    
    # Set timestamp formatting
    formatter = matplotlib.ticker.FuncFormatter(lambda s, x: time.strftime('%M:%S', time.gmtime(s)))
    ax.xaxis.set_major_formatter(formatter)
    
    # Draw stimulus time
    for idx, row in event.iterrows():
        if row['Direction'] == -1:
            ax.axvspan(row['Start_Time'], row['End_Time'], facecolor='r', alpha=0.2)
        elif row['Direction'] == 1:
            ax.axvspan(row['Start_Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
    
    # Save file
    file_name = 'AVelocity1D_'+freq
    plt.savefig(os.path.join(path,file_name+'.pdf'), format='pdf')
    plt.savefig(os.path.join(path,file_name+'.png'), format='png')
    plt.close()
    
def plotSize(event, CRP, freq, path):
    '''
        Plot pupil radius
    '''
    # Get epoch title
    title = 'Pupil radius - Freq: '+freq+'c/d'
    
    # Plot pupil position    
    ax = CRP.plot(x='Time', y='a_corr',label='r',color='dodgerblue',title=title)
    
    # Set axis limits and label
    ax.set_ylim(0.9*np.min(boundary['aMin']),1.1*np.max(boundary['aMax']))
    plt.ylabel('r [mm]')
    
    # Set timestamp formatting
    formatter = matplotlib.ticker.FuncFormatter(lambda s, x: time.strftime('%M:%S', time.gmtime(s)))
    ax.xaxis.set_major_formatter(formatter)
    
    # Draw stimulus time
    for idx, row in event.iterrows():
        if row['Direction'] == -1:
            ax.axvspan(row['Start_Time'], row['End_Time'], facecolor='r', alpha=0.2)
        elif row['Direction'] == 1:
            ax.axvspan(row['Start_Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)

    # Save file
    file_name = 'Size_'+freq
    plt.savefig(os.path.join(path,file_name+'.pdf'), format='pdf')
    plt.savefig(os.path.join(path,file_name+'.png'), format='png')
    plt.close()

def plot3DAccel(event, mpu, freq, path):
    '''
        Plot x/y/z acceleration of the head
    '''
    # Get epoch title
    title = 'Accelerometer linear acceleration - Freq: '+freq+'c/d'
    
    # Plot pupil position
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
    
    mpu.plot(ax=ax1, x='Time', y=['ax'],label=['a_x'],color='dodgerblue',title=title)
    mpu.plot(ax=ax2, x='Time', y=['ay'],label=['a_x'],color='orangered')
    mpu.plot(ax=ax3, x='Time', y=['az'],label=['a_x'],color='forestgreen')
    
    # Set axis limits and label
    ax1.set_ylim(1.1*min(boundary['axMin'],boundary['ayMin'],boundary['azMin']),1.1*max(boundary['axMax'],boundary['ayMax'],boundary['azMax']))
    ax2.set_ylim(1.1*min(boundary['axMin'],boundary['ayMin'],boundary['azMin']),1.1*max(boundary['axMax'],boundary['ayMax'],boundary['azMax']))
    ax3.set_ylim(1.1*min(boundary['axMin'],boundary['ayMin'],boundary['azMin']),1.1*max(boundary['axMax'],boundary['ayMax'],boundary['azMax']))
    
    ax1.set(ylabel=r'$a_x$ [m/s2]')
    ax2.set(ylabel=r'$a_y$ [m/s2]')
    ax3.set(ylabel=r'$a_z$ [m/s2]')
    
    # Set timestamp formatting
    formatter = matplotlib.ticker.FuncFormatter(lambda s, x: time.strftime('%M:%S', time.gmtime(s)))
    ax1.xaxis.set_major_formatter(formatter)
    
    # Draw stimulus time
    for idx, row in event.iterrows():
        if row['Direction'] == -1:
            ax1.axvspan(row['Start_Time'], row['End_Time'], facecolor='r', alpha=0.2)
            ax2.axvspan(row['Start_Time'], row['End_Time'], facecolor='r', alpha=0.2)
            ax3.axvspan(row['Start_Time'], row['End_Time'], facecolor='r', alpha=0.2)
        elif row['Direction'] == 1:
            ax1.axvspan(row['Start_Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
            ax2.axvspan(row['Start_Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
            ax3.axvspan(row['Start_Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
    
    # Save file
    file_name = 'Accel3D_'+freq
    plt.savefig(os.path.join(path,file_name+'.pdf'), format='pdf')
    plt.savefig(os.path.join(path,file_name+'.png'), format='png')
    plt.close()
    
def plot3DGyro(event, mpu, freq, path):
    '''
        Plot x/y/z angular acceleration of the head
    '''
    # Get epoch title
    title = 'Gyroscope angular velocity - Freq: '+freq+'c/d'
    
    # Plot pupil position
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
    
    mpu.plot(ax=ax1,x='Time', y=['gx'],label=['g_x'],color='dodgerblue',title=title)
    mpu.plot(ax=ax2,x='Time', y=['gy'],label=['g_y'],color='orangered')
    mpu.plot(ax=ax3,x='Time', y=['gz'],label=['g_z'],color='forestgreen')
    
    # Set axis limits and label
    ax1.set_ylim(1.1*min(boundary['gxMin'],boundary['gyMin'],boundary['gzMin']),1.1*max(boundary['gxMax'],boundary['gyMax'],boundary['gzMax']))
    ax2.set_ylim(1.1*min(boundary['gxMin'],boundary['gyMin'],boundary['gzMin']),1.1*max(boundary['gxMax'],boundary['gyMax'],boundary['gzMax']))
    ax3.set_ylim(1.1*min(boundary['gxMin'],boundary['gyMin'],boundary['gzMin']),1.1*max(boundary['gxMax'],boundary['gyMax'],boundary['gzMax']))
    
    ax1.set(ylabel=r'$\dot{\theta}$ [deg/s]')
    ax2.set(ylabel=r'$\dot{\theta}$ [deg/s]')
    ax3.set(ylabel=r'$\dot{\theta}$ [deg/s]')
    
    # Set timestamp formatting
    formatter = matplotlib.ticker.FuncFormatter(lambda s, x: time.strftime('%M:%S', time.gmtime(s)))
    ax1.xaxis.set_major_formatter(formatter)
    
    # Draw stimulus time
    for idx, row in event.iterrows():
        if row['Direction'] == -1:
            ax1.axvspan(row['Start_Time'], row['End_Time'], facecolor='r', alpha=0.2)
            ax2.axvspan(row['Start_Time'], row['End_Time'], facecolor='r', alpha=0.2)
            ax3.axvspan(row['Start_Time'], row['End_Time'], facecolor='r', alpha=0.2)
        elif row['Direction'] == 1:
            ax1.axvspan(row['Start_Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
            ax2.axvspan(row['Start_Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
            ax3.axvspan(row['Start_Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
    
    # Save file
    file_name = 'Gyro3D_'+freq
    plt.savefig(os.path.join(path,file_name+'.pdf'), format='pdf')
    plt.savefig(os.path.join(path,file_name+'.png'), format='png')
    plt.close()
    
def runEpoch(result_path):
    for filename in os.listdir(result_path):
        if filename.startswith('Epoch'):
            # Get epoch info
            event, CRP, mpu, freq, path = loadInfo(result_path, filename)
            
            # Plot pupil position
            plotPosition(event, CRP, freq, path)
            plot2DAngle(event, CRP, freq, path)
            plotDistance(event, CRP, freq, path)
            plot1DAngle(event, CRP, freq, path)
            plot2DVel(event, CRP, freq, path)
            plot2DAVel(event, CRP, freq, path)
            plot1DVel(event, CRP, freq, path)
            plot1DAVel(event, CRP, freq, path)
            plotSize(event, CRP, freq, path)
            plot3DAccel(event, mpu, freq, path)
            plot3DGyro(event, mpu, freq, path)
    
def processHalf(result_path):
    '''
        Processing pipeling for low and high protocol
    '''
    # Find data max and min for plot boundaries
    getBoundary(result_path)
    
    # Iterate through epoch
    runEpoch(result_path)
    
# Main function
def main(result_path):
    
    # Iterate over epoch
    plt.ioff()
    
    # Process low protocol
    #processHalf(result_path['low'])
    
    # Process high protocol
    processHalf(result_path['high'])
    print('\nPlot generated.')
    
# Code initialization
if __name__ == '__main__':
    # Parameters
    main_dir = r'C:\Users\HeLab\Documents\Ruijia\Project\EyeTracking\Data\v2\WT4'
    run_name = 'run1'
    low_main_path = os.path.join(main_dir,run_name, 'low')
    high_main_path = os.path.join(main_dir,run_name, 'high')
    main_path = {'low' : low_main_path,
                 'high' : high_main_path}
    
    output_dir = 'RESULT'
    low_output_path = os.path.join(low_main_path, output_dir)
    high_output_path = os.path.join(high_main_path, output_dir) 
    result_path = {'low' : low_output_path,
                   'high' : high_output_path}   
    
    ### Call main() function
    main(result_path)