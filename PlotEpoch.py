# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:22:27 2020

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
    }

# Functions definition
def getBoundary(path):
    '''
        Return min and max of the dataset
    '''
    global boundary
    
    CRP = pd.read_csv(os.path.join(path,'CRP.csv'))
    
    boundary['aMin'] = np.min(CRP['a_corr'])
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
    
def loadInfo(path, filename):
    '''
        Retrieve epoch info and associated data
    '''
    # Get epoch info
    freq = filename.split('_')[1]
    direction = filename.split('_')[2]
    tail = filename.split('_')[3]
    epoch_info = [freq,direction]
    
    #load data
    filename_path = os.path.join(path,filename)
    
    event_name = 'event_'+freq+'_'+direction+'_'+tail+'.csv'
    event = pd.read_csv(os.path.join(filename_path,event_name))
    
    data_name = 'CRP_'+freq+'_'+direction+'_'+tail+'.csv'
    data = pd.read_csv(os.path.join(path,filename,data_name))

    return event, data, epoch_info, filename_path

def plotPosition(event, CRP, info, path):
    '''
        Plot x/y position of the pupil center
    '''
    # Get epoch title
    if int(info[1]) == 1:
        direction = 'CW'
    else:
        direction = 'CCW'
    title = 'Pupil position - Freq: '+info[0]+'c/d & Dir: '+direction
    
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
            ax1.axvspan(row['Time'], row['End_Time'], facecolor='r', alpha=0.2)
            ax2.axvspan(row['Time'], row['End_Time'], facecolor='r', alpha=0.2)
        else:
            ax1.axvspan(row['Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
            ax2.axvspan(row['Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
            
    
def plot2DAngle(event, CRP, info, path):
    '''
        Plot x/y position of the pupil center
    '''
    # Get epoch title
    if int(info[1]) == 1:
        direction = 'CW'
    else:
        direction = 'CCW'
    title = 'Pupil angle - Freq: '+info[0]+'c/d & Dir: '+direction
    
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
            ax1.axvspan(row['Time'], row['End_Time'], facecolor='r', alpha=0.2)
            ax2.axvspan(row['Time'], row['End_Time'], facecolor='r', alpha=0.2)
        else:
            ax1.axvspan(row['Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
            ax2.axvspan(row['Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
    
def plotDistance(event, CRP, info, path):
    '''
        Plot distance of pupil
    '''
    # Get epoch title
    if int(info[1]) == 1:
        direction = 'CW'
    else:
        direction = 'CCW'
    title = 'Pupil distance to eye center - Freq: '+info[0]+'c/d & Dir: '+direction
    
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
            ax.axvspan(row['Time'], row['End_Time'], facecolor='r', alpha=0.2)
        if row['Direction'] == 1:
            ax.axvspan(row['Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
    
def plot1DAngle(event, CRP, info, path):
    '''
        Plot distance of pupil center to CR
    '''
    # Get epoch title
    if int(info[1]) == 1:
        direction = 'CW'
    else:
        direction = 'CCW'
    title = 'Pupil total angle - Freq: '+info[0]+'c/d & Dir: '+direction
    
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
            ax.axvspan(row['Time'], row['End_Time'], facecolor='r', alpha=0.2)
        if row['Direction'] == 1:
            ax.axvspan(row['Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
    
def plot2DVel(event, CRP, info, path):
    '''
        Plot x/y velocity of the pupil center
    '''
    # Get epoch title
    if int(info[1]) == 1:
        direction = 'CW'
    else:
        direction = 'CCW'
    title = 'Pupil velocity - Freq: '+info[0]+'c/d & Dir: '+direction
    
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
            ax1.axvspan(row['Time'], row['End_Time'], facecolor='r', alpha=0.2)
            ax2.axvspan(row['Time'], row['End_Time'], facecolor='r', alpha=0.2)
        else:
            ax1.axvspan(row['Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
            ax2.axvspan(row['Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
    
def plot2DAVel(event, CRP, info, path):
    '''
        Plot x/y velocity of the pupil center
    '''
    # Get epoch title
    if int(info[1]) == 1:
        direction = 'CW'
    else:
        direction = 'CCW'
    title = 'Pupil angular velocity - Freq: '+info[0]+'c/d & Dir: '+direction
    
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
            ax1.axvspan(row['Time'], row['End_Time'], facecolor='r', alpha=0.2)
            ax2.axvspan(row['Time'], row['End_Time'], facecolor='r', alpha=0.2)
        else:
            ax1.axvspan(row['Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
            ax2.axvspan(row['Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
    
def plot1DVel(event, CRP, info, path):
    '''
        Plot resultant velocity of the pupil center
    '''
    # Get epoch title
    if int(info[1]) == 1:
        direction = 'CW'
    else:
        direction = 'CCW'
    title = 'Pupil total velocity - Freq: '+info[0]+'c/d & Dir: '+direction
    
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
            ax.axvspan(row['Time'], row['End_Time'], facecolor='r', alpha=0.2)
        else:
            ax.axvspan(row['Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
    
def plot1DAVel(event, CRP, info, path):
    '''
        Plot resultant velocity of the pupil center
    '''
    # Get epoch title
    if int(info[1]) == 1:
        direction = 'CW'
    else:
        direction = 'CCW'
    title = 'Pupil total angular velocity - Freq: '+info[0]+'c/d & Dir: '+direction
    
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
            ax.axvspan(row['Time'], row['End_Time'], facecolor='r', alpha=0.2)
        else:
            ax.axvspan(row['Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)
    
def plotSize(event, CRP, info, path):
    '''
        Plot pupil radius
    '''
    # Get epoch title
    if int(info[1]) == 1:
        direction = 'CW'
    else:
        direction = 'CCW'
    title = 'Pupil radius - Freq.: '+info[0]+'c/d & Dir.: '+direction
    
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
            ax.axvspan(row['Time'], row['End_Time'], facecolor='r', alpha=0.2)
        else:
            ax.axvspan(row['Time'], row['End_Time'], facecolor='lightblue', alpha=0.6)

# Main function
def main(main_path, result_path, epoch_name):
    # Find data max and min for plot boundaries
    getBoundary(result_path)
    
    # Iterate over epoch
    event, CRP, epoch_info, path = loadInfo(result_path, epoch_name)
    
    # Plot pupil position
    plotPosition(event, CRP, epoch_info, path)
    plot2DAngle(event, CRP, epoch_info, path)
    plotDistance(event, CRP, epoch_info, path)
    plot1DAngle(event, CRP, epoch_info, path)
    plot2DVel(event, CRP, epoch_info, path)
    plot2DAVel(event, CRP, epoch_info, path)
    plot1DVel(event, CRP, epoch_info, path)
    plot1DAVel(event, CRP, epoch_info, path)
    plotSize(event, CRP, epoch_info, path)
    
    print('\nPlot generated.')
    
# Code initialization
if __name__ == '__main__':
    # Parameters
    main_path = r'C:\Users\HeLab\Documents\Ruijia\Project\EyeTracking\Data\v1\623-neg'

    run_file = 'RUN1_2020-01-19-14-42'
    
    epoch_name = 'Epoch_0.378_1_1'
    
    result_dir = 'RESULT'
    result_path = os.path.join(main_path, result_dir, run_file)
    
    ### Call main() function
    main(main_path, result_path, epoch_name)