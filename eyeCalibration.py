# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:02:29 2020

@author: HeLab
"""
import os
import cv2
import json
import numpy as np

# Parameters and bool triggers
pause = True #trigger for pausing the video

ROIpoint = [(0,1),(1,0)] #cropped ROI coordinates
cropping = False #trigger for cropping rectangle display

# Functions definition
def getROI(event,x,y,flags,param):
    '''
        Capture mouse position for click, drag, and drop
    '''
    global ROIpoint, cropping

    # Mouse first click
    if event == cv2.EVENT_LBUTTONDOWN:
        cropping = True
        ROIpoint[0] = (x,y)
        ROIpoint[1] = (x+1,y+1)
    # Mouse drag
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            ROIpoint[1] = (x,y)
    # Mouse drop
    elif event == cv2.EVENT_LBUTTONUP:
        ROIpoint[1] = (x,y)
        cropping = False

    return ROIpoint

def findPerp(frame):
    '''
        Draw perpendicular line to the eye axis
    '''
    # Calculate and draw eye axis
    # Line equation: y = slope*x + intercept
    dist = np.sqrt((ROIpoint[0][0]-ROIpoint[1][0])**2+(ROIpoint[0][1]-ROIpoint[1][1])**2)/2.
    try:
        slope = (ROIpoint[1][1]-ROIpoint[0][1])/(ROIpoint[1][0]-ROIpoint[0][0])
        intercept = ROIpoint[0][1] - slope*ROIpoint[0][0]
        axis_x = [slope,intercept]
    except:
        pass
    center_point = (((ROIpoint[0][0]+ROIpoint[1][0])/2),((ROIpoint[0][1]+ROIpoint[1][1])/2))
    cv2.circle(frame,(int(center_point[0]),int(center_point[1])),2,(0,0,255),2)
    
    # Find perpendicular that goes through center_point
    # Reciprocal slope
    try:
        if abs(slope) > 100:
            x1 = center_point[0] + dist
            y1 = center_point[1]
            
            x2 = center_point[0] - dist
            y2 = center_point[1]
            axis_y = [0,center_point[0]]
        elif slope != 0:
            slope_perp = -1./slope
            inter_perp = center_point[1]-(slope_perp*center_point[0])
            axis_y = [slope_perp,inter_perp]
            
            A = 1. + (slope_perp**2)
            B = (2*slope_perp*inter_perp) - (2*center_point[0]) - (2*slope_perp*center_point[1])
            C = (center_point[0]**2) + (inter_perp**2) - (2*inter_perp*center_point[1]) + (center_point[1]**2) - (dist**2)
            
            delta = (B**2) - (4*A*C)

            x1 = (-B + np.sqrt(delta))/(2*A)
            y1 = (x1*slope_perp) + inter_perp
            
            x2 = (-B - np.sqrt(delta))/(2*A)
            y2 = (x2*slope_perp) + inter_perp
        else:
            y1 = center_point[1] + dist
            x1 = center_point[0]
            
            y2 = center_point[1] - dist
            x2 = center_point[0]
            axis_y = [center_point[0],0]
    
    except:
        axis_x = [None,None]
        axis_y = [None,None]
        
        center_point = (None,None)
        
        x1 = None
        y1 = None
        
        x2 = None
        y2 = None

    return axis_x,axis_y,center_point,x1,y1,x2,y2

# Main function
def main(main_path, video_path):
    global pause
    
    cv2.namedWindow('Select_Calibration')
    cv2.setMouseCallback('Select_Calibration',getROI)
    
    cap = cv2.VideoCapture(video_path)
    frame_idx = -1
    
    while(cap.isOpened()):
        if pause == True:
            ret, frame = cap.read()
    
            # Stop the loop if all frames have been played
            if not ret:
                break
            
            # Flip frame
            frame = cv2.flip(frame, 0)
            
            # Add frame idx
            frame_idx = frame_idx + 1
            frame_temp = frame.copy()
            
            # Find perp
            axis_x,axis_y,center_point,x1,y1,x2,y2 = findPerp(frame_temp)
            
            # Draw axis
            cv2.line(frame_temp, ROIpoint[0], ROIpoint[1], (0, 0, 255), 2)
            try:
                cv2.line(frame_temp, (int(x1),int(y1)),(int(x2),int(y2)), (0, 0, 255), 2)
            except:
                pass
            cv2.imshow('Select_Calibration',frame_temp)
        else:
            frame_temp = frame.copy()
            
            # Find perp
            axis_x,axis_y,center_point,x1,y1,x2,y2 = findPerp(frame_temp)
            
            # Draw axis
            cv2.line(frame_temp, ROIpoint[0], ROIpoint[1], (0, 0, 255), 2)
            try:
                cv2.line(frame_temp, (int(x1),int(y1)),(int(x2),int(y2)), (0, 0, 255), 2)
            except:
                pass
            cv2.imshow('Select_Calibration',frame_temp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            if pause == True:
                pause = False
            else:
                pause = True
                
        # Press Esc to exit the window
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            print('Tracking manually terminated.')
            break

    cap.release
    cv2.destroyAllWindows()
    
    calibration = {
        'frame_idx' : frame_idx,
        'slope_1' : axis_x[0],
        'inter_1' : axis_x[1],
        'ROI_x1' : ROIpoint[0][0],
        'ROI_x2' : ROIpoint[1][0],
        'ROI_y1' : ROIpoint[0][1],
        'ROI_y2' : ROIpoint[1][1],
        'C_x' : center_point[0],
        'C_y' : center_point[1],
        'slope_2' : axis_y[0],
        'inter_2' : axis_y[1],
        'Perp_x1' : x1,
        'Perp_x2' : x2,
        'Perp_y1' : y1,
        'Perp_y2' : y2
        }
    
    with open(os.path.join(main_path,'calibration.txt'), 'w') as file:
        file.write(json.dumps(calibration))
    
# Code initialization 
if __name__ == '__main__':
    # Parameters
    main_path = r'C:\Users\HeLab\Documents\Ruijia\Project\EyeTracking\Data\v2\WT5\run1\low'

    video_dir = 'IN_VIDEO'     
    video_name = 'video.h264'
    video_path = os.path.join(main_path, video_dir, video_name)
    
    ### Call main() function
    main(main_path, video_path)