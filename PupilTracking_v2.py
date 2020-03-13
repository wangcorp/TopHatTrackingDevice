# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:28:02 2019

@author: Ruijia Wang <w.ruijia@gmail.com>
Inspired by the eye tracking algorithm of A. Meyer, et al. (2018)

Meyer, A. F., Poort, J., O’Keefe, J., Sahani, M., & Linden, J. F. (2018)
A head-mounted camera system integrates detailed behavioral monitoring
with multichannel electrophysiology in freely moving mice.
Neuron, 100(1), 46-60.


"""

# Imports
import os
import csv
import cv2 
import time
import numpy as np

from datetime import datetime

# Fast image inpainting based on coherence transport
# Folmar Bornemann and Tom Marz, written in C
# Adapted in Cython by A. Mayer, et al. (2018)
from inpaintBCT import inpaintBCT


# Tracking parameters
displayParams = {
    'SHOW_MODE' : False, #display debug master switch
    'SHOW_ROI' : False, # display cropped unprocessed
    'SHOW_PROCESSED' : True, #display processed frame
    'SHOW_BLOB' : True, #display blob and CR detection
    'SHOW_INPAINT' : True, #display inpainting result
    'SHOW_THRESH' : True, #display image thresholding
    'SHOW_NORM' : True, #display image after normalization
    'SHOW_CANNY' : True, #display canny edge detection
    'SHOW_CONTOURS' : True, #display contours detection
    'SHOW_ELLIPSE' : True, #display ellipse fitting
    
    'RENDER_MODE' : False, #display render master switch
    'ROI_MODE' : True, #display cropped video
    'GLOBAL_MODE' : False, #display uncropped video
    'CrosshairSize' : 3, #length of crosshair
    'CrosshairWidth' : 1, #width of crosshair
    }

saveParams = {
    'SAVE_MODE' : True, #save master switch
    'SAVE_DATA' : True, #save tracking data
    'SAVE_ROI' : True, #save cropped video
    'SAVE_GLOBAL' : False, # save uncropped video
    }

sleepParams = {
    'Bool' : False, #slow down video framerate
    'Time' : 1/20 #pause time between each frame
    }

preProcessParams = {
    'ClipLimit' : 4.0, #clahe clip limit
    'ClaheSize' : (25,25), #clahe grid size 
    'GaussianKernel' : (7,7) #gaussian kernel size
    }

paintParams = {
    'Scaling' : 110, #percentage of radius scaling for inpainting 114
    'Epsilon' : 7, #homogeneity within the mask
    'Kappa' : 5, #binarity of the color to the center
    'Rho' : 2, #reach of color to the mask center
    'Sigma' : 1, #width of color
    'Thresh' : 0, #inpaintBCT
    'GaussianKernel' : (7,7), #gaussian kernel size
    }

blobParams = {
    'MinTh' : 120, #min threshold for blob detection
    'MaxTh' : 255, #max threshold for blob detection
    'MinSize' : 5, #min size of detected flare/glare 
    'MaxSize' : 180, #max size of detected flare/glare
    'MinConv' : 0.6, #blob min convexity ratio
    'Color' : 255 #blob color (white)
    }

CRParams = {
    'MinTh' : 180, #min threshold for blob detection
    'MaxTh' : 255, #max threshold for blob detection
    'MinSize' : 60, #min size of detected flare/glare
    'MaxSize' : 160, #max size of detected flare/glare
    'MinConv' : 0.85, #blob min convexity ratio
    'Color' : 255 #blob color (white)
    }

pupilParams = {
    'NORM_BOOL' : False, #switch for image normalization
    'ThreshLow' : 90, #lower value for truncated thresholding (100)
    'Canny1' : 25, #threshold 1 for canny edge detection (25)
    'Canny2' : 75, #threshold 2 for canny edge detection (no norm 50/100 norm)
    'APSize' : 7, #size of aperture for canny edge detection (7)
    'MaxIntensity' : 150, #Max intensity of an ellipse area
    }

# pupilParams = {
#     'NORM_BOOL' : False, #switch for image normalization
#     'ThreshLow' : 140, #lower value for truncated thresholding 100
#     'CannyRatio' : 0.2, #constant for automatic canny threshold
#     'APSize' : 5 #size of aperture for canny edge detection 
#     }

contourParams = {
    'MinSize' : 20, #min number of point in contours
    'MorphKernel' : (7,7), #kernel size for morphocological expansion
    'StdMax' : 4, #max contours to center euclidean distance std
    'RatioMin' : 0.6, #min bounding box h/w ration
    }

# Global variables
ROIpoint = [(0,0),(0,0)] #cropped ROI coordinates
cropping = False #trigger for cropping rectangle display

first_blob = True #logic switch for first blob detection
last_keypoint = [cv2.KeyPoint(0,0,0)] #keypoint used for cleaning

last_ellipse = tuple([(0,0),(0,0),0])

# Value storage
CR_raw = []
CR_clean = []

ellipse_raw = []
ellipse_clean = []

# Functions definition
def getInfo(cap):
    '''
        Get info about the video processed
    '''
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))

    videoInfo = {
            'lenght' : length,
            'width' : width,
            'height' : height,
            'fps' : fps
            }
    
    return videoInfo;

def outputFull(save_path, cap):
    '''
        Create video output to save frames
    '''
    # Get frame info for video capture
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS)

    # Save video
    video_name = 'full.avi'
    out = cv2.VideoWriter(os.path.join(save_path,video_name),
                          cv2.VideoWriter_fourcc('M','J','P','G'),
                          fps, (frame_width,frame_height))
    return out

def outputROI(save_path, cap):
    '''
        Create video output to save frames
    '''
    # Get frame info for video capture
    frame_width  = int(abs(ROIpoint[0][0]-ROIpoint[1][0]))
    frame_height = int(abs(ROIpoint[0][1]-ROIpoint[1][1]))
    fps          = cap.get(cv2.CAP_PROP_FPS)

    # Save video
    video_name = 'ROI.avi'
    out = cv2.VideoWriter(os.path.join(save_path,video_name),
                          cv2.VideoWriter_fourcc('M','J','P','G'),
                          fps, (frame_width,frame_height))
    return out

def saveInfo(save_path, cap):
    '''
        Save info about the original video file
    '''
    # Video file specs
    info = getInfo(cap)
    # Add ROI position
    ymin, ymax, xmin, xmax = sortROI()
    info.update({
            'ROI_x_min' : xmin,
            'ROI_x_max' : xmax,
            'ROI_y_min' : ymin,
            'ROI_y_max' : ymax
            })
    
    # Write file
    file_path = os.path.join(save_path, 'Info.csv')
    with open(file_path, 'w') as f:
        for key in info:
            f.write("%s,%s\n"%(key,info[key]))
    
def saveParameters(save_path):
    '''
        Save values of all parameters used for tracking
    '''
    file_path = os.path.join(save_path, 'Param.csv')
    with open(file_path, 'w') as f:
        for key in preProcessParams:
            f.write("%s,%s,%s\n"%('preProcessParams',key,preProcessParams[key]))
        for key in paintParams:
            f.write("%s,%s,%s\n"%('paintParams',key,paintParams[key]))
        for key in blobParams:
            f.write("%s,%s,%s\n"%('blobParams',key,blobParams[key]))
        for key in CRParams:
            f.write("%s,%s,%s\n"%('CRParams',key,CRParams[key]))
        for key in pupilParams:
            f.write("%s,%s,%s\n"%('pupilParams',key,pupilParams[key]))
        for key in contourParams:
            f.write("%s,%s,%s\n"%('contourParams',key,contourParams[key]))
    
def saveCR(save_path):
    '''
        Save cleaned and raw position of the corneal reflection
    '''
    clean_path = os.path.join(save_path, 'CR_clean.csv')
    raw_path = os.path.join(save_path, 'CR_raw.csv')
    
    with open(clean_path, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, CR_clean[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(CR_clean)

    with open(raw_path, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, CR_raw[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(CR_raw)
    
def saveEllipse(save_path):
    '''
        Save cleaned and raw ellipse fitted from the pupil
    '''
    clean_path = os.path.join(save_path, 'Pupil_clean.csv')
    raw_path = os.path.join(save_path, 'Pupil_raw.csv')
    
    with open(clean_path, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, ellipse_clean[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(ellipse_clean)
    
    with open(raw_path, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, ellipse_raw[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(ellipse_raw)
    
def setSavePath(output_directory, run_name):
    '''
        Set up path to save files
    '''
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    t = datetime.now()
    data_name = run_name + '_' + t.strftime("%Y-%m-%d-%H-%M")
    data_path = os.path.join(output_directory, data_name)

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    return data_path

def getROI(event,x,y,flags,param):
    '''
        Capture mouse position for click, drag, and drop
    '''
    global ROIpoint, cropping

    # Mouse first click
    if event == cv2.EVENT_LBUTTONDOWN:
        cropping = True
        ROIpoint[0] = (x,y)
        ROIpoint[1] = (x,y)
    # Mouse drag
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            ROIpoint[1] = (x,y)
    # Mouse drop
    elif event == cv2.EVENT_LBUTTONUP:
        ROIpoint[1] = (x,y)
        cropping = False

    return ROIpoint

def getROICenter():
    '''
        Return the middle coordinate of the ROI
    '''
    roi_middle = [np.mean([ROIpoint[0][0], ROIpoint[1][0]]),
                  np.mean([ROIpoint[0][1], ROIpoint[1][1]])]
    return roi_middle

def sortROI():
    '''
        Sort the coordinates by size
    '''
    ymin = min([ROIpoint[0][1],ROIpoint[1][1]])
    ymax = max([ROIpoint[0][1],ROIpoint[1][1]])

    xmin = min([ROIpoint[0][0],ROIpoint[1][0]])
    xmax = max([ROIpoint[0][0],ROIpoint[1][0]])

    return ymin, ymax, xmin, xmax

def selectROI(cap):
    '''
        The ROI can be selected by dragging the mouse after holding
        the left click. The operatio can be repeated until satisfaction.
        Press Esc to exit the ROI selection window.
    '''
    
    cv2.namedWindow('Select_ROI')
    cv2.setMouseCallback('Select_ROI',getROI)

    print('Please select the ROI by dragging the mouse with a left click.')

    while(cap.isOpened()):
        ret, frame = cap.read()
    
        if not ret:
            print('ROI selection timed out')
            break
        
        frame = cv2.flip(frame, 0)
        
        cv2.rectangle(frame, ROIpoint[0], ROIpoint[1], (0, 0, 255), 2)
        cv2.imshow('Select_ROI', frame)

        # Press Esc to exit the window
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            print("ROI: ",ROIpoint[0], " x " , ROIpoint[1] )
            break

    cap.release()
    cv2.destroyAllWindows()

def applyGaussianBlur(frame, kernel):
    '''
        Apply gaussian blur to frame with givevn kernel
    '''
    # Gaussian blur (low pass filter)
    frame_blur = cv2.GaussianBlur(frame,kernel,0)

    return frame_blur

def preProcessing(frame):
    '''
        Global pre-processing (gray scale, low pass filter, adaptive histogram)
    '''
    # Gray Grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Adaptive histogram equallization
    clahe = cv2.createCLAHE(clipLimit=preProcessParams['ClipLimit'], tileGridSize=preProcessParams['ClaheSize'])
    frame_clahe = clahe.apply(frame_gray)

    # Gaussian blur (low pass filter)
    frame_blur = applyGaussianBlur(frame_clahe,preProcessParams['GaussianKernel'])

    return frame_blur

def adjustKeypoint(kp):
    '''
        Adjust a keypoint coordinates to the uncropped window
    '''
    ymin, ymax, xmin, xmax = sortROI()

    kp_corr = cv2.KeyPoint(kp[0].pt[0]+xmin, kp[0].pt[1]+ymin,
                            kp[0].size, kp[0].angle, kp[0].response,
                            kp[0].octave, kp[0].class_id)
    return kp_corr

def adjustEllipse(ellipse):
    '''
        Adjust the ellipse coordinates to the uncropped window
    '''
    ymin, ymax, xmin, xmax = sortROI()
    
    temp = list(ellipse)
    temp[0] = (temp[0][0]+xmin,temp[0][1]+ymin)
    temp[1] = (temp[1][0],temp[1][1])
    ellipse = tuple(temp)
    
    return ellipse

def renderGlobal(frame, CR, ellipse):
    '''
        Show the global frame with CR and pupil tracking
    '''
    # Draw corrected corneal reflection
    CR = adjustKeypoint(CR)
    frame = cv2.line(frame,
                    (int(CR.pt[0]-displayParams['CrosshairSize']),
                     int(CR.pt[1])),
                    (int(CR.pt[0]+displayParams['CrosshairSize']),
                     int(CR.pt[1])),
                     (255,0,0),displayParams['CrosshairWidth'])
    
    frame = cv2.line(frame,
                    (int(CR.pt[0]),
                     int(CR.pt[1])-displayParams['CrosshairSize']),
                    (int(CR.pt[0]),
                     int(CR.pt[1]+displayParams['CrosshairSize'])),
                     (255,0,0),displayParams['CrosshairWidth'])
    
    # Draw corrected ellipse
    try:
        ellipse = adjustEllipse(ellipse)
        cv2.ellipse(frame,ellipse,(0,0,255),1)
    except:
        pass
    
    if displayParams['RENDER_MODE'] == True and displayParams['GLOBAL_MODE'] == True:
        cv2.imshow("GLOBAL", frame)

    return frame

def renderROI(frame, CR, ellipse):
    '''
        Show the cropped frame with CR and pupil tracking
    '''
    # Draw corrected corneal reflection
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    frame = cv2.line(frame,
                    (int(CR[0].pt[0]-displayParams['CrosshairSize']),
                     int(CR[0].pt[1])),
                    (int(CR[0].pt[0]+displayParams['CrosshairSize']),
                     int(CR[0].pt[1])),
                     (255,0,0),displayParams['CrosshairWidth'])
    
    frame = cv2.line(frame,
                    (int(CR[0].pt[0]),
                     int(CR[0].pt[1])-displayParams['CrosshairSize']),
                    (int(CR[0].pt[0]),
                     int(CR[0].pt[1]+displayParams['CrosshairSize'])),
                     (255,0,0),displayParams['CrosshairWidth'])
    
    # Draw corrected ellipse
    try:
        cv2.ellipse(frame,ellipse,(0,0,255),1)
    except:
        pass
    
    if displayParams['RENDER_MODE'] == True and displayParams['ROI_MODE'] == True:
        cv2.imshow("ROI", frame)
        
    return frame

def cleanEllipse(ellipse_list, frame):
    '''
        Keep the ellipse with the lowest intensity or discard if none
    '''
    
    # Get mean intensity in the elliipse area
    mask_idx = None
    intensity = 0
    
    for idx, item in enumerate(ellipse_list):
        mask = np.zeros_like(frame)
        cv2.ellipse(mask,item,color=(255,255,255),thickness=-1)
        mean_intensity = cv2.mean(frame,mask=mask)
        if mean_intensity[0] > intensity:
            intensity = mean_intensity[0]
            mask_idx = idx
            
    # Get minor axis size of selected ellipse
    minor_axis = ellipse_list[mask_idx][1][1]
        
    # Return ellipse or error according to criteria
    if intensity > pupilParams['MaxIntensity'] or minor_axis < 10:
        raise ValueError('Intensity too high.')
    else:
        return ellipse_list[mask_idx]

def cleanKeyPoints(keypoints, last_keypoint):
    '''
        Preliminary cleaning of the blob found during tracking.
        Handles different case were none, one, or more blobs are found.
        Cleaned data returned have no null blobs.
        Raw data contains null blobs for further handling.
    '''
    global first_blob

    # Find middle point of ROI
    roi_middle = getROICenter()

    # Case if no blob have been detected yet (first one)
    if first_blob == True:
        if len(keypoints) == 1:
            first_blob = False
            return [keypoints[0]], [keypoints[0]]

        elif len(keypoints) > 1:
            ref_kp = cv2.KeyPoint(0,0,0)
            for key in keypoints:
                ref_dist = np.linalg.norm(np.array(roi_middle)-np.array([ref_kp.pt[0],ref_kp.pt[1]]))
                key_dist = np.linalg.norm(np.array(roi_middle)-np.array([key.pt[0],key.pt[1]]))
                if key_dist < ref_dist:
                    ref_kp = key
            return [ref_kp], [ref_kp]

        else:
            return [last_keypoint[0]], [last_keypoint[0]]

    # Case if blobs have already been detected
    elif first_blob == False:
        if len(keypoints) == 1:
            return [keypoints[0]], [keypoints[0]]

        elif len(keypoints) > 1:
            ref_kp = last_keypoint[0]
            loop_start = True
            for key in keypoints:
                if loop_start == True:
                    ref_dist = np.linalg.norm(np.array([key.pt[0],key.pt[1]])-np.array([ref_kp.pt[0],ref_kp.pt[1]]))
                    selected_kp = key
                    loop_start = False
                else:
                    key_dist = np.linalg.norm(np.array([key.pt[0],key.pt[1]])-np.array([ref_kp.pt[0],ref_kp.pt[1]]))
                    if key_dist < ref_dist:
                        ref_dist = key_dist
                        selected_kp = key
            return [selected_kp], [selected_kp]
        else:
                # Adjust last_keypoint position to ROI
            return [cv2.KeyPoint(0,0,0)], [last_keypoint[0]]

def getBlob(frame, params, clean=False):
    '''
        Create a blob detector and return detected blobs on frame
    '''
    global last_keypoint

    blobParams = cv2.SimpleBlobDetector_Params()

    blobParams.minThreshold = params['MinTh']
    blobParams.maxThreshold = params['MaxTh']

    blobParams.filterByArea = True
    blobParams.minArea = params['MinSize']
    blobParams.maxArea = params['MaxSize']

    blobParams.filterByConvexity = True
    blobParams.minConvexity = params['MinConv']

    blobParams.filterByColor = True
    blobParams.blobColor = params['Color']

    blob_detector = cv2.SimpleBlobDetector_create(blobParams)
    blob_keypoints = blob_detector.detect(frame)

    if clean == False:
        return blob_keypoints

    else:
        blob_keypoints_raw, blob_keypoints_clean = cleanKeyPoints(blob_keypoints,last_keypoint)
        last_keypoint = blob_keypoints_clean
        
        return blob_keypoints_raw, blob_keypoints_clean

def updateBlob(CR_kp_raw, CR_kp_clean, blob_keypoints, frame):
    CR_raw.append({'x': CR_kp_raw[0].pt[0], 'y': CR_kp_raw[0].pt[1]})
    CR_clean.append({'x': CR_kp_clean[0].pt[0], 'y': CR_kp_clean[0].pt[1]})
        
    if displayParams['SHOW_MODE'] == True and displayParams['SHOW_BLOB'] == True:
        roi_CR = cv2.drawKeypoints(frame, blob_keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        roi_CR = cv2.drawKeypoints(roi_CR, CR_kp_clean, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Blob", roi_CR)

def applyInpainting(frame, keypoints):
    '''
        Inpaint blobs based on coherance transport
    '''
    # Create a mask for regions to inpaint (detected blobs)
    mask_inpainting = np.zeros_like(frame)

    if len(keypoints) > 0:
        frame = np.reshape(frame, (frame.shape[0],frame.shape[1],1))
    
        for blob in keypoints:
            cv2.circle(mask_inpainting, (int(np.round(blob.pt[0])), int(np.round(blob.pt[1]))),
                        int(np.round(blob.size*paintParams['Scaling']/100)),
                        (255,255,255), -1)
    
        # Inpainting using coherence transport (Folkmar Bornemann and Tom Maerz)
        frame_BCT = inpaintBCT.inpaintBCT(
                        np.asfortranarray(frame.astype(np.float64)),
                        np.asfortranarray(mask_inpainting.astype(np.float64)),
                        paintParams['Epsilon'], paintParams['Kappa'],
                        paintParams['Sigma'], paintParams['Rho'],
                        paintParams['Thresh'])
        
        frame_BCT = frame_BCT.astype(np.uint8)
        
        # Apply Gaussian blur
        frame_BCT = applyGaussianBlur(frame_BCT, paintParams['GaussianKernel'])
        
        if displayParams['SHOW_MODE'] == True and displayParams['SHOW_INPAINT'] == True:
            cv2.imshow("Inpaint", frame_BCT)
        
        return frame_BCT
    else:
        if displayParams['SHOW_MODE'] == True and displayParams['SHOW_INPAINT'] == True:
            cv2.imshow("Inpaint", frame)
            
        return frame

def pupilProcessing(frame):
    '''
        Apply truncated thresholding, lp filter, normalization,
        and canny edges detection.
    '''
    
    _, frame_thresh = cv2.threshold(frame,pupilParams['ThreshLow'],
                                  255,cv2.THRESH_TRUNC)
    
    if displayParams['SHOW_MODE'] == True and displayParams['SHOW_THRESH'] == True:
        cv2.imshow("Thresh",frame_thresh)

    frame_norm = frame_thresh
    if pupilParams['NORM_BOOL'] == True:
        cv2.normalize(frame_thresh, frame_norm, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        if displayParams['SHOW_MODE'] == True and displayParams['SHOW_NORM'] == True:
            cv2.imshow("Normalized",frame_norm)
    
    ## Canny edges
    # frame_median = np.median(frame_norm)
    # low = int(max(np.min(frame_norm),(1-pupilParams['CannyRatio'])*frame_median))
    # high = int(min(np.max(frame_norm),(1.0+pupilParams['CannyRatio'])*frame_median))
    #
    # frame_edged = cv2.Canny(frame_norm.astype(np.uint8),
    #                         low, high,
    #                         pupilParams['APSize'],
    #                         L2gradient=True)
    
    # Canny edges
    frame_edged = cv2.Canny(frame_norm.astype(np.uint8),
                        pupilParams['Canny1'],
                        pupilParams['Canny2'],
                        pupilParams['APSize'],
                        L2gradient=True)
    
    if displayParams['SHOW_MODE'] == True and displayParams['SHOW_CANNY'] == True:
        cv2.imshow('Canny Edges After Contouring', frame_edged)
       
    return frame_edged

def getContours(frame):
    '''
        Get contours and associated hierarchy from a frame.
        Return numpy arrays instead of lists
    '''
    # Extract contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,contourParams['MorphKernel'])
    dilated = cv2.dilate(frame, kernel)
    contours, _ = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours = np.asarray(contours)
    
    # Remove contours too small (nb of points)
    s_mask = [idx for (idx,item) in enumerate(contours) if len(item)>contourParams['MinSize']]
    contours_filtered = contours[s_mask]
    
    # Find centroid of contours and filter based on euclidean distance variation
    c_mask = []
    
    for idx, item in enumerate(contours_filtered):
        distance = []
        center = np.sum(item,axis=0)/len(item)
        for point in item:
            distance.append(np.linalg.norm(center[0]-point[0]))
        std = np.std(distance)
        if std < contourParams['StdMax']:
            c_mask.append(idx)
    
    contours_filtered = contours_filtered[c_mask]
    
    # Remove contours non-circular contours
    b_mask = []
    
    for idx, item in enumerate(contours_filtered):
        rect = cv2.minAreaRect(item)
        ratio = np.min([rect[1][0],rect[1][1]])/np.max([rect[1][0],rect[1][1]])
        if ratio > contourParams['RatioMin']:
            b_mask.append(idx)
    
    contours_filtered = contours_filtered[b_mask]
    
    if displayParams['SHOW_MODE'] == True and displayParams['SHOW_CONTOURS'] == True:
        frame_contours = np.zeros_like(frame)
        cv2.drawContours(frame_contours, contours_filtered, -1, (255,0,0), 1)
        cv2.imshow("Contours", frame_contours)
        
    return contours_filtered

def buildEllipseDict(e, empty=False):
    '''
        Convert ellipse tuple into dict for storage    
    '''
    if empty == False:
        ellipse_dict = {
            'xc': e[0][0], #center x coordinate
            'yc': e[0][1], #center y coordinate
            'a': e[1][0]/2, #semi-major axis
            'b': e[1][1]/2, #semi-minor axis
            'theta': e[2] #rotation angle
            }
    else:
        ellipse_dict = {
            'xc': 0.0, #center x coordinate
            'yc': 0.0, #center y coordinate
            'a': 0.0, #semi-major axis
            'b': 0.0, #semi-minor axis
            'theta': 0.0 #rotation angle
            }    
    return ellipse_dict

def fitEllipse(contours, frame):
    '''
        Fit an ellipse on the pupil contour
    '''
    global last_ellipse, ellipse_raw, ellipse_clean
    
    try:
        # Fit ellipse
        ellipse_list = []
        for item in contours:
            ellipse_list.append(cv2.fitEllipse(item))
        ellipse = cleanEllipse(ellipse_list, frame)
        last_ellipse = ellipse
        
        # Correct ellipse for contours dilation
        temp = list(ellipse)
        temp[1] = (temp[1][0]-contourParams['MorphKernel'][0]+1,temp[1][1]-contourParams['MorphKernel'][0]+1)
        ellipse = tuple(temp)
        ellipse_dict = buildEllipseDict(ellipse)
        ellipse_raw.append(ellipse_dict); ellipse_clean.append(ellipse_dict)
    except:
        ellipse_dict = buildEllipseDict(None, empty=True)
        ellipse_raw.append(ellipse_dict)
        ellipse = last_ellipse
        ellipse_dict = buildEllipseDict(ellipse)
        ellipse_clean.append(ellipse_dict)
    
    if displayParams['SHOW_MODE'] == True and displayParams['SHOW_ELLIPSE'] == True:
        roi_ellipse = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
        try:
            cv2.ellipse(roi_ellipse,ellipse,(0,0,255),1)
        except:
            pass
        cv2.imshow('roi_ellipse',roi_ellipse)
        
    return ellipse

def eyeTracking(save_path, cap):
    '''
        Track pupil center and size, and corneal reflection point
    '''
    global CR_raw, CR_clean
    
    #Creating capture for video
    if saveParams['SAVE_MODE'] == True and saveParams['SAVE_ROI'] == True:
        out_roi = outputROI(save_path, cap)
    if saveParams['SAVE_MODE'] == True and saveParams['SAVE_GLOBAL'] == True:
        out_full = outputFull(save_path, cap)    
    
    # Process video frame by frame
    print('\nTracking in progress...')
    while(cap.isOpened()):
        ret, frame = cap.read()

        # Stop the loop if all frames have been played
        if not ret:
            print('\nTracking finished.')
            break

        # rotate frame to the right orientation
        frame = cv2.flip(frame, 0)
        
        # Sort ROI
        ymin, ymax, xmin, xmax = sortROI()
        
        if displayParams['SHOW_MODE'] == True and displayParams['SHOW_ROI'] == True:
            cv2.imshow('original_ROI',frame[ymin:ymax, xmin:xmax])

        # Preprocessing of global image
        frame_prep = preProcessing(frame)
        roi = frame_prep[ymin:ymax, xmin:xmax]

        if displayParams['SHOW_MODE'] == True and displayParams['SHOW_PROCESSED'] == True:
            cv2.imshow('processed_ROI',roi)        

        # Detect white blobs and CR for reference
        blob_keypoints = getBlob(roi, blobParams, clean=False)
        CR_kp_raw, CR_kp_clean = getBlob(roi, CRParams, clean=True)
        updateBlob(CR_kp_raw, CR_kp_clean, blob_keypoints, roi)

        # Inpaint corneal reflections
        roi_paint = applyInpainting(roi, blob_keypoints)

        # Pupil processing
        roi_proc = pupilProcessing(roi_paint)
        
        # Extract pupil contours
        contours = getContours(roi_proc)
        
        # Fit ellipse on pupill contour
        ellipse = fitEllipse(contours, roi)
        
        # Render cropped frame with CR and ellipse
        frame_roi = renderROI(roi, CR_kp_clean, ellipse)
        if saveParams['SAVE_MODE'] == True and saveParams['SAVE_ROI'] == True:
            out_roi.write(frame_roi)
        
        # Render uncropped frame with CR and ellipse
        frame_full = renderGlobal(frame, CR_kp_clean, ellipse)
        if saveParams['SAVE_MODE'] == True and saveParams['SAVE_GLOBAL'] == True:
            out_full.write(frame_full)
            
        # Press Esc to exit the window
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            print('Tracking manually terminated.')
            break

        if sleepParams['Bool'] == True:
            time.sleep(sleepParams['Time'])

    # Clean up after loop
    cap.release
    cv2.destroyAllWindows() 
        
    if saveParams['SAVE_MODE'] == True and saveParams['SAVE_GLOBAL'] == True:
        out_full.release
    if saveParams['SAVE_MODE'] == True and saveParams['SAVE_ROI'] == True:
        out_roi.release

# Main function
def main(input_directory, output_directory, file_name, run_name):
    # Get path of video file
    file_path = os.path.join(input_directory, file_name)

    # Set save path
    if saveParams['SAVE_MODE'] == True:
        save_path = setSavePath(output_directory, run_name)
    else:
        save_path = None

    # Select the ROI for pupil analysis
    cap_for_roi = cv2.VideoCapture(file_path)
    selectROI(cap_for_roi)

    # Eye tracking (CR and pupil)
    cap_for_track = cv2.VideoCapture(file_path)
    eyeTracking(save_path, cap_for_track)

    # Save tracking data
    if saveParams['SAVE_MODE'] == True:
        saveInfo(save_path, cap_for_track)
        saveParameters(save_path)
        saveCR(save_path)
        saveEllipse(save_path)

# Code initialization 
if __name__ == '__main__':
    # Saving folder name
    run_name = 'RUN1'
    
    # Data main directory
    main_directory = r'C:\Users\HeLab\Documents\Ruijia\Project\EyeTracking\Data\v2\WT5\run1\low'
    
    # Input directory containing video
    input_directory_name = 'IN_VIDEO'     
    input_directory = os.path.join(main_directory, input_directory_name)
    
    # Output directory
    output_directory_name = 'OUT_VIDEO'
    output_directory = os.path.join(main_directory, output_directory_name)

    # Video file name
    file_name = 'video.h264'
    
    ### Call main() function
    main(input_directory, output_directory, file_name, run_name)