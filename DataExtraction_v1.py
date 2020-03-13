# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:58:52 2020

@author: Ruijia Wang <w.ruijia@gmail.com>
"""
import os
import ProcessData
import EpochData
import EpochVideo
import PlotData

# Code initialization 
if __name__ == '__main__':
    # Parameters
    main_path = r'C:\Users\HeLab\Documents\Ruijia\Project\EyeTracking\Data\v1\624-1339450-OKR'
    
    info_dir = 'OUT_INFO'
    info_path = os.path.join(main_path, info_dir)
    
    video_dir = 'IN_VIDEO'
    video_name = 'video_2019-11-25-15-55-02.h264'
    video_path = os.path.join(main_path, video_dir, video_name)
    
    data_dir = 'OUT_VIDEO'
    data_file = 'RUN1_2020-01-19-15-16'
    data_path = os.path.join(main_path, data_dir , data_file)
     
    result_dir = 'RESULT'
    result_path = os.path.join(main_path, result_dir, data_file)
    
    # Script execution
    print('\n-------Processing data-------')
    ProcessData.main(main_path, info_path, data_path, result_path)
    print('\n-------Epoching data-------')
    EpochData.main(main_path, info_path, data_path, result_path)
    print('\n-------Plot data-------')
    PlotData.main(main_path, info_path, data_path, result_path)
    print('\n-------Epoching video-------')
    EpochVideo.main(main_path, info_path, video_path, data_path, result_path) 