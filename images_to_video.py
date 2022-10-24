import cv2
import numpy as np
import os
import requests
from datetime import datetime

def im_to_vid(directory,name = "video",push_to_dashboard = False): 
    all_files = os.listdir(directory)
    all_files.sort()
    for filename in all_files:
        filename = os.path.join(directory, filename)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        break
    
    n = 0
    
    now = datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")
    f_name = os.path.join("./temp_videos",'{}_{}.mp4'.format(now,name))
    temp_name = f = os.path.join("./temp_videos",'temp.mp4')
    
    out = cv2.VideoWriter(temp_name,cv2.VideoWriter_fourcc(*'mp4v'), 8, size)
     
    for filename in all_files:
        filename = os.path.join(directory, filename)
        img = cv2.imread(filename)
        out.write(img)
        print("Wrote frame {}".format(n))
        n += 1
        
        # if n > 30:
        #     break
    out.release()
    
    #os.system("/usr/bin/ffmpeg -i {} -vcodec libx264 {}".format(temp_name,f_name))
    os.system("/remote/i24_common/miniconda3/envs/tracking/bin/ffmpeg -i {} -vcodec libx264 {}".format(temp_name,f_name))
    if push_to_dashboard:
        
        
        #snow = now.strftime("%Y-%m-%d_%H-%M-%S")
        url = 'http://viz-dev.isis.vanderbilt.edu:5991/upload?type=boxes_raw'
        files = {'upload_file': open(f_name,'rb')}
        ret = requests.post(url, files=files)
        print(f_name)
        print(ret)
        if ret.status_code == 200:
            print('Uploaded!')
            
            
file = "/home/worklab/Data/cv/KITTI/data_tracking_image_2/training/image_02/0000"
file = "/home/worklab/Documents/derek/track_i24/output/temp_frames"
file = "/home/worklab/Documents/derek/LBT-count/vid"
file = "/home/worklab/Documents/derek/3D-playground/video/6"
file = "/home/derek/Desktop/temp_frames"


    
file  = './temp_frames'
im_to_vid(file,name = "name_name_name",push_to_dashboard = True)


if True:
# just push a video
    import requests
    f_name = "/home/derek/Desktop/2022-09-16_16-31-37_latest_greatest.mp4"
    url = 'http://viz-dev.isis.vanderbilt.edu:5991/upload?type=boxes_raw'
    files = {'upload_file': open(f_name,'rb')}
    ret = requests.post(url, files=files)
    print(f_name)
    print(ret)
    if ret.status_code == 200:
        print('Uploaded!')
