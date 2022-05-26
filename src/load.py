import configparser
import torch

import os
import numpy as np
import time
import re

import cv2
from PIL import Image

from torchvision.transforms import functional as F

class DummyNoiseLoader():
    """
        Parses the config and returns random noise in the right shape
    """
    
    
    def __init__(self,mapping_file):
        
        # all we need to know which cameras go on which GPU
        
        cp = configparser.ConfigParser()
        cp.read(mapping_file)
        mapping = dict(cp["DEFAULT"])
       
        for key in mapping.keys():
            parsed_val = int(mapping[key])
            mapping[key] = parsed_val
    
        self.cam_devices = mapping

        self.dummy_time = 0         
            
    def get_frames(self,target):
        
        frames = [[] for i in range(torch.cuda.device_count())]
        for cam_id in self.cam_devices:
            dev_id = self.cam_devices[cam_id]
            frames[dev_id].append(torch.rand(3,1080//2,1920//2).to(torch.device("cuda:{}".format(dev_id))))
            
        # stack each list
        out = []
        for lis in frames:
            out.append(torch.stack(lis))
        ts = torch.tensor([self.dummy_time for i in range(len(self.cam_devices))])    
        
        return out,ts