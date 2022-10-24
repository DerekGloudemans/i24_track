import os ,sys
import numpy as np
import random 
import cv2
import time
import torch
from torch.utils import data
from torch import optim
import collections
import torch.nn as nn
import torchvision.transforms.functional as F
import torchvision
from scipy.optimize import linear_sum_assignment
from datetime import datetime

random.seed(0)
torch.manual_seed(0)

checkpoint_file = "./data/weights/multi_model34_e8_0.pt"
testZ_im = "test.png"
from src.detect.retinanet_3D_multi_frame.retinanet.model import resnet34 as Retinanet3D_multi_frame

# load detector
depth = 18
num_classes = 8
checkpoint_file = None #"cp/multi_model34_e8_0.pt"

retinanet = Retinanet3D_multi_frame(8)
conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
conv1.weight.data[:,:3,:,:] = retinanet.conv1.weight.data.clone()
conv1.weight.data[:,3:,:,:] = retinanet.conv1.weight.data.clone()
retinanet.conv1 = conv1

# load checkpoint if necessary
try:
    if checkpoint_file is not None:
        retinanet.load_state_dict(torch.load(checkpoint_file).state_dict())
except:
    retinanet.load_state_dict(torch.load(checkpoint_file))

# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
retinanet = retinanet.to(device)


# eval mode
retinanet.training = False
retinanet.eval()
retinanet.freeze_bn()

retinanet = retinanet.half()
for layer in retinanet.modules():
    if isinstance(layer, torch.nn.BatchNorm2d):
        layer.float()
        
        

