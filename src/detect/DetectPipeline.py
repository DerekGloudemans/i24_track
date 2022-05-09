import torch
from i24_configparse.parse import parse_cfg

import os
os.environ["user_config_directory"] = "/home/worklab/Documents/i24/i24_track/config"


# imports for specific detectors
from retinanet_3D.retinanet.model import resnet50 as Retinanet3D

class DetectPipeline():
    """
    The DetectPipeline provides a wrapper class for all operations carried out on a single
    GPU for a single frame; generally, this includes prepping frames for detection,
    performing inference with self.detector, and any subsequent post-detection steps
    such as low-confidence filtering or best detection selection based on priors
    """
    
    def __init__(self):
        raise NotImplementedError
        
    def prep_frames(self,frames):
        """
        Receives a stack of frames as input and returns a stack of frames as output.
        All frames in each stack are expected to be on the same device as the DetectPipeline object. 
        In the default case, no changes are made
        param: frames (tensor of shape [B,3,H,W])
        return: frames (tensor of shape [B,3,H,W])
        """
        return frames
        
    def detect(self,frames):
        with torch.no_grad():
            result = self.detector(frames)
        return result
        
    def post_detect(self,detection_result):
        return detection_result

    def __call__(self,frames):
        prepped_frames = self.prep_frames(frames)
        detection_result = self.detect(prepped_frames)
        output = self.post_detect(detection_result)
        return output


    
    
class RetinanetFullFramePipeline(DetectPipeline):
    """
    DetectPipeline that applies Retinanet Detector on full object frames
    """
    
    def __init__(self):
        
        # load configuration file parameters
        self = parse_cfg("0",obj = self)

        
        # initialize detector
        self.detector = Retinanet3D(self.n_classes)
    
        # quantize model         
        if self.quantize:
            self.detector = self.detector.half()
        
        # load detector weights
        self.detector.load_state_dict(torch.load(self.weights_file))

        
        # configure detector
        self.device = torch.device("cuda:{}".format(self.device_id))
        torch.cuda.set_device(self.device_id)
        self.detector = self.detector.to(self.device)
        self.detector.eval()
    
        

        
    def post_detect(self,detection_result):
        reg_boxes, classes = detection_result
        confs,classes = torch.max(classes, dim = 2) 
        return [reg_boxes,confs,classes]
    
        
class RetinanetCropFramePipeline(DetectPipeline):
        
    def __init__(self):
        # load configuration file parameters
        self = parse_cfg("0",obj = self)

        
        # initialize detector
        self.detector = Retinanet3D(self.n_classes)
    
        # quantize model         
        if self.quantize:
            self.detector = self.detector.half()
        
        # load detector weights
        self.detector.load_state_dict(torch.load(self.weights_file))

        
        # configure detector
        self.device = torch.device("cuda:{}".format(self.device_id))
        torch.cuda.set_device(self.device_id)
        self.detector = self.detector.to(self.device)
        self.detector.eval()
        
        
    def prep_frames(self):
        pass
        # TODO - Derek Implement cropping
    
    def post_detect(self,detection_result):
        pass
        # TODO - Derek implement post-detection local-to-global mapping