import torch
from i24_configparse.parse import parse_cfg

import os
os.environ["user_config_directory"] = "/home/worklab/Documents/i24/i24_track/config"


# imports for specific detectors
from .retinanet_3D.retinanet.model import resnet50 as Retinanet3D


def get_Pipeline(name):
    """
    getter function that takes a string (class name) input and returns an instance
    of the named class
    """
    if name == "RetinanetFullFramePipeline":
        pipeline = RetinanetFullFramePipeline()
    elif name == "RetinanetCropFramePipeline":
        pipeline = RetinanetCropFramePipeline()
    else:
        raise NotImplementedError("No DetectPipeline child class named {}".format(name))
    
    return pipeline
    

class DetectPipeline():
    """
    The DetectPipeline provides a wrapper class for all operations carried out on a single
    GPU for a single frame; generally, this includes prepping frames for detection,
    performing inference with self.detector, and any subsequent post-detection steps
    such as low-confidence filtering or best detection selection based on priors
    """
    
    def __init__(self):
        raise NotImplementedError
        
    def prep_frames(self,frames,priors = None):
        """
        Receives a stack of frames as input and returns a stack of frames as output.
        All frames in each stack are expected to be on the same device as the DetectPipeline object. 
        In the default case, no changes are made. Note that if priors are used, they should 
        be converted to im pixels here
        param: frames (tensor of shape [B,3,H,W])
        return: frames (tensor of shape [B,3,H,W])
        """
        
        return frames
        
    def detect(self,frames):
        with torch.no_grad():
            result = self.detector(frames)
        return result
        
    def post_detect(self,detection_result,priors = None):
        
        """
        At a minimum this function should apply hg to convert to state/space
        """
        return detection_result

    def __call__(self,frames,priors = None):
        [ids,priors,frame_idx,cam_name] = priors
        
        prepped_frames = self.prep_frames(frames,priors = priors)
        detection_result = self.detect(prepped_frames)
        detections,classes,confs  = self.post_detect(detection_result,priors = priors)
        
        # Associate
        output = self.associate(ids,priors,detections,self.hg)
        
        return output


    
    
class RetinanetFullFramePipeline(DetectPipeline):
    """
    DetectPipeline that applies Retinanet Detector on full object frames
    """
    
    def __init__(self,hg,device_id=-1):
        
        # load configuration file parameters
        self = parse_cfg("DEFAULT",obj = self)

        
        # initialize detector
        self.detector = Retinanet3D(self.n_classes)
    
        # quantize model         
        if self.quantize:
            self.detector = self.detector.half()
        
        # load detector weights
        self.detector.load_state_dict(torch.load(self.weights_file))

        
        # configure detector
        self.device = torch.device("cuda:{}".format(device_id) if device_id != -1 else "cpu")
        torch.cuda.set_device(self.device_id)
        self.detector = self.detector.to(self.device)
        self.detector.eval()
    
        self.hg = hg # store homography

        
    def post_detect(self,detection_result):
        reg_boxes, classes = detection_result
        confs,classes = torch.max(classes, dim = 2) 
        
        # TODO - convert detections from image space to state space
        # Hmm, somehow we need to know which frame each index is, so we probably need to pass that list in
        detections = self.hg.im_to_state()
        
        return [detections,confs,classes]
    
    
    
    
        
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