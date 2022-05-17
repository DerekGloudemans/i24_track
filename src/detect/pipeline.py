import torch
from i24_configparse.parse import parse_cfg
from ..util.bbox import im_nms,space_nms


import os
os.environ["user_config_directory"] = "/home/worklab/Documents/i24/i24_track/config"


# imports for specific detectors
from .retinanet_3D.retinanet.model import resnet50 as Retinanet3D


def get_Pipeline(name,hg):
    """
    getter function that takes a string (class name) input and returns an instance
    of the named class
    """
    if name == "RetinanetFullFramePipeline":
        pipeline = RetinanetFullFramePipeline(hg)
    elif name == "RetinanetCropFramePipeline":
        pipeline = RetinanetCropFramePipeline(hg)
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

    def __call__(self,frames,priors):
        
        [ids,priors,frame_idx,cam_names] = priors
        
        prepped_frames = self.prep_frames(frames,priors = priors)
        detection_result = self.detect(prepped_frames)
        confs,classes,detections,detection_cam_names  = self.post_detect(detection_result,priors = priors)
        
        # Associate
        matchings = self.associate(ids,priors,detections,self.hg)
        return detections,confs,classes,detection_cam_names,matchings
    
    def set_cam_names(self,cam_names):
        self.this_device_cam_names = cam_names
    
    
    
    
    
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
        if False and self.quantize:
            self.detector = self.detector.half()
        
        # load detector weights
        self.detector.load_state_dict(torch.load(self.weights_file))
    
        self.hg = hg # store homography

        self.cam_names = None
        
    def set_device(self,device_id = -1):
        # configure detector
        self.device = torch.device("cuda:{}".format(device_id) if device_id != -1 else "cpu")
        if device_id != -1:
            torch.cuda.set_device(device_id)
        self.detector = self.detector.to(self.device)
        self.detector.eval()
        
        torch.cuda.set_device(self.device)
        
    def detect(self,frames):
        with torch.no_grad():
            result = self.detector(frames,MULTI_FRAME = True)
        return result
    
    def post_detect(self,detection_result,priors = None):
        confs,classes,detections,detection_idxs = detection_result # detection_idx = which frame from idx each detection is from
        #confs,classes = torch.max(classes, dim = 2) 
        detection_cam_names = [] # dummy value in case no objects returned
        
        # low confidence filter
        if len(confs) > 0:
            mask           = torch.where(confs > self.min_conf,torch.ones(confs.shape,device = confs.device),torch.zeros(confs.shape,device = confs.device)).nonzero()
            confs          = confs[mask]
            classes        = classes[mask]
            detections     = detections[mask]
            detection_idxs = detection_idxs[mask]
        
        # im space NMS 
        if len(confs) > 0 and self.im_nms_iou < 1:
            mask           = im_nms(detections,confs,threshold = self.im_nms_iou,groups = detection_idxs)
            confs          = confs[mask]
            classes        = classes[mask]
            detections     = detections[mask]
            detection_idxs = detection_idxs[mask]
        
        if len(confs) > 0:
            detection_cam_names = [self.cam_names[i] for i in detection_idxs]
            
            # Use the guess and refine method to get box heights
            detections = self.hg.im_to_state(detections,name = detection_cam_names,classes = classes)
            
            # state space NMS
            mask           = space_nms(detections,confs,threshold = self.im_nms_iou)
            confs          = confs[mask]
            classes        = classes[mask]
            detections     = detections[mask]
            detection_cam_names = detection_cam_names[mask]  
        
        # finally, move back to cpu
        detections = detections.cpu()
        confs = confs.cpu()
        classes = classes.cpu()
        
        return [detections,confs,classes,detection_cam_names]
    
    
    
    
        
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