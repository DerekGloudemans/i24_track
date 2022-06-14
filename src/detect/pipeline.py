import torch
from torchvision.ops import roi_align
from i24_configparse import parse_cfg
from ..util.bbox import im_nms,space_nms

# imports for specific detectors
from .retinanet_3D.retinanet.model import resnet50 as Retinanet3D

from i24_logger.log_writer import logger,catch_critical

@catch_critical()
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
    
    @catch_critical()
    def __call__(self,frames,priors):
        
        [ids,priors,frame_idx,cam_names] = priors
        
        # no frames assigned to this GPU
        if frames.shape[0] == 0:
            del frames
            return torch.empty([0,6]) , torch.empty(0), torch.empty(0), [], []
        
        prepped_frames = self.prep_frames(frames,priors = priors)
        detection_result = self.detect(prepped_frames)
        detections,confs,classes,detection_cam_names  = self.post_detect(detection_result,priors = priors)
        
        # Associate
        #matchings = self.associate(ids,priors,detections,self.hg)
        matchings = []
        del frames
        return detections,confs,classes,detection_cam_names,matchings
    
    def set_cam_names(self,cam_names):
        self.cam_names = cam_names
    
    @catch_critical()
    def set_device(self,device_id = -1):
        # configure detector
        self.device = torch.device("cuda:{}".format(device_id) if device_id != -1 else "cpu")
        if device_id != -1:
            torch.cuda.set_device(device_id)
        self.detector = self.detector.to(self.device)
        self.detector.eval()
        
        torch.cuda.set_device(self.device)
    
    
    
class RetinanetFullFramePipeline(DetectPipeline):
    """
    DetectPipeline that applies Retinanet Detector on full object frames
    """
    
    def __init__(self,hg):
        
        # load configuration file parameters
        self = parse_cfg("TRACK_CONFIG_SECTION",obj = self)

        
        # initialize detector
        self.detector = Retinanet3D(self.n_classes)
    
        # quantize model         
        if self.quantize:
            self.detector = self.detector.half()
        
        # load detector weights
        self.detector.load_state_dict(torch.load(self.weights_file))
    
        self.hg = hg # store homography

        self.cam_names = None
        
        
    def detect(self,frames):
        
        #logger.debug("Log message test from pipeline: {}".format(self.device))
        
        with torch.no_grad():
            result = self.detector(frames,MULTI_FRAME = True)
        return result
    
    def post_detect(self,detection_result,priors = None):
        confs,classes,detections,detection_idxs = detection_result # detection_idx = which frame from idx each detection is from
        #confs,classes = torch.max(classes, dim = 2) 
        detection_cam_names = [] # dummy value in case no objects returned
        
        
        # reshape detections to form [d,8,2] 
        detections = detections.reshape(-1,10,2)
        detections = detections[:,:8,:] # drop 2D boxes
        
        
        # low confidence filter
        if len(confs) > 0:
            mask           = torch.where(confs > self.min_conf,torch.ones(confs.shape,device = confs.device),torch.zeros(confs.shape,device = confs.device)).nonzero().squeeze(1)
            confs          = confs[mask]
            classes        = classes[mask]
            detections     = detections[mask,:,:]
            detection_idxs = detection_idxs[mask]
        
        # im space NMS 
        if len(confs) > 0 and self.im_nms_iou < 1:
            mask           = im_nms(detections,confs,threshold = self.im_nms_iou,groups = detection_idxs)
            confs          = confs[mask]
            classes        = classes[mask]
            detections     = detections[mask,:,:]
            detection_idxs = detection_idxs[mask]
        
        if len(confs) > 0:
            detection_cam_names = [self.cam_names[i] for i in detection_idxs]
            
            # Use the guess and refine method to get box heights
            detections = self.hg.im_to_state(detections,name = detection_cam_names,classes = classes)
            detections = self.hg.state_to_space(detections)
            # state space NMS
            mask           = space_nms(detections,confs,threshold = self.space_nms_iou)
            confs          = confs[mask]
            classes        = classes[mask]
            detections     = detections[mask,:,:]
            detection_idxs = detection_idxs[mask]
            detection_cam_names = [self.cam_names[i] for i in detection_idxs]
            
            if len(confs) > 0:
                detections = self.hg.space_to_state(detections)
        
        # finally, move back to cpu
        detections = detections.cpu()
        confs = confs.cpu()
        classes = classes.cpu()
                
        return [detections,confs,classes,detection_cam_names]
    
    
    
    
        
class RetinanetCropFramePipeline(DetectPipeline):
        
    def __init__(self,hg,):
        #  load configuration file parameters
        self = parse_cfg("TRACK_CONFIG_SECTION",obj = self)

        
        # initialize detector
        self.detector = Retinanet3D(self.n_classes)
    
        # quantize model         
        if self.quantize:
            self.detector = self.detector.half()
        
        # load detector weights
        self.detector.load_state_dict(torch.load(self.weights_file))
    
        self.hg = hg # store homography

        self.cam_names = None
        
    @catch_critical()
    def __call__(self,frames,priors):
        
        #[ids,priors,frame_idx,cam_names] = priors
        
        # no frames assigned to this GPU
        if frames.shape[0] == 0:
            del frames
            return torch.empty([0,6]) , torch.empty(0), torch.empty(0), [], []
        
        
        prepped_frames,crop_boxes = self.prep_frames(frames,priors = priors)
        detection_result = self.detect(prepped_frames)
        detections,classes,confs,detection_cam_names,matchings  = self.post_detect(detection_result,priors,crop_boxes)
        

    
    
        del frames
        return detections,confs,classes,detection_cam_names,matchings
        
    
    def prep_frames(self,frames,priors):
        """
        priors - list of [ids,priors (nx6 tensor), frame_idx in stack, cam_name for prior]
        frames - 
        """
        ids,objs,frame_idxs,cam_names = priors
        
        # 1. State to im
        objs_im = self.hg.state_to_im(objs,name = cam_names)
    
        # 2. Get expanded boxes
        crop_boxes = self._get_crop_boxes(objs_im)
        
        # 3. Crop
        cidx = frame_idxs.unsqueeze(1).to(self.device).double()
        torch_boxes = torch.cat((cidx,crop_boxes),dim = 1)
        crops = roi_align(frames,torch_boxes.float(),(self.crop_size,self.crop_size))
        
        return crops,crop_boxes
        
        
    def detect(self,frames):
        with torch.no_grad():
            result = self.detector(frames,LOCALIZE = True)
        return result
    
    def post_detect(self,detection_result,priors,crop_boxes):

        ids,objs,frame_idxs,cam_names = priors
        
        reg_boxes, classes = detection_result
        confs,classes = torch.max(classes, dim = 2)

    
        # 1. Keep top k boxes
        top_idxs = torch.topk(confs,self.keep_boxes,dim = 1)[1]
        row_idxs = torch.arange(reg_boxes.shape[0]).unsqueeze(1).repeat(1,top_idxs.shape[1])
        
        reg_boxes = reg_boxes[row_idxs,top_idxs,:,:]
        confs =  confs[row_idxs,top_idxs]
        classes = classes[row_idxs,top_idxs] 
        
        # 2. local to global mapping
        reg_boxes = self.local_to_global(reg_boxes,crop_boxes)
        
        # 3. Convert to space
        n_objs = reg_boxes.shape[0]
        cam_names_repeated = [cam for cam in cam_names for i in range(reg_boxes.shape[1])]
        reg_boxes = reg_boxes.reshape(-1,8,2)
        reg_boxes_state = self.hg.im_to_state(reg_boxes,name = cam_names_repeated)
        
 
        # 4. Select best box
        detections, classes, confs = self.select_best_box(objs,reg_boxes_state,confs,classes,n_objs)
        
        
        # note that cam_names and ids directly become detection cameras and detection ids - implicit matching
        
        return detections,classes,confs,cam_names,ids
        
        
    def _get_crop_boxes(self,objects):
            """
            Given a set of objects, returns boxes to crop them from the frame
            objects - [n,8,2] array of x,y, corner coordinates for 3D bounding boxes
            
            returns [n,4] array of xmin,xmax,ymin,ymax for cropping each object
            """
            
            # find xmin,xmax,ymin, and ymax for 3D box points
            minx = torch.min(objects[:,:,0],dim = 1)[0]
            miny = torch.min(objects[:,:,1],dim = 1)[0]
            maxx = torch.max(objects[:,:,0],dim = 1)[0]
            maxy = torch.max(objects[:,:,1],dim = 1)[0]
            
            w = maxx - minx
            h = maxy - miny
            scale = torch.max(torch.stack([w,h]),dim = 0)[0] * self.box_expansion_ratio
            
            # find a tight box around each object in xysr formulation
            minx2 = (minx+maxx)/2.0 - scale/2.0
            maxx2 = (minx+maxx)/2.0 + scale/2.0
            miny2 = (miny+maxy)/2.0 - scale/2.0
            maxy2 = (miny+maxy)/2.0 + scale/2.0
            
            crop_boxes = torch.stack([minx2,miny2,maxx2,maxy2]).transpose(0,1).to(self.device)
            return crop_boxes
            