from i24_configparse.parse import parse_cfg
import configparser
from homography import Homography_Wrapper
import torch
import re
    
    
class DeviceMap():
    """
    DeviceMap objects have two main functions:            
        1.) camera mapping - given a TrackState object, and frame times for each camera
            takes a TrackState and a frame_time from each camera, and outputs a 
            set of object - camera query pairs. This set may be underfull
            (not every object is queried) or overfull (some objects are queried more than once).
            
        2.) GPU mapping - map_devices function returns manages the "pointers"
            to GPU-side frames, returning the correct GPU device and frame index for each camera
            
            Together, these two functions create a set of object queries indexing frames per GPU, which
            can then be passed to the DetectorBank to split across devices and perform any cropping etc.
    """
    
    def __init__(self,map_fn = None):
        
        # load config
        self = parse_cfg("DEFAULT",obj = self)

        # load self.cam_extents
        self._parse_cameras(self.camera_extents_file)
        
        # convert camera extents to tensor
        cam_ids = []
        extents = []
        [(cam_ids.append(key),extents.append(self.cam_extents[key])) for key in self.cam_extents.keys()]
        self.cam_ids = cam_ids
        self.cam_extents = torch.tensor(extents)
        
        
        priority_dict = {"c1":1,
                    "c2":100,
                    "c3":1000,
                    "c4":1000,
                    "c5":100,
                    "c6":1}

        self.priority = [priority_dict[re.search("c\d",cam).group(0)] for cam in self.cam_ids]

        
        # load self.cam_devices
        self._parse_device_mapping(self.camera_mapping_file)
        
        # load mapping function
        if map_fn is None:
            map_fn = self.default_map_cameras
        self.map_cameras = map_fn
        
        
    def _parse_cameras(self,extents_file):
        """
        This function is likely to change in future versions. For now, config file is expected to 
        express camera range as minx,miny,maxx,maxy e.g. p1c1=100,-10,400,120
        :param extents_file - (str) name of file with camera extents
        :return dict with same information in list form p1c1:[100,-10,400,120]
        """
        
        cp = configparser.ConfigParser()
        cp.read(extents_file)
        extents = dict(cp["DEFAULT"])
       
        for key in extents.keys():
            parsed_val = [int(item) for item in extents[key].split(",")]
            extents[key] = parsed_val
    
        self.cam_extents = extents
                                
    def _parse_device_mapping(self,mapping_file):
        """
        This function is likely to change in future versions. For now, config file is expected to 
        express camera device as integer e.g. p1c1=3
        :param mapping_file - (str) name of file with camera mapping
        :return dict with same information p1c1:3
        """
        cp = configparser.ConfigParser()
        cp.read(mapping_file)
        mapping = dict(cp["DEFAULT"])
       
        for key in mapping.keys():
            parsed_val = int(mapping[key])
            mapping[key] = parsed_val
    
        self.cam_devices = mapping       
    
    def map_cameras(self):
        raise NotImplementedError
    
    def map_devices(self,cameras):
        """
        :param cameras - list of camera names of size n
        :return gpus - tensor of GPU IDs (int) for each of n input cameras
        """
        
        gpus = torch.tensor([self.cam_devices[camera] for camera in cameras])
        return gpus
    

    
class HeuristicDeviceMap1(DeviceMap):
    def map_cameras(self,tstate,ts):
        """
        MAPPING:
            constraints:
                object is within camera range
                camera reports timestamp
            preferences:
                interior camers (c3 and c4, then c2 and c5, then c1 and c6)
                camera with center of FOV closest to object
            
        :param - tstate - TrackState object
        :param - ts - list of size [n_cameras] with reported timestamp for each, or torch.nan if ts is invalid
        
        :return list of size n_objs with camera name for each
        """
        
        # TODO - may need to map ts into the correct order of self.cam_ids
        
        # get object positions as tensor
        ids,states = tstate() # [n_objects,state_size]
        
        # store useful dimensions
        n_c = len(self.cam_ids)
        n_o = len(self.ids)
        
        
        
        ## create is_visible, [n_objects,n_cameras]

        # broadcast both to [n_objs,n_cameras,2]
        states_brx = states[:,0].unsqueeze(1).unsqueeze(2).expand(n_o,n_c,2)
        states_bry = states[:,1].unsqueeze(1).unsqueeze(2).expand(n_o,n_c,2)
        cams_brx = self.extents[:,[0,2]].unsqueeze(0).expand(n_o,n_c,2)
        cams_bry = self.extents[:,[1,3]].unsqueeze(0).expand(n_o,n_c,2)
        
        # get map of where state is within range
        x_pass = torch.sign(torch.mul(states_brx,cams_brx))
        x_pass = (torch.mul(x_pass[:,0],x_pass[:,1]) -1 )* -2  # 1 if inside, 0 if outside
        y_pass = torch.sign(torch.mul(states_bry,cams_bry))
        y_pass = (torch.mul(y_pass[:,0],y_pass[:,1]) -1 )* -2  # 1 if inside, 0 if outside
        
        is_visible = torch.mul(x_pass,y_pass)
        
        
        
        ## create ts_valid, [n_objs,n_cameras]
        ts_valid = torch.nan_to_num(ts).clamp(0,1).int().unsqueeze(0).expand(n_o,n_c)
        
        # create priority, [n_objs,n_cameras]
        priority = self.priority.unsqueeze(0).expand(n_o,n_c)
        
        # create distance, [n_objs,n_cameras]        
        center_x = cams_brx.sum(dim = 1)
        center_y = cams_bry.sum(dim = 1)
        
        dist = ((center_x - states_brx[:,:,0]).pow(2) + (center_y - states_bry[:,:,0]).pow(2)).sqrt()
        
        score = 1/dist * priority * ts_valid * is_visible
        
        cam_map = score.argmax(dim = 1)
        
        
        #TODO need to unmap cameras to idxs here?
        
        return cam_map
    