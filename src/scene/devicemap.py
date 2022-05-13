from i24_configparse.parse import parse_cfg
import configparser
import torch
import re
    


def get_DeviceMap(name):
    """
    getter function that takes a string (class name) input and returns an instance
    of the named class
    """
    if name == "DeviceMap":
        dmap = DeviceMap()
    elif name == "HeuristicDeviceMap":
        dmap = HeuristicDeviceMap()
    else:
        raise NotImplementedError("No DeviceMap child class named {}".format(name))
    
    return dmap


    
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
        cam_names = []
        extents = []
        [(cam_names.append(key),extents.append(self.cam_extents[key])) for key in self.cam_extents.keys()]
        self.cam_names = cam_names
        self.cam_extents = torch.tensor(extents)

        
        # load self.cam_devices
        self._parse_device_mapping(self.camera_mapping_file)
        
        # load mapping function
        if map_fn is None:
            map_fn = self.default_map_cameras
        self.map_cameras = map_fn
        
        
        # note that self.cam_names is THE ordering of cameras, and all other camera orderings should be relative to this ["p1c1","p1c2", etc]
        # self.gpu_cam_ids is THE ordering of cameras per GPU (list of lists) [0,0,0,1,1,1 etc.]
        # and self.cam_devices[i] contains device index for camera i in self.cam_ids [[p1c1,p1c2],[p1c3,p1c4] etc]
        self.gpu_cam_names = [[] for i in range(self.n_devices)]
        [self.gpu_cam_names[self.cam_devices[i]].append(self.cam_names[i]) for i in range(len(self.cam_names))]
        
        
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
    
    def map_devices(self,cam_map):
        """
        :param cameras - list of camera names of size n
        :return gpus - tensor of GPU IDs (int) for each of n input cameras
        """
        
        gpu_map = torch.tensor([self.cam_devices[camera] for camera in cam_map])
        return gpu_map
    
    def __call__(self,tstate,ts):
        cam_map = self.map_cameras(tstate,ts)
        gpu_map = self.map_devices(cam_map)
        
        # get times
        
        return 
    
    def route_objects(self,cam_map,gpu_map,obj_ids,obj_priors):
        """
        
        """
        pass
        

    
class HeuristicDeviceMap(DeviceMap):
    
    def __init__(self):
        super(HeuristicDeviceMap, self).__init__()
        
        # add camera priority
        priority_dict = {"c1":1,
                    "c2":100,
                    "c3":1000,
                    "c4":1000,
                    "c5":100,
                    "c6":1}

        self.priority = [priority_dict[re.search("c\d",cam).group(0)] for cam in self.cam_names]
    
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
    