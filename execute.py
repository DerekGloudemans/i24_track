import torch


from i24_configparse.parse    import parse_cfg
from src.track.Tracker        import get_Tracker
from src.track.TrackState     import TrackState
from src.detect.detectors     import get_Pipeline
from src.scene.Map            import get_DeviceMap
from src.scene.Homography     import Homography_Wrapper
from src.detect.DetectorBank  import DetectorBank
from src.load                 import MCLoader
from src.db_write             import DBWriter


# get available devices
devices = torch.get_devices()



# load parameters
params = parse_cfg("run.spec","DEBUG")

# verify config notion of GPUs matches torch.cuda notion of available devices


# initialize multi-camera loader
loader = MCLoader()


# intialize DeviceMap
dmap = get_DeviceMap(params.device_map)

# initialize pipelines
pipelines = params.pipelines.split(",")
pipelines = [get_Pipeline(item) for item in pipelines]


# initialize tracker
tracker = get_Tracker

# add Tracker.associate function to each pipeline

# initialize Homography object
hg = Homography_Wrapper()

# initialize DetectorBank
dbank = DetectorBank(devices,pipelines,hg)



# initialize DBWriter object
db_writer = DBWriter()


# intialize empty TrackState Object
tstate = TrackState()


# main loop
while True:
    
    # compute target time
    target_time = None
    
    # select pipeline for this frame
    pipeline_idx = 0
    
    # get frames and timestamps
    frames,timestamps = loader.get_frames(target_time)

    
    device_idxs,camera_idxs,obj_times = dmap.apply(tstate,timestamps)
    obj_ids,priors,selected_obj_idxs = tracker.preprocess(tstate,obj_times)
    
    # slice only objects we care to pass to DeviceBank on this set of frames
    # DEREK NOTE may run into trouble here since dmap and preprocess implicitly relies on the list ordering of tstate
    obj_ids     =     obj_ids[selected_obj_idxs]
    priors      =      priors[selected_obj_idxs,:]
    device_idxs = device_idxs[selected_obj_idxs]
    camera_idxs = camera_idxs[selected_obj_idxs]
    
    
    # map and mov
    cam_idx_names = None # map idxs to names here
    #device_priors = dmap.map_priors(priors,device_idxs,camera_idxs)
    
    detections,associations = dbank(obj_ids,priors,frames,device_idxs,camera_idxs,pipeline_idx = 0)
    
    terminated_objects = tracker.post_process(detections,associations)
    
    db_writer(terminated_objects)
    
    # optionally, plot outputs
    
    