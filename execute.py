import torch.multiprocessing as mp
import time

if __name__ == "__main__":
    #     try:
    #         mp.set_start_method('spawn')
    #     except RuntimeError:
    #         pass
    
    
    
    import torch
    import os
    
    from i24_configparse.parse    import parse_cfg
    from src.track.tracker        import get_Tracker, get_Associator
    from src.track.trackstate     import TrackState
    from src.detect.pipeline      import get_Pipeline
    from src.scene.devicemap      import get_DeviceMap
    from src.scene.homography     import HomographyWrapper
    from src.detect.devicebank    import DeviceBank
    from src.load                 import DummyNoiseLoader,MCLoader
    #from src.db_write             import DBWriter
    
    
    ctx = mp.get_context('spawn')

    
    os.environ["user_config_directory"] = "/home/worklab/Documents/i24/i24_track/config/lambda_quad"
    
    
    # load parameters
    params = parse_cfg("DEFAULT",cfg_name="execute.config")
    
    # verify config notion of GPUs matches torch.cuda notion of available devices
    # get available devices
    
    # TODO fix this once you redo configparse
    params.cuda_devices = [int(i) for i in range(int(params.cuda_devices))]
    assert max(params.cuda_devices) < torch.cuda.device_count()

    
    # intialize DeviceMap
    dmap = get_DeviceMap(params.device_map)
    
    # initialize multi-camera loader
    #loader = DummyNoiseLoader(dmap.camera_mapping_file)
    in_dir = "/home/worklab/Data/dataset_beta/sequence_1"
    loader = MCLoader(in_dir,dmap.camera_mapping_file,ctx)
    
    # initialize Homography object
    hg = HomographyWrapper()
    
    # initialize pipelines
    pipelines = params.pipelines[1:-1].split(",")
    pipelines = [get_Pipeline(item,hg) for item in pipelines]
    
    associators = params.associators[1:-1].split(",")
    associators = [get_Associator(item) for item in associators]
    
    # initialize tracker
    tracker = get_Tracker(params.tracker)
    
    # add Associate function to each pipeline
    for i in range(len(pipelines)):
        assoc = associators[i]
        pipelines[i].associate = associators[i]
    
    # initialize DetectorBank
    dbank = DeviceBank(params.cuda_devices,pipelines,dmap.gpu_cam_names,ctx)
    
    
    
    # initialize DBWriter object
    #db_writer = DBWriter()
    
    
    # intialize empty TrackState Object
    tstate = TrackState()
    
    
    # main loop
    frames_processed = 0
    start_time = time.time()
    while True:
        fps = frames_processed/(time.time() - start_time)
        print("\rTracking frame {} ({:.2f} bps average)".format(frames_processed,fps), end = '\r', flush = True)
    
        
        # compute target time
        target_time = None
        
        # select pipeline for this frame
        pipeline_idx = 0
        
        # get frames and timestamps
        frames,timestamps = loader.get_frames(target_time)
    
        
        camera_idxs,device_idxs,obj_times = dmap(tstate,timestamps)
        obj_ids,priors,selected_obj_idxs = tracker.preprocess(tstate,obj_times)
        
        
         # slice only objects we care to pass to DeviceBank on this set of frames
        # DEREK NOTE may run into trouble here since dmap and preprocess implicitly relies on the list ordering of tstate
        if len(obj_ids) > 0:
            obj_ids     =     obj_ids[selected_obj_idxs]
            priors      =      priors[selected_obj_idxs,:]
            device_idxs = device_idxs[selected_obj_idxs]
            camera_idxs = camera_idxs[selected_obj_idxs]
        
        
        # prep input stack by grouping priors by gpu
        cam_idx_names = None # map idxs to names here
        prior_stack =  dmap.route_objects(obj_ids,priors,device_idxs,camera_idxs,run_device_ids = params.cuda_devices)
        
        # test on a single on-process pipeline
        # pipelines[0].set_device(0)
        # pipelines[0].set_cam_names(dmap.gpu_cam_names[0])
        # test = pipelines[0](frames[0],prior_stack[0])
        
        # TODO select correct pipeline based on pipeline pattern logic parameter
        detections,confs,classes,detection_cam_names,associations = dbank(prior_stack,frames,pipeline_idx = 0)
        
        # THIS MAY BE SLOW SINCE ITS DOUBLE INDEXING
        detection_times = [timestamps[dmap.cam_idxs[cam_name]] for cam_name in detection_cam_names]
    
        terminated_objects = tracker.postprocess(detections,detection_times,classes,confs,associations,tstate)
        
        frames_processed += 1
        
    
        #db_writer(terminated_objects)
        
        # optionally, plot outputs
        
    