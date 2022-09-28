import torch.multiprocessing as mp
import socket
import _pickle as pickle
import numpy as np
import torch
import os,shutil
import time

os.environ["USER_CONFIG_DIRECTORY"] = "/home/derek/Documents/i24/i24_track/config/lambda_cerulean_eval2"

from i24_logger.log_writer         import logger,catch_critical,log_warnings

#mp.set_sharing_strategy('file_system')



from src.util.bbox                 import space_nms,estimate_ts_offsets
from src.util.misc                 import plot_scene,colors,Timer
from i24_configparse               import parse_cfg
from src.track.tracker             import get_Tracker, get_Associator
from src.track.trackstate          import TrackierState as TrackState
from src.detect.pipeline           import get_Pipeline
from src.scene.devicemap           import get_DeviceMap
from src.scene.homography          import HomographyWrapper,Homography
from src.detect.devicebank         import DeviceBank
#from src.load.gpu_load_multi_presynced       import MCLoader, ManagerClock
from src.db_write                  import WriteWrapperConf as WriteWrapper

#from src.log_init                  import logger
#from i24_logger.log_writer         import logger,catch_critical,log_warnings



#from src.load.cpu_load             import DummyNoiseLoader #,MCLoader
#from src.load.gpu_load             import MCLoader




@log_warnings()
def parse_cfg_wrapper(run_config):
    params = parse_cfg("TRACK_CONFIG_SECTION",
                       cfg_name=run_config, SCHEMA=False)
    return params



@catch_critical()
def checkpoint(tstate,next_target_time,collection_overwrite,save_file = "working_checkpoint.cpkl"):
    """
    Saves the trackstate and next target_time as a pickled object such that the 
    state of tracker can be reloaded for no loss in tracking progress
    
    :param   tstate - TrackState object
    :param   next_target_time - float
    :return  None
    """
    
    with open(save_file,"wb") as f:
        pickle.dump([next_target_time,tstate,collection_overwrite],f)
    logger.debug("Checkpointed TrackState object, time:{}s".format(next_target_time))

        
@catch_critical()
def load_checkpoint(target_time,tstate,collection_overwrite,save_file = "working_checkpoint.cpkl"):
    """
    Loads the trackstate and next target_time from pickled object such that the 
    state of tracker can be reloaded for no loss in tracking progress. Requires 
    input time and tstate such that objects can be naively passed through if no 
    save file exists
    
    :param   tstate - TrackState object
    :param   next_target_time - float
    :return  None
    """    
    if os.path.exists(save_file):
        with open(save_file,"rb") as f:
            target_time,tstate,collection_overwrite = pickle.load(f)
        
        logger.debug("Loaded checkpointed TrackState object, time:{}s".format(target_time))
        
    else:
        logger.debug("No checkpoint file exists, starting tracking from max min video timestamp")
        
    return target_time,tstate,collection_overwrite
        
@catch_critical()
def soft_shutdown(target_time,tstate,collection_overwrite,cleanup = []):
    logger.warning("Soft Shutdown initiated. Either SIGINT or KeyboardInterrupt recieved")
    checkpoint(tstate,target_time,collection_overwrite,save_file = "working_checkpoint.cpkl")
    
    for i in range(len(cleanup)-1,-1,-1):
        del cleanup[i]
    
    logger.debug("Soft Shutdown complete. All processes should be terminated")
    raise KeyboardInterrupt()

def main(collection_overwrite = None):  
    
    
        
        
    ctx = mp.get_context('spawn')
    
    from i24_logger.log_writer         import logger,catch_critical,log_warnings
    logger.set_name("Tracking Main")
    
    #%% Old run settings and setup
    
    # if False: 
    #     run_config = "execute.config"       
    #     #os.environ["user_config_directory"] = "/home/derek/Documents/i24/i24_track/config/lambda_cerulean"
    #     #os.environ["TRACK_CONFIG_SECTION"] = "DEFAULT"
    #     in_dir = "/home/derek/Data/dataset_beta/sequence_1"
        
        
    #     # load parameters
    #     params = parse_cfg("TRACK_CONFIG_SECTION",
    #                         cfg_name=run_config, SCHEMA=False)
        
    #     # verify config notion of GPUs matches torch.cuda notion of available devices
    #     # get available devices
        
    #     # # initialize logger
    #     # try:
    #     #     log_params = {
    #     #                 "log_name":"Tracking Session: {}".format(np.random.randint(0,1000000)),
    #     #                 "processing_environment":os.environ["TRACK_CONFIG_SECTION"],
    #     #                 "logstash_address":(params.log_host_ip,params.log_host_port),
    #     #                 "connect_logstash": (True if "logstash" in params.log_mode else False),
    #     #                 "connect_syslog":(True if "syslog" in params.log_mode else False),
    #     #                 "connect_file": (True if "file" in params.log_mode else False),
    #     #                 "connect_console":(True if "sysout" in params.log_mode else False),
    #     #                 "console_log_level":params.log_level
    #     #                 }
            
    #     #     logger = connect_automatically(user_settings = log_params)
    #     #     logger.debug("Logger initialized with custom log settings",extra = log_params)
    #     # except:
    #     #     logger.debug("Logger initialized with default parameters")
        
        
        
    #     # TODO fix this once you redo configparse
    #     params.cuda_devices = [int(i) for i in range(int(params.cuda_devices))]
    #     assert max(params.cuda_devices) < torch.cuda.device_count()
        
    #     # intialize DeviceMap
    #     dmap = get_DeviceMap(params.device_map)
    
    #     from src.load.gpu_load             import MCLoader
    #     loader = MCLoader(in_dir, dmap.camera_mapping_file, ctx)
    
    
    
    #%% run settings    
    tm = Timer()
    tm.split("Init")
    
    run_config = "execute.config"       
    #mask = ["p2c1", "p2c3","p2c5","p3c1"] #["p46c01","p46c02", "p46c03", "p46c04", "p46c05","p46c06"]
    mask = None
    
    # load parameters
    params = parse_cfg_wrapper(run_config)
    
    in_dir = params.input_directory
    
    # # initialize logger
    # try:
    #     log_params = {
    #                 "log_name":"Tracking Session: {}".format(np.random.randint(0,1000000)),
    #                 "processing_environment":os.environ["TRACK_CONFIG_SECTION"],
    #                 "logstash_address":(params.log_host_ip,params.log_host_port),
    #                 "connect_logstash": (True if "logstash" in params.log_mode else False),
    #                 "connect_syslog":(True if "syslog" in params.log_mode else False),
    #                 "connect_file": (True if "file" in params.log_mode else False),
    #                 "connect_console":(True if "sysout" in params.log_mode else False),
    #                 "console_log_level":params.log_level
    #                 }
        
    #     logger = connect_automatically(user_settings = log_params)
    #     logger.debug("Logger initialized with custom log settings",extra = log_params)
    # except:
    #     logger.debug("Logger initialized with default parameters")
    
    # verify config notion of GPUs matches torch.cuda notion of available devices
    # get available devices
    
    
    
    # TODO fix this once you redo configparse
    params.cuda_devices = [int(i) for i in range(int(params.cuda_devices))]
    assert max(params.cuda_devices) < torch.cuda.device_count()
    
    # intialize DeviceMap
    dmap = get_DeviceMap(params.device_map)
    
    # intialize empty TrackState Object
    tstate = TrackState()
    target_time = None
    
    # load checkpoint
    target_time,tstate,collection_overwrite = load_checkpoint(target_time,tstate,collection_overwrite)
    
    try:
        ts_file = params.timestamp_file
        from src.load.gpu_load_GT_presynced import MCLoader
        loader = MCLoader(in_dir, ts_file, dmap.camera_mapping_file, ctx,Hz = params.nominal_framerate)
        #loader.get_frames()
        print("Using GT Loader")
    except:
        from src.load.gpu_load_multi_presynced       import MCLoader
        loader = MCLoader(in_dir,dmap.camera_mapping_file,dmap.cam_names, ctx,start_time = target_time,Hz = params.nominal_framerate)
        print("Using Multi Loader")
        target_time = loader.start_time
    
    logger.debug("Initialized {} loader processes.".format(len(loader.device_loaders)))
    
    #%% more init stuff 
    
    # initialize Homography object
    hg = HomographyWrapper(hg1 = params.eb_homography_file,hg2 = params.wb_homography_file)
     
    if params.track:
        # initialize pipelines
        pipelines = params.pipelines
        pipelines = [get_Pipeline(item, hg) for item in pipelines]
        
        associators = params.associators
        associators = [get_Associator(item) for item in associators]
        
        # initialize tracker
        tracker = get_Tracker(params.tracker)
        
        # add Associate function to each pipeline
        # for i in range(len(pipelines)):
        #     assoc = associators[i]
        #     pipelines[i].associate = associators[i]
        
        # initialize DetectorBank
        dbank = DeviceBank(params.cuda_devices, pipelines, dmap.gpu_cam_names, ctx)
        
    # initialize DBWriter object
    if params.write_db:
        dbw = WriteWrapper(collection_overwrite = collection_overwrite)
    else:
        dbw = []
    

    
    # get frames and timestamps
    frames, timestamps = loader.get_frames()
    
    # initialize processing sync clock
    start_ts = max(timestamps)

        
    nom_framerate = params.nominal_framerate 
    #clock  = ManagerClock(start_ts,params.desired_processing_speed, nom_framerate)
    target_time = start_ts
    #target_time = clock.tick()
    
    
    # initial sync-up of all cameras
    # TODO - note this means we always skip at least one frame at the beginning of execution
    # frames,timestamps = loader.get_frames(target_time)
    # ts_trunc = [item - start_ts for item in timestamps]
    
    frames_processed = 0
    term_objects = 0
    
    # # plot first frame
    # if params.plot:
    #     plot_scene(tstate, frames, ts_trunc, dmap.gpu_cam_names,
    #          hg, colors,extents=dmap.cam_extents_dict, mask=mask,fr_num = frames_processed,detections = None)
    
    
    ts_offsets_all = []
    
    #%% Main Processing Loop
    start_time = time.time()
    
    logger.debug("Initialization Complete. Starting tracking at {}s".format(start_time))
    
    end_time = np.inf
    if params.end_time != -1:
        end_time = params.end_time
        
    # readout headers
    try:
        print("\n\nFrame:    Since Start:  Frame BPS:    Sync Timestamp:     Max ts Deviation:     Active Objects:    Written Objects:")
        while target_time < end_time:
                

            
            if params.track: # shortout actual processing
                
                # select pipeline for this frame
                pidx = frames_processed % len(params.pipeline_pattern)
                pipeline_idx = params.pipeline_pattern[pidx]
        
                if pipeline_idx != -1: # -1 means skip frame
                    
                    # get next frames and timestamps
                    tm.split("Get Frames")
                    frames, timestamps = loader.get_frames()
                    #print(frames_processed,timestamps[0],target_time) # now we expect almost an exactly 30 fps framerate and exactly 30 fps target framerate
                    
                    if frames is None:
                        logger.warning("Ran out of input. Tracker is shutting down")
                        break #out of input
                    ts_trunc = [item - start_ts for item in timestamps]
                    
                    # for obj in initializations:
                    #     obj["timestamp"] -= start_ts
                    initializations = None
                        
                    ### WARNING! TIME ERROR INJECTION
                    # ts_trunc[3] += 0.05
                    # ts_trunc[10] += .1
                    
                    tm.split("Predict")
                    camera_idxs, device_idxs, obj_times, selected_obj_idxs = dmap(tstate, ts_trunc)
                    obj_ids, priors, _ = tracker.preprocess(
                        tstate, obj_times)
                    
                    # slice only objects we care to pass to DeviceBank on this set of frames
                    # DEREK NOTE may run into trouble here since dmap and preprocess implicitly relies on the list ordering of tstate
                    if len(obj_ids) > 0:
                        obj_ids     =     obj_ids[selected_obj_idxs]
                        priors      =      priors[selected_obj_idxs,:]
                        device_idxs = device_idxs[selected_obj_idxs]
                        camera_idxs = camera_idxs[selected_obj_idxs]
                    
                    # prep input stack by grouping priors by gpu
                    tm.split("Map")
                    cam_idx_names = None  # map idxs to names here
                    prior_stack = dmap.route_objects(
                        obj_ids, priors, device_idxs, camera_idxs, run_device_ids=params.cuda_devices)
            
                    # test on a single on-process pipeline
                    # pipelines[pipeline_idx].set_device(1)
                    # pipelines[pipeline_idx].set_cam_names(dmap.gpu_cam_names[1])
                    # test = pipelines[pipeline_idx](frames[1],prior_stack[1])
            
            
                    # TODO - full frame detections should probably get full set of objects?
                    # TODO select correct pipeline based on pipeline pattern logic parameter
                    tm.split("Detect {}".format(pipeline_idx),SYNC = True)
                    detections, confs, classes, detection_cam_names, associations = dbank(
                        prior_stack, frames, pipeline_idx=pipeline_idx)
                    
                    detections = detections.float()
                    confs = confs.float()
                    
                    # THIS MAY BE SLOW SINCE ITS DOUBLE INDEXING
                    detection_times = torch.tensor(
                        [ts_trunc[dmap.cam_idxs[cam_name]] for cam_name in detection_cam_names])
                    
                    if True and pipeline_idx == 0:
                        keep = dmap.filter_by_extents(detections,detection_cam_names)
                        detections = detections[keep,:]
                        confs = confs[keep]
                        classes = classes[keep]
                        detection_times = detection_times[keep]
                        detection_cam_names = [detection_cam_names[_] for _ in keep]
                    
                    if False:
                        ts_offsets = estimate_ts_offsets(detections,detection_cam_names,detection_times,tstate,hg)
                        if ts_offsets is not None:
                            ts_offsets_all.append(ts_offsets)
                            #logger.info("Camera clock offset estimates: MAX {}s".format(max(np.max(np.array(ts_offsets.values())))),extra = ts_offsets)
                        
                    tm.split("Associate",SYNC = True)
                    if pipeline_idx == 0:
                        detections_orig = detections.clone()
                        if True and len(detections) > 0:
                            # do nms across all device batches to remove dups
                            space_new = hg.state_to_space(detections)
                            keep = space_nms(space_new,confs)
                            detections = detections[keep,:]
                            classes = classes[keep]
                            confs = confs[keep]
                            detection_times = detection_times[keep]
                        
                        # overwrite associations here
                        associations = associators[0](obj_ids,priors,detections,hg)
            
                    tm.split("Postprocess")
                    terminated_objects,cause_of_death = tracker.postprocess(
                        detections, detection_times, classes, confs, associations, tstate, hg = hg,measurement_idx =0)
                    term_objects += len(terminated_objects)
        
                    tm.split("Write DB")
                    if params.write_db:
                        dbw.insert(terminated_objects,cause_of_death = cause_of_death,time_offset = start_ts)
                    #print("Active Trajectories: {}  Terminated Trajectories: {}   Documents in database: {}".format(len(tstate),len(terminated_objects),len(dbw)))
        
    
            # optionally, plot outputs
            if params.plot:
                tm.split("Plot")
                #detections = None
                priors = None
                plot_scene(tstate, 
                           frames, 
                           ts_trunc, 
                           dmap.gpu_cam_names,
                           hg, 
                           colors,
                           extents=dmap.cam_extents_dict, 
                           mask=mask,
                           fr_num = frames_processed,
                           detections = detections,
                           priors = priors,
                           save_crops_dir=None)
    
            
            # text readout update
            tm.split("Bookkeeping")
            fps = (max(timestamps) - start_ts)/(time.time() - start_time) * 30
            max_dev = max(timestamps) - min(timestamps)
            #max_dev = max(dev)
            print("\r{}        {:.3f}s       {:.2f}        {:.3f}              {:.3f}                {}               {}".format(frames_processed, time.time() - start_time,fps,max(timestamps), max_dev, len(tstate), len(dbw)), end='\r', flush=True)
        
            # get next target time
            #target_time = clock.tick()
            frames_processed += 1
            
    
            if frames_processed % 50 == 1:
                metrics = {
                    "frame bps": fps,
                    "frame batches processed":frames_processed,
                    "run time":time.time() - start_time,
                    "scene time processed":target_time - start_ts,
                    "active objects":len(tstate),
                    "total terminated objects":term_objects,
                    "avg skipped frames per processed frame": nom_framerate*(target_time - start_ts)/frames_processed -1
                    }
                logger.info("Tracking Status Log",extra = metrics)
                logger.info("Time Utilization: {}".format(tm),extra = tm.bins())
                
            if params.checkpoint and frames_processed % 500 == 0:
                checkpoint(target_time,tstate,collection_overwrite)
       
        
        #checkpoint(target_time,tstate,collection_overwrite)
        logger.info("Finished tracking over input time range. Shutting down.")
        with open ("ts_offset_file.cpkl","wb") as f:
            pickle.dump(ts_offsets_all,f)
        
        if True: # Flush tracker objects
            residual_objects,COD = tracker.flush(tstate)
            dbw.insert(residual_objects,cause_of_death = COD,time_offset = start_ts)
            logger.info("Flushed all active objects to database",extra = metrics)


        if collection_overwrite is not None:
            # cache settings in new folder
            cache_dir = "./data/config_cache/{}".format(collection_overwrite)
            #os.mkdir(cache_dir)
            shutil.copytree(os.environ["USER_CONFIG_DIRECTORY"],cache_dir)    
            logger.debug("Cached run settings in {}".format(cache_dir))
        
    except KeyboardInterrupt:
        logger.debug("Keyboard Interrupt recieved. Initializing soft shutdown")

        if True: # Flush tracker objects
            residual_objects,COD = tracker.flush(tstate)
            try:
                dbw.insert(residual_objects,cause_of_death = COD,time_offset = start_ts)
                logger.info("Flushed all active objects to database",extra = metrics)
            except:
                logger.warning("Failed to flush active objects at end of tracking. Is write_db = False?")
            
        soft_shutdown(target_time, tstate,collection_overwrite,cleanup = [dbw,loader,dbank])
     
    
    
    return fps
        
     
if __name__ == "__main__":
    
    main()
    
    if True:
        import cv2
        import os
        import requests
        from datetime import datetime

        def im_to_vid(directory,name = "video",push_to_dashboard = False): 
            all_files = os.listdir(directory)
            all_files.sort()
            for filename in all_files:
                filename = os.path.join(directory, filename)
                img = cv2.imread(filename)
                height, width, layers = img.shape
                size = (width,height)
                break
            
            n = 0
            
            now = datetime.now()
            now = now.strftime("%Y-%m-%d_%H-%M-%S")
            f_name = os.path.join("/home/derek/Desktop",'{}_{}.mp4'.format(now,name))
            temp_name =  os.path.join("/home/derek/Desktop",'temp.mp4')
            
            out = cv2.VideoWriter(temp_name,cv2.VideoWriter_fourcc(*'mp4v'), 8, size)
             
            for filename in all_files:
                filename = os.path.join(directory, filename)
                img = cv2.imread(filename)
                out.write(img)
                print("Wrote frame {}".format(n))
                n += 1
                
                # if n > 30:
                #     break
            out.release()
            
            os.system("/usr/bin/ffmpeg -i {} -vcodec libx264 {}".format(temp_name,f_name))
            
            if push_to_dashboard:
                
                
                #snow = now.strftime("%Y-%m-%d_%H-%M-%S")
                url = 'http://viz-dev.isis.vanderbilt.edu:5991/upload?type=boxes_raw'
                files = {'upload_file': open(f_name,'rb')}
                ret = requests.post(url, files=files)
                print(f_name)
                print(ret)
                if ret.status_code == 200:
                    print('Uploaded!')
                    
                    
        file = "/home/worklab/Data/cv/KITTI/data_tracking_image_2/training/image_02/0000"
        file = "/home/worklab/Documents/derek/track_i24/output/temp_frames"
        file = "/home/worklab/Documents/derek/LBT-count/vid"
        file = "/home/worklab/Documents/derek/3D-playground/video/6"
        file = "/home/derek/Desktop/temp_frames"


            
        #file  = '/home/worklab/Desktop/temp'
        im_to_vid(file,name = "latest_greatest",push_to_dashboard = True)