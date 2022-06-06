import torch.multiprocessing as mp
import time

import numpy as np
import cv2
from torchvision import transforms

from src.util.bbox import im_nms,space_nms


def plot(tstate, frames, ts, gpu_cam_names, hg, colors, mask=None, extents=None, fr_num = 0):
    """
    Plots the set of active cameras, or a subset thereof
    tstate - TrackState object
    ts     - stack of camera timestamps
    frames - stack of frames as pytorch tensors
    hg     - Homography Wrapper object
    mask   - None or list of camera names to be plotted
    extents - None or cam extents from dmap.cam_extents_dict
    """

    # Internal settings that shouldn't change after initial tuning
    PLOT_TOLERANCE = 50  # feet
    MONITOR_SIZE = (2160, 3840)
    denorm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                  std=[1/0.229, 1/0.224, 1/0.225])
    #mask = ["p1c1"]

    # 1. prep frames
    # move to CPU
    frames = [item.cpu() for item in frames]

    # stack all frames into single list
    frames = torch.cat(frames, dim=0)
    #ts =  torch.cat(ts,dim = 0)
    cam_names = [item for sublist in gpu_cam_names for item in sublist]


    print(cam_names)
    # get mask
    if mask is not None:
        keep = []
        keep_cam_names = []
        for idx, cam in enumerate(cam_names):
            if cam in mask:
                keep.append(idx)
                keep_cam_names.append(cam)

        # mask relevant cameras
        cam_names = keep_cam_names
        ts = [ts[idx] for idx in keep]
        frames = frames[keep, ...]

    class_by_id = tstate.get_classes()

    print(cam_names)
    
    # 2. plot boxes
    # for each frame
    plot_frames = []
    for f_idx in range(len(frames)):

        # get the reported position of each object from tstate for that time
        ids, boxes = tstate(ts[f_idx],with_direction=True)
        classes = torch.tensor([class_by_id[id.item()] for id in ids])
        
        if extents is not None and len(boxes) > 0:
            xmin, xmax, _ = extents[cam_names[f_idx]]

            # select objects that fall within that camera's space range (+ some tolerance)
            keep_obj = torch.mul(torch.where(boxes[:, 0] > xmin - PLOT_TOLERANCE, 1, 0), torch.where(
                boxes[:, 0] < xmax + PLOT_TOLERANCE, 1, 0)).nonzero().squeeze(1)
            boxes = boxes[keep_obj,:]
            ids = ids[keep_obj]
            classes = classes[keep_obj]
            classes = [hg.hg1.class_dict[cls.item()] for cls in classes]
            
        # convert frame into cv2 image
        fr = (denorm(frames[f_idx]).numpy().transpose(1, 2, 0)*255)[:,:,::-1]
        #fr = frames[f_idx].numpy().transpose(1,2,0)
        # use hg to plot the boxes and IDs in that camera
        if boxes is not None and len(boxes) > 0:
            
            labels = ["{}: {}".format(classes[i],ids[i]) for i in range(len(ids))]
            color_slice = colors[ids%colors.shape[0],:]
            #color_slice = [colors[id,:] for id in ids]
            #color_slice = np.stack(color_slice)
            if color_slice.ndim == 1:
                 color_slice = color_slice[np.newaxis,:]
            
            fr = hg.plot_state_boxes(
                fr.copy(), boxes, name=cam_names[f_idx], labels=labels,thickness = 3, color = color_slice)

        # plot timestamp
        fr = cv2.putText(fr.copy(), "Timestamp: {:.3f}s".format(ts[f_idx]), (10,70), cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),3)
        fr = cv2.putText(fr.copy(), "Camera: {}".format(cam_names[f_idx]), (10,30), cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),3)
        
        # append to array of frames
        plot_frames.append(fr)

    # 3. tile frames
    n_ims = len(plot_frames)
    n_row = int(np.round(np.sqrt(n_ims)))
    n_col = int(np.ceil(n_ims/n_row))

    cat_im = np.zeros([1080*n_row, 1920*n_col, 3])
    for im_idx, original_im in enumerate(plot_frames):
        row = im_idx // n_col
        col = im_idx % n_col
        cat_im[row*1080:(row+1)*1080, col*1920:(col+1)*1920, :] = original_im

    # resize to fit on standard monitor
    trunc_h = cat_im.shape[0] / MONITOR_SIZE[0]
    trunc_w = cat_im.shape[1] / MONITOR_SIZE[1]
    trunc = max(trunc_h, trunc_w)
    new_size = (int(cat_im.shape[1]//trunc), int(cat_im.shape[0]//trunc))
    cat_im = cv2.resize(cat_im, new_size) / 255.0

    cv2.imwrite("/home/derek/Desktop/temp_frames/{}.png".format(str(fr_num).zfill(4)),cat_im*255)
    # plot
    cv2.imshow("frame", cat_im)
    # cv2.setWindowTitle("frame",str(self.frame_num))
    key = cv2.waitKey(1)
    if key == ord("p"):
        cv2.waitKey(0)
    elif key == ord("q"):
        cv2.destroyAllWindows()
        shutdown()

def shutdown():
    raise KeyboardInterrupt("Manual Shutdown triggered")


if __name__ == "__main__":
    #     try:
    #         mp.set_start_method('spawn')
    #     except RuntimeError:
    #         pass

    import torch
    import os

    from i24_configparse import parse_cfg
    from src.track.tracker import get_Tracker, get_Associator
    from src.track.trackstate import TrackState
    from src.detect.pipeline import get_Pipeline
    from src.scene.devicemap import get_DeviceMap
    from src.scene.homography import HomographyWrapper,Homography
    from src.detect.devicebank import DeviceBank
    from src.load import DummyNoiseLoader #,MCLoader
    #from src.gpu_load             import MCLoader
    from src.gpu_load_abstractor  import MCLoader, ManagerClock
    from src.db_write import WriteWrapper

    ctx = mp.get_context('spawn')

    colors = np.random.randint(0,255,[1000,3])

    os.environ["user_config_directory"] = "/home/derek/Documents/i24/i24_track/config/lambda_cerulean_2"
    os.environ["TRACK_CONFIG_SECTION"] = "DEFAULT"

    # load parameters
    params = parse_cfg("TRACK_CONFIG_SECTION",
                       cfg_name="execute.config", SCHEMA=False)

    # verify config notion of GPUs matches torch.cuda notion of available devices
    # get available devices

    # TODO fix this once you redo configparse
    params.cuda_devices = [int(i) for i in range(int(params.cuda_devices))]
    assert max(params.cuda_devices) < torch.cuda.device_count()

    # intialize DeviceMap
    dmap = get_DeviceMap(params.device_map)

    # initialize multi-camera loader
    #loader = DummyNoiseLoader(dmap.camera_mapping_file)


    #in_dir = "/home/derek/Data/dataset_beta/sequence_1"
    #loader = MCLoader(in_dir, dmap.camera_mapping_file, ctx)

    in_dir = "/home/derek/Data/cv/video/06-02-2022/batch1"
    loader = MCLoader(in_dir, dmap.camera_mapping_file,dmap.cam_names, ctx)

    # initialize Homography object
    hg = HomographyWrapper(hg1 = params.eb_homography_file,hg2 = params.wb_homography_file)

    # initialize pipelines
    pipelines = params.pipelines
    pipelines = [get_Pipeline(item, hg) for item in pipelines]

    associators = params.associators
    associators = [get_Associator(item) for item in associators]

    # initialize tracker
    tracker = get_Tracker(params.tracker)

    # add Associate function to each pipeline
    for i in range(len(pipelines)):
        assoc = associators[i]
        pipelines[i].associate = associators[i]

    # initialize DetectorBank
    dbank = DeviceBank(params.cuda_devices, pipelines, dmap.gpu_cam_names, ctx)

    # initialize DBWriter object
    dbw = WriteWrapper()

    # intialize empty TrackState Object
    tstate = TrackState()

    # get frames and timestamps
    frames, timestamps = loader.get_frames(target_time = None)
    
    # initialize processing sync clock
    start_ts = max(timestamps)
    desired_processing_speed = 0    # for now, no constraint
    nom_framerate = 30
    clock  = ManagerClock(start_ts,desired_processing_speed, nom_framerate)
    target_time = start_ts

    # initial sync-up of all cameras
    # TODO - note this means we always skip at least one frame at the beginning of execution
    frames,timestamps = loader.get_frames(target_time)
    
    # main loop
    frames_processed = 0
    start_time = time.time()
    
    # readout headers
    print("Frame:    Since Start:  Frame BPS:    Sync Timestamp:     Max ts Deviation:     Active Objects:")
    while True:
        
        if False: # shortout actual processing

            # select pipeline for this frame
            pipeline_idx = 0
    
            camera_idxs, device_idxs, obj_times = dmap(tstate, timestamps)
            obj_ids, priors, selected_obj_idxs = tracker.preprocess(
                tstate, obj_times)
    
            # slice only objects we care to pass to DeviceBank on this set of frames
            # DEREK NOTE may run into trouble here since dmap and preprocess implicitly relies on the list ordering of tstate
            if len(obj_ids) > 0:
                obj_ids     =     obj_ids[selected_obj_idxs]
                priors      =      priors[selected_obj_idxs,:]
                device_idxs = device_idxs[selected_obj_idxs]
                camera_idxs = camera_idxs[selected_obj_idxs]
    
            # prep input stack by grouping priors by gpu
            cam_idx_names = None  # map idxs to names here
            prior_stack = dmap.route_objects(
                obj_ids, priors, device_idxs, camera_idxs, run_device_ids=params.cuda_devices)
    
            # test on a single on-process pipeline
            # pipelines[0].set_device(0)
            # pipelines[0].set_cam_names(dmap.gpu_cam_names[0])
            # test = pipelines[0](frames[0],prior_stack[0])
    
    
            # TODO - full frame detections should probably get full set of objects?
    
            # TODO select correct pipeline based on pipeline pattern logic parameter
        
       
            detections, confs, classes, detection_cam_names, associations = dbank(
                prior_stack, frames, pipeline_idx=0)
            
            # THIS MAY BE SLOW SINCE ITS DOUBLE INDEXING
            detection_times = torch.tensor(
                [timestamps[dmap.cam_idxs[cam_name]] for cam_name in detection_cam_names])
            
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
    
            
            terminated_objects = tracker.postprocess(
                detections, detection_times, classes, confs, associations, tstate, hg = hg)

            dbw.insert(terminated_objects)
            #print("Active Trajectories: {}  Terminated Trajectories: {}   Documents in database: {}".format(len(tstate),len(terminated_objects),len(dbw)))

        frames_processed += 1

        # optionally, plot outputs
        if True:
            #mask = ["p1c1","p1c2", "p1c3", "p1c4", "p1c5","p1c6"]
            mask = None
            #mask = ["p48c01","p48c02","p48c03","p48c04","p48c05","p48c06"]
            plot(tstate, frames, timestamps, dmap.gpu_cam_names,
                 hg, colors,extents=dmap.cam_extents_dict, mask=mask,fr_num = frames_processed)

        
        # text readout update
        fps = frames_processed/(time.time() - start_time)
        dev = [np.abs(t-target_time) for t in timestamps]
        max_dev = max(dev)
        print("\r{}        {:.3f}s       {:.2f}        {:.3f}              {:.3f}                {}".format(frames_processed, time.time() - start_time,fps,target_time, max_dev, len(tstate)), end='\r', flush=True)
    
        # get next target time
        target_time = clock.tick(timestamps)
        
        # get next frames and timestamps
        frames, timestamps = loader.get_frames(target_time)