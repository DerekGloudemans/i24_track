from torchvision import transforms
import torch
import numpy as np
import cv2


colors = np.random.randint(0,255,[1000,3])


# def apply_config(obj,cfg,case = "DEFAULT"):
# 	# read config file	
# 	config = configparser.ConfigParser()
# 	config.read(cfg)
# 	params = config[case]

# 	# set object attributes according to config case
# 	# getany() automatically parses type (int,float,bool,string) from config
# 	[setattr(obj,key,config.getany(case,key)) for key in params.keys()]


def plot_scene(tstate, frames, ts, gpu_cam_names, hg, colors, mask=None, extents=None, fr_num = 0):
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