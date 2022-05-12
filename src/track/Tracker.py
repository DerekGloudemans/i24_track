import torch
from .TrackState import TrackState


class Associator():
    
    def __init__(self):
        pass
    
    def __call__(self):
        pass


class Tracker():
    """
    
    """
    def __init__(self):
        
        # parse config file
        pass        
        

    def preprocess(self,tstate,obj_times):
        
        """
        Receives a TrackState object as input, as well as the times for each object
        Applies kf.predict to objects
        Theoretically, additional logic could happen here i.e. a subset of objects
        could be selected for update based on covarinace, etc.
        
        :param tstate - TrackState object
        :param obj_times - tensor of times of same length as number of active objects
        
        :return obj_ids - tensor of int IDs for object priors
        :return priors - tensor of all TrackState object priors
        :return selected_idxs - tensor of idxs on which to perform measurement update (naively, all of them)
        """
        
        dts = tstate.get_dts(obj_times)
        tstate.predict(dt = dts)   
        
        obj_ids,priors = tstate()
        
        selected_idxs = self.select_idxs(tstate,obj_times)
        
        return obj_ids,priors,selected_idxs
        
    def select_idxs(self,tstate,obj_times):
        return torch.tensor([i for i in range(len(tstate))])
        

    def postprocess(self,detections,detection_times,classes,confs,assigned_ids,tstate,meas_idx = 0):
        """
        Updates KF representation of objects where assigned_id is not -1 (unassigned)
        Adds other objects as new detections
        For all TrackState objects, checks confidences and fslds and removes inactive objects
        :param detections -tensor of size [n_detections,state_size]
        :param detection_times - tensor of size [n_detections] with frame time
        :param classes - tensor of size [n_detections] with integer class prediction for each
        :param confs - tensor of size [n_detections] of confidences in range[0,1]
        :param assigned_ids - tensor of size [n_detections] of IDs, or -1 if no id assigned
        :param tstate - TrackState object
        : fn_idx - int specifying which association / measurement index was used
        
        :return - stale_objects - dictionary of object histories indexed by object ID
        """
        
        # get IDs and times for update
        update_idxs = torch.nonzero(assigned_ids + 1) 
        update_ids = assigned_ids[update_idxs]
        update_times = detection_times[update_idxs]
                

        # roll existing objects forward to the detection times
        dts = tstate.get_dts(update_times)
        tstate.predict(dt = dts,idxs = update_ids)
    
        # update assigned detections
        update_detections = detections[update_idxs,:]
        update_classes = classes[update_idxs]
        update_confs = confs[update_idxs]
        tstate.update(update_detections,update_ids,update_classes,update_confs, measurement_idx = meas_idx)
        
        
        # add unassigned detections
        new_idxs = [i for i in range(len(update_ids))]
        for i in update_idxs:
            new_idxs.remove(i)
          
        # add new detections as new objects
        new_idxs = torch.tensor(new_idxs)
        new_detections = detections[new_idxs,:]
        new_classes = classes[new_idxs]
        new_confs = confs[new_idxs]
        new_times = detection_times[new_idxs]
        
        # create direction tensor based on location
        directions = torch.where(new_detections[:,1] > 60, torch.zeros(new_idxs.shape)-1,torch.ones(new_idxs.shape))
        
        tstate.add(new_detections,directions,new_times,new_classes,new_confs)
        
        # remove objects
        stale_objects = self.remove(tstate)
        
        return stale_objects

    def remove(self,tstate):
        """
        
        """
        # remove stale objects
        removals = []
        for id in tstate()[0]:
            if tstate.fsld[id] > self. fsld_max:
                removals.append[id]
                
        stale_objects = tstate.remove(removals)
        return stale_objects
    
    
