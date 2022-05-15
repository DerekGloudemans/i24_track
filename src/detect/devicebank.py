import torch.multiprocessing as mp
import torch

class DeviceBank():
    """
    DeviceBank  maintains a list of DeviceHandler objects, one per GPU device.
    Each DeviceHandler object communicates with a separate process to execute Pipelines
    in parallel. The main purpose of the DeviceBank is as a simple input-splitter and output-combiner,
    as well as to initialize each DeviceHandler with the correct parameters
    """
    
    def __init__(self,device_ids,pipelines):
        
        self.device_ids = device_ids
        
        self.send_queues = []
        self.receive_queues = []
        self.handlers = []
        
        self.manager = mp.Manager()
        #mp.set_start_method('spawn')
        ctx = mp.get_context('spawn')
        
        
         
        # create handler objects
        for dev_id in range(len(device_ids)):
            in_queue = ctx.queue()
            out_queue = ctx.queue()
            handler = ctx.Process(target=handle_device, args=(in_queue,out_queue,pipelines,dev_id))
            handler.start()
            
            self.handlers.append(handler)
            self.send_queues.append(in_queue)
            self.receive_queues.append(out_queue)
            
            
        self.ctx = ctx   
        
    def __call__(self,prior_stack,frames,pipeline_idx=0):
        """
        :prior_stack - list of items (obj_ids,priors,cam_idx), one for each device
                      where obj_ids - tensor of size [n_objs_on_gpu]
                            priors  - tensor of size [n_objs_on_gpu,state_size]
                            cam_idx - tensor of size [n_obs_on_gpu] with index into cameras on that GPU
        :param frames - list of tensors - [n_cameras_on_gpu,channels,im_height,im_width], one for each device
        :param pipeline_idx - int
        
        
        """
        
        # issue commands
        for i in range(prior_stack.shape[0]):
            self.handler_list[i](prior_stack[i],frames[i],pipeline_idx = pipeline_idx)
        
        # get results
        result = []
        for i in range(prior_stack.shape[0]):
            self.handler_list[i].take()
        
        # concat and return results
    
    def __del__():
        for pid in self.handlers:
            pid.terminate()
            pid.join()
            
        # for q in self.send_queues:
        #     del q
        # for q in self.receive_queues:
        #     del q
        
        # del self.ctx 
        # del self.manager
            
            
        # TODO log DeviceBank shutdown
        

def handle_device(in_queue,out_queue,pipelines,device_id):
   
    # intialize
    device = torch.cuda.device("cuda:{}".format(device_id) if device_id != -1 else "cpu")
        
    while True:
        
        # try to get set of inputs from in_queue
        # each item on this queue is (prior_stack,frames,pipeline_idx)
        try: inputs = in_queue.get(timeout = 0) 
        except TimeoutError: continue
    
        if inputs is None:
            break
            #TODO log device handler shutdown
        
        (prior_data,frames,pipeline_idx) = inputs
        
        result = pipelines[pipeline_idx](frames,priors = prior_data)
        out_queue.put(result)
        