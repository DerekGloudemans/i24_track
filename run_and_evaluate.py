from execute import main
from src.scene.homography           import HomographyWrapper,Homography

from tests.sup_metrics     import evaluate
from tests.unsup_statistics   import call
from src.util.names import adj_list, noun_list

from i24_database_api.db_writer import DBWriter
import numpy as np
import os
import time
import _pickle as pickle

def db_cleanup(dbw,coll_name):
    dbw.delete_collection([coll_name])

    inp = input("Do you want to remove all RAW_TRACKS collections? Type YES if so:")
    if inp == "YES":
        existing_collections = [col["name"] for col in list(dbw.db.list_collections())] # list all collections
        remove = []
        for item in existing_collections:
            if "RAW*" in item:
                remove.append(item)
        inp2 = input("{} collections to be removed ({}). Continue? Type YES if so:".format(len(remove),remove))
        if inp2 == "YES":
            if len(remove) > 0:
                dbw.delete_collection(remove)
# db_param = {
#       "default_host": "10.2.218.56",
#       "default_port": 27017,
#       "host":"10.2.218.56",
#       "port":27017,
#       "username":"i24-data",
#       "password":"mongodb@i24",
#       "default_username": "i24-data",
#       "readonly_user":"i24-data",
#       "default_password": "mongodb@i24",
#       "db_name": "trajectories",      
#       "server_id": 1,
#       "session_config_id": 1,
#       "trajectory_database":"trajectories",
#       "timestamp_database":"transformed"
#       }

db_param = {
      "default_host": "10.80.4.91",
      "default_port": 27017,
      #"host":"10.80.4.91",
      # "port":27017,
      # "username":"i24-data",
      # "password":"mongodb@i24",
      "default_username": "mongo-admin",
      "default_password": "i24-data-access",
      "db_name": "trajectories",      
      "server_id": 1,
      "session_config_id": 1,
      "trajectory_database":"trajectories",
      "timestamp_database":"transformed"
      }

if __name__ == "__main__":
    run_config = "/home/derek/Documents/i24/i24_track/config/lambda_cerulean_eval2"
    gt_coll = "groundtruth_scene_2_57"
    TAG = "GT2" 
    n_GPUs = 4
    n_cameras = 17
        

    if TAG not in ["GT1","GT2","GT3","Unsup"]:
        raise ValueError("Tag! You're not it.")
        

    dbw   = DBWriter(db_param,collection_name = db_param["db_name"])
    existing_collections = [col["name"] for col in list(dbw.db.list_collections())] # list all collections
    existing_collections.sort()
    print(existing_collections)
    
    # generate comment
    comment = input("Description of run settings / test for storage with evaluation results: ")    
    
    bps = -1
    try:
        coll_name
    except:
        while True:
            config_short = run_config.split("/")[-1]
            adj = adj_list[np.random.randint(0,len(adj_list))]
            noun = noun_list[np.random.randint(0,len(noun_list))]
            coll_name = "{}_{}--RAW_{}".format(adj,noun,TAG)
            if coll_name not in existing_collections:
                break
            
    print("\nFound unique database collection name: {}".format(coll_name))    
        
    # set config directory
    os.environ["USER_CONFIG_DIRECTORY"] = run_config
    os.environ["user_config_directory"] = run_config
    
    # ### run
    bps = main(coll_name)
    
    result = {}
    result["name"] = coll_name
    result["description"] = comment
    result["bps"]  = bps
    result["config"] = run_config
    result["n_cameras"] = n_cameras
    result["n_gpus"] = n_GPUs

    ### Save results dict in /data/run_results
    save_name = "/home/derek/Documents/i24/trajectory-eval-toolkit/data/run_results/{}_run_results.cpkl".format(coll_name)
    with open(save_name, 'wb') as f:
        pickle.dump(result, f)

    time.sleep(15)

    os.chdir("../I24_postprocessing")
    from postprocess import main as postprocess_main
    postprocess_main(collection_name = coll_name)
    
    
    os.chdir("../trajectory-eval-toolkit")
    from evaluate import main as evaluate_main
    evaluate_main(gt_coll= gt_coll, TAG = TAG)