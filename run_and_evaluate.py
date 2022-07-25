
from execute import main
from tests.compare_collections      import evaluate
from src.scene.homography          import HomographyWrapper,Homography

from i24_database_api.db_writer import DBWriter
import numpy as np
import os
import time
import json


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

adj_list = ["admissible",
            "ostentatious",
            "modest",
            "loquacious",
            "gregarious",
            "cantankerous",
            "bionic",
            "demure",
            "thrifty",
            "quizzical",
            "pragmatic",
            "sibilant",
            "visionary",
            "morose",
            "jubilant",
            "apathetic",
            "stalwart",
            "paradoxical",
            "tantalizing",
            "specious",
            "tautological",
            "hollistic",
            "super",
            "pristine",
            "wobbly",
            "lovely"]

noun_list = ["anteater",
             "zebra",
             "anaconda",
             "aardvark",
             "bison",
             "wallaby",
             "heron",
             "stork",
             "cyborg",
             "vulcan",
             "sssnek",
             "beluga",
             "panda",
             "lynx",
             "panther",
             "housecat",
             "osprey",
             "bovine",
             "jackalope",
             "yeti",
             "doggo",
             "cheetah",
             "squirrel",
             "axylotl",
             "kangaroo"
             ]

if __name__ == "__main__":
    run_config = "/home/derek/Documents/i24/i24_track/config/lambda_cerulean_eval1"
    gt_coll = "groundtruth_scene_1"
    collection_cleanup = False
    
    
    
    # manage database  get a new database collection name
    param = {
      "default_host": "10.2.218.56",
      "default_port": 27017,
      "default_username": "i24-data",
      "readonly_user":"i24-data",
      "default_password": "mongodb@i24",
      "db_name": "trajectories",
      "raw_collection": "NA",
      
      "server_id": 1,
      "session_config_id": 1
    }
    dbw   = DBWriter(param,collection_name = param["db_name"])
    existing_collections = [col["name"] for col in list(dbw.db.list_collections())] # list all collections
    print(existing_collections)
    
    coll_name = "tantalizing_bison--RAW_TRACKS"
    
    if False: # can short circuit if there is already an existing collection
        while True:
            config_short = run_config.split("/")[-1]
            adj = adj_list[np.random.randint(0,len(adj_list))]
            noun = noun_list[np.random.randint(0,len(noun_list))]
            coll_name = "{}_{}--RAW_TRACKS".format(adj,noun)
            if coll_name not in existing_collections:
                print("Found unique database collection name: {}".format(coll_name))     
                break
            
        # generate comment
        comment = input("Description of run settings / test for storage with evaluation results: ")
        
        # set config directory
        os.environ["USER_CONFIG_DIRECTORY"] = run_config
        os.environ["user_config_directory"] = run_config
        
        # ### run
        start = time.time()
        main(coll_name)
        elapsed = time.time() - start
    
    description = "Temp Dumy"
    runtime = 10000
    
    ### evaluate
    metrics = evaluate(gt_collection = gt_coll,pred_collection = coll_name, sample_freq = 30, break_sample = 2700)
    result = {}
    result["metrics"] = metrics
    result["name"] = coll_name
    result["gt"] = gt_coll
    result["description"] = comment
    result["runtime"] = elapsed
    
    
    ### Save results dict in /data/eval_results
    save_name = "./data/eval_results/{}.json".format(coll_name)
    with open(save_name, 'w') as f:
        json.dump(result, f)

    ### clean up
    if collection_cleanup: db_cleanup(dbw,coll_name)
