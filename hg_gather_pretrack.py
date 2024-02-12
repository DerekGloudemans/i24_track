from i24_rcs import I24_RCS
from i24_logger import i24_logger
import signal,os

# Required entry point for executing the module;
# parameters/arguments are supplied by the ServerControl based on the configuration
def __process_entry__(
                      hg_dir               = '', # roughly local/run_data/runID/     substructure PXXCXX/  PXXCXX.cpkl, PXXCXX.png 
                      aerial_ref_dir       = '', # roughly i24_common/aerial_ref_dir/vX.X /   MUST contain a file as rcs_base_v1-1-0.cpkl
                      cams                 = [],
                      hg_local_path        = '', # absolute path as /local/run_data/runID/hg_nodeX.cpkl
                      hg_archive_path      = ''  # absolute path as /data/hg_save/runID/hg_nodeX.cpkl
                      ):
    
    """
    Pre-caches all I24_RCS data such that a single save file can be loaded per node
    
    hg_dir - str - directory containing .cpkl files for each camera and side of roadway
    aerial_ref_dir - str - directory contain a file as rcs_base_v1-1-0.cpkl (a RCS save file with no correspondences)
    cams - [str] list of cameras to include in the cache file
    hg_local_path - str - absolute path to local file at which to save single node homography cache
    hg_archive_path - str - absolute path to shared file at which to save single node homography cache

    """
    
    
    # set up logger
    from i24_logger.log_writer         import logger,catch_critical,log_warnings
    logger.set_name("Homography gather pretrack")
    
    
    # load rcs base file
    base_file = None
    aerial_files = os.listdir(aerial_ref_dir)
    for file in aerial_files:
        if "rcs_base" in file and ".cpkl" in file:
            base_file = os.path.join(aerial_ref_dir,file)
            break
    if base_file is None:
        logger.error("No rcs_base file in specified aerial_ref_dir. You need to manaually generate this base file (with no correspondences)")
    
    # initialzze hg object
    hg = I24_RCS(save_file =base_file)
    
    # add correspondences
    hg.load_correspondences(hg_dir)
    

    # make sure cams <-> correspondences mapping is 1 to 1    
    removals = []
    for corr in hg.correspondence.keys():
        if corr.split("_")[0] not in cams:
            removals.append(corr)
    for removal in removals:
        hg.correspondence.pop(removal,None)
        logger.warning("Removed camera {} from homography save file because it was not specifed in the camera list.".format(removal))
        
    for cam in cams:
        if cam not in hg.correspondence.keys():
            logger.warning("Camera {} specified but no correspondence exists, make sure .cpkl file exists at {}".format(cam,hg_dir))
            
    # save files
    hg.save(hg_local_path)
    logger.info("Saved local homography cache file at {}. Process complete".format(hg_local_path))

    hg.save(hg_archive_path)
    logger.info("Saved shared homography cache file at {}. Process complete".format(hg_archive_path))
    
    