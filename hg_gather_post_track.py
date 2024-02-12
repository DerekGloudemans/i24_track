from i24_rcs import I24_RCS
from i24_logger import i24_logger
import signal,os

# Required entry point for executing the module;
# parameters/arguments are supplied by the ServerControl based on the configuration
def __process_entry__(
                      runID                = '',
                      hg_archive_dir       = ''  # absolute path as /data/hg_save/runID/hg_nodeX.cpkl
                      ):
    
    """
    runID - str 
    hg_archive_dir - str - directory with save files for a single node homography cache each
    """
    
    
    # set up logger
    from i24_logger.log_writer         import logger,catch_critical,log_warnings
    logger.set_name("Homography gather pretrack")
    
    hg = None
    
    # for each file
    for file in os.list_dir(hg_archive_dir):
        path = os.path.join(hg_archive_dir,file)
        
        if hg is None:
            hg = I24_RCS(path)
        else:
            
            temp_hg = I24_RCS(path)
            
            # append all correspondences to same file
            for corr in temp_hg.correspondence.keys():
                if corr not in hg.correspondence.keys():
                    hg.correspondence[corr] = temp_hg.correspondence[corr].copy()
    
            
    # save file
    out_path = os.path.join("hg_{}.cpkl".format(runID))
    hg.save(out_path)
    logger.info("Saved full homography cache file at {}. Process complete.".format(out_path))

    
    
    