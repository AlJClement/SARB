from omegaconf import OmegaConf
from preprocessing import *
from feature_extraction import *
import torch
from torch.utils.data import Dataset, DataLoader
import datetime
from helpers import *
from object_detection import *

def run_object_detection(config):
    
    #logger data setup
    t_start=datetime.datetime.now()
    t_start_str = datetime.datetime.now().strftime("%m_%d_%Y_%H%M%S")

    os.makedirs(os.path.join(config.output.loc,'object_detection'), exist_ok=True)
    log=logger.setup_logger(os.path.join(config.output.loc,'object_detection','log_'+t_start_str))

    log.info('START TIME:'+t_start_str)
    log_dict(log, config)

    #load data into data loader (imports all data into a dataloader)
    dataset=SARB_dataloader(config)    
    data = DataLoader(dataset, batch_size=config.data.batch_size, shuffle=False, drop_last = True)
    
    t_loader=datetime.datetime.now()
    total_time=t_loader-t_start
    log.info('END DATALOAD TIME:'+t_loader.strftime("%m_%d_%Y_%H%M%S"))
    log.info('DIFFERENCE:'+ str(total_time))

    ## run object detection from selected method
    ObjectDetection = eval(f"{config.object_detection.method}")
    ObjectDetection(config, data)._detect_objects()
    time_compare = datetime.datetime.now()
    total_time = time_compare-t_loader
    log.info('END COMPARE TIME:'+time_compare.strftime("%m_%d_%Y_%H%M%S"))
    log.info('DIFFERENCE:'+ str(total_time))

    return

# Load the Omega config YAML file
config_file = './configs/run_feats.yaml'
# with open(config_file, 'r') as file:
#     config = yaml.safe_load(file)
config = OmegaConf.load(config_file)
run_object_detection(config)