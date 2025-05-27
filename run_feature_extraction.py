from omegaconf import OmegaConf
from preprocessing import *
from feature_extraction import *
import torch
from torch.utils.data import Dataset, DataLoader
import datetime
from helpers import *

def run_feature_extraction(config):
    
    #logger data setup
    t_start=datetime.datetime.now()
    t_start_str = datetime.datetime.now().strftime("%m_%d_%Y_%H%M%S")
    #logger.setup_logger(os.path.join(config.output.loc,config.feature_extraction.method,'log_'+t_start_str))0-0
    log=logger.setup_logger(os.path.join(config.output.loc,config.feature_extraction.method,'log_'+t_start_str))
    log.info('START TIME:'+t_start_str)
    log_dict(log, config)

    #load data into data loader (imports all data into a dataloader)
    dataset=SARB_dataloader(config)    
    data = DataLoader(dataset, batch_size=config.data.batch_size, shuffle=False, drop_last = True)
    
    t_loader=datetime.datetime.now()
    total_time=t_loader-t_start
    log.info('END DATALOAD TIME:'+t_loader.strftime("%m_%d_%Y_%H%M%S"))
    log.info('DIFFERENCE:'+ str(total_time))

    

    # #load feature extractor from config file
    # feat_extractor = eval(f"{config.feature_extraction.method}")
    
    # #run extractor to get output as dict of multiple features
    # feat_dict = feat_extractor(config)._get_feature_dict(dict_arr)

    #compare healthy and unhealthy features from dictconda 
    Compare(config, data, log)._report()
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
run_feature_extraction(config)