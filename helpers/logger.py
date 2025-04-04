import logging
import os 

def setup_logger(log_path):
    os.makedirs(os.path.dirname(log_path),exist_ok=True)
    logging.basicConfig(filename=log_path,
                        format='%(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger

def log_dict(logger, data: dict):
    for key, value in data.items():
        logger.info(f"{key}: {value}")
