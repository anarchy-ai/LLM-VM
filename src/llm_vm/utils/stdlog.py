import logging

def setup_logger(name, log_level=logging.INFO):
    
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format)
    
    logger = logging.getLogger(name)
    
    return logger
