import logging
import numpy as np
import pandas as pd

def check_type(obj, name:str ="Given object", logger: logging.Logger = None) -> None:
    """
    Checks whether the given object is a NumPy array or a pandas DataFrame.
    
    Args:
        obj: The object to check.
        name (str): The name of the object. Default is "Given object".
        logger (logging.Logger): The logger instance to log messages. Default is None.
        
    Returns:
        None
    """
    logger = logger if logger is not None else get_logger()
    if isinstance(obj, np.ndarray):
        logger.info(f"{name} is a NumPy array.\n")
    elif isinstance(obj, pd.DataFrame):
        logger.info(f"{name} is a pandas DataFrame.\n")
    else:
        logger.warning(f"{name} is neither a NumPy array nor a pandas DataFrame.\n")

def get_logger(name: str = 'default_logger', log_file: str = 'default_log.log', level: int = logging.INFO) -> logging.Logger:
    """
    Returns a logger instance. If the logger already exists, it will return the existing one.
    
    Args:
        name (str): The name of the logger.
        log_file (str): The file where logs will be written.
        level (int): The logging level. Default is logging.INFO.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create a logger or retrieve the existing one
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if the logger already exists
    if not logger.handlers:
        # Create a file handler to write to a file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        # Create a log format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        logger.addHandler(file_handler)

    return logger
