import logging
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
