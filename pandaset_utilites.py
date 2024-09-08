import pandas as pd
import numpy as np
from general_utilities import *
import logging

def cleanup_lidar_data_and_labels(lidar_data: pd.DataFrame, labels:pd.DataFrame, logger: logging.Logger = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filters out only those points in the lidar data which have a corresponding semantic label
    
    Args:
        lidar_data (pd.DataFrame): The LiDAR data as a pandas DataFrame.
        labels (pd.DataFrame): The semantic labels as a pandas DataFrame.
        logger (logging.Logger): The logger instance to log messages. Default is None.
        
    Returns:
        pd.DataFrame: The filtered LiDAR data without labels.
        pd.Series: The labels column.
    """
    # if logger is None:
    #     logger = get_logger()
    logger = logger if logger is not None else get_logger()
    
    check_type(lidar_data, "lidar_data", logger)
    check_type(labels, "labels", logger)
    # Step 1: Concatenate the labels DataFrame to the lidar_data DataFrame along columns
    lidar_data_with_labels = pd.concat([lidar_data, labels], axis=1)

    # Step 2: Filter the DataFrame to keep only the rows where d == 0
    filtered_lidar_data = lidar_data_with_labels[lidar_data_with_labels['d'] == 0]

    # Display the final filtered DataFrame
    check_type(filtered_lidar_data,"filtered_lidar_data")
    logger.info(f"Filtered LiDAR data with labels (d=0):\n{filtered_lidar_data}")
    
    # Step 3: Extract the labels column (the last column) into a separate variable
    lidar_labels = filtered_lidar_data['class']  # The last column

    # Step 4: Extract the rest of the DataFrame into another variable
    lidar_data = filtered_lidar_data.iloc[:, :-1]  # All columns except the last one
    
    # Display the results
    check_type(lidar_labels,"lidar_labels", logger)
    logger.info("Labels Column (last column):")
    logger.info(lidar_labels)
    logger.info(f"lidar_labels shape: {lidar_labels.shape}\n")
    
    check_type(lidar_data,"lidar_data",logger)
    logger.info("\nLiDAR Data without Labels (all except the last column):")
    logger.info(lidar_data)
    
    return lidar_data, lidar_labels