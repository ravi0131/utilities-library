import pandas as pd
import numpy as np
from .gen_utils import *
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

import os
from typing import List, Tuple

def _check_lidar_frames(base_path, expected_frame_count=80) -> Tuple[bool, List[str]]:
    """
    Checks if all scenes have the expected number of LiDAR frames.

    :param base_path: Path to the dataset folder containing scenes (e.g., 'PANDASET').
    :param expected_frame_count: The expected number of LiDAR frames for each scene (default is 79).
    :return: Tuple (bool, list), where bool is True if all scenes are valid, and list contains scenes with incorrect frames.
    """
    invalid_scenes = []

    # Iterate through each scene directory inside the base path
    for scene_dir in sorted(os.listdir(base_path)):
        scene_path = os.path.join(base_path, scene_dir, 'lidar')
        
        # Only proceed if the 'lidar' directory exists
        if os.path.exists(scene_path) and os.path.isdir(scene_path):
            lidar_files = [f for f in os.listdir(scene_path) if f.endswith('.pkl')]
            num_frames = len(lidar_files)

            # Check if the number of frames matches the expected count
            if num_frames != expected_frame_count:
                invalid_scenes.append((scene_dir, num_frames))

    # If invalid_scenes is empty, all scenes are valid
    if not invalid_scenes:
        return True, []
    else:
        return False, invalid_scenes

def does_every_scene_have_80_lidar_frames(path_to_dataset: str) -> Tuple[bool, List[str]]:
    """
    Check if every scene in the dataset has 80 LiDAR frames.

    :param path_to_dataset: Path to the dataset directory.
    :return: Tuple where the first element is a boolean indicating if every scene has 80 LiDAR frames,
             and the second element is a list of scene names that do not have 80 LiDAR frames.
    """
    all_valid, invalid_scenes_list = _check_lidar_frames(path_to_dataset, expected_frame_count=80)
    return all_valid, invalid_scenes_list

does_every_scene_have_80_lidar_frames(dataset_path)
