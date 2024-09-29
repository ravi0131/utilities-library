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

import open3d as o3d
import matplotlib.colors as mcolors

def visualize_beam_ids(lidar_points: np.ndarray, beam_ids: np.ndarray) -> None:
    """
    Visualize a point cloud with beam IDs assigned as colors
    Args:
        lidar_points (np.ndarray): 3D points in Cartesian
        beam_ids (np.ndarray): Beam IDs assigned to each point
    Returns:
        None
    """

    # Create a point cloud in Open3D and assign colors based on beam IDs
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_points)

    # Use a discrete colormap with enough unique colors
    unique_beam_ids = np.unique(beam_ids)
    n_unique_beam_ids = len(unique_beam_ids)

    # Generate a colormap with N unique colors
    colors_list = list(mcolors.CSS4_COLORS.values())  # Get a large list of CSS colors
    if n_unique_beam_ids > len(colors_list):
        raise ValueError("Not enough unique colors available. Increase the color list or choose another colormap.")

    # Create a color dictionary that maps each beam ID to a unique color
    beam_id_to_color = {beam_id: colors_list[i % len(colors_list)] for i, beam_id in enumerate(unique_beam_ids)}

    # Map the beam IDs to the corresponding colors
    colors = np.array([mcolors.to_rgb(beam_id_to_color[beam_id]) for beam_id in beam_ids])

    # Set the colors to the point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])