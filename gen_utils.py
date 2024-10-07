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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import open3d as o3d

def visualize_beam_ids(lidar_points: np.ndarray, beam_ids: np.ndarray) -> None:
    """
    Viszaliizes the LiDAR points with colors based on beam IDs.
    
    Args:
        lidar_points (n,3): 3D points in Cartesian coordinates (x, y, z).
        beam_ids (n,1): Beam IDs corresponding to each LiDAR point.
    
    Returns:
        None
    
    Notes:
        - The beam IDs are used to color the points for better visualization.
        - The beam IDs are normalized to fit the colormap range [0, 63].
        - The 'hsv' colormap from matplotlib is used for better distinction of beam IDs.
        - The point cloud is visualized using Open3D with larger point size.
    """
    # Ensure beam_ids has the correct shape
    if beam_ids.shape[1] != 1:
        raise ValueError("Beam IDs must be an (n, 1) array.")
    
    # Reshape beam_ids to be a flat array for color mapping
    beam_ids = beam_ids.flatten()
    
    # Check shapes for debugging
    print("lidar_points shape:", lidar_points.shape)
    print("beam_ids shape after flattening:", beam_ids.shape)

    # Ensure beam_ids has the correct length
    if lidar_points.shape[0] != beam_ids.shape[0]:
        raise ValueError("Error: The number of beam IDs does not match the number of lidar points.")

    # Create a point cloud in Open3D and assign colors based on beam IDs
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_points)

    # Use the 'hsv' colormap from matplotlib for better distinction of beam IDs
    cmap = plt.get_cmap('hsv')

    # Normalize beam IDs to fit the colormap range
    norm = mcolors.Normalize(vmin=0, vmax=63)  # Assuming beam IDs are in the range [0, 63]

    # Assign colors based on the normalized beam IDs
    colors = cmap(norm(beam_ids))[:, :3]  # Extract RGB channels
    colors = colors.reshape(-1, 3)  # Ensure the shape is (n, 3)

    # Check color array shape for debugging
    print("colors shape:", colors.shape)

    # Set the colors to the point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud with a larger point size
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    # Set rendering options for better visualization
    render_option = vis.get_render_option()
    render_option.point_size = 2.0  # Increase point size for better visibility
    render_option.background_color = np.array([1, 1, 1])  # Set background color to white for contrast

    vis.run()
    vis.destroy_window()