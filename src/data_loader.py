"""
This module provides functions for loading LiDAR point cloud data from .parquet files.
It includes a master function to load all datasets required for the project at once.

Author: Adesh
Date: 2025-07-07
"""
import pandas as pd
import os
import sys

def load_lidar_data(file_path: str) -> pd.DataFrame:
    """
    Loads a single LiDAR .parquet file into a DataFrame, ensuring it contains
    the required 'x', 'y', and 'z' columns.

    Args:
        file_path (str): The full path to the .parquet file.

    Returns:
        pd.DataFrame: A DataFrame containing the LiDAR point cloud data.
        
    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        ValueError: If the loaded DataFrame is missing required columns.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_parquet(file_path)

    required_columns = ['x', 'y', 'z']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Input DataFrame from '{file_path}' is missing required columns: {missing}")

    return df[required_columns]

def load_all_lidar_datasets() -> dict:
    """
    Loads all four LiDAR datasets (easy, medium, hard, extrahard) from the
    'data' directory into a dictionary of DataFrames.

    This function is designed to be called from other modules in the pipeline.

    Returns:
        dict: A dictionary where keys are the difficulty levels ('easy', 'medium', etc.)
              and values are the corresponding pandas DataFrames.
    """
    print("--- Loading All LiDAR Datasets ---")
    
    # This path logic works in both .py scripts and Jupyter notebooks
    try:
        current_dir = os.path.dirname(os.path.realpath(__file__))
    except NameError:
        current_dir = os.getcwd()
        
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    data_dir = os.path.join(project_root, 'data')
    
    if not os.path.isdir(data_dir):
        print(f"❌ Critical Error: Data directory not found at the expected path: {data_dir}")
        return {}

    datasets = {}
    for difficulty in ['easy', 'medium', 'hard', 'extrahard']:
        try:
            file_path = os.path.join(data_dir, f"lidar_cable_points_{difficulty}.parquet")
            datasets[difficulty] = load_lidar_data(file_path)
            print(f"  ✅ Loaded '{difficulty}' dataset ({len(datasets[difficulty]):,} points).")
        except Exception as e:
            print(f"  ❌ Could not load '{difficulty}' dataset: {e}")
            datasets[difficulty] = pd.DataFrame()
    
    return datasets

# if __name__ == "__main__":
#     # This block is commented out as requested.
#     # It serves as an example of how to use the functions in this module.
# 
#     print("--- Testing data_loader.py ---")
#     
#     # Example of loading all datasets
#     all_data = load_all_lidar_datasets()
#     
#     if all_data:
#         print("\n--- Summary of Loaded Data ---")
#         for name, df in all_data.items():
#             if not df.empty:
#                 print(f"  - {name.upper()}: {df.shape[0]} points, Columns: {df.columns.tolist()}")
#             else:
#                 print(f"  - {name.upper()}: Failed to load.")
#     else:
#         print("No data was loaded.")