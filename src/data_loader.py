"""
This module provides functions for loading LiDAR point cloud data from .parquet files.
It ensures that the loaded data contains the expected 'x', 'y', and 'z' columns,
which are essential for subsequent processing in the catenary modeling pipeline.
"""
import pandas as pd
import os

def load_lidar_data(file_path: str) -> pd.DataFrame:
    """
    Loads LiDAR point cloud data from a .parquet file. 

    This function performs checks to ensure the file exists and is a valid
    parquet file, and that it contains the necessary 'x', 'y', and 'z'
    coordinates as specified for the LiDAR datasets. 

    Args:
        file_path (str): The full path to the .parquet file. 

    Returns:
        pd.DataFrame: A DataFrame containing the LiDAR points with 'x', 'y', and 'z' columns.

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
        IOError: If there's an error reading the parquet file.
        ValueError: If the loaded DataFrame does not contain 'x', 'y', and 'z' columns.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' was not found.")

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        raise IOError(f"Error loading parquet file '{file_path}': {e}")

    # The document states the files have 'x', 'y', and 'z' columns.
    required_columns = ['x', 'y', 'z']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"DataFrame from '{file_path}' is missing required columns: {missing_cols}. Expected 'x', 'y', 'z'.")

    return df[required_columns] # Ensure only the required columns are returned

if __name__ == "__main__":
    # This block allows you to test the data_loader independently and serves as example usage.

    # Assume your .parquet files are in a 'data' directory one level up from 'src/catenary_model'
    # Adjust this path based on where you put your 'data' folder relative to where you run this script
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    data_dir = os.path.join(project_root, 'data')

    print("--- Testing Data Loader ---")

    # Example usage for each dataset:
    file_names = {
        "easy": "lidar_cable_points_easy.parquet",
        "medium": "lidar_cable_points_medium.parquet",
        "hard": "lidar_cable_points_hard.parquet",
        "extrahard": "lidar_cable_points_extrahard.parquet"
    }

    loaded_dataframes = {}

    for name, filename in file_names.items():
        file_path = os.path.join(data_dir, filename)
        try:
            print(f"\nAttempting to load {filename}...")
            df = load_lidar_data(file_path)
            loaded_dataframes[name] = df
            print(f"Successfully loaded {name.upper()} data. Shape: {df.shape}")
            print(f"First 5 rows of {name.upper()} data:\n{df.head()}")
        except (FileNotFoundError, ValueError, IOError) as e:
            print(f"Error loading {name.upper()} data from '{file_path}': {e}")
            print("Please ensure the .parquet files are in the 'data/' directory.")
        except Exception as e:
            print(f"An unexpected error occurred while loading {name.upper()} data: {e}")

    print("\n--- Data Loading Test Complete ---")

    if loaded_dataframes:
        print("\nSummary of loaded data:")
        for name, df in loaded_dataframes.items():
            print(f"- {name.upper()}: {df.shape[0]} points")