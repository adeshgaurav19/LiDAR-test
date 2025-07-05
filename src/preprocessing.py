"""
This module provides functions for basic preprocessing of LiDAR point cloud data.
Currently, it supports merging multiple LiDAR datasets into a single DataFrame
and assigning a unique identifier to points from each original file.
"""

import pandas as pd
import os
from data_loader import load_lidar_data # Assuming data_loader.py is in the same 'src' directory

def merge_lidar_files(file_paths: dict) -> pd.DataFrame:
    """
    Merges multiple LiDAR point cloud DataFrames into a single DataFrame,
    adding an 'original_file' column to identify the source of each point.

    Args:
        file_paths (dict): A dictionary where keys are descriptive names (e.g., 'easy', 'medium')
                           and values are the full paths to the .parquet files.

    Returns:
        pd.DataFrame: A concatenated DataFrame with 'x', 'y', 'z', and 'original_file' columns.
                      'original_file' will contain the key from the input dictionary.

    Raises:
        ValueError: If file_paths dictionary is empty or no files can be loaded.
    """
    if not file_paths:
        raise ValueError("The 'file_paths' dictionary cannot be empty.")

    all_dfs = []
    print("\n--- Merging LiDAR Files ---")
    for name, path in file_paths.items():
        try:
            df = load_lidar_data(path)
            if not df.empty:
                df['original_file'] = name # Add the identifier column
                all_dfs.append(df)
                print(f"  Loaded and added '{name}' data ({len(df):,} points).")
            else:
                print(f"  Skipping '{name}': Loaded DataFrame was empty.")
        except Exception as e:
            print(f"  Error loading '{name}' from '{path}': {e}. Skipping this file.")

    if not all_dfs:
        raise ValueError("No LiDAR data could be loaded and merged from the provided file paths.")

    merged_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nSuccessfully merged all loaded data. Total points: {len(merged_df):,}")
    print(f"Columns in merged DataFrame: {merged_df.columns.tolist()}")
    return merged_df

def save_processed_data(df: pd.DataFrame, output_path: str):
    """
    Saves a DataFrame to a .parquet file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        output_path (str): The full path where the .parquet file will be saved.
    """
    try:
        df.to_parquet(output_path, index=False)
        print(f"Processed data successfully saved to '{output_path}'.")
    except Exception as e:
        print(f"Error saving processed data to '{output_path}': {e}")
        raise IOError(f"Failed to save DataFrame to {output_path}")

def read_processed_data(file_path: str) -> pd.DataFrame:
    """
    Reads processed LiDAR data from a .parquet file, expecting 'x', 'y', 'z', and 'original_file' columns.

    Args:
        file_path (str): The full path to the processed .parquet file.

    Returns:
        pd.DataFrame: The DataFrame containing the processed LiDAR points.

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
        ValueError: If the loaded DataFrame does not contain expected columns.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' was not found.")

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        raise IOError(f"Error loading parquet file '{file_path}': {e}")

    required_columns = ['x', 'y', 'z', 'original_file']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Processed DataFrame from '{file_path}' is missing required columns: {missing_cols}. Expected 'x', 'y', 'z', 'original_file'.")

    return df

if __name__ == "__main__":
    # This block allows you to test the preprocessing module independently

    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    data_dir = os.path.join(project_root, 'data')
    processed_data_dir = os.path.join(project_root, 'data', 'processed') # Create a subdir for processed data

    # Ensure the processed data directory exists
    os.makedirs(processed_data_dir, exist_ok=True)

    print("--- Testing Preprocessing Module: Merging & Saving ---")

    file_paths_to_merge = {
        "easy": os.path.join(data_dir, "lidar_cable_points_easy.parquet"),
        "medium": os.path.join(data_dir, "lidar_cable_points_medium.parquet"),
        "hard": os.path.join(data_dir, "lidar_cable_points_hard.parquet"),
        "extrahard": os.path.join(data_dir, "lidar_cable_points_extrahard.parquet")
    }

    output_merged_file = os.path.join(processed_data_dir, "all_lidar_data_merged.parquet")

    try:
        # 1. Merge the files
        merged_df = merge_lidar_files(file_paths_to_merge)
        print(f"\nMerged DataFrame head:\n{merged_df.head()}")
        print(f"Value counts for 'original_file':\n{merged_df['original_file'].value_counts()}")

        # 2. Save the merged file
        save_processed_data(merged_df, output_merged_file)

        # 3. Read the saved file back to verify
        print(f"\n--- Verifying Re-read of Processed Data ---")
        re_read_df = read_processed_data(output_merged_file)
        print(f"Successfully re-read data. Shape: {re_read_df.shape}")
        print(f"Re-read DataFrame head:\n{re_read_df.head()}")
        print(f"Re-read Value counts for 'original_file':\n{re_read_df['original_file'].value_counts()}")
        
        # Verify the content
        if merged_df.equals(re_read_df):
            print("\nVerification successful: Original merged DataFrame matches re-read DataFrame.")
        else:
            print("\nWarning: Original merged DataFrame DOES NOT match re-read DataFrame.")


    except (ValueError, FileNotFoundError, IOError) as e:
        print(f"Error during preprocessing test: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during preprocessing test: {e}")

    print("\n--- Preprocessing Module Test Complete ---")