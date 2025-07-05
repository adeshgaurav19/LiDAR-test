"""
This module provides functions for clustering LiDAR point cloud data to identify individual wires.
It includes implementations for DBSCAN and HDBSCAN, offering different approaches to density-based clustering.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import hdbscan # New import for HDBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import time # To time clustering operations

# Assume data_loader.py is in the same 'src' directory
# (We'll use it in the example usage block)
# from data_loader import load_lidar_data

def perform_dbscan_clustering(points_df: pd.DataFrame, eps: float, min_samples: int) -> pd.DataFrame:
    """
    Performs DBSCAN clustering on 3D LiDAR points (x, y, z).

    DBSCAN is a density-based clustering algorithm that can discover clusters
    of arbitrary shape in spatial data with noise.

    Args:
        points_df (pd.DataFrame): A DataFrame with 'x', 'y', 'z' columns
                                  representing the 3D LiDAR points.
        eps (float): The maximum distance between two samples for one to be
                     considered as in the neighborhood of the other. This is
                     the most critical DBSCAN parameter for point cloud data,
                     representing the maximum radius of the neighborhood.
        min_samples (int): The number of samples (or total weight) in a neighborhood
                           for a point to be considered as a core point. This
                           controls the minimum size of a cluster.

    Returns:
        pd.DataFrame: The input DataFrame with an additional 'cluster_id' column.
                      Noise points are assigned a cluster_id of -1.

    Raises:
        ValueError: If the input DataFrame does not contain 'x', 'y', 'z' columns.
    """
    required_columns = ['x', 'y', 'z']
    if not all(col in points_df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in points_df.columns]
        raise ValueError(f"Input DataFrame is missing required columns: {missing_cols}. Expected 'x', 'y', 'z'.")

    # Extract the 3D coordinates
    points = points_df[required_columns].values

    print(f"  Starting DBSCAN clustering with eps={eps:.4f}, min_samples={min_samples}...")
    start_time = time.time()
    db = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = db.fit_predict(points)
    end_time = time.time()
    print(f"  DBSCAN finished in {end_time - start_time:.2f} seconds.")

    result_df = points_df.copy()
    result_df['cluster_id'] = clusters
    n_clusters = result_df['cluster_id'].nunique() - (1 if -1 in result_df['cluster_id'].unique() else 0)
    print(f"  DBSCAN found {n_clusters} clusters.")
    if -1 in result_df['cluster_id'].unique():
        noise_points = (result_df['cluster_id'] == -1).sum()
        print(f"  {noise_points:,} points identified as noise (cluster_id = -1).")
    print(f"  Cluster ID counts (first 10, including noise):\n{result_df['cluster_id'].value_counts().sort_index().head(10).to_string()}")


    return result_df

def perform_hdbscan_clustering(points_df: pd.DataFrame, min_cluster_size: int, min_samples: int = None) -> pd.DataFrame:
    """
    Performs HDBSCAN clustering on 3D LiDAR points (x, y, z).

    HDBSCAN is a powerful density-based clustering algorithm that can find clusters
    of varying densities and is robust to noise. It does not require an 'eps' parameter.

    Args:
        points_df (pd.DataFrame): A DataFrame with 'x', 'y', 'z' columns
                                  representing the 3D LiDAR points.
        min_cluster_size (int): The smallest size grouping that a cluster can be.
                                Similar to min_samples in DBSCAN for overall cluster size.
        min_samples (int, optional): The number of samples in a neighborhood for a point
                                     to be considered a core point. Defaults to `min_cluster_size`.
                                     Higher values lead to more conservative clustering (more noise).

    Returns:
        pd.DataFrame: The input DataFrame with an additional 'cluster_id' column.
                      Noise points are assigned a cluster_id of -1.

    Raises:
        ValueError: If the input DataFrame does not contain 'x', 'y', 'z' columns.
    """
    required_columns = ['x', 'y', 'z']
    if not all(col in points_df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in points_df.columns]
        raise ValueError(f"Input DataFrame is missing required columns: {missing_cols}. Expected 'x', 'y', 'z'.")

    # Extract the 3D coordinates
    points = points_df[required_columns].values

    print(f"  Starting HDBSCAN clustering with min_cluster_size={min_cluster_size}, min_samples={min_samples if min_samples is not None else min_cluster_size}...")
    start_time = time.time()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    clusters = clusterer.fit_predict(points)
    end_time = time.time()
    print(f"  HDBSCAN finished in {end_time - start_time:.2f} seconds.")


    result_df = points_df.copy()
    result_df['cluster_id'] = clusters
    n_clusters = result_df['cluster_id'].nunique() - (1 if -1 in result_df['cluster_id'].unique() else 0)
    print(f"  HDBSCAN found {n_clusters} clusters.")
    if -1 in result_df['cluster_id'].unique():
        noise_points = (result_df['cluster_id'] == -1).sum()
        print(f"  {noise_points:,} points identified as noise (cluster_id = -1).")
    print(f"  Cluster ID counts (first 10, including noise):\n{result_df['cluster_id'].value_counts().sort_index().head(10).to_string()}")

    return result_df


def estimate_dbscan_eps(points_df: pd.DataFrame, n_neighbors: int = 5) -> float:
    """
    Estimates a suitable 'eps' value for DBSCAN using the K-distance graph method.
    (No changes to this function, it's specific to DBSCAN)
    """
    required_columns = ['x', 'y', 'z']
    if not all(col in points_df.columns for col in required_columns):
        raise ValueError(f"Input DataFrame is missing required columns: {points_df.columns.tolist()}. Expected 'x', 'y', 'z'.")

    points = points_df[required_columns].values
    print(f"  Estimating DBSCAN 'eps' using K-distance graph (k={n_neighbors})...")

    # Find the k-nearest neighbors for each point
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
    distances, indices = nbrs.kneighbors(points)

    # Sort distances to the k-th nearest neighbor
    k_distances = np.sort(distances[:, n_neighbors-1], axis=0)

    plt.figure(figsize=(12, 6))
    plt.plot(k_distances)
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"Distance to {n_neighbors}-th nearest neighbor (k-distance)")
    plt.title(f"K-Distance Graph for DBSCAN 'eps' Estimation (k={n_neighbors})")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    print(f"  Please inspect the K-distance graph above. Look for a 'knee' or 'elbow' point.")
    print(f"  The Y-value at this elbow is a good candidate for 'eps'.")
    # A simple way to suggest a value (e.g., 90th percentile, but manual inspection is better)
    suggested_eps = np.percentile(k_distances, 90) # Just a heuristic, not a definitive method
    print(f"  A heuristic-based suggestion for 'eps' (e.g., 90th percentile): {suggested_eps:.4f}")
    return suggested_eps


if __name__ == "__main__":
    # This block allows you to test the clustering module independently.

    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    data_dir = os.path.join(project_root, 'data')
    processed_data_dir = os.path.join(data_dir, 'processed')

    sys.path.append(os.path.join(project_root, 'src'))
    from data_loader import load_lidar_data
    from preprocessing import merge_lidar_files

    print("--- Testing Clustering Module (DBSCAN & HDBSCAN) ---")

    # --- Step 1: Prepare data (merge if not already done) ---
    merged_file_path = os.path.join(processed_data_dir, "all_lidar_data_merged.parquet")
    merged_df = None

    if os.path.exists(merged_file_path):
        try:
            merged_df = pd.read_parquet(merged_file_path)
            print(f"\nLoaded pre-merged data from '{merged_file_path}'. Shape: {merged_df.shape}")
        except Exception as e:
            print(f"\nCould not load pre-merged data: {e}. Attempting to merge fresh data.")

    if merged_df is None or merged_df.empty:
        file_paths_to_merge = {
            "easy": os.path.join(data_dir, "lidar_cable_points_easy.parquet"),
            "medium": os.path.join(data_dir, "lidar_cable_points_medium.parquet"),
            "hard": os.path.join(data_dir, "lidar_cable_points_hard.parquet"),
            "extrahard": os.path.join(data_dir, "lidar_cable_points_extrahard.parquet")
        }
        try:
            merged_df = merge_lidar_files(file_paths_to_merge)
        except Exception as e:
            print(f"Error merging data for clustering test: {e}")
            merged_df = pd.DataFrame()


    if not merged_df.empty:
        # --- DBSCAN Testing ---
        print("\n--- Running DBSCAN Test ---")
        suggested_min_samples_dbscan = 10 # Adjust as needed
        estimated_eps = estimate_dbscan_eps(merged_df, n_neighbors=suggested_min_samples_dbscan)

        chosen_eps_dbscan = estimated_eps # Start with suggestion
        chosen_min_samples_dbscan = suggested_min_samples_dbscan

        try:
            clustered_df_dbscan = perform_dbscan_clustering(merged_df,
                                                            eps=chosen_eps_dbscan,
                                                            min_samples=chosen_min_samples_dbscan)
            print("\n--- DBSCAN Cluster Visualization (Sample) ---")
            non_noise_clusters_dbscan = clustered_df_dbscan[clustered_df_dbscan['cluster_id'] != -1]
            if not non_noise_clusters_dbscan.empty:
                top_clusters_ids_dbscan = non_noise_clusters_dbscan['cluster_id'].value_counts().nlargest(3).index.tolist()
                print(f"Visualizing top {len(top_clusters_ids_dbscan)} largest non-noise clusters from DBSCAN: {top_clusters_ids_dbscan}")

                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                colors = plt.cm.get_cmap('tab10', len(top_clusters_ids_dbscan)) # Use tab10 for distinct colors

                for i, cluster_id in enumerate(top_clusters_ids_dbscan):
                    cluster_data = non_noise_clusters_dbscan[non_noise_clusters_dbscan['cluster_id'] == cluster_id]
                    ax.scatter(cluster_data['x'], cluster_data['y'], cluster_data['z'],
                               s=2, alpha=0.8, color=colors(i), label=f'DBSCAN Cluster {cluster_id}')
                ax.set_title(f'Sample of DBSCAN Clustered LiDAR Wires (eps={chosen_eps_dbscan:.4f}, min_samples={chosen_min_samples_dbscan})')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.legend()
                plt.tight_layout()
                plt.show()
            else:
                print("No non-noise clusters found by DBSCAN to visualize.")

        except Exception as e:
            print(f"Error during DBSCAN test: {e}")

        # --- HDBSCAN Testing ---
        print("\n--- Running HDBSCAN Test ---")
        # HDBSCAN parameters: min_cluster_size is the primary one.
        # min_samples (optional) can be used to make clustering more conservative (more noise)
        chosen_min_cluster_size_hdbscan = 10 # Smallest cluster size to consider a wire
        chosen_min_samples_hdbscan = None # Defaults to min_cluster_size; set higher for more noise

        try:
            clustered_df_hdbscan = perform_hdbscan_clustering(merged_df,
                                                              min_cluster_size=chosen_min_cluster_size_hdbscan,
                                                              min_samples=chosen_min_samples_hdbscan)

            print("\n--- HDBSCAN Cluster Visualization (Sample) ---")
            non_noise_clusters_hdbscan = clustered_df_hdbscan[clustered_df_hdbscan['cluster_id'] != -1]
            if not non_noise_clusters_hdbscan.empty:
                top_clusters_ids_hdbscan = non_noise_clusters_hdbscan['cluster_id'].value_counts().nlargest(3).index.tolist()
                print(f"Visualizing top {len(top_clusters_ids_hdbscan)} largest non-noise clusters from HDBSCAN: {top_clusters_ids_hdbscan}")

                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                colors = plt.cm.get_cmap('Set1', len(top_clusters_ids_hdbscan)) # Use Set1 for distinct colors

                for i, cluster_id in enumerate(top_clusters_ids_hdbscan):
                    cluster_data = non_noise_clusters_hdbscan[non_noise_clusters_hdbscan['cluster_id'] == cluster_id]
                    ax.scatter(cluster_data['x'], cluster_data['y'], cluster_data['z'],
                               s=2, alpha=0.8, color=colors(i), label=f'HDBSCAN Cluster {cluster_id}')
                ax.set_title(f'Sample of HDBSCAN Clustered LiDAR Wires (min_cluster_size={chosen_min_cluster_size_hdbscan})')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.legend()
                plt.tight_layout()
                plt.show()
            else:
                print("No non-noise clusters found by HDBSCAN to visualize.")

        except ImportError:
            print("HDBSCAN not installed. Please run `pip install hdbscan` to test HDBSCAN clustering.")
        except Exception as e:
            print(f"Error during HDBSCAN test: {e}")

    else:
        print("\nSkipping clustering tests: No merged data available.")

    print("\n--- Clustering Module Test Complete ---")