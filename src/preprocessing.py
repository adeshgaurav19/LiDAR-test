"""
This module provides functions for both preprocessing and Exploratory Data Analysis (EDA)
of LiDAR point cloud data.

Author: Adesh
Date: 2025-07-07
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from data_loader import load_lidar_data

# ==============================================================================
# PREPROCESSING FUNCTIONS
# ==============================================================================


def merge_lidar_files(file_paths: dict) -> pd.DataFrame:
    """Merges multiple LiDAR DataFrames, adding an 'original_file' column."""
    if not file_paths:
        raise ValueError("The 'file_paths' dictionary cannot be empty.")
    all_dfs = []
    print("\n--- Merging LiDAR Files ---")
    for name, path in file_paths.items():
        try:
            df = load_lidar_data(path)
            if not df.empty:
                df["original_file"] = name
                all_dfs.append(df)
                print(f"  ✅ Loaded and added '{name}' data ({len(df):,} points).")
        except Exception as e:
            print(f"  ❌ Error loading '{name}': {e}.")
    if not all_dfs:
        raise ValueError("No LiDAR data could be loaded.")
    return pd.concat(all_dfs, ignore_index=True)


def save_processed_data(df: pd.DataFrame, output_path: str):
    """Saves a DataFrame to a .parquet file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"✅ Processed data successfully saved to '{output_path}'.")
    except Exception as e:
        raise IOError(f"Error saving processed data to '{output_path}': {e}")


# ==============================================================================
# EDA PLOTTING FUNCTIONS
# ==============================================================================


def plot_3d_scatter(df: pd.DataFrame, title: str) -> plt.Figure:
    """Generates a 3D scatter plot of the point cloud."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(df["x"], df["y"], df["z"], s=1, c="royalblue", alpha=0.5)
    ax.set_title(f"3D Point Cloud: {title}", fontsize=16)
    return fig


def plot_2d_projections(df: pd.DataFrame, title: str) -> plt.Figure:
    """
    Generates 2D projections of the point cloud (XY, XZ, YZ).
    This corrected version manually sets up each plot for robustness and clarity.
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle(f"2D Projections: {title}", fontsize=16)

    # XY Projection (Top-down view)
    axes[0].scatter(df["x"], df["y"], s=1, alpha=0.5, color="salmon")
    axes[0].set_title("XY Projection (Top-down)", fontsize=13)
    axes[0].set_xlabel("X Coordinate")
    axes[0].set_ylabel("Y Coordinate")
    axes[0].set_aspect("equal", adjustable="box")  # Keep it true to scale
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # XZ Projection (Front view)
    axes[1].scatter(df["x"], df["z"], s=1, alpha=0.5, color="mediumseagreen")
    axes[1].set_title("XZ Projection (Front/Catenary View)", fontsize=13)
    axes[1].set_xlabel("X Coordinate")
    axes[1].set_ylabel("Z Coordinate")
    axes[1].set_aspect(
        "auto", adjustable="box"
    )  # Auto aspect is better for visualizing sag
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # YZ Projection (Side view)
    axes[2].scatter(df["y"], df["z"], s=1, alpha=0.5, color="mediumpurple")
    axes[2].set_title("YZ Projection (Side View)", fontsize=13)
    axes[2].set_xlabel("Y Coordinate")
    axes[2].set_ylabel("Z Coordinate")
    axes[2].set_aspect("auto", adjustable="box")
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout
    return fig


def plot_distributions(df: pd.DataFrame, title: str) -> plt.Figure:
    """Generates histograms for coordinate distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(f"Coordinate Distributions: {title}", fontsize=16)
    for i, coord in enumerate(["x", "y", "z"]):
        axes[i].hist(df[coord], bins=80, color="skyblue", edgecolor="black", alpha=0.7)
        axes[i].set_title(f"{coord.upper()} Distribution")
    return fig


def plot_boxplots(df: pd.DataFrame, title: str) -> plt.Figure:
    """Generates box plots for coordinates."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Coordinate Box Plots: {title}", fontsize=16)
    for i, coord in enumerate(["x", "y", "z"]):
        axes[i].boxplot(
            df[coord],
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue"),
        )
        axes[i].set_title(f"{coord.upper()} Coordinate")
    return fig


def generate_eda_plots(df: pd.DataFrame, dataset_name: str, output_dir: str) -> dict:
    """Generates, saves, and returns all EDA plots for a dataset."""
    if df.empty:
        return {}
    os.makedirs(output_dir, exist_ok=True)
    plot_functions = {
        "3d_scatter": plot_3d_scatter,
        "2d_projections": plot_2d_projections,
        "distributions": plot_distributions,
        "boxplots": plot_boxplots,
    }
    figures = {}
    print(f"\n--- Generating EDA plots for '{dataset_name}' dataset ---")
    for name, func in plot_functions.items():
        try:
            fig = func(df, dataset_name.upper())
            save_path = os.path.join(output_dir, f"{dataset_name}_{name}.png")
            fig.savefig(save_path, bbox_inches="tight")
            print(f"  ✅ Saved {name} to '{save_path}'")
            plt.close(fig)
            figures[name] = fig
        except Exception as e:
            print(f"  ❌ Failed to generate {name}: {e}")
    return figures


# ==============================================================================
# POINT CLOUD ALIGNMENT FUNCTIONS
# ==============================================================================


def align_point_cloud(df: pd.DataFrame):
    """
    Rotates the point cloud to align with the X-axis and returns the
    aligned data along with the rotation matrix for reverse transformation.
    """
    print("  - Aligning point cloud...")
    points_2d = df[["x", "y"]].values

    # Use PCA to find the main direction
    pca = PCA(n_components=2).fit(points_2d)
    direction_vector = pca.components_[0]
    angle = np.arctan2(direction_vector[1], direction_vector[0])

    # Create the rotation matrix
    rotation_matrix = np.array(
        [[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]]
    )

    # Apply the rotation
    rotated_points_2d = points_2d.dot(rotation_matrix.T)

    df_aligned = pd.DataFrame(rotated_points_2d, columns=["x", "y"])
    df_aligned["z"] = df["z"].values

    print(f"  - Cloud rotated by {-np.degrees(angle):.2f} degrees.")
    return df_aligned, rotation_matrix


def reverse_alignment(clusters: list, rotation_matrix: np.ndarray):
    """
    Applies the inverse rotation to transform clusters back to original coordinates.
    """
    realigned_clusters = []

    for cluster in clusters:
        # Apply inverse rotation to XY coordinates
        original_xy = cluster[:, :2].dot(rotation_matrix)

        # Combine with original Z coordinates
        realigned_cluster = np.hstack([original_xy, cluster[:, 2].reshape(-1, 1)])
        realigned_clusters.append(realigned_cluster)

    return realigned_clusters
