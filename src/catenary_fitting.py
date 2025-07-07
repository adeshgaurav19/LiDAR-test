"""
Catenary Fitting for Wire Point Clouds
======================================

This module provides functions for fitting catenary curves to 3D wire point clouds.
It includes PCA-based plane fitting, robust catenary parameter estimation, 
and visualization capabilities.

Author: Adesh
Date: 2025-07-07
"""

import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit, minimize
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from clustering import (
    align_point_cloud,
    reverse_alignment,
    strategy_for_easy,
    strategy_for_medium_extrahard,
    strategy_hough_transform,
)
from data_loader import load_all_lidar_datasets
from preprocessing import align_point_cloud, reverse_alignment

warnings.filterwarnings("ignore")

# ==============================================================================
# CATENARY FITTING FUNCTIONS
# ==============================================================================


def find_best_2d_plane(points_3d):
    """
    Find the best 2D plane for a wire using PCA.

    Args:
        points_3d: numpy array of shape (n_points, 3) containing 3D coordinates

    Returns:
        tuple: (plane_normal, plane_center, u_axis, v_axis, projected_2d_points)
    """
    # Center the points
    centroid = np.mean(points_3d, axis=0)
    centered_points = points_3d - centroid

    # Use PCA to find the best fitting plane
    pca = PCA(n_components=3)
    pca.fit(centered_points)

    # The plane normal is the third principal component (least variance)
    plane_normal = pca.components_[2]

    # The first two components define the plane's coordinate system
    u_axis = pca.components_[0]  # Primary direction (along wire)
    v_axis = pca.components_[1]  # Secondary direction (sag direction)

    # Project points onto the 2D plane
    projected_2d = centered_points.dot(np.column_stack([u_axis, v_axis]))

    return plane_normal, centroid, u_axis, v_axis, projected_2d


def catenary_2d(x, a, x0, y0):
    """
    2D catenary equation: y = a * cosh((x - x0) / a) + y0

    Args:
        x: x-coordinates
        a: catenary parameter (related to tension and weight)
        x0: x-coordinate of the vertex (lowest point)
        y0: y-coordinate of the vertex (lowest point)
    """
    return a * np.cosh((x - x0) / a) + y0


def fit_catenary_to_2d_points(points_2d, method="robust"):
    """
    Fit a catenary curve to 2D projected points.

    Args:
        points_2d: numpy array of shape (n_points, 2) with 2D coordinates
        method: 'robust' or 'direct' fitting method

    Returns:
        tuple: (fitted_parameters, fit_quality_score)
    """
    x, y = points_2d[:, 0], points_2d[:, 1]

    if len(x) < 3:
        return None, 0

    # Sort points by x-coordinate
    sort_idx = np.argsort(x)
    x_sorted, y_sorted = x[sort_idx], y[sort_idx]

    # Initial parameter estimates
    x0_init = x_sorted[np.argmin(y_sorted)]  # x of lowest point
    y0_init = np.min(y_sorted)  # lowest y value

    # Estimate 'a' parameter from the span and sag
    x_span = np.max(x_sorted) - np.min(x_sorted)
    y_sag = np.max(y_sorted) - np.min(y_sorted)
    a_init = max(x_span / 4, y_sag / 2, 1.0)  # Reasonable initial guess

    initial_params = [a_init, x0_init, y0_init]

    try:
        if method == "robust":
            # Robust fitting using minimize with custom loss
            def catenary_loss(params):
                a, x0, y0 = params
                if a <= 0:  # Constraint: a must be positive
                    return 1e10
                y_pred = catenary_2d(x_sorted, a, x0, y0)
                residuals = y_sorted - y_pred
                # Use Huber loss for robustness
                huber_delta = np.std(residuals)
                huber_loss = np.where(
                    np.abs(residuals) <= huber_delta,
                    0.5 * residuals**2,
                    huber_delta * np.abs(residuals) - 0.5 * huber_delta**2,
                )
                return np.sum(huber_loss)

            result = minimize(
                catenary_loss,
                initial_params,
                bounds=[(0.1, None), (None, None), (None, None)],
                method="L-BFGS-B",
            )

            if result.success:
                fitted_params = result.x
            else:
                return None, 0

        else:  # Direct curve fitting
            fitted_params, _ = curve_fit(
                catenary_2d, x_sorted, y_sorted, p0=initial_params, maxfev=5000
            )

        # Calculate fit quality (R²)
        y_pred = catenary_2d(x_sorted, *fitted_params)
        ss_res = np.sum((y_sorted - y_pred) ** 2)
        ss_tot = np.sum((y_sorted - np.mean(y_sorted)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return fitted_params, r_squared

    except Exception as e:
        print(f"Catenary fitting failed: {e}")
        return None, 0


def project_catenary_to_3d(x_range, catenary_params, plane_center, u_axis, v_axis):
    """
    Project the fitted 2D catenary back to 3D space.

    Args:
        x_range: x-coordinates for the catenary curve
        catenary_params: [a, x0, y0] parameters
        plane_center: 3D center of the plane
        u_axis: first axis of the plane (u direction)
        v_axis: second axis of the plane (v direction)

    Returns:
        numpy array of 3D points representing the catenary curve
    """
    a, x0, y0 = catenary_params

    # Generate 2D catenary points
    y_catenary = catenary_2d(x_range, a, x0, y0)
    points_2d = np.column_stack([x_range, y_catenary])

    # Transform back to 3D
    points_3d = plane_center + points_2d[:, 0:1] * u_axis + points_2d[:, 1:2] * v_axis

    return points_3d


# ==============================================================================
# ENHANCED STRATEGY FUNCTIONS WITH CATENARY FITTING
# ==============================================================================


def fit_catenary_to_wire_clusters(wire_clusters, dataset_name="Unknown"):
    """
    Fit catenary curves to each wire cluster.

    Args:
        wire_clusters: List of numpy arrays, each containing 3D points of a wire
        dataset_name: Name of the dataset for logging

    Returns:
        dict: Results containing fitted parameters and 3D curves for each wire
    """
    print(f"\n--- Fitting Catenary Curves for {dataset_name} ---")

    results = {
        "wire_params": [],
        "wire_curves_3d": [],
        "fit_qualities": [],
        "plane_info": [],
    }

    for i, wire_points in enumerate(wire_clusters):
        print(f"  Processing Wire {i+1}: {len(wire_points)} points")

        if len(wire_points) < 5:
            print(f"    ❌ Not enough points for Wire {i+1}")
            continue

        # Find best 2D plane
        plane_normal, plane_center, u_axis, v_axis, points_2d = find_best_2d_plane(
            wire_points
        )

        # Fit catenary to 2D projection
        catenary_params, fit_quality = fit_catenary_to_2d_points(
            points_2d, method="robust"
        )

        if catenary_params is None:
            print(f"    ❌ Catenary fitting failed for Wire {i+1}")
            continue

        a, x0, y0 = catenary_params
        print(
            f"    ✅ Wire {i+1}: a={a:.3f}, x0={x0:.3f}, y0={y0:.3f}, R²={fit_quality:.3f}"
        )

        # Generate 3D catenary curve
        x_min, x_max = np.min(points_2d[:, 0]), np.max(points_2d[:, 0])
        x_range = np.linspace(x_min, x_max, 100)
        curve_3d = project_catenary_to_3d(
            x_range, catenary_params, plane_center, u_axis, v_axis
        )

        # Store results
        results["wire_params"].append(catenary_params)
        results["wire_curves_3d"].append(curve_3d)
        results["fit_qualities"].append(fit_quality)
        results["plane_info"].append(
            {
                "normal": plane_normal,
                "center": plane_center,
                "u_axis": u_axis,
                "v_axis": v_axis,
            }
        )

    return results


def save_plot(fig, difficulty: str, output_dir: str):
    """Saves a Matplotlib figure to the specified output directory."""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a descriptive filename
    filename = f"{difficulty}_final_visualization.png"
    filepath = os.path.join(output_dir, filename)

    # Save the figure
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"    🖼️  Plot saved to: {filepath}")


def save_results_to_json(data: dict, output_path: str):
    """
    Saves the final results dictionary to a JSON file, converting all
    nested NumPy arrays to lists to ensure compatibility.
    """
    serializable_data = {}

    # Loop through each difficulty level ('easy', 'hard', etc.)
    for difficulty, results in data.items():

        # This dictionary will hold the converted data for one difficulty level
        converted_results = {}

        # Convert the list of 'wire_clusters' arrays
        converted_results["wire_clusters"] = [
            w.tolist() for w in results["wire_clusters"]
        ]

        # Convert the nested data within 'catenary_results'
        catenary_res = results["catenary_results"]
        converted_catenary = {
            "wire_params": [p.tolist() for p in catenary_res["wire_params"]],
            "wire_curves_3d": [c.tolist() for c in catenary_res["wire_curves_3d"]],
            "fit_qualities": catenary_res["fit_qualities"],
            "plane_info": [
                {
                    "normal": info["normal"].tolist(),
                    "center": info["center"].tolist(),
                    "u_axis": info["u_axis"].tolist(),
                    "v_axis": info["v_axis"].tolist(),
                }
                for info in catenary_res["plane_info"]
            ],
        }
        converted_results["catenary_results"] = converted_catenary

        serializable_data[difficulty] = converted_results

    # Ensure the output directory exists and save the file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(serializable_data, f, indent=4)
    print(f"💾 All results data successfully saved to '{output_path}'")


def visualize_catenary_results(
    original_df,
    wire_clusters,
    catenary_results,
    title="Catenary Fitting Results",
    output_dir=None,
    difficulty="unknown",
):
    """
    Creates a comprehensive visualization of the catenary fitting results
    and saves it using the helper function.
    """
    fig = plt.figure(figsize=(15, 10))

    # --- 3D Plot (logic unchanged) ---
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(
        original_df["x"],
        original_df["y"],
        original_df["z"],
        c="lightgray",
        s=1,
        alpha=0.3,
        label="Original Points",
    )
    colors = ["red", "blue", "green", "orange", "purple", "brown"]
    for i, (wire_points, curve_3d, fit_quality) in enumerate(
        zip(
            wire_clusters,
            catenary_results["wire_curves_3d"],
            catenary_results["fit_qualities"],
        )
    ):
        color = colors[i % len(colors)]
        ax1.scatter(
            wire_points[:, 0],
            wire_points[:, 1],
            wire_points[:, 2],
            c=color,
            s=10,
            alpha=0.7,
        )
        ax1.plot(
            curve_3d[:, 0],
            curve_3d[:, 1],
            curve_3d[:, 2],
            c=color,
            linewidth=3,
            linestyle="--",
            label=f"Wire {i+1} Catenary (R²={fit_quality:.2f})",
        )
    ax1.set_title(f"{title} - 3D View")
    ax1.legend()
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # --- 2D Plot (logic unchanged) ---
    ax2 = fig.add_subplot(122)
    ax2.scatter(original_df["x"], original_df["z"], c="lightgray", s=1, alpha=0.3)
    for i, (wire_points, curve_3d) in enumerate(
        zip(wire_clusters, catenary_results["wire_curves_3d"])
    ):
        color = colors[i % len(colors)]
        ax2.scatter(wire_points[:, 0], wire_points[:, 2], c=color, s=10, alpha=0.7)
        ax2.plot(curve_3d[:, 0], curve_3d[:, 2], c=color, linewidth=3, linestyle="--")
    ax2.set_title(f"{title} - X-Z Projection")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")

    plt.tight_layout()

    # --- MODIFIED: Call the save_plot function instead of plt.show() ---
    if output_dir:
        save_plot(fig, difficulty, output_dir)

    plt.close(fig)  # Close the figure to free up memory


# ==============================================================================
# INTEGRATION WITH EXISTING STRATEGY FUNCTIONS
# ==============================================================================


def run_complete_analysis_with_catenary():
    """
    Master function that runs all steps and saves all outputs (plots and data).
    """
    print("🚀 Starting Complete Wire Analysis with Catenary Fitting")

    # --- Setup output directories ---
    try:
        current_dir = os.path.dirname(os.path.realpath(__file__))
    except NameError:
        current_dir = os.getcwd()
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    plot_output_dir = os.path.join(project_root, "output", "plots")
    results_output_dir = os.path.join(project_root, "output", "results")

    # (The clustering logic remains the same)
    lidar_dataframes = load_all_lidar_datasets()
    if not lidar_dataframes:
        return None
    all_results = {}
    for difficulty, df in lidar_dataframes.items():
        print(f"\n{'='*80}\n🔍 ANALYZING '{difficulty.upper()}' DATASET")
        if df.empty:
            continue
        wire_clusters, rotation_matrix = None, None
        if difficulty == "easy":
            wire_clusters = strategy_for_easy(df)
        else:
            aligned_df, rotation_matrix = align_point_cloud(df)
            if difficulty == "medium":
                wire_clusters = strategy_for_medium_extrahard(
                    aligned_df, eps=0.75, min_samples=25
                )
            elif difficulty == "hard":
                wire_clusters = strategy_hough_transform(aligned_df)
            elif difficulty == "extrahard":
                wire_clusters = strategy_for_medium_extrahard(
                    aligned_df, eps=0.8, min_samples=15
                )
            if wire_clusters and rotation_matrix is not None:
                wire_clusters = reverse_alignment(wire_clusters, rotation_matrix)
        if not wire_clusters or not all(len(c) > 0 for c in wire_clusters):
            continue
        print(f"✅ Found {len(wire_clusters)} wire clusters")
        catenary_results = fit_catenary_to_wire_clusters(
            wire_clusters, difficulty.upper()
        )
        if not catenary_results["wire_params"]:
            continue

        # --- Visualize and Save Plot ---
        plot_path = os.path.join(
            plot_output_dir, f"{difficulty}_final_visualization.png"
        )
        visualize_catenary_results(
            df,
            wire_clusters,
            catenary_results,
            f"{difficulty.upper()} Dataset",
            plot_path,
        )

        # Store results for the final JSON dump
        all_results[difficulty] = {
            "wire_clusters": wire_clusters,
            "catenary_results": catenary_results,
        }

    # --- Save the consolidated results dictionary to a single JSON file ---
    if all_results:
        final_results_path = os.path.join(
            results_output_dir, "all_pipeline_results.json"
        )
        # This now calls the new, robust save function
        save_results_to_json(all_results, final_results_path)

    return all_results


"""
if __name__ == "__main__":
    print("Catenary Fitting Module Loaded Successfully! 🎯")
    print("\nAvailable functions:")
    print("- find_best_2d_plane()")
    print("- fit_catenary_to_2d_points()")
    print("- fit_catenary_to_wire_clusters()")
    print("- visualize_catenary_results()")
    print("- run_complete_analysis_with_catenary()")
    print("- test_catenary_fitting()")
    

    
    # To run the complete analysis, first implement the placeholder functions:
    results = run_complete_analysis_with_catenary()
"""
