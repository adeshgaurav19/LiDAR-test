"""
LiDAR point cloud processing for powerline wire detection.
Uses different clustering methods for different dataset difficulties.
Saves plots to output/plots directory.

Author: Adesh
Date: 2025-07-07
"""

import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_loader import load_all_lidar_datasets, load_lidar_data
from preprocessing import align_point_cloud, reverse_alignment

# Optional OpenCV import for Hough Transform
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Hough Transform strategy will be skipped.")


def ensure_output_directory():
    """Create output/plots directory if it doesn't exist."""
    output_dir = os.path.join(os.getcwd(), "output", "plots")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def strategy_for_easy(df: pd.DataFrame):
    """Segment-and-Trace strategy for clean, unaligned 'Easy' dataset."""
    print("--- Applying Strategy 1: Segment-and-Trace ---")
    points = df[["x", "y", "z"]].values
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    all_wire_clusters, steps = [], np.arange(x_min, x_max, 0.5)

    for i in range(len(steps) - 1):
        s_mask = (points[:, 0] >= steps[i]) & (points[:, 0] < steps[i + 1])
        s_pts = points[s_mask]
        if len(s_pts) < 5:
            all_wire_clusters.append([])
            continue

        labels = DBSCAN(eps=0.4, min_samples=3).fit_predict(s_pts[:, 1:])
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        s_clusters = [
            {"c": np.mean(s_pts[labels == l], axis=0), "p": s_pts[labels == l]}
            for l in unique_labels
        ]
        s_clusters.sort(key=lambda x: x["c"][1])
        all_wire_clusters.append(s_clusters)

    n_per_slice = [len(s) for s in all_wire_clusters if s]
    if not n_per_slice:
        return None

    n_wires = Counter(n_per_slice).most_common(1)[0][0]
    start_idx = next(
        (i for i, s in enumerate(all_wire_clusters) if len(s) == n_wires), -1
    )
    if start_idx == -1:
        return None

    traces = [[cluster] for cluster in all_wire_clusters[start_idx]]

    for i in range(start_idx + 1, len(all_wire_clusters)):
        avail = list(all_wire_clusters[i])
        if not avail:
            continue

        for trace in traces:
            last_c = trace[-1]["c"]
            if not avail:
                break
            dists = [distance.euclidean(last_c, c["c"]) for c in avail]
            best_idx = np.argmin(dists)
            if dists[best_idx] < 5.0:
                trace.append(avail.pop(best_idx))

    return [np.vstack([seg["p"] for seg in wt]) for wt in traces if wt]


def strategy_for_medium_extrahard(df: pd.DataFrame, eps=0.75, min_samples=15):
    """Global DBSCAN strategy for 'MEDIUM' and 'EXTRAHARD' datasets."""
    print(
        f"--- Applying Strategy 2: Global DBSCAN (eps={eps}, min_samples={min_samples}) ---"
    )
    points = df[["x", "y", "z"]].values
    scaled_points = StandardScaler().fit_transform(points)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled_points)
    return [points[labels == l] for l in set(labels) if l != -1]


def strategy_hough_transform(df: pd.DataFrame):
    """Hough Transform strategy for sparse 'HARD' dataset."""
    print("--- Applying Strategy 3: Hough Transform ---")
    if not CV2_AVAILABLE:
        print("❌ OpenCV (cv2) is not installed. Skipping this strategy.")
        return None

    points = df[["x", "y", "z"]].values

    # Project points to 2D image
    x, y = points[:, 0], points[:, 1]
    resolution = 0.1
    x_offset, y_offset = np.min(x), np.min(y)
    img_width = int((np.max(x) - x_offset) / resolution) + 1
    img_height = int((np.max(y) - y_offset) / resolution) + 1
    image = np.zeros((img_height, img_width), dtype=np.uint8)
    img_x = ((x - x_offset) / resolution).astype(int)
    img_y = ((y - y_offset) / resolution).astype(int)
    image[img_y, img_x] = 255

    # Apply Hough Line Transform
    lines = cv2.HoughLinesP(
        image, rho=1, theta=np.pi / 180, threshold=10, minLineLength=30, maxLineGap=25
    )

    if lines is None:
        print("❌ Hough Transform could not detect any lines.")
        return None

    # Group 3D points based on proximity to detected lines
    temp_clusters = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        p1 = np.array([x1 * resolution + x_offset, y1 * resolution + y_offset])
        p2 = np.array([x2 * resolution + x_offset, y2 * resolution + y_offset])

        if np.all(p1 == p2):
            continue

        d = np.abs(np.cross(p2 - p1, p1 - points[:, :2])) / np.linalg.norm(p2 - p1)
        wire_points_mask = d < (resolution * 5)

        if np.sum(wire_points_mask) > 0:
            temp_clusters.append(points[wire_points_mask])

    if not temp_clusters:
        return None

    # Merge overlapping clusters
    merged_clusters = []
    temp_clusters.sort(key=len, reverse=True)

    while temp_clusters:
        base_cluster = temp_clusters.pop(0)
        base_set = set(map(tuple, base_cluster))

        other_indices = []
        for i, other_cluster in enumerate(temp_clusters):
            other_set = set(map(tuple, other_cluster))
            if len(base_set.intersection(other_set)) > 0.5 * len(other_set):
                base_set.update(other_set)
                other_indices.append(i)

        for i in sorted(other_indices, reverse=True):
            temp_clusters.pop(i)

        merged_clusters.append(np.array(list(base_set)))

    # Return the 3 largest clusters
    merged_clusters.sort(key=len, reverse=True)
    return merged_clusters[:3]


def cluster_hard_data_fallback(df: pd.DataFrame):
    """Fallback strategy for hard dataset when OpenCV is not available."""
    print("--- Applying Fallback Strategy for Hard Dataset ---")
    points = df[["x", "y", "z"]].values

    # Initial loose clustering
    scaled_points = StandardScaler().fit_transform(points)
    labels = DBSCAN(eps=1.5, min_samples=3).fit_predict(scaled_points)

    # Get initial groups
    initial_groups = []
    for label in set(labels):
        if label != -1:
            group_points = points[labels == label]
            if len(group_points) > 10:
                initial_groups.append(group_points)

    if not initial_groups:
        return []

    # Split large groups
    final_clusters = []
    for group in initial_groups:
        if len(group) > 100:
            group_scaled = StandardScaler().fit_transform(group)
            sub_labels = DBSCAN(eps=0.6, min_samples=5).fit_predict(group_scaled)

            for sub_label in set(sub_labels):
                if sub_label != -1:
                    sub_group = group[sub_labels == sub_label]
                    if len(sub_group) > 15:
                        final_clusters.append(sub_group)
        else:
            final_clusters.append(group)

    # Keep only 3 largest clusters
    final_clusters.sort(key=len, reverse=True)
    return final_clusters[:3]


def save_plot(fig, difficulty: str, output_dir: str):
    """Save plot to output directory."""
    filename = f"lidar_wires_{difficulty}.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"  - Plot saved to: {filepath}")


def visualize_results(
    df: pd.DataFrame, clusters: list, difficulty: str, output_dir: str
):
    """Create and save visualization of clustering results."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Plot original data in background
    ax.scatter(
        df["x"], df["y"], df["z"], c="lightgray", s=1, alpha=0.1, label="Original Data"
    )

    # Plot clustered wires
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink"]
    for i, wire_cloud in enumerate(clusters):
        color = colors[i % len(colors)]
        ax.scatter(
            wire_cloud[:, 0],
            wire_cloud[:, 1],
            wire_cloud[:, 2],
            c=color,
            s=5,
            label=f"Wire {i+1} ({len(wire_cloud)} pts)",
        )

    ax.set_title(
        f"Wire Detection Results - '{difficulty.upper()}' Dataset", fontsize=14
    )
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.legend()

    # Save plot
    save_plot(fig, difficulty, output_dir)


def run_all_strategies():
    """
    Main function that processes all datasets with appropriate strategies,
    saves plots, and returns results.
    """
    # Setup output directory
    output_dir = ensure_output_directory()
    print(f"📁 Output directory: {output_dir}")

    # Load all datasets
    lidar_dataframes = load_all_lidar_datasets()

    if not lidar_dataframes:
        print("❌ No datasets loaded.")
        return None, None

    all_results = {}
    plot_dataframes = {}

    for difficulty, df in lidar_dataframes.items():
        print(f"\n{'='*80}")
        print(f"PROCESSING '{difficulty.upper()}' DATASET ({len(df)} points)")

        if df.empty:
            print(f"⏩ Skipping '{difficulty}' dataset because it is empty.")
            continue

        final_clusters = None
        rotation_matrix_to_reverse = None
        df_for_plot = df.copy()  # Original data for plotting

        # Apply strategy based on difficulty
        if difficulty == "easy":
            final_clusters = strategy_for_easy(df)

        else:
            # Align data for other difficulties
            aligned_df, rotation_matrix_to_reverse = align_point_cloud(df)

            if difficulty == "medium":
                final_clusters = strategy_for_medium_extrahard(
                    aligned_df, eps=0.75, min_samples=25
                )
            elif difficulty == "hard":
                if CV2_AVAILABLE:
                    final_clusters = strategy_hough_transform(aligned_df)
                else:
                    final_clusters = cluster_hard_data_fallback(aligned_df)
            elif difficulty == "extrahard":
                final_clusters = strategy_for_medium_extrahard(
                    aligned_df, eps=0.8, min_samples=15
                )

        # Reverse alignment if needed
        if final_clusters and rotation_matrix_to_reverse is not None:
            final_clusters = reverse_alignment(
                final_clusters, rotation_matrix_to_reverse
            )

        # Process results
        if final_clusters and all(len(c) > 0 for c in final_clusters):
            print(f"✅ Strategy successful! Found {len(final_clusters)} wires.")
            for i, cluster in enumerate(final_clusters):
                print(f"  - Wire {i+1}: {len(cluster)} points")

            all_results[difficulty] = final_clusters
            plot_dataframes[difficulty] = df_for_plot

            # Create and save visualization
            visualize_results(df_for_plot, final_clusters, difficulty, output_dir)

        else:
            print(f"❌ Strategy for '{difficulty}' failed to produce valid results.")

    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"📊 Successfully processed {len(all_results)} datasets")
    print(f"📁 Plots saved to: {output_dir}")

    return all_results, plot_dataframes


'''
def main():
    """Main entry point."""
    print("🚀 Starting LiDAR Wire Detection Processing...")
    print("="*80)
    
    try:
        results, plot_data = run_all_strategies()
        
        if results:
            print("\n📈 SUMMARY:")
            for difficulty, clusters in results.items():
                print(f"  - {difficulty.upper()}: {len(clusters)} wires detected")
        else:
            print("\n❌ No successful results.")
            
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
