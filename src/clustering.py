"""
LiDAR point cloud processing for powerline wire detection.
Uses different clustering methods for different dataset difficulties.
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial import distance
from collections import Counter

def load_lidar_data(file_path: str) -> pd.DataFrame:
    """Load LiDAR data from parquet file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_parquet(file_path)

def align_data(df: pd.DataFrame) -> pd.DataFrame:
    """Align point cloud using PCA rotation."""
    print("  - Aligning point cloud...")
    points_2d = df[['x', 'y']].values
    pca = PCA(n_components=2).fit(points_2d)
    angle = np.arctan2(pca.components_[0][1], pca.components_[0][0])
    rotation_matrix = np.array([[np.cos(-angle), -np.sin(-angle)], 
                               [np.sin(-angle), np.cos(-angle)]])
    rotated_points = points_2d.dot(rotation_matrix.T)
    aligned_df = pd.DataFrame(rotated_points, columns=['x', 'y'])
    aligned_df['z'] = df['z'].values
    return aligned_df

def slice_and_trace(df: pd.DataFrame, eps: float, gap_limit: float) -> list:
    """Slice data into segments and trace wires through them."""
    print(f"--- Using slice and trace method (eps={eps}, gap_limit={gap_limit}) ---")
    points = df[['x', 'y', 'z']].values
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    
    # Create slices
    slices = []
    slice_positions = np.arange(x_min, x_max, 0.5)
    
    for i in range(len(slice_positions) - 1):
        mask = (points[:, 0] >= slice_positions[i]) & (points[:, 0] < slice_positions[i+1])
        slice_points = points[mask]
        
        if len(slice_points) < 5:
            slices.append([])
            continue
            
        # Cluster points in this slice
        labels = DBSCAN(eps=eps, min_samples=3).fit_predict(slice_points[:, 1:])
        
        clusters = []
        for label in set(labels):
            if label != -1:
                cluster_points = slice_points[labels == label]
                center = np.mean(cluster_points, axis=0)
                clusters.append({'center': center, 'points': cluster_points})
        
        # Sort by y-coordinate
        clusters.sort(key=lambda x: x['center'][1])
        slices.append(clusters)
    
    # Find most common number of wires
    wire_counts = [len(s) for s in slices if s]
    if not wire_counts:
        return None
    
    target_count = Counter(wire_counts).most_common(1)[0][0]
    
    # Find starting slice
    start_idx = -1
    for i, s in enumerate(slices):
        if len(s) == target_count:
            start_idx = i
            break
    
    if start_idx == -1:
        return None
    
    # Initialize traces
    traces = [[cluster] for cluster in slices[start_idx]]
    
    # Trace forward
    for i in range(start_idx + 1, len(slices)):
        available = list(slices[i])
        if not available:
            continue
            
        for trace in traces:
            if not available:
                break
            last_center = trace[-1]['center']
            distances = [distance.euclidean(last_center, c['center']) for c in available]
            best_idx = np.argmin(distances)
            
            if distances[best_idx] < gap_limit:
                trace.append(available.pop(best_idx))
    
    # Combine points from each trace
    result = []
    for trace in traces:
        if trace:
            wire_points = np.vstack([segment['points'] for segment in trace])
            result.append(wire_points)
    
    return result

def global_cluster(df: pd.DataFrame, eps: float, min_samples: int) -> list:
    """Apply DBSCAN clustering to entire dataset."""
    print(f"--- Using global clustering (eps={eps}, min_samples={min_samples}) ---")
    points = df[['x', 'y', 'z']].values
    scaled_points = StandardScaler().fit_transform(points)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled_points)
    
    clusters = []
    for label in set(labels):
        if label != -1:
            clusters.append(points[labels == label])
    
    return clusters

def cluster_hard_data(df: pd.DataFrame) -> list:
    """Special clustering for hard dataset to get 3 separate wire clusters."""
    print("--- Using hard dataset clustering ---")
    points = df[['x', 'y', 'z']].values
    
    # Step 1: Initial loose clustering
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
    
    # Step 2: Split large groups
    final_clusters = []
    for group in initial_groups:
        if len(group) > 100:  # Large group, split it
            group_scaled = StandardScaler().fit_transform(group)
            sub_labels = DBSCAN(eps=0.6, min_samples=5).fit_predict(group_scaled)
            
            for sub_label in set(sub_labels):
                if sub_label != -1:
                    sub_group = group[sub_labels == sub_label]
                    if len(sub_group) > 15:
                        final_clusters.append(sub_group)
        else:
            final_clusters.append(group)
    
    # Step 3: Keep only 3 largest clusters
    final_clusters.sort(key=len, reverse=True)
    return final_clusters[:3]

def run_clustering():
    """Main function to process all datasets."""
    # Setup paths
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    data_dir = os.path.join(project_root, 'data')

    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at '{data_dir}'")
        sys.exit(1)
    
    # Load datasets
    datasets = {}
    for difficulty in ['easy', 'medium', 'hard', 'extrahard']:
        try:
            file_path = os.path.join(data_dir, f"lidar_cable_points_{difficulty}.parquet")
            datasets[difficulty] = load_lidar_data(file_path)
        except Exception as e:
            print(f"Could not load '{difficulty}' dataset: {e}")
            datasets[difficulty] = pd.DataFrame()

    # Process each dataset
    for difficulty, df in datasets.items():
        print(f"\n{'='*60}")
        print(f"PROCESSING '{difficulty.upper()}' DATASET")
        
        if df.empty:
            print(f"Skipping '{difficulty}' - no data")
            continue

        clusters = None
        plot_data = df
        
        if difficulty == 'easy':
            # Easy dataset - use original data with slice and trace
            clusters = slice_and_trace(df, eps=0.4, gap_limit=5.0)
        
        else:
            # Other datasets - align first
            aligned_df = align_data(df)
            plot_data = aligned_df
            
            if difficulty == 'medium':
                clusters = global_cluster(aligned_df, eps=0.75, min_samples=25)
            elif difficulty == 'hard':
                clusters = cluster_hard_data(aligned_df)
            elif difficulty == 'extrahard':
                clusters = global_cluster(aligned_df, eps=0.8, min_samples=15)
        
        # Show results
        if clusters and all(len(c) > 0 for c in clusters):
            print(f"Success! Found {len(clusters)} wires")
            for i, cluster in enumerate(clusters):
                print(f"  Wire {i+1}: {len(cluster)} points")
            
            # Plot results
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot original data
            ax.scatter(plot_data['x'], plot_data['y'], plot_data['z'], 
                      c='gray', s=1, alpha=0.3)
            
            # Plot clusters
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            for i, cluster in enumerate(clusters):
                color = colors[i % len(colors)]
                ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], 
                          c=color, s=5, label=f'Wire {i+1}')
            
            ax.set_title(f"Results for '{difficulty.upper()}' Dataset")
            ax.legend()
            plt.show()
        else:
            print(f"Failed to cluster '{difficulty}' dataset")

if __name__ == "__main__":
    run_clustering()