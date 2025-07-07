# src/pipeline.py

"""
Main end-to-end pipeline for the LiDAR Wire Detection and Modeling project.

This script orchestrates the entire workflow:
1.  Loads all raw LiDAR datasets.
2.  (Optional) Generates and saves Exploratory Data Analysis (EDA) plots.
3.  Applies tailored preprocessing (alignment) and clustering strategies for each dataset.
4.  Fits true catenary models to the identified wire clusters.
5.  Visualizes the final models and saves all plots and results to disk.
"""
import os
import sys
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt

# --- Add the 'src' directory to the Python path ---
try:
    current_dir = os.path.dirname(os.path.realpath(__file__))
except NameError:
    current_dir = os.getcwd()
src_dir = os.path.abspath(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

# --- Import all necessary functions from the project modules ---
from data_loader import load_all_lidar_datasets
from preprocessing import align_point_cloud, generate_eda_plots,reverse_alignment
from clustering import strategy_for_easy, strategy_for_medium_extrahard, strategy_hough_transform
from catenary_fitting import fit_catenary_to_wire_clusters, visualize_catenary_results, save_results_to_json


def run_end_to_end_pipeline(run_eda=True):
    """
    Orchestrates the entire workflow from data loading to final model reporting.
    """
    print("🚀 Starting End-to-End LiDAR Wire Analysis Pipeline...")
    
    # --- 1. Setup Paths and Load Data ---
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    data_dir = os.path.join(project_root, 'data')
    output_plot_dir = os.path.join(project_root, 'output', 'plots')
    results_output_dir = os.path.join(project_root, 'output', 'results')
    
    lidar_dataframes = load_all_lidar_datasets()
    if not lidar_dataframes:
        print("❌ No datasets loaded. Halting pipeline.")
        return

    # --- 2. Exploratory Data Analysis (EDA) ---
    if run_eda:
        print("\n" + "="*80 + "\nSTEP 1: GENERATING EDA PLOTS\n" + "="*80)
        for difficulty, df in lidar_dataframes.items():
            if not df.empty:
                generate_eda_plots(df, difficulty, os.path.join(output_plot_dir, 'eda'))

    # --- 3. Clustering & Catenary Fitting ---
    print("\n" + "="*80 + "\nSTEP 2: CLUSTERING & MODEL FITTING\n" + "="*80)
    all_results = {}
    for difficulty, df in lidar_dataframes.items():
        print(f"\n--- PROCESSING '{difficulty.upper()}' DATASET ---")
        if df.empty:
            print("⏩ Skipping empty dataset."); continue
        
        # Apply the correct, tailored clustering strategy
        wire_clusters, rotation_matrix = None, None
        df_for_viz = df
        
        if difficulty == 'easy':
            wire_clusters = strategy_for_easy(df)
        else:
            aligned_df, rotation_matrix = align_point_cloud(df)
            if difficulty == 'medium': wire_clusters = strategy_for_medium_extrahard(aligned_df, eps=0.75, min_samples=25)
            elif difficulty == 'hard': wire_clusters = strategy_hough_transform(aligned_df)
            elif difficulty == 'extrahard': wire_clusters = strategy_for_medium_extrahard(aligned_df, eps=0.8, min_samples=15)
            if wire_clusters and rotation_matrix is not None:
                wire_clusters = reverse_alignment(wire_clusters, rotation_matrix)
        
        if not wire_clusters or not all(len(c) > 0 for c in wire_clusters):
            print(f"❌ Clustering failed for '{difficulty}'"); continue
        print(f"✅ Found {len(wire_clusters)} wire clusters.")
        
        # Fit catenary models to the results
        catenary_results = fit_catenary_to_wire_clusters(wire_clusters, difficulty.upper())
        if not catenary_results['wire_params']:
            print(f"❌ Catenary fitting failed for '{difficulty}'"); continue
            
        # Visualize results and save the plot
        plot_path = os.path.join(output_plot_dir, f"{difficulty}_final_model.png")
        visualize_catenary_results(df_for_viz, wire_clusters, catenary_results, f"Final Model for {difficulty.upper()}", plot_path)
        
        # Store results for final report
        all_results[difficulty] = {
            'wire_clusters': wire_clusters,
            'catenary_results': catenary_results
        }

    # --- 4. Save Final Report ---
    if all_results:
        final_results_path = os.path.join(results_output_dir, "all_pipeline_results.json")
        save_results_to_json(all_results, final_results_path)
        print("\n\n✅✅✅ PIPELINE COMPLETE ✅✅✅")
    else:
        print("\n\n❌❌❌ PIPELINE FAILED: No results were generated. ❌❌❌")

if __name__ == "__main__":
    # To run the entire project, simply execute this script.
    run_end_to_end_pipeline(run_eda=True)