"""LiDAR Powerline Analysis Dashboard

This module implements an interactive Dash web application for visualizing and analyzing
LiDAR powerline data. It provides three main views: Exploratory Data Analysis (EDA),
clustering results, and fitted catenary models. The app uses Dash Bootstrap Components
for a polished, responsive UI and Plotly for interactive 3D visualizations.

Key features:
- Loads LiDAR point cloud data from parquet files for different difficulty levels
  (easy, medium, hard, extrahard).
- Generates EDA plots (e.g., histograms, scatter plots) using Matplotlib, saved to a
  temporary directory and displayed as images.
- Applies tailored clustering strategies to identify wire clusters in the point cloud.
- Fits catenary models to clusters and visualizes them in 3D alongside a results table.
- Uses Matplotlib's 'Agg' backend to ensure thread-safe plot generation on macOS.

The app is structured with a dropdown for dataset selection and tabs for navigating
between EDA, clustering, and final model views. Error handling is robust, with alerts
displayed for any issues during data loading, clustering, or plot generation.

Note: Ensure the 'data' directory contains the required parquet files and the 'src'
directory includes the custom modules (data_loader, preprocessing, clustering,
catenary_fitting). For better performance, consider modifying `generate_eda_plots` to
return Matplotlib figures directly instead of saving to disk.

Author: Adesh
Date: 2025-07-07
"""

import os
import sys

import dash
import dash_bootstrap_components as dbc
import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output

# Use Agg backend to avoid macOS threading issues with Matplotlib's GUI.
matplotlib.use("Agg")
import base64
import glob
import tempfile
from io import BytesIO

import matplotlib.pyplot as plt

# --- Set up the project path to access our custom modules ---
try:
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.realpath(__file__))
except NameError:
    # Fallback for interactive environments like Jupyter
    current_dir = os.getcwd()

# Add 'src' directory to Python's path so we can import our modules
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from catenary_fitting import fit_catenary_to_wire_clusters, project_catenary_to_3d
from clustering import (
    run_all_strategies,
    strategy_for_easy,
    strategy_for_medium_extrahard,
    strategy_hough_transform,
    visualize_results,
)

# --- Import our custom functions ---
# These are from our project modules in the src/ directory
from data_loader import load_lidar_data
from preprocessing import align_point_cloud, generate_eda_plots, reverse_alignment

# ==============================================================================
# DASH APP SETUP
# ==============================================================================
# Initialize the Dash app with Bootstrap for a polished look
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # For deployment, if needed


# --- Helper function to convert saved PNG files to base64 ---
def image_file_to_base64(file_path):
    """
    Convert a PNG file to base64 for embedding in the Dash app.
    Returns None if the file can't be read to avoid crashes.
    """
    try:
        with open(file_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("ascii")
            return f"data:image/png;base64,{encoded}"
    except Exception as e:
        print(f"Warning: Couldn't read image {file_path}: {e}")
        return None


# --- Define the app layout ---
app.layout = dbc.Container(
    [
        # Header with a clean, centered title
        dbc.Row(
            dbc.Col(
                html.H1(
                    "LiDAR Powerline Analysis Dashboard",
                    className="text-center text-primary mt-4 mb-4",
                )
            )
        ),
        # Dataset selection dropdown in a nice card
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.H5("Select Dataset", className="card-title"),
                                dcc.Dropdown(
                                    id="dataset-dropdown",
                                    options=[
                                        {"label": "Easy Dataset", "value": "easy"},
                                        {"label": "Medium Dataset", "value": "medium"},
                                        {"label": "Hard Dataset", "value": "hard"},
                                        {
                                            "label": "Extra Hard Dataset",
                                            "value": "extrahard",
                                        },
                                    ],
                                    value="easy",  # Default to 'easy' dataset
                                    className="mb-3",
                                ),
                            ]
                        )
                    ],
                    className="shadow-sm mb-4",  # Subtle shadow for visual appeal
                )
            )
        ),
        # Tabs to switch between EDA, clustering, and final models
        dbc.Row(
            dbc.Col(
                dbc.Tabs(
                    [
                        dbc.Tab(label="Exploratory Data Analysis", tab_id="eda"),
                        dbc.Tab(label="Clustering Results", tab_id="clustering"),
                        dbc.Tab(label="Final Models", tab_id="final"),
                    ],
                    id="tabs",
                    active_tab="eda",  # Start with EDA tab
                    className="mb-4",
                )
            )
        ),
        # Container for tab content
        dbc.Row(dbc.Col(html.Div(id="tab-content", className="mt-4"))),
    ],
    fluid=True, 
)


# ==============================================================================
# CALLBACK LOGIC
# ==============================================================================
@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"), Input("dataset-dropdown", "value")],
)
def render_tab_content(active_tab, selected_difficulty):
    """
    Main callback to update the dashboard content based on the selected tab and dataset.
    Handles EDA, clustering, and final model visualization.
    """
    # Check if a dataset is selected
    if not selected_difficulty:
        return dbc.Alert(
            "Please select a dataset to analyze.",
            color="warning",
            className="text-center",
        )

    # --- Load the LiDAR data ---
    try:
        data_dir = os.path.join(current_dir, "data")
        file_path = os.path.join(
            data_dir, f"lidar_cable_points_{selected_difficulty}.parquet"
        )
        df = load_lidar_data(file_path)
    except Exception as e:
        return dbc.Alert(f"Error loading data: {str(e)}", color="danger")

    # --- EDA Tab: Show exploratory plots ---
    if active_tab == "eda":
        try:
            # Use a temporary directory since generate_eda_plots requires output_dir
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"Generating EDA plots for '{selected_difficulty}' dataset...")
                generate_eda_plots(df, selected_difficulty, temp_dir)

                # Grab all PNG files from the temp directory
                eda_image_files = glob.glob(os.path.join(temp_dir, "*.png"))
                if not eda_image_files:
                    return dbc.Alert(
                        f"No EDA plots generated for the '{selected_difficulty}' dataset.",
                        color="warning",
                    )

                # Convert images to base64 for display
                eda_images = [
                    img
                    for img in [
                        image_file_to_base64(img_path) for img_path in eda_image_files
                    ]
                    if img is not None
                ]

                if not eda_images:
                    return dbc.Alert(
                        "Failed to load EDA plots from temporary directory.",
                        color="warning",
                    )

            # Layout for EDA plots
            return html.Div(
                [
                    html.H4(
                        f"EDA for {selected_difficulty.upper()} Dataset",
                        className="mb-4",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Img(
                                    src=img,
                                    style={"width": "100%", "margin-bottom": "20px"},
                                ),
                                md=6,
                            )
                            for img in eda_images
                        ],
                        justify="center",
                    ),
                ]
            )
        except Exception as e:
            return dbc.Alert(f"Error generating EDA plots: {str(e)}", color="danger")

    # --- Clustering Tab: Show wire clusters ---
    elif active_tab == "clustering":
        try:
            # Initialize variables
            wire_clusters, rotation_matrix = None, None
            aligned_df = df

            # Apply the appropriate clustering strategy
            if selected_difficulty == "easy":
                wire_clusters = strategy_for_easy(df)
            else:
                aligned_df, rotation_matrix = align_point_cloud(df)
                if selected_difficulty == "medium":
                    wire_clusters = strategy_for_medium_extrahard(
                        aligned_df, eps=0.75, min_samples=25
                    )
                elif selected_difficulty == "hard":
                    wire_clusters = strategy_hough_transform(aligned_df)
                elif selected_difficulty == "extrahard":
                    wire_clusters = strategy_for_medium_extrahard(
                        aligned_df, eps=0.8, min_samples=15
                    )
                # Reverse alignment if needed
                if wire_clusters and rotation_matrix is not None:
                    wire_clusters = reverse_alignment(wire_clusters, rotation_matrix)

            if not wire_clusters:
                return dbc.Alert(
                    f"Clustering failed for the '{selected_difficulty}' dataset.",
                    color="warning",
                )

            # Create a 3D scatter plot for clusters
            fig_clustering = go.Figure()
            fig_clustering.add_trace(
                go.Scatter3d(
                    x=df["x"],
                    y=df["y"],
                    z=df["z"],
                    mode="markers",
                    marker=dict(size=1.5, color="lightgray", opacity=0.3),
                    name="Original Cloud",
                )
            )
            # Color palette for wires
            colors = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
            ]
            for i, wire_cloud in enumerate(wire_clusters):
                color = colors[i % len(colors)]
                fig_clustering.add_trace(
                    go.Scatter3d(
                        x=wire_cloud[:, 0],
                        y=wire_cloud[:, 1],
                        z=wire_cloud[:, 2],
                        mode="markers",
                        marker=dict(size=2, color=color),
                        name=f"Wire {i+1} Points",
                    )
                )
            fig_clustering.update_layout(
                title=f"Clustering Results for {selected_difficulty.upper()} Dataset",
                margin=dict(l=0, r=0, b=0, t=40),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                scene=dict(
                    xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"
                ),
            )

            return html.Div(
                [
                    html.H4(
                        f"Clustering Results for {selected_difficulty.upper()} Dataset",
                        className="mb-4",
                    ),
                    dcc.Graph(figure=fig_clustering, style={"height": "80vh"}),
                ]
            )
        except Exception as e:
            return dbc.Alert(
                f"Error generating clustering plots: {str(e)}", color="danger"
            )

    # --- Final Models Tab: Show fitted catenary models ---
    elif active_tab == "final":
        try:
            # Run clustering (same as above)
            wire_clusters, rotation_matrix = None, None
            df_for_plot = df
            if selected_difficulty == "easy":
                wire_clusters = strategy_for_easy(df)
            else:
                aligned_df, rotation_matrix = align_point_cloud(df)
                if selected_difficulty == "medium":
                    wire_clusters = strategy_for_medium_extrahard(
                        aligned_df, eps=0.75, min_samples=25
                    )
                elif selected_difficulty == "hard":
                    wire_clusters = strategy_hough_transform(aligned_df)
                elif selected_difficulty == "extrahard":
                    wire_clusters = strategy_for_medium_extrahard(
                        aligned_df, eps=0.8, min_samples=15
                    )
                if wire_clusters and rotation_matrix is not None:
                    wire_clusters = reverse_alignment(wire_clusters, rotation_matrix)

            if not wire_clusters:
                return dbc.Alert(
                    f"Clustering failed for the '{selected_difficulty}' dataset.",
                    color="warning",
                )

            # Fit catenary models to the clusters
            catenary_results = fit_catenary_to_wire_clusters(
                wire_clusters, selected_difficulty.upper()
            )
            if not catenary_results["wire_params"]:
                return dbc.Alert(
                    f"Catenary fitting failed for the '{selected_difficulty}' dataset.",
                    color="warning",
                )

            # Build results table
            results_list = []
            for i, (params, quality) in enumerate(
                zip(catenary_results["wire_params"], catenary_results["fit_qualities"])
            ):
                results_list.append(
                    {
                        "Wire ID": i + 1,
                        "Points": len(wire_clusters[i]),
                        'Catenary "a"': f"{params[0]:.2f}",
                        "Trough (x0, y0)": f"({params[1]:.2f}, {params[2]:.2f})",
                        "R-Squared": f"{quality:.3f}",
                    }
                )
            results_df = pd.DataFrame(results_list)

            # Create 3D visualization with original points, clusters, and catenary curves
            fig_3d = go.Figure()
            fig_3d.add_trace(
                go.Scatter3d(
                    x=df["x"],
                    y=df["y"],
                    z=df["z"],
                    mode="markers",
                    marker=dict(size=1.5, color="lightgray", opacity=0.3),
                    name="Original Cloud",
                )
            )
            colors = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
            ]
            for i, wire_cloud in enumerate(wire_clusters):
                color = colors[i % len(colors)]
                fig_3d.add_trace(
                    go.Scatter3d(
                        x=wire_cloud[:, 0],
                        y=wire_cloud[:, 1],
                        z=wire_cloud[:, 2],
                        mode="markers",
                        marker=dict(size=2, color=color),
                        name=f"Wire {i+1} Points",
                    )
                )
                if i < len(catenary_results["wire_curves_3d"]):
                    curve_3d = np.array(catenary_results["wire_curves_3d"][i])
                    fig_3d.add_trace(
                        go.Scatter3d(
                            x=curve_3d[:, 0],
                            y=curve_3d[:, 1],
                            z=curve_3d[:, 2],
                            mode="lines",
                            line=dict(width=7, color=color),
                            name=f"Wire {i+1} Model",
                        )
                    )
            fig_3d.update_layout(
                title=f"Final 3D Models for {selected_difficulty.upper()} Dataset",
                margin=dict(l=0, r=0, b=0, t=40),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                scene=dict(
                    xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"
                ),
            )

            # Layout with plot on left and table on right
            return html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Graph(figure=fig_3d, style={"height": "80vh"}), md=8
                            ),
                            dbc.Col(
                                [
                                    html.H4("Fitted Model Results", className="mt-5"),
                                    dash_table.DataTable(
                                        data=results_df.to_dict("records"),
                                        columns=[
                                            {"name": c, "id": c}
                                            for c in results_df.columns
                                        ],
                                        style_table={"overflowX": "auto"},
                                        style_cell={
                                            "textAlign": "left",
                                            "padding": "5px",
                                        },
                                        style_header={
                                            "backgroundColor": "rgb(230, 230, 230)",
                                            "fontWeight": "bold",
                                        },
                                    ),
                                ],
                                md=4,
                            ),
                        ]
                    )
                ]
            )
        except Exception as e:
            return dbc.Alert(f"Error generating final models: {str(e)}", color="danger")


# ==============================================================================
# RUN THE APP
# ==============================================================================
if __name__ == "__main__":
    # Run in debug mode for development; turn off for production
    app.run(debug=True)
