# LiDAR Powerline Wire Detection and Catenary Modeling

A Python package for identifying individual powerline wires from drone-based LiDAR point cloud data and generating 3D catenary models. This project was developed as a technical assessment for a data science internship.

HF Link - https://huggingface.co/spaces/adzee19/LiDAR

## 🎯 Project Overview

This project provides an end-to-end pipeline that processes raw LiDAR point cloud datasets to automatically detect and cluster individual powerline wires. For each identified wire, the pipeline then fits a robust 3D catenary model. The final solution is a sophisticated, **multi-strategy system** that adapts its approach based on the specific challenges of each dataset, including noise, large gaps, varying point densities, and close wire proximity.

The entire workflow is orchestrated by a main `pipeline.py` script and can be explored through a fully interactive `Dash` web application.

## 🚀 Features

-   **Adaptive Clustering**: Automatically selects the best clustering strategy for different data characteristics.
-   **Robust Preprocessing**: Includes a PCA-based alignment step to standardize the orientation of powerlines.
-   **Numerically Stable Catenary Fitting**: Implements a robust fitting algorithm for the true catenary equation to ensure accurate model parameters.
-   **Comprehensive Outputs**: Generates visual plots and structured JSON files for all results.
-   **Automated Unit Testing**: Core logic is validated by a suite of tests using Python's `unittest` framework.
-   **Interactive Dashboard**: Includes a `Dash` web application for interactive exploration of the final results.
  
## My Approach: An Iterative Journey

This project was a deep dive into the practical challenges of working with real-world LiDAR data. My final solution was the result of an iterative process of experimentation, failure analysis, and research.

### 1. Initial Exploration & Simple Clustering

I began by researching the fundamentals of catenary equations and the common structures of 3D point cloud data. My initial strategy was to apply standard, out-of-the-box clustering algorithms to the entire point cloud.

-   **Algorithms Tested**: I started with `KMeans` but quickly found it unsuitable as it assumes spherical clusters. I then moved to density-based methods, starting with `DBSCAN` and then the more advanced `HDBSCAN`.
-   **Issues Faced**: This global approach suffered from several critical failures:
    * **Fragmentation**: Clusters would often break partway along a single wire.
    * **Noise Classification**: A significant number of wire points, especially in the `Hard` dataset, were incorrectly classified as noise.
    * **Incorrect Grouping**: The algorithms would sometimes group one side of all the wires into a single cluster and the other side into a second cluster, or merge all wires into one large cluster entirely.

### 2. A Shift in Strategy: Learning from Research

Realizing that a simple clustering approach was insufficient, I began looking for more specialized techniques. I reviewed approximately 10 academic research papers on the topic of powerline extraction from LiDAR data.

The key breakthrough came from the paper **"A Hierarchical Clustering Method to Repair Gaps in Point Clouds of Powerline Corridor for Powerline Extraction"** (https://www.mdpi.com/2072-4292/13/8/1502). This inspired the "Segment-and-Trace" method (`strategy_for_easy`), which involved slicing the data and connecting wire segments. This new approach worked perfectly for the `Easy` dataset but still struggled with the more complex files.

### 3. The Final Pipeline: A Tailored, Multi-Strategy Solution

The final, successful pipeline acknowledges that there is no single best algorithm. Instead, it uses a master function to apply a unique, tailored strategy to each dataset based on its specific challenges.

-   **Pre-Alignment**: A crucial insight was that most strategies performed better if the data was first standardized. The `align_point_cloud` function was created to do this using PCA, ensuring the wires run parallel to the X-axis. For datasets that require it, this step is reversed after clustering using the `reverse_alignment` function to return the wires to their original orientation.

-   **Dataset-Specific Clustering**: The `pipeline.py` script directs each dataset to a specific, fine-tuned clustering strategy from `clustering.py`:
    * **Easy**: The `strategy_for_easy` function uses the Segment-and-Trace method on the original, unaligned data.
    * **Medium & Extra-Hard**: These datasets are pre-aligned and then processed with `strategy_for_medium_extrahard`, which uses a global DBSCAN with tailored `eps` and `min_samples` parameters to handle noise and proximity.
    * **Hard**: This sparse, gappy dataset is pre-aligned and then processed with `strategy_hough_transform`, which uses an image-based line-finding algorithm to identify wires despite the missing data.

### 4. Catenary Model Fitting: From 3D to 2D and Back

The final modeling stage, handled by `catenary_fitting.py`, was also an iterative process.

-   **The Challenge - Finding the Right Plane**: The first step is to project the 3D wire points onto a 2D plane. Initial attempts using a direct PCA (`find_best_2d_plane`) sometimes failed, producing a "flattened" top-down view instead of the desired catenary U-shape. This happened when the side-to-side noise in the data was greater than the vertical sag.

-   **The Solution - Robust Projection**: The final, successful approach was to use PCA to find only the primary direction of the wire (the `u-axis`). Then, it uses vector mathematics to construct a guaranteed "downward-facing" sag axis (the `v-axis`). This ensures a correct 2D projection every time.

-   **The Catenary Model**: The `fit_catenary_to_2d_points` function fits the true catenary equation (`y = a * cosh((x - x0) / a) + y0`) to the projected 2D points. To ensure a stable and accurate fit, it uses a **robust optimization method** (`scipy.optimize.minimize` with a Huber loss function), which is less sensitive to outliers than a standard curve fit. It also uses intelligent initial guesses for the `a`, `x0`, and `y0` parameters to guide the optimizer.

-   **3D Reconstruction**: The final fitted 2D curve is then projected back into 3D space using the `project_catenary_to_3d` function, creating the final, smooth 3D model of the wire.

### 5. Final Deliverables

With the core processing pipeline finalized, the project was completed by building the surrounding infrastructure:

* **Automated Testing**: A suite of unit tests was created in the `/tests` directory using Python's `unittest` framework. These tests verify the functionality of key components like the data loader, the alignment logic, and the catenary fitting algorithm on synthetic data.
* **Interactive Application**: A `Dash` web application (`app.py`) was built to provide a user-friendly interface for exploring the results. The app allows a user to select any dataset and view the EDA plots, the intermediate clustering results, and the final 3D fitted models in an organized, tabbed layout.

## 📁 Project Structure

```
LiDAR/
├── .github/workflows/
│   └── python-tests.yml
├── data/
│   └── lidar_cable_points_easy.parquet, etc.
├── output/
│   ├── plots/
│   └── results/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── catenary_fitting.py
│   └── pipeline.py
├── tests/
│   ├── test_data_loader.py
│   └── test_catenary_fitting.py
│   └── test_preprocessing.py
├── app.py
├── requirements.txt
└── README.md
```

## 🧠 Algorithm Overview

The core of this project is a **multi-strategy pipeline** that evolved through an iterative process of diagnosing and solving the unique challenges posed by each dataset.

### 1. Preprocessing: PCA Alignment

The key insight was that clustering performance dramatically improves if the data is first standardized. The `align_point_cloud` function uses Principal Component Analysis (PCA) to find the dominant direction of the wires and rotates the entire point cloud so they run parallel to the X-axis. For datasets that require it, this step is reversed after clustering to return the wires to their original orientation.

### 2. Clustering: A Tailored Approach

No single algorithm could handle all datasets. The pipeline therefore uses a different, fine-tuned strategy for each case.

| Dataset | Challenge(s) | Applied Strategy (`clustering.py`) | Rationale |
| :--- | :--- | :--- | :--- |
| **Easy** | Clean, Continuous | `strategy_for_easy` | A **Segment-and-Trace** method works best on the *original, unaligned data*. |
| **Medium** | Multiple, Close Wires | `strategy_global_dbscan` | After **PCA Alignment**, a standard **Global DBSCAN** can cleanly separate the parallel wire groups. |
| **Hard** | Extreme Gaps, Sparse | `strategy_hough_transform` | After **PCA Alignment**, an **Image-based Hough Transform** is used to find the lines despite large gaps in the data. |
| **Extra-Hard**| Noise and Gaps | `strategy_global_dbscan` | After **PCA Alignment**, a more lenient **Global DBSCAN** captures the noisy, less-dense points without classifying them as noise. |

### 3. Catenary Model Fitting

For each successfully identified wire cluster, a true 3D catenary model is generated using the functions in `catenary_fitting.py`.

* **Plane Fitting**: First, the `find_best_2d_plane` function uses PCA to identify the 2D plane that the 3D wire points lie on.
* **Projection**: The 3D points are then projected onto this plane, resulting in a clean 2D point set that follows a catenary curve.
* **Robust Curve Fitting**: The `fit_catenary_to_2d_points` function fits the true catenary equation (`y = a * cosh((x - x0) / a) + y0`) to the 2D points. It uses a robust optimization method (`scipy.optimize.minimize`) that is less sensitive to outliers than a standard fit.
* **Model Validation**: The quality of each fit is measured by its **R-squared (R²)** value, which indicates how well the model describes the data.
* **3D Reconstruction**: Finally, the fitted 2D curve is projected back into 3D space using the `project_catenary_to_3d` function, creating the final, smooth 3D model of the wire.

## 🧪 Testing

The project includes a suite of automated unit tests in the `tests/` directory to ensure the reliability and correctness of the core logic.

* `test_data_loader.py`: Verifies that data loading handles valid files, missing files, and corrupted data correctly.
* `test_preprocessing.py`: Checks that the PCA alignment correctly rotates point clouds.
* `test_catenary_fitting.py`: This contains two critical tests:
    1.  It confirms that the **3D-to-2D projection** correctly preserves the shape of the wire.
    2.  It creates a **perfect, synthetic catenary curve** with known parameters, then asserts that the `fit_catenary_to_2d_points` function can accurately recover those exact original parameters with an R² score > 0.999.

To run all tests, navigate to the project root and run:
```bash
python -m unittest discover tests/
```
- The codebase is automatically formatted using `black` and `isort` and linted with `flake8`, `pydocstyle`, and `pylint` to ensure high code quality and consistency.

## 🖥️ Interactive Dashboard

The project includes a web application for interactive exploration of the results.

* **To run the app**:
    ```bash
    python app.py
    ```
* **Features**:
    * A **dropdown menu** to select any of the four datasets.
    * A **tabbed interface** to view the results from each stage of the pipeline.
    * **Tab 1: EDA**: An interactive 3D plot of the raw, original point cloud.
    * **Tab 2: Clustering**: A 3D plot showing the result of the clustering strategy, with each identified wire in a different color.
    * **Tab 3: Model Fitting**: The final view, showing the smooth, fitted 3D catenary models overlaid on the clustered points, along with a table of the final model parameters (curvature `a`, trough position `x0`, `y0`, and R² value).