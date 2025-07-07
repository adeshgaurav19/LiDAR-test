# LiDAR Powerline Wire Detection and Catenary Modeling

A Python package for identifying individual powerline wires from drone-based LiDAR point cloud data and generating 3D catenary models.

## 🎯 Project Overview

This project processes LiDAR point cloud datasets to automatically detect and cluster individual powerline wires, then fits catenary models to each wire. The solution handles various real-world challenges including noise, gaps, varying point densities, and closely spaced wires.

## 🚀 Features

- Automatic wire detection and clustering from 3D LiDAR point clouds
- Adaptive algorithm selection based on dataset characteristics
- 3D catenary model fitting for each detected wire
- Robust handling of noisy, sparse, and dense point cloud data
- Visualization tools for results inspection

## 📁 Project Structure

```
lidar-wire-detection/
├── src/
│   ├── __init__.py
│   ├── clustering/
│   │   ├── __init__.py
│   │   ├── dbscan_clustering.py
│   │   ├── slice_trace.py
│   │   └── vertical_separation.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── alignment.py
│   │   └── data_loader.py
│   ├── modeling/
│   │   ├── __init__.py
│   │   └── catenary_fitting.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── visualization.py
│   └── main.py
├── data/
│   ├── raw/
│   └── processed/
├── tests/
│   ├── __init__.py
│   ├── test_clustering.py
│   ├── test_preprocessing.py
│   └── test_modeling.py
├── notebooks/
│   ├── exploration.ipynb
│   └── results_analysis.ipynb
├── results/
│   ├── plots/
│   └── models/
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/lidar-wire-detection.git
cd lidar-wire-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## 📊 Usage

### Basic Usage

```python
from src.main import WireDetector

# Initialize detector
detector = WireDetector()

# Process a dataset
wires, catenary_models = detector.process_dataset('path/to/your/data.parquet')

# Visualize results
detector.visualize_results(wires, catenary_models)
```

### Command Line Interface

```bash
# Process single dataset
python -m src.main --input data/raw/easy_dataset.parquet --output results/

# Process all datasets
python -m src.main --input data/raw/ --output results/ --batch
```

## 🧠 Algorithm Overview

### The Challenge
Each LiDAR dataset presents unique challenges:
- **Clean data**: Continuous wires with consistent point density
- **Noisy data**: Scattered points with gaps and inconsistent coverage  
- **Dense clustering**: Multiple wires in close proximity
- **Sparse data**: Large gaps between wire sections

### Our Solution: Adaptive Method Selection

We developed four distinct approaches and automatically select the best method for each dataset:

#### 1. Simple Global Clustering
- **Method**: Direct DBSCAN on entire point cloud
- **Best for**: Clean, well-separated datasets
- **Pros**: Fast and simple
- **Cons**: Fails with varying densities

#### 2. Slice-and-Trace Method
- **Method**: Partition into slices → find cross-sections → connect across slices
- **Best for**: Continuous, well-separated wires
- **Pros**: Handles local density variations
- **Cons**: Wire-jumping with close wires, fails on large gaps

#### 3. PCA-Aligned Clustering  
- **Method**: Align wires using PCA → apply global clustering
- **Best for**: Most datasets with misaligned wires
- **Pros**: Dramatically improves wire separation
- **Cons**: Can degrade clean, pre-aligned data

#### 4. Vertical Separation Method
- **Method**: Sort by elevation → find natural gaps → split into groups
- **Best for**: Sparse data with clear vertical separation
- **Pros**: Robust to noise and gaps
- **Cons**: Requires vertically separated wires

### Dataset-Specific Solutions

| Dataset | Method | Rationale |
|---------|--------|-----------|
| Easy | Slice-and-trace (unaligned) | Clean, continuous data works perfectly |
| Medium | PCA alignment + DBSCAN | Close wires separate after alignment |
| Hard | PCA alignment + Y-separation | Sparse data needs vertical separation |
| Extra-Hard | PCA alignment + lenient DBSCAN | Noisy data needs loose parameters |

## 📈 Results

<!-- Results section - to be filled -->

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_clustering.py

# Run with coverage
pytest --cov=src tests/
```

## 📋 Requirements

```python
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Additional dependencies
# (Full list in requirements.txt)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Documentation

<!-- Documentation links - to be filled -->

## 📄 License

<!-- License information - to be filled -->

## 🔗 References

- [A Hierarchical Clustering Method to Repair Gaps in Point Clouds of Powerline Corridor](https://www.mdpi.com/2072-4292/13/8/1502)
- Additional academic references in `/docs/references.md`

## 📞 Contact

<!-- Contact information - to be filled -->

## 🙏 Acknowledgments

<!-- Acknowledgments - to be filled -->

---

**Note**: This project was developed as part of a data science internship technical assessment.