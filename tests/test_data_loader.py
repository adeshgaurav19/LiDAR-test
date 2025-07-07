# tests/test_data_loader.py

import unittest
import pandas as pd
import os
import tempfile
import sys

# Add src to path to allow imports
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, src_path)

from data_loader import load_lidar_data

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.TemporaryDirectory()
        self.data_path = self.test_dir.name

    def tearDown(self):
        # Clean up the temporary directory
        self.test_dir.cleanup()

    def test_load_success(self):
        """Test successful loading of a valid parquet file."""
        file_path = os.path.join(self.data_path, "valid.parquet")
        df = pd.DataFrame({'x': [1], 'y': [2], 'z': [3]})
        df.to_parquet(file_path)
        
        loaded_df = load_lidar_data(file_path)
        self.assertIsInstance(loaded_df, pd.DataFrame)
        self.assertListEqual(list(loaded_df.columns), ['x', 'y', 'z'])

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_lidar_data("non_existent_file.parquet")

    def test_missing_columns(self):
        """Test that ValueError is raised if a required column is missing."""
        file_path = os.path.join(self.data_path, "invalid.parquet")
        df = pd.DataFrame({'x': [1], 'y': [2]}) # Missing 'z' column
        df.to_parquet(file_path)
        
        with self.assertRaises(ValueError):
            load_lidar_data(file_path)

if __name__ == '__main__':
    unittest.main()