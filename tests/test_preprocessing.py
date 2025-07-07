# tests/test_preprocessing.py

import unittest
import pandas as pd
import numpy as np
import os
import sys

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, src_path)

from preprocessing import align_point_cloud

class TestPreprocessing(unittest.TestCase):

    def test_alignment_no_rotation(self):
        """Test alignment on data that is already aligned with the X-axis."""
        df = pd.DataFrame({'x': [-1, 0, 1], 'y': [0, 0, 0], 'z': [1, 0, 1]})
        aligned_df, _ = align_point_cloud(df)
        
        # The x-coordinates should be almost identical
        np.testing.assert_allclose(df['x'].values, aligned_df['x'].values, atol=1e-6)
        # The new y-coordinates should be close to zero
        np.testing.assert_allclose(aligned_df['y'].values, [0, 0, 0], atol=1e-6)

    def test_alignment_90_degree_rotation(self):
        """Test alignment on data aligned with the Y-axis."""
        df = pd.DataFrame({'x': [0, 0, 0], 'y': [-1, 0, 1], 'z': [1, 0, 1]})
        aligned_df, _ = align_point_cloud(df)
        
        # The new x-coordinates should now match the old y-coordinates
        np.testing.assert_allclose(aligned_df['x'].values, df['y'].values, atol=1e-6)

if __name__ == '__main__':
    unittest.main()