import unittest
import numpy as np
import pandas as pd
import os
import sys

# Add the 'src' directory to the Python path to allow for module imports
try:
    current_dir = os.path.dirname(os.path.realpath(__file__))
except NameError:
    current_dir = os.getcwd()

src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

# Import the functions to be tested
from catenary_fitting import find_best_2d_plane, fit_catenary_to_2d_points, catenary_2d

class TestCatenaryFitting(unittest.TestCase):
    
    def test_find_best_2d_plane(self):
        """
        Tests if the plane fitting function correctly projects 3D points to 2D.
        """
        print("\n--- Testing find_best_2d_plane ---")
        # Create a simple 3D plane of points (e.g., a tilted square)
        x = np.array([-1, 1, -1, 1])
        y = np.array([-1, -1, 1, 1])
        z = 0.5 * x + 0.5 * y  # Define a simple plane
        points_3d = np.vstack([x, y, z]).T
        
        _, _, _, _, projected_2d = find_best_2d_plane(points_3d)
        
        self.assertIsNotNone(projected_2d, "Projection should not be None")
        self.assertEqual(projected_2d.shape, (4, 2), "Projected points should have 2 dimensions")
        print("✅ test_find_best_2d_plane passed.")

    def test_fit_catenary_on_perfect_data(self):
        """
        Tests if the fitting function can accurately recover known catenary
        parameters from a perfect, noise-free curve.
        """
        print("\n--- Testing fit_catenary_to_2d_points ---")
        # 1. Define true, known parameters for a catenary curve
        a_true, x0_true, y0_true = 100.0, 5.0, 10.0
        
        # 2. Generate perfect 2D data using the catenary equation
        x_data = np.linspace(-50, 60, 200)
        y_data = catenary_2d(x_data, a_true, x0_true, y0_true)
        points_2d = np.vstack([x_data, y_data]).T
        
        # 3. Call the fitting function
        fitted_params, r_squared = fit_catenary_to_2d_points(points_2d, method='robust')
        
        # 4. Check the results
        self.assertIsNotNone(fitted_params, "Fitting should return valid parameters")
        self.assertGreater(r_squared, 0.999, "R-squared for a perfect fit should be > 0.999")
        
        # Check if the recovered parameters are very close to the true ones
        np.testing.assert_allclose(fitted_params, [a_true, x0_true, y0_true], rtol=1e-3)
        print("✅ test_fit_catenary_on_perfect_data passed.")

if __name__ == '__main__':
    unittest.main()