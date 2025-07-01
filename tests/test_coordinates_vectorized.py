"""
Comprehensive tests for vectorized coordinate transformation functions.

This test suite ensures that vectorized functions produce identical results
to the original scalar implementations for all coordinate systems and edge cases.
"""

import numpy as np
import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geopack
from geopack.coordinates_vectorized import (
    gsmgse_vectorized, geigeo_vectorized, magsm_vectorized,
    smgsm_vectorized, geomag_vectorized, geogsm_vectorized,
    gswgsm_vectorized
)
from geopack.coordinates_vectorized_complex import (
    sphcar_vectorized, bspcar_vectorized, bcarsp_vectorized
)


class TestCoordinatesVectorized(unittest.TestCase):
    
    def setUp(self):
        """Initialize geopack for consistent global variables."""
        # Set a specific time for reproducible results
        ut = 1577836800  # 2020-01-01 00:00:00 UTC
        geopack.recalc(ut)
    
    def test_scalar_compatibility(self):
        """Test that scalar inputs return scalar outputs."""
        # Test single scalar point
        x, y, z = 5.0, 3.0, 2.0
        
        # Test gsmgse
        result = gsmgse_vectorized(x, y, z, 1)
        self.assertIsInstance(result[0], (int, float))
        self.assertIsInstance(result[1], (int, float))
        self.assertIsInstance(result[2], (int, float))
        
        # Test sphcar
        r, theta, phi = 5.0, 0.5, 1.0
        result = sphcar_vectorized(r, theta, phi, 1)
        self.assertIsInstance(result[0], (int, float))
        self.assertIsInstance(result[1], (int, float))
        self.assertIsInstance(result[2], (int, float))
    
    def test_array_compatibility(self):
        """Test that array inputs return array outputs."""
        # Test array of points
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        z = np.array([7.0, 8.0, 9.0])
        
        # Test gsmgse
        result = gsmgse_vectorized(x, y, z, 1)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)
        self.assertIsInstance(result[2], np.ndarray)
        self.assertEqual(result[0].shape, x.shape)
    
    def test_gsmgse_accuracy(self):
        """Test GSM-GSE transformation accuracy against scalar version."""
        # Test points including edge cases
        test_points = [
            (0.0, 0.0, 0.0),      # Origin
            (10.0, 0.0, 0.0),     # X-axis
            (0.0, 10.0, 0.0),     # Y-axis
            (0.0, 0.0, 10.0),     # Z-axis
            (5.0, 3.0, 2.0),      # General point
            (-5.0, -3.0, -2.0),   # Negative values
            (1e-10, 1e-10, 1e-10), # Very small values
            (1e10, 1e10, 1e10),   # Very large values
        ]
        
        for x, y, z in test_points:
            # Forward transformation (GSM to GSE)
            scalar_result = geopack.gsmgse(x, y, z, 1)
            vector_result = gsmgse_vectorized(x, y, z, 1)
            
            np.testing.assert_allclose(
                scalar_result, vector_result,
                rtol=1e-15, atol=1e-15,
                err_msg=f"Forward transformation failed at point ({x}, {y}, {z})"
            )
            
            # Reverse transformation (GSE to GSM)
            scalar_result = geopack.gsmgse(x, y, z, -1)
            vector_result = gsmgse_vectorized(x, y, z, -1)
            
            np.testing.assert_allclose(
                scalar_result, vector_result,
                rtol=1e-15, atol=1e-15,
                err_msg=f"Reverse transformation failed at point ({x}, {y}, {z})"
            )
    
    def test_all_simple_transformations(self):
        """Test all simple coordinate transformations for accuracy."""
        # Test multiple points at once
        n_points = 100
        np.random.seed(42)
        x = np.random.uniform(-10, 10, n_points)
        y = np.random.uniform(-10, 10, n_points)
        z = np.random.uniform(-10, 10, n_points)
        
        transformations = [
            (geopack.gsmgse, gsmgse_vectorized, "GSM-GSE"),
            (geopack.geigeo, geigeo_vectorized, "GEI-GEO"),
            (geopack.magsm, magsm_vectorized, "MAG-SM"),
            (geopack.smgsm, smgsm_vectorized, "SM-GSM"),
            (geopack.geomag, geomag_vectorized, "GEO-MAG"),
            (geopack.geogsm, geogsm_vectorized, "GEO-GSM"),
            (geopack.gswgsm, gswgsm_vectorized, "GSW-GSM"),
        ]
        
        for scalar_func, vector_func, name in transformations:
            # Test forward transformation
            for i in range(n_points):
                scalar_result = scalar_func(x[i], y[i], z[i], 1)
                vector_result = vector_func(x[i], y[i], z[i], 1)
                
                np.testing.assert_allclose(
                    scalar_result, vector_result,
                    rtol=1e-14, atol=1e-14,
                    err_msg=f"{name} forward transformation failed at point {i}"
                )
            
            # Test reverse transformation
            for i in range(n_points):
                scalar_result = scalar_func(x[i], y[i], z[i], -1)
                vector_result = vector_func(x[i], y[i], z[i], -1)
                
                np.testing.assert_allclose(
                    scalar_result, vector_result,
                    rtol=1e-14, atol=1e-14,
                    err_msg=f"{name} reverse transformation failed at point {i}"
                )
    
    def test_sphcar_accuracy(self):
        """Test spherical-Cartesian transformation accuracy."""
        # Test special cases
        test_cases = [
            # (r, theta, phi) for forward transformation
            (0.0, 0.0, 0.0),      # Origin
            (1.0, 0.0, 0.0),      # North pole
            (1.0, np.pi, 0.0),    # South pole
            (1.0, np.pi/2, 0.0),  # Equator, x-axis
            (1.0, np.pi/2, np.pi/2), # Equator, y-axis
            (5.0, 0.7, 1.2),      # General point
        ]
        
        for r, theta, phi in test_cases:
            # Forward transformation (spherical to Cartesian)
            scalar_result = geopack.sphcar(r, theta, phi, 1)
            vector_result = sphcar_vectorized(r, theta, phi, 1)
            
            np.testing.assert_allclose(
                scalar_result, vector_result,
                rtol=1e-15, atol=1e-15,
                err_msg=f"Spherical to Cartesian failed at ({r}, {theta}, {phi})"
            )
            
            # Use the Cartesian result for reverse transformation
            x, y, z = scalar_result
            
            # Reverse transformation (Cartesian to spherical)
            scalar_result = geopack.sphcar(x, y, z, -1)
            vector_result = sphcar_vectorized(x, y, z, -1)
            
            np.testing.assert_allclose(
                scalar_result, vector_result,
                rtol=1e-15, atol=1e-15,
                err_msg=f"Cartesian to spherical failed at ({x}, {y}, {z})"
            )
    
    def test_bspcar_bcarsp_accuracy(self):
        """Test field component transformation accuracy."""
        # Test points with various field configurations
        test_cases = [
            # (theta, phi, br, btheta, bphi)
            (np.pi/4, np.pi/3, 1.0, 0.5, 0.3),
            (0.0, 0.0, 1.0, 0.0, 0.0),  # North pole
            (np.pi, 0.0, 1.0, 0.0, 0.0), # South pole
            (np.pi/2, 0.0, 0.0, 1.0, 0.0), # Equator
        ]
        
        for theta, phi, br, btheta, bphi in test_cases:
            # Test bspcar (spherical to Cartesian)
            scalar_result = geopack.bspcar(theta, phi, br, btheta, bphi)
            vector_result = bspcar_vectorized(theta, phi, br, btheta, bphi)
            
            np.testing.assert_allclose(
                scalar_result, vector_result,
                rtol=1e-15, atol=1e-15,
                err_msg=f"bspcar failed at theta={theta}, phi={phi}"
            )
            
            # Get position for bcarsp test
            r = 5.0  # arbitrary radius
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            bx, by, bz = scalar_result
            
            # Test bcarsp (Cartesian to spherical)
            scalar_result = geopack.bcarsp(x, y, z, bx, by, bz)
            vector_result = bcarsp_vectorized(x, y, z, bx, by, bz)
            
            np.testing.assert_allclose(
                scalar_result, vector_result,
                rtol=1e-14, atol=1e-14,
                err_msg=f"bcarsp failed at position ({x}, {y}, {z})"
            )
    
    def test_vectorized_performance(self):
        """Test that vectorized functions handle arrays efficiently."""
        # Create large arrays
        n = 10000
        x = np.random.uniform(-10, 10, n)
        y = np.random.uniform(-10, 10, n)
        z = np.random.uniform(-10, 10, n)
        
        # Test that vectorized function works with arrays
        result = gsmgse_vectorized(x, y, z, 1)
        self.assertEqual(result[0].shape, (n,))
        self.assertEqual(result[1].shape, (n,))
        self.assertEqual(result[2].shape, (n,))
        
        # Verify a few random points match scalar version
        indices = np.random.choice(n, 10)
        for i in indices:
            scalar_result = geopack.gsmgse(x[i], y[i], z[i], 1)
            vector_result = (result[0][i], result[1][i], result[2][i])
            
            np.testing.assert_allclose(
                scalar_result, vector_result,
                rtol=1e-14, atol=1e-14
            )
    
    def test_edge_cases_bcarsp(self):
        """Test edge cases for bcarsp where rho = 0."""
        # Points on z-axis where rho = 0
        test_cases = [
            (0.0, 0.0, 5.0, 1.0, 2.0, 3.0),   # Positive z-axis
            (0.0, 0.0, -5.0, 1.0, 2.0, 3.0),  # Negative z-axis
            (0.0, 0.0, 0.0, 1.0, 2.0, 3.0),   # Origin
        ]
        
        for x, y, z, bx, by, bz in test_cases:
            scalar_result = geopack.bcarsp(x, y, z, bx, by, bz)
            vector_result = bcarsp_vectorized(x, y, z, bx, by, bz)
            
            # Use assert_allclose with equal_nan=True to handle nan comparisons
            np.testing.assert_allclose(
                scalar_result, vector_result,
                rtol=1e-14, atol=1e-14, equal_nan=True,
                err_msg=f"bcarsp edge case failed at ({x}, {y}, {z})"
            )
    
    def test_roundtrip_transformations(self):
        """Test that forward + reverse transformations return to original values."""
        # Test point
        x0, y0, z0 = 5.0, 3.0, 2.0
        
        # GSM -> GSE -> GSM
        x1, y1, z1 = gsmgse_vectorized(x0, y0, z0, 1)
        x2, y2, z2 = gsmgse_vectorized(x1, y1, z1, -1)
        np.testing.assert_allclose([x0, y0, z0], [x2, y2, z2], rtol=1e-14)
        
        # Spherical -> Cartesian -> Spherical
        r0, theta0, phi0 = 5.0, 0.7, 1.2
        x1, y1, z1 = sphcar_vectorized(r0, theta0, phi0, 1)
        r1, theta1, phi1 = sphcar_vectorized(x1, y1, z1, -1)
        np.testing.assert_allclose([r0, theta0, phi0], [r1, theta1, phi1], rtol=1e-14)


if __name__ == '__main__':
    unittest.main()