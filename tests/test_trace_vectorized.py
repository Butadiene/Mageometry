"""
Comprehensive test suite for vectorized field line tracing.

Tests accuracy, edge cases, and performance of the vectorized implementation
against the scalar version.
"""

import numpy as np
import unittest
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import geopack
from geopack.trace_field_lines_vectorized import trace_vectorized


class TestTraceVectorized(unittest.TestCase):
    """Test vectorized field line tracing implementation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment with recalc."""
        # Use a fixed time for reproducible results
        ut = 100.0  # 1970-01-01 00:01:40
        cls.ps = geopack.recalc(ut)
        
    def test_scalar_compatibility(self):
        """Test that scalar inputs return scalar outputs."""
        x0, y0, z0 = 5.0, 0.0, 0.0
        
        # Test with scalar input
        xf, yf, zf, status = trace_vectorized(x0, y0, z0, return_full_path=False)
        
        self.assertIsInstance(xf, (float, np.floating))
        self.assertIsInstance(yf, (float, np.floating))
        self.assertIsInstance(zf, (float, np.floating))
        self.assertIsInstance(status, (int, np.integer))
        
    def test_array_inputs(self):
        """Test with array inputs."""
        x0 = np.array([5.0, 6.0, 7.0])
        y0 = np.array([0.0, 1.0, 0.0])
        z0 = np.array([0.0, 0.0, 1.0])
        
        xf, yf, zf, status = trace_vectorized(x0, y0, z0, return_full_path=False)
        
        self.assertEqual(len(xf), 3)
        self.assertEqual(len(yf), 3)
        self.assertEqual(len(zf), 3)
        self.assertEqual(len(status), 3)
        
    def test_accuracy_vs_scalar(self):
        """Compare accuracy with scalar implementation."""
        # Test points in different regions
        test_points = [
            (5.0, 0.0, 0.0),    # Equatorial
            (3.0, 0.0, 3.0),    # High latitude
            (7.0, 2.0, 1.0),    # Off-axis
            (4.0, -1.0, 2.0),   # Different quadrant
        ]
        
        max_error = 0.0
        
        for x0, y0, z0 in test_points:
            # Scalar version
            xf_s, yf_s, zf_s, xx_s, yy_s, zz_s = geopack.geopack.trace(
                x0, y0, z0, dir=1, parmod=2, exname='t89'
            )
            
            # Vectorized version (single point)
            xf_v, yf_v, zf_v, status = trace_vectorized(
                x0, y0, z0, dir=1, parmod=2, exname='t89', return_full_path=False
            )
            
            # Calculate relative error
            dist_scalar = np.sqrt(xf_s**2 + yf_s**2 + zf_s**2)
            dist_vector = np.sqrt(xf_v**2 + yf_v**2 + zf_v**2)
            
            if dist_scalar > 0:
                rel_error = abs(dist_vector - dist_scalar) / dist_scalar
                max_error = max(max_error, rel_error)
                
                # Check position match
                self.assertAlmostEqual(xf_s, xf_v, places=6,
                    msg=f"X mismatch at ({x0}, {y0}, {z0})")
                self.assertAlmostEqual(yf_s, yf_v, places=6,
                    msg=f"Y mismatch at ({x0}, {y0}, {z0})")
                self.assertAlmostEqual(zf_s, zf_v, places=6,
                    msg=f"Z mismatch at ({x0}, {y0}, {z0})")
        
        print(f"Maximum relative error: {max_error:.2e}")
        self.assertLess(max_error, 1e-6, "Relative error exceeds tolerance")
        
    def test_full_path_return(self):
        """Test returning full field line paths."""
        x0, y0, z0 = 5.0, 0.0, 0.0
        
        # Test with full path
        xf, yf, zf, xx, yy, zz, status = trace_vectorized(
            x0, y0, z0, return_full_path=True
        )
        
        # Check that paths are returned
        self.assertIsInstance(xx, np.ndarray)
        self.assertIsInstance(yy, np.ndarray)
        self.assertIsInstance(zz, np.ndarray)
        
        # First point should be starting position
        self.assertAlmostEqual(xx[0], x0)
        self.assertAlmostEqual(yy[0], y0)
        self.assertAlmostEqual(zz[0], z0)
        
        # Last valid point should be near final position
        self.assertAlmostEqual(xx[-1], xf, places=3)
        self.assertAlmostEqual(yy[-1], yf, places=3)
        self.assertAlmostEqual(zz[-1], zf, places=3)
        
    def test_status_codes(self):
        """Test different status code scenarios."""
        # Test successful trace to inner boundary
        x0, y0, z0 = 5.0, 0.0, 0.0
        _, _, _, status = trace_vectorized(x0, y0, z0)
        self.assertEqual(status, 0, "Expected successful trace status")
        
        # Test outer boundary hit
        x0, y0, z0 = 25.0, 0.0, 0.0  # Far outside
        _, _, _, status = trace_vectorized(x0, y0, z0, rlim=10)
        self.assertEqual(status, 1, "Expected outer boundary status")
        
        # Test max iterations (use very small maxloop)
        x0, y0, z0 = 5.0, 0.0, 0.0
        _, _, _, status = trace_vectorized(x0, y0, z0, maxloop=5)
        self.assertEqual(status, 2, "Expected max iterations status")
        
    def test_edge_cases(self):
        """Test various edge cases."""
        # Test starting inside Earth - should not trace (starts inside boundary)
        x0, y0, z0 = 0.5, 0.0, 0.0
        xf, yf, zf, status = trace_vectorized(x0, y0, z0, r0=1.0)
        # Starting inside r0 means immediate termination
        self.assertEqual(xf, x0, "Should not move from starting position")
        self.assertEqual(yf, y0, "Should not move from starting position")
        self.assertEqual(zf, z0, "Should not move from starting position")
        
        # Test multiple traces with mixed lengths
        x0 = np.array([5.0, 3.0, 8.0, 2.0])
        y0 = np.zeros(4)
        z0 = np.array([0.0, 2.0, 1.0, 0.5])
        
        xf, yf, zf, status = trace_vectorized(x0, y0, z0)
        
        # All should complete successfully
        self.assertEqual(len(status), 4)
        self.assertTrue(np.all(status >= 0), "All traces should complete")
        
    def test_different_models(self):
        """Test with different field models."""
        x0, y0, z0 = 5.0, 0.0, 0.0
        
        # Test available models
        models = ['t89']  # Start with T89
        
        # Check which other models are available
        try:
            from geopack.vectorized import t96_vectorized
            models.append('t96')
        except ImportError:
            pass
            
        for model in models:
            # Set appropriate parameters
            if model == 't89':
                parmod = 2  # Kp index
            else:
                parmod = [2.0, -5.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            
            xf, yf, zf, status = trace_vectorized(
                x0, y0, z0, parmod=parmod, exname=model
            )
            
            # Should complete successfully
            self.assertEqual(status, 0, f"Model {model} failed to trace")
            
            # Final position should be near Earth surface
            r_final = np.sqrt(xf**2 + yf**2 + zf**2)
            self.assertAlmostEqual(r_final, 1.0, delta=0.1,
                msg=f"Model {model} final radius {r_final:.3f} not near 1.0")
                
    def test_direction_parameter(self):
        """Test tracing in both directions."""
        x0, y0, z0 = 5.0, 0.0, 1.0
        
        # Trace antiparallel (north to south)
        xf1, yf1, zf1, status1 = trace_vectorized(x0, y0, z0, dir=1)
        
        # Trace parallel (south to north)
        xf2, yf2, zf2, status2 = trace_vectorized(x0, y0, z0, dir=-1)
        
        # Should reach Earth surface
        self.assertEqual(status1, 0)
        self.assertEqual(status2, 0)
        
        # Both should reach near Earth radius
        r1 = np.sqrt(xf1**2 + yf1**2 + zf1**2)
        r2 = np.sqrt(xf2**2 + yf2**2 + zf2**2)
        self.assertAlmostEqual(r1, 1.0, delta=0.1, msg="Dir=1 should reach Earth surface")
        self.assertAlmostEqual(r2, 1.0, delta=0.1, msg="Dir=-1 should reach Earth surface")
        
        # They should reach different locations
        dist = np.sqrt((xf1-xf2)**2 + (yf1-yf2)**2 + (zf1-zf2)**2)
        self.assertGreater(dist, 0.5, "Different directions should reach different points")
        
    def test_batch_processing(self):
        """Test processing many field lines at once."""
        # Create a grid of starting points
        n = 10
        x_grid = np.linspace(3, 8, n)
        y_grid = np.zeros(n)
        z_grid = np.linspace(-2, 2, n)
        
        # Time batch processing
        t_start = time.time()
        xf, yf, zf, status = trace_vectorized(x_grid, y_grid, z_grid)
        t_batch = time.time() - t_start
        
        # Check all completed
        self.assertEqual(len(status), n)
        self.assertTrue(np.all(status >= 0), "Some traces failed")
        
        # Compare with scalar loop timing (just a few for comparison)
        n_compare = min(3, n)
        t_start = time.time()
        for i in range(n_compare):
            geopack.geopack.trace(x_grid[i], y_grid[i], z_grid[i])
        t_scalar = (time.time() - t_start) * n / n_compare  # Extrapolate
        
        speedup = t_scalar / t_batch
        print(f"Batch processing {n} traces: {speedup:.1f}x speedup")
        
        # For small batches, speedup might be modest
        if n >= 10:
            self.assertGreater(speedup, 2.0, "Insufficient speedup for batch")


class TestMemoryEfficiency(unittest.TestCase):
    """Test memory efficiency features."""
    
    def test_memory_usage_endpoints_only(self):
        """Test memory usage with endpoints-only mode."""
        # Large batch
        n = 1000
        x0 = np.random.uniform(3, 8, n)
        y0 = np.random.uniform(-2, 2, n)
        z0 = np.random.uniform(-2, 2, n)
        
        # Should use minimal memory (no full paths)
        xf, yf, zf, status = trace_vectorized(
            x0, y0, z0, return_full_path=False
        )
        
        # Check output sizes
        self.assertEqual(len(xf), n)
        self.assertEqual(len(yf), n)
        self.assertEqual(len(zf), n)
        self.assertEqual(len(status), n)
        
    def test_memory_usage_full_paths(self):
        """Test memory usage with full path storage."""
        # Smaller batch for full paths
        n = 100
        x0 = np.random.uniform(3, 8, n)
        y0 = np.random.uniform(-2, 2, n)  
        z0 = np.random.uniform(-2, 2, n)
        
        xf, yf, zf, xx, yy, zz, status = trace_vectorized(
            x0, y0, z0, return_full_path=True, maxloop=500
        )
        
        # Check shapes
        self.assertEqual(xx.shape[0], n)
        self.assertEqual(yy.shape[0], n)
        self.assertEqual(zz.shape[0], n)
        self.assertLessEqual(xx.shape[1], 500)
        
        # Check masking works (not all traces use all steps)
        self.assertTrue(np.ma.is_masked(xx))
        self.assertTrue(np.ma.is_masked(yy))
        self.assertTrue(np.ma.is_masked(zz))


class BenchmarkPerformance(unittest.TestCase):
    """Benchmark performance at different scales."""
    
    def test_performance_scaling(self):
        """Test performance scaling with number of traces."""
        sizes = [1, 10, 100]
        
        print("\nPerformance Scaling:")
        print("N traces | Time (s) | Traces/sec | Speedup")
        print("-" * 45)
        
        # Baseline: time for single scalar trace
        x0, y0, z0 = 5.0, 0.0, 0.0
        t_start = time.time()
        geopack.geopack.trace(x0, y0, z0)
        t_scalar_single = time.time() - t_start
        
        for n in sizes:
            # Generate random starting points
            np.random.seed(42)  # Reproducible
            x0 = np.random.uniform(3, 8, n)
            y0 = np.random.uniform(-2, 2, n)
            z0 = np.random.uniform(-2, 2, n)
            
            # Time vectorized version
            t_start = time.time()
            xf, yf, zf, status = trace_vectorized(x0, y0, z0)
            t_vec = time.time() - t_start
            
            # Calculate metrics
            traces_per_sec = n / t_vec
            speedup = (t_scalar_single * n) / t_vec
            
            print(f"{n:8d} | {t_vec:8.4f} | {traces_per_sec:10.1f} | {speedup:7.1f}x")
            
            # Verify all completed
            self.assertTrue(np.all(status >= 0))


if __name__ == '__main__':
    unittest.main(verbosity=2)