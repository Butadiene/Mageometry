#!/usr/bin/env python
"""
Comprehensive accuracy evaluation for T04 vectorized implementation.
Tests accuracy across wide parameter space.
"""

import numpy as np
import time
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from geopack.t04 import t04 as t04_scalar
from geopack.t04_vectorized import t04_vectorized

def evaluate_t04_accuracy():
    """Evaluate accuracy across parameter space."""
    print("\nT04 Vectorized Accuracy Evaluation")
    print("=" * 70)
    print(f"Evaluation started: {datetime.now()}")
    
    # Define parameter ranges
    # T04 parameters: [Pdyn, Dst, ByIMF, BzIMF, W1, W2, W3, W4, W5, W6]
    pdyn_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    dst_values = [-100, -50, -20, 0, 20]
    by_values = [-10, -5, 0, 5, 10]
    bz_values = [-10, -5, 0, 5, 10]
    w_values = [0.0, 0.5, 1.0, 1.5, 2.0]  # Storm indices
    ps_values = [-0.3, -0.15, 0.0, 0.15, 0.3]
    
    # Spatial grid
    n_spatial = 20
    x_range = np.linspace(-15, 10, n_spatial)
    y_range = np.linspace(-10, 10, n_spatial)
    z_range = np.linspace(-5, 5, n_spatial)
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    n_points = len(x_flat)
    
    print(f"\nTest configuration:")
    print(f"  Pdyn values: {len(pdyn_values)}")
    print(f"  Dst values: {len(dst_values)}")
    print(f"  IMF By values: {len(by_values)}")
    print(f"  IMF Bz values: {len(bz_values)}")
    print(f"  Storm W values: {len(w_values)}")
    print(f"  Tilt angles: {len(ps_values)}")
    print(f"  Spatial points: {n_points}")
    
    # Storage for results
    max_abs_error = 0.0
    max_rel_error = 0.0
    all_errors = []
    error_count = 0
    total_tests = 0
    
    print("\nRunning accuracy tests...")
    
    # Test subset of parameter combinations
    for i, pdyn in enumerate(pdyn_values[:3]):  # Test first 3 values
        for dst in dst_values[::2]:  # Every other value
            for by in by_values[::2]:
                for bz in bz_values[::2]:
                    for w in w_values[::2]:
                        # Use same W values for simplicity
                        parmod = np.array([pdyn, dst, by, bz, w, w, w, w, w, w])
                        
                        for ps in ps_values[::2]:
                            # Calculate with scalar version
                            bx_scalar = np.zeros(n_points)
                            by_scalar = np.zeros(n_points)
                            bz_scalar = np.zeros(n_points)
                            
                            for idx in range(n_points):
                                bx_scalar[idx], by_scalar[idx], bz_scalar[idx] = \
                                    t04_scalar(parmod, ps, x_flat[idx], y_flat[idx], z_flat[idx])
                            
                            # Calculate with vectorized version
                            bx_vector, by_vector, bz_vector = t04_vectorized(parmod, ps, x_flat, y_flat, z_flat)
                            
                            # Calculate errors
                            dx = np.abs(bx_vector - bx_scalar)
                            dy = np.abs(by_vector - by_scalar)
                            dz = np.abs(bz_vector - bz_scalar)
                            
                            # Find maximum errors
                            max_dx = np.max(dx)
                            max_dy = np.max(dy)
                            max_dz = np.max(dz)
                            
                            # Update maximum absolute error
                            max_abs_error = max(max_abs_error, max_dx, max_dy, max_dz)
                            
                            # Calculate relative errors where field is significant
                            b_scalar_mag = np.sqrt(bx_scalar**2 + by_scalar**2 + bz_scalar**2)
                            significant = b_scalar_mag > 1.0  # Only where field > 1 nT
                            
                            if np.any(significant):
                                rel_errors = np.sqrt(dx[significant]**2 + dy[significant]**2 + dz[significant]**2) / b_scalar_mag[significant]
                                if len(rel_errors) > 0:
                                    max_rel_error = max(max_rel_error, np.max(rel_errors))
                                    all_errors.extend(rel_errors)
                            
                            total_tests += 1
                            
                            # Count significant errors
                            if max(max_dx, max_dy, max_dz) > 1e-6:
                                error_count += 1
        
        print(f"  Completed Pdyn = {pdyn:.1f} nPa ({i+1}/{len(pdyn_values[:3])})")
    
    # Calculate statistics
    if all_errors:
        mean_rel_error = np.mean(all_errors)
        median_rel_error = np.median(all_errors)
        percentile_95 = np.percentile(all_errors, 95)
        percentile_99 = np.percentile(all_errors, 99)
    else:
        mean_rel_error = median_rel_error = percentile_95 = percentile_99 = 0.0
    
    # Performance test
    print("\nRunning performance test...")
    
    # Large array test
    n_perf = 10000
    x_perf = np.random.uniform(-10, 10, n_perf)
    y_perf = np.random.uniform(-10, 10, n_perf)
    z_perf = np.random.uniform(-5, 5, n_perf)
    parmod_perf = np.array([2.0, -20.0, 5.0, -5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    ps_perf = 0.2
    
    # Time scalar version (subset)
    n_scalar_test = min(100, n_perf)
    t0 = time.time()
    for i in range(n_scalar_test):
        t04_scalar(parmod_perf, ps_perf, x_perf[i], y_perf[i], z_perf[i])
    scalar_time = (time.time() - t0) * (n_perf / n_scalar_test)
    
    # Time vectorized version
    t0 = time.time()
    t04_vectorized(parmod_perf, ps_perf, x_perf, y_perf, z_perf)
    vector_time = time.time() - t0
    
    speedup = scalar_time / vector_time
    
    # Print results
    print("\n" + "=" * 70)
    print("ACCURACY RESULTS")
    print("=" * 70)
    print(f"Total parameter combinations tested: {total_tests}")
    print(f"Total spatial points per test: {n_points}")
    print(f"Total field calculations: {total_tests * n_points:,}")
    
    print(f"\nMaximum absolute error: {max_abs_error:.2e} nT")
    print(f"Maximum relative error: {max_rel_error:.2e}")
    print(f"\nRelative error statistics (where |B| > 1 nT):")
    print(f"  Mean: {mean_rel_error:.2e}")
    print(f"  Median: {median_rel_error:.2e}")
    print(f"  95th percentile: {percentile_95:.2e}")
    print(f"  99th percentile: {percentile_99:.2e}")
    
    print(f"\nTests with error > 1e-6 nT: {error_count}/{total_tests} ({100*error_count/total_tests:.1f}%)")
    
    print("\n" + "=" * 70)
    print("PERFORMANCE RESULTS")
    print("=" * 70)
    print(f"Array size: {n_perf:,} points")
    print(f"Scalar time (estimated): {scalar_time:.2f} s")
    print(f"Vector time: {vector_time:.3f} s")
    print(f"Speedup: {speedup:.1f}x")
    print(f"Processing rate: {n_perf/vector_time:,.0f} points/second")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if max_abs_error < 1e-7 and max_rel_error < 1e-6:
        print("✓ EXCELLENT accuracy - suitable for production use")
    elif max_abs_error < 1e-5 and max_rel_error < 1e-4:
        print("✓ GOOD accuracy - acceptable for most applications")
    else:
        print("✗ POOR accuracy - needs improvement")
    
    print(f"\nEvaluation completed: {datetime.now()}")
    
    return max_abs_error, max_rel_error, speedup

if __name__ == "__main__":
    # Suppress warnings
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # Run evaluation
    max_abs_error, max_rel_error, speedup = evaluate_t04_accuracy()