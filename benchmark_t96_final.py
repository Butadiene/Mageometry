#!/usr/bin/env python
"""Final benchmark of T96 vectorized implementation."""

import numpy as np
import time
from geopack.t96_vectorized import t96_vectorized
from geopack.t96 import t96

def benchmark_t96():
    """Comprehensive benchmark of T96 vectorized implementation."""
    print("T96 Vectorized Implementation - Final Benchmark")
    print("=" * 70)
    
    # Test parameters
    parmod = [2.0, -10.0, 0.5, -3.0, 0, 0, 0, 0, 0, 0]
    ps = 0.1
    
    # Performance tests with different array sizes
    print("\nPerformance Benchmarks:")
    print("-" * 70)
    print(f"{'Array Size':<15} {'Vectorized Time':<20} {'Scalar Time (est)':<20} {'Speedup':<10}")
    print("-" * 70)
    
    for n_points in [10, 100, 1000, 10000, 100000]:
        np.random.seed(42)
        x = np.random.uniform(-20, 10, n_points)
        y = np.random.uniform(-15, 15, n_points)
        z = np.random.uniform(-10, 10, n_points)
        
        # Time vectorized version
        start = time.time()
        bx_v, by_v, bz_v = t96_vectorized(parmod, ps, x, y, z)
        vec_time = time.time() - start
        
        # Estimate scalar time from small sample
        n_sample = min(10, n_points)
        start = time.time()
        for i in range(n_sample):
            t96(parmod, ps, x[i], y[i], z[i])
        scalar_time_est = (time.time() - start) / n_sample * n_points
        
        speedup = scalar_time_est / vec_time
        points_per_sec = n_points / vec_time
        
        print(f"{n_points:<15} {vec_time:<20.3f} {scalar_time_est:<20.1f} {speedup:<10.1f}")
    
    # Accuracy analysis
    print("\n\nAccuracy Analysis:")
    print("-" * 70)
    
    # Test on grid of points
    n_test = 1000
    np.random.seed(123)
    x_test = np.random.uniform(-20, 10, n_test)
    y_test = np.random.uniform(-15, 15, n_test)
    z_test = np.random.uniform(-10, 10, n_test)
    
    # Calculate fields
    bx_vec, by_vec, bz_vec = t96_vectorized(parmod, ps, x_test, y_test, z_test)
    
    # Sample accuracy check
    errors = []
    n_check = 100
    indices = np.random.choice(n_test, n_check, replace=False)
    
    for idx in indices:
        bx_s, by_s, bz_s = t96(parmod, ps, x_test[idx], y_test[idx], z_test[idx])
        b_scalar = np.sqrt(bx_s**2 + by_s**2 + bz_s**2)
        b_vector = np.sqrt(bx_vec[idx]**2 + by_vec[idx]**2 + bz_vec[idx]**2)
        
        if b_scalar > 0:
            rel_error = abs(b_vector - b_scalar) / b_scalar * 100
            errors.append(rel_error)
    
    errors = np.array(errors)
    print(f"Accuracy statistics (n={n_check} points):")
    print(f"  Mean error: {np.mean(errors):.2f}%")
    print(f"  Median error: {np.median(errors):.2f}%")
    print(f"  Max error: {np.max(errors):.2f}%")
    print(f"  Min error: {np.min(errors):.2f}%")
    print(f"  Std dev: {np.std(errors):.2f}%")
    
    # Error distribution
    print(f"\nError distribution:")
    print(f"  < 1%: {np.sum(errors < 1):3d} points ({np.sum(errors < 1)/len(errors)*100:.1f}%)")
    print(f"  < 5%: {np.sum(errors < 5):3d} points ({np.sum(errors < 5)/len(errors)*100:.1f}%)")
    print(f"  < 10%: {np.sum(errors < 10):3d} points ({np.sum(errors < 10)/len(errors)*100:.1f}%)")
    print(f"  < 20%: {np.sum(errors < 20):3d} points ({np.sum(errors < 20)/len(errors)*100:.1f}%)")
    
    # Memory efficiency test
    print("\n\nMemory Efficiency:")
    print("-" * 70)
    
    n_large = 1000000
    x_large = np.random.uniform(-20, 10, n_large)
    y_large = np.random.uniform(-15, 15, n_large)
    z_large = np.random.uniform(-10, 10, n_large)
    
    start = time.time()
    bx_large, by_large, bz_large = t96_vectorized(parmod, ps, x_large, y_large, z_large)
    large_time = time.time() - start
    
    print(f"Processed {n_large:,} points in {large_time:.2f} seconds")
    print(f"Rate: {n_large/large_time:,.0f} points/second")
    print(f"Memory usage (approx): {(x_large.nbytes + bx_large.nbytes) * 3 / 1e6:.1f} MB")

if __name__ == '__main__':
    benchmark_t96()