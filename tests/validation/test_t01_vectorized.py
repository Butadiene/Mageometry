#!/usr/bin/env python
"""
Test script to validate T01 vectorized implementation against scalar version.
Tests both accuracy and performance.
"""

import numpy as np
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from geopack.t01 import t01
from geopack.t01_vectorized import t01_vectorized

def test_single_point():
    """Test single point calculation for exact match."""
    print("\n" + "="*60)
    print("Testing single point calculation...")
    print("="*60)
    
    # Test parameters
    parmod = np.array([2.0, -10.0, 3.0, -5.0, 0.5, 0.8, 0.0, 0.0, 0.0, 0.0])
    ps = 0.1
    x, y, z = 5.0, 2.0, 1.0
    
    # Calculate with both versions
    bx_scalar, by_scalar, bz_scalar = t01(parmod, ps, x, y, z)
    bx_vector, by_vector, bz_vector = t01_vectorized(parmod, ps, x, y, z)
    
    # Compare results
    print(f"Position: ({x}, {y}, {z})")
    print(f"Scalar result: Bx={bx_scalar:.6f}, By={by_scalar:.6f}, Bz={bz_scalar:.6f}")
    print(f"Vector result: Bx={bx_vector:.6f}, By={by_vector:.6f}, Bz={bz_vector:.6f}")
    
    # Calculate differences
    dx = abs(bx_vector - bx_scalar)
    dy = abs(by_vector - by_scalar)
    dz = abs(bz_vector - bz_scalar)
    
    print(f"\nAbsolute differences:")
    print(f"  ΔBx = {dx:.2e}")
    print(f"  ΔBy = {dy:.2e}")
    print(f"  ΔBz = {dz:.2e}")
    
    # Check if differences are within tolerance
    tolerance = 1e-6
    if dx < tolerance and dy < tolerance and dz < tolerance:
        print(f"\n✓ Single point test PASSED (tolerance: {tolerance})")
    else:
        print(f"\n✗ Single point test FAILED (tolerance: {tolerance})")
        print(f"  Maximum difference: {max(dx, dy, dz):.2e}")

def test_array_accuracy():
    """Test array calculations for accuracy."""
    print("\n" + "="*60)
    print("Testing array calculations accuracy...")
    print("="*60)
    
    # Test parameters
    parmod = np.array([2.0, -10.0, 3.0, -5.0, 0.5, 0.8, 0.0, 0.0, 0.0, 0.0])
    ps = 0.1
    
    # Create test grid
    n_points = 100
    np.random.seed(42)  # For reproducibility
    x = np.random.uniform(-10, 10, n_points)
    y = np.random.uniform(-10, 10, n_points)
    z = np.random.uniform(-5, 5, n_points)
    
    # Calculate with scalar version (loop)
    bx_scalar = np.zeros(n_points)
    by_scalar = np.zeros(n_points)
    bz_scalar = np.zeros(n_points)
    
    print(f"Calculating {n_points} points with scalar version...")
    t0 = time.time()
    for i in range(n_points):
        bx_scalar[i], by_scalar[i], bz_scalar[i] = t01(parmod, ps, x[i], y[i], z[i])
    scalar_time = time.time() - t0
    print(f"Scalar calculation time: {scalar_time:.3f} seconds")
    
    # Calculate with vectorized version
    print(f"Calculating {n_points} points with vectorized version...")
    t0 = time.time()
    bx_vector, by_vector, bz_vector = t01_vectorized(parmod, ps, x, y, z)
    vector_time = time.time() - t0
    print(f"Vector calculation time: {vector_time:.3f} seconds")
    
    # Performance improvement
    speedup = scalar_time / vector_time
    print(f"\nSpeedup: {speedup:.1f}x")
    
    # Calculate differences
    dx = np.abs(bx_vector - bx_scalar)
    dy = np.abs(by_vector - by_scalar)
    dz = np.abs(bz_vector - bz_scalar)
    
    # Statistics
    print(f"\nAccuracy statistics:")
    print(f"  Bx differences: mean={np.mean(dx):.2e}, max={np.max(dx):.2e}")
    print(f"  By differences: mean={np.mean(dy):.2e}, max={np.max(dy):.2e}")
    print(f"  Bz differences: mean={np.mean(dz):.2e}, max={np.max(dz):.2e}")
    
    # Relative errors where field is significant
    b_mag_scalar = np.sqrt(bx_scalar**2 + by_scalar**2 + bz_scalar**2)
    significant = b_mag_scalar > 1.0  # Only consider points with |B| > 1 nT
    
    if np.any(significant):
        rel_x = dx[significant] / np.abs(bx_scalar[significant] + 1e-10)
        rel_y = dy[significant] / np.abs(by_scalar[significant] + 1e-10)
        rel_z = dz[significant] / np.abs(bz_scalar[significant] + 1e-10)
        
        print(f"\nRelative errors (where |B| > 1 nT):")
        print(f"  Bx: mean={np.mean(rel_x):.2e}, max={np.max(rel_x):.2e}")
        print(f"  By: mean={np.mean(rel_y):.2e}, max={np.max(rel_y):.2e}")
        print(f"  Bz: mean={np.mean(rel_z):.2e}, max={np.max(rel_z):.2e}")
    
    # Overall assessment
    max_abs_error = max(np.max(dx), np.max(dy), np.max(dz))
    tolerance = 1e-4  # 0.01% relative error
    
    if max_abs_error < tolerance:
        print(f"\n✓ Array accuracy test PASSED (max error: {max_abs_error:.2e} < {tolerance})")
    else:
        print(f"\n✗ Array accuracy test FAILED (max error: {max_abs_error:.2e} > {tolerance})")
        
        # Find worst case
        worst_idx = np.argmax(dx + dy + dz)
        print(f"\nWorst case at index {worst_idx}:")
        print(f"  Position: ({x[worst_idx]:.3f}, {y[worst_idx]:.3f}, {z[worst_idx]:.3f})")
        print(f"  Scalar: ({bx_scalar[worst_idx]:.6f}, {by_scalar[worst_idx]:.6f}, {bz_scalar[worst_idx]:.6f})")
        print(f"  Vector: ({bx_vector[worst_idx]:.6f}, {by_vector[worst_idx]:.6f}, {bz_vector[worst_idx]:.6f})")

def test_edge_cases():
    """Test edge cases and special conditions."""
    print("\n" + "="*60)
    print("Testing edge cases...")
    print("="*60)
    
    parmod = np.array([2.0, -10.0, 3.0, -5.0, 0.5, 0.8, 0.0, 0.0, 0.0, 0.0])
    ps = 0.1
    
    # Test 1: Points on axes
    print("\n1. Testing points on coordinate axes:")
    test_points = [
        (10.0, 0.0, 0.0, "X-axis"),
        (0.0, 10.0, 0.0, "Y-axis"),
        (0.0, 0.0, 10.0, "Z-axis"),
        (0.0, 0.0, 0.0, "Origin")
    ]
    
    for x, y, z, label in test_points:
        try:
            bx_s, by_s, bz_s = t01(parmod, ps, x, y, z)
            bx_v, by_v, bz_v = t01_vectorized(parmod, ps, x, y, z)
            
            diff = np.sqrt((bx_v-bx_s)**2 + (by_v-by_s)**2 + (bz_v-bz_s)**2)
            print(f"  {label}: difference = {diff:.2e}")
        except Exception as e:
            print(f"  {label}: ERROR - {str(e)}")
    
    # Test 2: Very small and large distances
    print("\n2. Testing extreme distances:")
    extreme_points = [
        (0.1, 0.1, 0.1, "Very close"),
        (100.0, 50.0, 20.0, "Very far")
    ]
    
    for x, y, z, label in extreme_points:
        try:
            bx_s, by_s, bz_s = t01(parmod, ps, x, y, z)
            bx_v, by_v, bz_v = t01_vectorized(parmod, ps, x, y, z)
            
            diff = np.sqrt((bx_v-bx_s)**2 + (by_v-by_s)**2 + (bz_v-bz_s)**2)
            print(f"  {label}: difference = {diff:.2e}")
        except Exception as e:
            print(f"  {label}: ERROR - {str(e)}")
    
    # Test 3: Mixed scalar and array inputs
    print("\n3. Testing mixed input types:")
    x_arr = np.array([1.0, 2.0, 3.0])
    y_scalar = 1.0
    z_arr = np.array([0.5, 1.0, 1.5])
    
    try:
        bx_v, by_v, bz_v = t01_vectorized(parmod, ps, x_arr, y_scalar, z_arr)
        print(f"  Mixed inputs: Output shape = {bx_v.shape}")
        print(f"  ✓ Mixed input handling works")
    except Exception as e:
        print(f"  ✗ Mixed input handling failed: {str(e)}")

def test_performance_scaling():
    """Test performance scaling with array size."""
    print("\n" + "="*60)
    print("Testing performance scaling...")
    print("="*60)
    
    parmod = np.array([2.0, -10.0, 3.0, -5.0, 0.5, 0.8, 0.0, 0.0, 0.0, 0.0])
    ps = 0.1
    
    sizes = [10, 100, 1000, 10000]
    times = []
    
    for n in sizes:
        # Create random points
        np.random.seed(42)
        x = np.random.uniform(-10, 10, n)
        y = np.random.uniform(-10, 10, n)
        z = np.random.uniform(-5, 5, n)
        
        # Time the calculation
        t0 = time.time()
        bx, by, bz = t01_vectorized(parmod, ps, x, y, z)
        dt = time.time() - t0
        times.append(dt)
        
        print(f"  {n:6d} points: {dt:8.4f} seconds ({n/dt:8.0f} points/sec)")
    
    # Check if scaling is approximately linear
    if len(times) > 1:
        scaling_factor = times[-1] / times[0]
        size_factor = sizes[-1] / sizes[0]
        efficiency = size_factor / scaling_factor
        
        print(f"\nScaling efficiency: {efficiency:.1f}x")
        if efficiency > 0.8:
            print("✓ Good linear scaling")
        else:
            print("✗ Poor scaling efficiency")

if __name__ == "__main__":
    print("T01 Vectorization Validation Tests")
    print("==================================")
    
    try:
        test_single_point()
        test_array_accuracy()
        test_edge_cases()
        test_performance_scaling()
        
        print("\n" + "="*60)
        print("All tests completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: Test failed with exception:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()