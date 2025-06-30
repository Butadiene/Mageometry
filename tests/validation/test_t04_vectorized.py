#!/usr/bin/env python
"""
Test script to validate T04 vectorized implementation against scalar version.
"""

import numpy as np
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from geopack.t04 import t04 as t04_scalar
from geopack.t04_vectorized import t04_vectorized

def test_single_point():
    """Test single point calculation."""
    print("\n" + "="*60)
    print("Testing single point calculation...")
    print("="*60)
    
    # T04 parameters: [Pdyn, Dst, ByIMF, BzIMF, W1, W2, W3, W4, W5, W6]
    parmod = np.array([2.0,    # Solar wind pressure (nPa)
                       -20.0,  # Dst (nT)
                       5.0,    # IMF By (nT)
                       -5.0,   # IMF Bz (nT)
                       0.5,    # W1
                       0.6,    # W2
                       0.7,    # W3
                       0.8,    # W4
                       0.9,    # W5
                       1.0])   # W6
    ps = 0.2  # Dipole tilt
    x, y, z = 5.0, 2.0, 1.0
    
    # Calculate with both versions
    bx_scalar, by_scalar, bz_scalar = t04_scalar(parmod, ps, x, y, z)
    bx_vector, by_vector, bz_vector = t04_vectorized(parmod, ps, x, y, z)
    
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
        return True
    else:
        print(f"\n✗ Single point test FAILED (tolerance: {tolerance})")
        return False

def test_array_interface():
    """Test array interface and scalar handling."""
    print("\n" + "="*60)
    print("Testing array interface...")
    print("="*60)
    
    parmod = np.array([2.0, -20.0, 5.0, -5.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ps = 0.2
    
    # Test 1: Scalar inputs should return scalars
    x, y, z = 5.0, 2.0, 1.0
    bx, by, bz = t04_vectorized(parmod, ps, x, y, z)
    print(f"Scalar input: type(bx) = {type(bx).__name__}")
    assert isinstance(bx, float), "Should return scalar for scalar input"
    
    # Test 2: Array inputs should return arrays
    x_arr = np.array([5.0, 6.0, 7.0])
    y_arr = np.array([2.0, 3.0, 4.0])
    z_arr = np.array([1.0, 1.5, 2.0])
    bx, by, bz = t04_vectorized(parmod, ps, x_arr, y_arr, z_arr)
    print(f"Array input: type(bx) = {type(bx).__name__}, shape = {bx.shape}")
    assert isinstance(bx, np.ndarray), "Should return array for array input"
    assert bx.shape == (3,), "Should have correct shape"
    
    # Test 3: Mixed inputs (broadcasting)
    bx, by, bz = t04_vectorized(parmod, ps, x_arr, 2.0, z_arr)
    print(f"Mixed input: type(bx) = {type(bx).__name__}, shape = {bx.shape}")
    assert isinstance(bx, np.ndarray), "Should return array for mixed input"
    assert bx.shape == (3,), "Should broadcast correctly"
    
    print("\n✓ Array interface test PASSED")
    return True

def test_accuracy():
    """Test accuracy across multiple points."""
    print("\n" + "="*60)
    print("Testing accuracy across multiple points...")
    print("="*60)
    
    parmod = np.array([2.0, -20.0, 5.0, -5.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ps = 0.2
    
    # Generate test points
    n_points = 100
    np.random.seed(42)
    x = np.random.uniform(-10, 10, n_points)
    y = np.random.uniform(-10, 10, n_points)
    z = np.random.uniform(-5, 5, n_points)
    
    # Calculate with scalar version
    bx_scalar = np.zeros(n_points)
    by_scalar = np.zeros(n_points)
    bz_scalar = np.zeros(n_points)
    
    t0 = time.time()
    for i in range(n_points):
        bx_scalar[i], by_scalar[i], bz_scalar[i] = t04_scalar(parmod, ps, x[i], y[i], z[i])
    scalar_time = time.time() - t0
    
    # Calculate with vectorized version
    t0 = time.time()
    bx_vector, by_vector, bz_vector = t04_vectorized(parmod, ps, x, y, z)
    vector_time = time.time() - t0
    
    # Calculate errors
    dx = np.abs(bx_vector - bx_scalar)
    dy = np.abs(by_vector - by_scalar)
    dz = np.abs(bz_vector - bz_scalar)
    
    # Statistics
    print(f"Points tested: {n_points}")
    print(f"Scalar time: {scalar_time:.3f} s")
    print(f"Vector time: {vector_time:.3f} s")
    print(f"Speedup: {scalar_time/vector_time:.1f}x")
    
    print(f"\nAccuracy statistics:")
    print(f"  Bx differences: mean={np.mean(dx):.2e}, max={np.max(dx):.2e}")
    print(f"  By differences: mean={np.mean(dy):.2e}, max={np.max(dy):.2e}")
    print(f"  Bz differences: mean={np.mean(dz):.2e}, max={np.max(dz):.2e}")
    
    max_error = max(np.max(dx), np.max(dy), np.max(dz))
    tolerance = 1e-4
    
    if max_error < tolerance:
        print(f"\n✓ Accuracy test PASSED (max error: {max_error:.2e} < {tolerance})")
        return True
    else:
        print(f"\n✗ Accuracy test FAILED (max error: {max_error:.2e} > {tolerance})")
        return False

def test_edge_cases():
    """Test edge cases."""
    print("\n" + "="*60)
    print("Testing edge cases...")
    print("="*60)
    
    parmod = np.array([2.0, -20.0, 5.0, -5.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ps = 0.2
    
    # Test points
    test_cases = [
        ((0.0, 0.0, 0.0), "Origin"),
        ((10.0, 0.0, 0.0), "X-axis"),
        ((0.0, 10.0, 0.0), "Y-axis"),
        ((0.0, 0.0, 10.0), "Z-axis"),
        ((0.1, 0.1, 0.1), "Near origin"),
        ((-15.0, 0.0, 0.0), "Tail boundary")
    ]
    
    all_passed = True
    for (x, y, z), label in test_cases:
        try:
            bx_s, by_s, bz_s = t04_scalar(parmod, ps, x, y, z)
            bx_v, by_v, bz_v = t04_vectorized(parmod, ps, x, y, z)
            
            error = np.sqrt((bx_v-bx_s)**2 + (by_v-by_s)**2 + (bz_v-bz_s)**2)
            
            if np.isnan(error) and np.isnan(bx_s):
                # Both return NaN - this is expected for origin
                print(f"{label:15}: Both return NaN (expected)")
            elif error < 1e-6:
                print(f"{label:15}: Error = {error:.2e} ✓")
            else:
                print(f"{label:15}: Error = {error:.2e} ✗")
                all_passed = False
        except Exception as e:
            print(f"{label:15}: ERROR - {str(e)}")
            all_passed = False
    
    if all_passed:
        print("\n✓ Edge cases test PASSED")
    else:
        print("\n✗ Edge cases test FAILED")
    
    return all_passed

if __name__ == "__main__":
    print("T04 Vectorized Implementation Test")
    print("==================================")
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    tests_passed = 0
    total_tests = 4
    
    # Run tests
    if test_single_point():
        tests_passed += 1
    
    if test_array_interface():
        tests_passed += 1
    
    if test_accuracy():
        tests_passed += 1
    
    if test_edge_cases():
        tests_passed += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    if tests_passed == total_tests:
        print("✓ All tests PASSED!")
    else:
        print("✗ Some tests FAILED!")
    print("="*60)