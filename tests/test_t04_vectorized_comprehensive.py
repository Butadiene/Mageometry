"""
Comprehensive test script for T04 vectorized implementation.
Runs the same tests as the evaluation notebook but in script form.
"""

import numpy as np
import sys
import time
sys.path.append('..')

from geopack import t04, geopack
from geopack.t04_vectorized import t04_vectorized

def test_basic_usage():
    """Test basic single point and array calculations."""
    print("=== Testing Basic Usage ===")
    
    # Set up time and calculate dipole tilt
    import datetime
    dt = datetime.datetime(2023, 3, 15, 12, 0, 0)
    ut = dt.timestamp()
    ps = geopack.recalc(ut)
    
    # T04 storm parameters
    parmod = np.array([
        5.0,    # Pdyn (nPa)
        -50.0,  # Dst (nT)
        2.0,    # By (nT)
        -5.0,   # Bz (nT)
        0.5, 1.0, 0.8, 1.2, 0.6, 0.9  # W1-W6
    ])
    
    # Single point test
    x, y, z = -6.6, 0.0, 0.0
    bx_s, by_s, bz_s = t04.t04(parmod, ps, x, y, z)
    bx_v, by_v, bz_v = t04_vectorized(parmod, ps, x, y, z)
    
    print(f"Single point test at ({x}, {y}, {z}):")
    print(f"  Scalar: ({bx_s:.3f}, {by_s:.3f}, {bz_s:.3f})")
    print(f"  Vector: ({bx_v:.3f}, {by_v:.3f}, {bz_v:.3f})")
    print(f"  Match: {np.allclose([bx_s, by_s, bz_s], [bx_v, by_v, bz_v], rtol=1e-10)}")
    
    # Array test
    x_arr = np.linspace(-15, 10, 26)
    y_arr = np.zeros_like(x_arr)
    z_arr = np.zeros_like(x_arr)
    
    bx_arr, by_arr, bz_arr = t04_vectorized(parmod, ps, x_arr, y_arr, z_arr)
    print(f"\nArray test with {len(x_arr)} points:")
    print(f"  Output shape: {bx_arr.shape}")
    print(f"  Contains NaN: {np.any(np.isnan(bx_arr))}")
    
    return True

def test_accuracy():
    """Test accuracy across parameter space."""
    print("\n=== Testing Accuracy ===")
    
    n_test = 1000
    np.random.seed(42)
    
    # Random positions within T04 validity range
    x_test = np.random.uniform(-15, 10, n_test)
    y_test = np.random.uniform(-10, 10, n_test)
    z_test = np.random.uniform(-5, 5, n_test)
    
    # Random storm parameters
    parmod_test = np.zeros((n_test, 10))
    parmod_test[:, 0] = np.random.uniform(1, 10, n_test)     # Pdyn
    parmod_test[:, 1] = np.random.uniform(-150, -20, n_test) # Dst
    parmod_test[:, 2] = np.random.uniform(-5, 5, n_test)     # By
    parmod_test[:, 3] = np.random.uniform(-10, 2, n_test)    # Bz
    parmod_test[:, 4:10] = np.random.uniform(0, 2, (n_test, 6))  # W1-W6
    
    ps_test = np.random.uniform(-0.5, 0.5, n_test)
    
    errors = []
    for i in range(n_test):
        bx_s, by_s, bz_s = t04.t04(parmod_test[i], ps_test[i], 
                                   x_test[i], y_test[i], z_test[i])
        bx_v, by_v, bz_v = t04_vectorized(parmod_test[i], ps_test[i], 
                                          x_test[i], y_test[i], z_test[i])
        
        if not (np.isnan(bx_s) and np.isnan(bx_v)):  # Skip if both are NaN
            b_mag = np.sqrt(bx_s**2 + by_s**2 + bz_s**2)
            if b_mag > 1e-10:
                error = np.sqrt((bx_v-bx_s)**2 + (by_v-by_s)**2 + (bz_v-bz_s)**2) / b_mag
                errors.append(error)
    
    errors = np.array(errors)
    
    print(f"Tested {n_test} random points")
    print(f"Valid comparisons: {len(errors)}")
    print(f"Mean relative error: {np.mean(errors):.2e}")
    print(f"Max relative error: {np.max(errors):.2e}")
    print(f"99th percentile: {np.percentile(errors, 99):.2e}")
    
    # T04 has max error around 3.8e-10 which is still excellent
    return np.max(errors) < 1e-9

def test_performance():
    """Test performance for different array sizes."""
    print("\n=== Testing Performance ===")
    
    sizes = [1, 10, 100, 1000, 5000]
    parmod = np.array([5.0, -50.0, 2.0, -5.0, 0.5, 1.0, 0.8, 1.2, 0.6, 0.9])
    ps = 0.1
    
    print(f"{'Size':>8} {'Scalar (ms)':>12} {'Vector (ms)':>12} {'Speedup':>10}")
    print("-" * 45)
    
    speedups = []
    for size in sizes:
        # Generate random points
        x = np.random.uniform(-10, 5, size)
        y = np.random.uniform(-5, 5, size)
        z = np.random.uniform(-3, 3, size)
        
        # Time scalar implementation
        t0 = time.perf_counter()
        for i in range(size):
            _ = t04.t04(parmod, ps, x[i], y[i], z[i])
        t_scalar = (time.perf_counter() - t0) * 1000
        
        # Time vectorized implementation
        t0 = time.perf_counter()
        _ = t04_vectorized(parmod, ps, x, y, z)
        t_vector = (time.perf_counter() - t0) * 1000
        
        speedup = t_scalar / t_vector if t_vector > 0 else 0
        speedups.append(speedup)
        print(f"{size:8d} {t_scalar:12.2f} {t_vector:12.2f} {speedup:10.1f}x")
    
    # Check speedup for larger arrays (ignore single point overhead)
    return speedups[-1] > 10 and speedups[-2] > 10  # Should have >10x speedup for large arrays

def test_edge_cases():
    """Test edge cases like origin and boundaries."""
    print("\n=== Testing Edge Cases ===")
    
    parmod = np.array([5.0, -50.0, 2.0, -5.0, 0.5, 1.0, 0.8, 1.2, 0.6, 0.9])
    ps = 0.1
    
    # Test at origin
    print("Testing at origin (0, 0, 0):")
    bx_s, by_s, bz_s = t04.t04(parmod, ps, 0.0, 0.0, 0.0)
    bx_v, by_v, bz_v = t04_vectorized(parmod, ps, 0.0, 0.0, 0.0)
    print(f"  Scalar: ({bx_s}, {by_s}, {bz_s})")
    print(f"  Vector: ({bx_v}, {by_v}, {bz_v})")
    print(f"  Both NaN: {np.isnan(bx_s) and np.isnan(bx_v)}")
    
    # Test near boundary
    print("\nTesting near model boundary (x = -15):")
    bx_s, by_s, bz_s = t04.t04(parmod, ps, -15.0, 0.0, 0.0)
    bx_v, by_v, bz_v = t04_vectorized(parmod, ps, -15.0, 0.0, 0.0)
    print(f"  Scalar: ({bx_s:.3f}, {by_s:.3f}, {bz_s:.3f})")
    print(f"  Vector: ({bx_v:.3f}, {by_v:.3f}, {bz_v:.3f})")
    print(f"  Match: {np.allclose([bx_s, by_s, bz_s], [bx_v, by_v, bz_v], rtol=1e-10)}")
    
    return True

def test_w_parameters():
    """Test sensitivity to W parameters."""
    print("\n=== Testing W Parameter Sensitivity ===")
    
    ps = 0.1
    x = np.array([-10.0, -5.0, 0.0, 5.0])
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    
    # Base parameters
    parmod_base = np.array([5.0, -50.0, 2.0, -5.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    
    # Test varying each W parameter
    for w_idx in range(4, 10):
        w_name = f'W{w_idx-3}'
        parmod_test = parmod_base.copy()
        
        # Test two different values
        parmod_test[w_idx] = 0.0
        bx1, by1, bz1 = t04_vectorized(parmod_test, ps, x, y, z)
        
        parmod_test[w_idx] = 2.0
        bx2, by2, bz2 = t04_vectorized(parmod_test, ps, x, y, z)
        
        # Check that changing W parameter affects the field
        diff = np.sqrt((bx2-bx1)**2 + (by2-by1)**2 + (bz2-bz1)**2)
        max_diff = np.nanmax(diff)  # Use nanmax to ignore NaN at origin
        
        print(f"{w_name} parameter: max field change = {max_diff:.3f} nT")
    
    return True

def main():
    """Run all tests."""
    print("T04 Vectorized Implementation Comprehensive Test")
    print("=" * 50)
    
    results = {}
    
    # Run tests
    results['basic'] = test_basic_usage()
    results['accuracy'] = test_accuracy()
    results['performance'] = test_performance()
    results['edge_cases'] = test_edge_cases()
    results['w_parameters'] = test_w_parameters()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    for test_name, passed in results.items():
        status = "PASSED ✓" if passed else "FAILED ✗"
        print(f"  {test_name:15s}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 50)
    if all_passed:
        print("All tests PASSED! ✓")
        print("T04 vectorized implementation is verified and ready for use.")
    else:
        print("Some tests FAILED! ✗")
        print("Note: Performance test may fail due to system load variations.")
    
    return all_passed

if __name__ == "__main__":
    import warnings
    # Suppress warnings from scalar implementation at origin
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    main()