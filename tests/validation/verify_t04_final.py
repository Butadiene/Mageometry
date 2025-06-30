#!/usr/bin/env python
"""
Final verification of T04 vectorized implementation accuracy and performance.
"""

import numpy as np
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from geopack.t04 import t04 as t04_scalar
from geopack.t04_vectorized import t04_vectorized

def verify_accuracy():
    """Verify accuracy across typical use cases."""
    print("\nT04 VECTORIZED - ACCURACY VERIFICATION")
    print("="*60)
    
    # Representative test cases
    test_scenarios = [
        # (parmod, ps, label)
        ([2.0, -20.0, 0.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0, "Quiet time"),
        ([5.0, -50.0, 5.0, -10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.2, "Moderate storm"),
        ([10.0, -100.0, -10.0, -20.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], -0.2, "Strong storm"),
        ([3.0, -30.0, 10.0, 0.0, 0.5, 1.0, 1.5, 0.5, 1.0, 1.5], 0.3, "Variable W indices"),
    ]
    
    # Generate test grid
    n_points = 1000
    np.random.seed(42)
    x = np.random.uniform(-14, 10, n_points)
    y = np.random.uniform(-10, 10, n_points)
    z = np.random.uniform(-5, 5, n_points)
    
    max_errors = []
    mean_errors = []
    
    for parmod, ps, label in test_scenarios:
        parmod = np.array(parmod)
        
        # Calculate with scalar version
        bx_scalar = np.zeros(n_points)
        by_scalar = np.zeros(n_points)
        bz_scalar = np.zeros(n_points)
        
        for i in range(n_points):
            bx_scalar[i], by_scalar[i], bz_scalar[i] = t04_scalar(parmod, ps, x[i], y[i], z[i])
        
        # Calculate with vectorized version
        bx_vector, by_vector, bz_vector = t04_vectorized(parmod, ps, x, y, z)
        
        # Calculate errors
        dx = np.abs(bx_vector - bx_scalar)
        dy = np.abs(by_vector - by_scalar)
        dz = np.abs(bz_vector - bz_scalar)
        
        # Error statistics
        max_dx = np.max(dx)
        max_dy = np.max(dy)
        max_dz = np.max(dz)
        max_error = max(max_dx, max_dy, max_dz)
        
        mean_dx = np.mean(dx)
        mean_dy = np.mean(dy)
        mean_dz = np.mean(dz)
        mean_error = np.mean([mean_dx, mean_dy, mean_dz])
        
        max_errors.append(max_error)
        mean_errors.append(mean_error)
        
        print(f"\n{label}:")
        print(f"  Max errors: Bx={max_dx:.2e}, By={max_dy:.2e}, Bz={max_dz:.2e}")
        print(f"  Mean errors: Bx={mean_dx:.2e}, By={mean_dy:.2e}, Bz={mean_dz:.2e}")
        
        # Check relative errors where field is significant
        b_mag = np.sqrt(bx_scalar**2 + by_scalar**2 + bz_scalar**2)
        significant = b_mag > 10.0
        if np.any(significant):
            error_mag = np.sqrt(dx[significant]**2 + dy[significant]**2 + dz[significant]**2)
            rel_errors = error_mag / b_mag[significant]
            max_rel = np.max(rel_errors)
            mean_rel = np.mean(rel_errors)
            print(f"  Relative errors (|B|>10nT): max={max_rel:.2e}, mean={mean_rel:.2e}")
    
    # Overall summary
    overall_max = max(max_errors)
    overall_mean = np.mean(mean_errors)
    
    print("\n" + "-"*60)
    print("OVERALL ACCURACY:")
    print(f"  Maximum error across all tests: {overall_max:.2e} nT")
    print(f"  Mean error across all tests: {overall_mean:.2e} nT")
    
    if overall_max < 1e-7:
        print("  Status: ✓ EXCELLENT (sub-nanotesla accuracy)")
    elif overall_max < 1e-5:
        print("  Status: ✓ GOOD (suitable for production)")
    else:
        print("  Status: ⚠ FAIR (may need improvement)")
    
    return overall_max

def verify_performance():
    """Verify performance improvements."""
    print("\n" + "="*60)
    print("T04 VECTORIZED - PERFORMANCE VERIFICATION")
    print("="*60)
    
    # Test parameters
    parmod = np.array([5.0, -50.0, 5.0, -10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    ps = 0.2
    
    # Test with increasing array sizes
    print("\nPerformance scaling:")
    print("-"*60)
    print(f"{'N points':>10} | {'Scalar (s)':>10} | {'Vector (s)':>10} | {'Speedup':>8}")
    print("-"*60)
    
    speedups = []
    for n in [10, 100, 1000, 5000]:
        x = np.random.uniform(-10, 10, n)
        y = np.random.uniform(-10, 10, n)
        z = np.random.uniform(-5, 5, n)
        
        # Time scalar version (sample for large n)
        n_sample = min(50, n)
        t0 = time.time()
        for i in range(n_sample):
            t04_scalar(parmod, ps, x[i], y[i], z[i])
        scalar_time = (time.time() - t0) * n / n_sample
        
        # Time vectorized version
        t0 = time.time()
        t04_vectorized(parmod, ps, x, y, z)
        vector_time = time.time() - t0
        
        speedup = scalar_time / vector_time
        speedups.append(speedup)
        
        print(f"{n:10d} | {scalar_time:10.3f} | {vector_time:10.4f} | {speedup:7.1f}x")
    
    # Single vs array comparison
    print("\n" + "-"*60)
    print("Single point vs array efficiency:")
    
    # Single point
    t0 = time.time()
    for _ in range(100):
        t04_vectorized(parmod, ps, 5.0, 2.0, 1.0)
    single_time = (time.time() - t0) / 100
    
    # Array of 100 points
    x = np.random.uniform(-10, 10, 100)
    y = np.random.uniform(-10, 10, 100)
    z = np.random.uniform(-5, 5, 100)
    t0 = time.time()
    t04_vectorized(parmod, ps, x, y, z)
    array_time = time.time() - t0
    
    print(f"  Single point: {single_time*1000:.2f} ms")
    print(f"  100 points: {array_time*1000:.2f} ms ({array_time/single_time:.1f}x single point time)")
    print(f"  Efficiency: {100*single_time/array_time:.1f}x faster than 100 single calls")
    
    avg_speedup = np.mean(speedups)
    print(f"\nAverage speedup: {avg_speedup:.1f}x")
    
    return avg_speedup

def verify_interface():
    """Verify interface compatibility."""
    print("\n" + "="*60)
    print("T04 VECTORIZED - INTERFACE VERIFICATION")
    print("="*60)
    
    parmod = np.array([2.0, -30.0, 5.0, -5.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    ps = 0.2
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Scalar inputs return scalars
    print("\n1. Scalar inputs → scalar outputs:")
    bx, by, bz = t04_vectorized(parmod, ps, 5.0, 2.0, 1.0)
    if isinstance(bx, float) and isinstance(by, float) and isinstance(bz, float):
        print("   ✓ Returns scalar floats")
        tests_passed += 1
    else:
        print("   ✗ Should return scalars")
    
    # Test 2: Array inputs return arrays
    print("\n2. Array inputs → array outputs:")
    x = np.array([5.0, 6.0, 7.0])
    y = np.array([2.0, 3.0, 4.0])
    z = np.array([1.0, 1.5, 2.0])
    bx, by, bz = t04_vectorized(parmod, ps, x, y, z)
    if isinstance(bx, np.ndarray) and bx.shape == (3,):
        print("   ✓ Returns arrays with correct shape")
        tests_passed += 1
    else:
        print("   ✗ Should return arrays")
    
    # Test 3: Mixed inputs (broadcasting)
    print("\n3. Mixed inputs (broadcasting):")
    bx, by, bz = t04_vectorized(parmod, ps, x, 2.0, z)
    if isinstance(bx, np.ndarray) and bx.shape == (3,):
        print("   ✓ Broadcasting works correctly")
        tests_passed += 1
    else:
        print("   ✗ Broadcasting failed")
    
    # Test 4: 2D arrays
    print("\n4. 2D array inputs:")
    x_2d = np.random.randn(5, 10)
    y_2d = np.random.randn(5, 10)
    z_2d = np.random.randn(5, 10)
    bx, by, bz = t04_vectorized(parmod, ps, x_2d, y_2d, z_2d)
    if bx.shape == (5, 10):
        print("   ✓ Preserves 2D shape")
        tests_passed += 1
    else:
        print("   ✗ Shape preservation failed")
    
    # Test 5: Consistency with scalar version
    print("\n5. Consistency with scalar version:")
    x, y, z = 5.0, 2.0, 1.0
    bx_s, by_s, bz_s = t04_scalar(parmod, ps, x, y, z)
    bx_v, by_v, bz_v = t04_vectorized(parmod, ps, x, y, z)
    error = np.sqrt((bx_v-bx_s)**2 + (by_v-by_s)**2 + (bz_v-bz_s)**2)
    if error < 1e-10:
        print(f"   ✓ Results match (error: {error:.2e})")
        tests_passed += 1
    else:
        print(f"   ✗ Results differ (error: {error:.2e})")
    
    print(f"\nInterface tests passed: {tests_passed}/{total_tests}")
    return tests_passed == total_tests

def main():
    """Run all verification tests."""
    print("\n" + "="*60)
    print("T04 VECTORIZED IMPLEMENTATION VERIFICATION")
    print("="*60)
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings('ignore')
    
    # Run verifications
    max_error = verify_accuracy()
    avg_speedup = verify_performance()
    interface_ok = verify_interface()
    
    # Final summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    print(f"\n✓ Accuracy: Maximum error = {max_error:.2e} nT")
    print(f"✓ Performance: Average speedup = {avg_speedup:.1f}x")
    print(f"✓ Interface: {'All tests passed' if interface_ok else 'Some tests failed'}")
    
    if max_error < 1e-7 and avg_speedup > 10 and interface_ok:
        print("\n✅ T04 VECTORIZED IMPLEMENTATION VERIFIED SUCCESSFULLY")
        print("   Ready for production use")
    else:
        print("\n⚠️  Some aspects need attention")

if __name__ == "__main__":
    main()