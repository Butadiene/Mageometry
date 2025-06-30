#!/usr/bin/env python
"""
Quick accuracy test for T04 vectorized implementation.
"""

import numpy as np
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from geopack.t04 import t04 as t04_scalar
from geopack.t04_vectorized import t04_vectorized

def quick_accuracy_test():
    """Quick accuracy test across key parameter combinations."""
    print("\nT04 Vectorized Quick Accuracy Test")
    print("=" * 60)
    
    # Test parameters
    test_cases = [
        # (Pdyn, Dst, ByIMF, BzIMF, W1-W6, ps, label)
        ([2.0, -20.0, 0.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0, "Quiet time"),
        ([5.0, -50.0, 5.0, -10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.2, "Moderate storm"),
        ([10.0, -100.0, -10.0, -15.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], -0.2, "Strong storm"),
        ([1.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.3, "Northward IMF"),
        ([3.0, -30.0, 10.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], -0.3, "By dominated"),
    ]
    
    # Test points
    x = np.array([-15.0, -10.0, -5.0, 0.0, 5.0, 10.0])
    y = np.array([-5.0, -2.0, 0.0, 2.0, 5.0, 8.0])
    z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0, 5.0])
    
    max_error = 0.0
    
    for parmod, ps, label in test_cases:
        print(f"\nTesting {label}:")
        print(f"  Parameters: Pdyn={parmod[0]:.1f}, Dst={parmod[1]:.0f}, By={parmod[2]:.0f}, Bz={parmod[3]:.0f}")
        
        # Calculate with both versions
        bx_s = np.zeros_like(x)
        by_s = np.zeros_like(x)
        bz_s = np.zeros_like(x)
        
        for i in range(len(x)):
            bx_s[i], by_s[i], bz_s[i] = t04_scalar(parmod, ps, x[i], y[i], z[i])
        
        bx_v, by_v, bz_v = t04_vectorized(parmod, ps, x, y, z)
        
        # Calculate errors
        dx = np.abs(bx_v - bx_s)
        dy = np.abs(by_v - by_s)
        dz = np.abs(bz_v - bz_s)
        
        max_dx = np.max(dx)
        max_dy = np.max(dy)
        max_dz = np.max(dz)
        max_case_error = max(max_dx, max_dy, max_dz)
        max_error = max(max_error, max_case_error)
        
        print(f"  Max errors: Bx={max_dx:.2e}, By={max_dy:.2e}, Bz={max_dz:.2e}")
        
        # Show a sample point
        i = 2  # Middle point
        print(f"  Sample at ({x[i]:.0f},{y[i]:.0f},{z[i]:.0f}):")
        print(f"    Scalar: Bx={bx_s[i]:12.6f}, By={by_s[i]:12.6f}, Bz={bz_s[i]:12.6f}")
        print(f"    Vector: Bx={bx_v[i]:12.6f}, By={by_v[i]:12.6f}, Bz={bz_v[i]:12.6f}")
    
    # Performance test
    print("\n" + "=" * 60)
    print("Performance Test")
    print("=" * 60)
    
    n = 1000
    x_perf = np.random.uniform(-10, 10, n)
    y_perf = np.random.uniform(-10, 10, n)
    z_perf = np.random.uniform(-5, 5, n)
    parmod = np.array([2.0, -20.0, 5.0, -5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    ps = 0.2
    
    # Scalar timing (subset)
    t0 = time.time()
    for i in range(min(50, n)):
        t04_scalar(parmod, ps, x_perf[i], y_perf[i], z_perf[i])
    scalar_time = (time.time() - t0) * n / min(50, n)
    
    # Vector timing
    t0 = time.time()
    t04_vectorized(parmod, ps, x_perf, y_perf, z_perf)
    vector_time = time.time() - t0
    
    print(f"Points: {n}")
    print(f"Scalar time (estimated): {scalar_time:.3f} s")
    print(f"Vector time: {vector_time:.4f} s")
    print(f"Speedup: {scalar_time/vector_time:.1f}x")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Maximum error across all tests: {max_error:.2e} nT")
    
    if max_error < 1e-7:
        print("✓ EXCELLENT accuracy")
    elif max_error < 1e-5:
        print("✓ GOOD accuracy")
    else:
        print("✗ Accuracy needs improvement")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    quick_accuracy_test()