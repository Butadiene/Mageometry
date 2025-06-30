#!/usr/bin/env python
"""
Comprehensive verification of T04 vectorized implementation.
Tests accuracy against scalar version and measures performance.
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

def verify_accuracy_detailed():
    """Detailed accuracy verification across parameter space."""
    print("\n" + "="*70)
    print("T04 VECTORIZED - COMPREHENSIVE ACCURACY VERIFICATION")
    print("="*70)
    print(f"Started: {datetime.now()}")
    
    # Define comprehensive parameter ranges
    # T04 parameters: [Pdyn, Dst, ByIMF, BzIMF, W1, W2, W3, W4, W5, W6]
    test_cases = [
        # Quiet conditions
        ([1.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0, "Quiet magnetosphere"),
        ([2.0, -10.0, 2.0, 3.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0.1, "Quiet with IMF"),
        
        # Moderate activity
        ([3.0, -30.0, 5.0, -5.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 0.2, "Moderate storm"),
        ([4.0, -40.0, -5.0, -8.0, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7], -0.1, "Moderate with southward IMF"),
        
        # Strong storms
        ([5.0, -80.0, 10.0, -10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.3, "Strong storm"),
        ([8.0, -100.0, -15.0, -20.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5], -0.2, "Intense storm"),
        
        # Extreme conditions
        ([10.0, -200.0, 20.0, -30.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 0.35, "Extreme storm"),
        ([15.0, -300.0, -25.0, -40.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], -0.35, "Super storm"),
        
        # Variable W indices
        ([5.0, -50.0, 5.0, -10.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], 0.15, "Variable W indices"),
        ([3.0, -40.0, -8.0, -5.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5], -0.15, "Decreasing W indices"),
    ]
    
    # Test positions covering magnetosphere
    test_positions = [
        # Near Earth
        (1.5, 0.0, 0.0, "Dayside near Earth"),
        (-2.0, 0.0, 0.0, "Nightside near Earth"),
        (0.0, 2.0, 0.0, "Dusk near Earth"),
        (0.0, -2.0, 0.0, "Dawn near Earth"),
        
        # Middle magnetosphere
        (5.0, 0.0, 0.0, "Dayside middle"),
        (-8.0, 0.0, 0.0, "Nightside middle"),
        (0.0, 6.0, 0.0, "Dusk middle"),
        (0.0, -6.0, 0.0, "Dawn middle"),
        
        # Outer magnetosphere
        (8.0, 4.0, 2.0, "Dayside flank"),
        (-12.0, -3.0, 1.0, "Tail lobe"),
        (-10.0, 0.0, -3.0, "Tail current sheet"),
        (6.0, -5.0, 2.0, "Dawn flank"),
        
        # High latitude
        (3.0, 0.0, 5.0, "North polar"),
        (3.0, 0.0, -5.0, "South polar"),
        (-5.0, 2.0, 4.0, "North tail lobe"),
        
        # Near boundary (model valid to X=-15)
        (-14.0, 0.0, 0.0, "Near tail boundary"),
        (-14.0, 3.0, 1.0, "Tail boundary flank"),
    ]
    
    # Statistics storage
    all_errors = []
    max_abs_error = 0.0
    max_rel_error = 0.0
    error_locations = []
    
    print("\nTesting accuracy across parameter space...")
    print("-" * 70)
    
    for parmod, ps, param_label in test_cases:
        parmod_array = np.array(parmod)
        param_errors = []
        
        for x, y, z, pos_label in test_positions:
            # Calculate with both versions
            bx_s, by_s, bz_s = t04_scalar(parmod_array, ps, x, y, z)
            bx_v, by_v, bz_v = t04_vectorized(parmod_array, ps, x, y, z)
            
            # Calculate errors
            dx = abs(bx_v - bx_s)
            dy = abs(by_v - by_s)
            dz = abs(bz_v - bz_s)
            
            # Total error magnitude
            error_mag = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Field magnitude
            b_mag = np.sqrt(bx_s**2 + by_s**2 + bz_s**2)
            
            # Relative error (only where field is significant)
            if b_mag > 1.0:
                rel_error = error_mag / b_mag
            else:
                rel_error = 0.0
            
            # Store results
            param_errors.append(error_mag)
            all_errors.append(error_mag)
            
            # Track maximum errors
            if error_mag > max_abs_error:
                max_abs_error = error_mag
                max_error_info = (param_label, pos_label, x, y, z, error_mag)
            
            if rel_error > max_rel_error:
                max_rel_error = rel_error
                max_rel_error_info = (param_label, pos_label, x, y, z, rel_error, b_mag)
            
            # Report significant errors
            if error_mag > 1e-6:
                error_locations.append((param_label, pos_label, error_mag))
        
        # Summary for this parameter set
        mean_error = np.mean(param_errors)
        max_param_error = np.max(param_errors)
        print(f"{param_label:20s}: mean={mean_error:.2e}, max={max_param_error:.2e}")
    
    # Overall statistics
    all_errors = np.array(all_errors)
    mean_error = np.mean(all_errors)
    median_error = np.median(all_errors)
    std_error = np.std(all_errors)
    percentile_95 = np.percentile(all_errors, 95)
    percentile_99 = np.percentile(all_errors, 99)
    
    print("\n" + "="*70)
    print("ACCURACY SUMMARY")
    print("="*70)
    print(f"Total comparisons: {len(all_errors)}")
    print(f"\nAbsolute error statistics (nT):")
    print(f"  Mean:     {mean_error:.2e}")
    print(f"  Median:   {median_error:.2e}")
    print(f"  Std Dev:  {std_error:.2e}")
    print(f"  95%ile:   {percentile_95:.2e}")
    print(f"  99%ile:   {percentile_99:.2e}")
    print(f"  Maximum:  {max_abs_error:.2e}")
    
    print(f"\nMaximum absolute error location:")
    print(f"  Parameters: {max_error_info[0]}")
    print(f"  Position: {max_error_info[1]} at ({max_error_info[2]}, {max_error_info[3]}, {max_error_info[4]})")
    print(f"  Error: {max_error_info[5]:.2e} nT")
    
    print(f"\nMaximum relative error: {max_rel_error:.2e}")
    if max_rel_error > 0:
        print(f"  Parameters: {max_rel_error_info[0]}")
        print(f"  Position: {max_rel_error_info[1]} at ({max_rel_error_info[2]}, {max_rel_error_info[3]}, {max_rel_error_info[4]})")
        print(f"  Relative error: {max_rel_error_info[5]:.2e}")
        print(f"  Field magnitude: {max_rel_error_info[6]:.1f} nT")
    
    # Accuracy verdict
    print("\n" + "-"*70)
    if max_abs_error < 1e-10:
        print("✓ EXCELLENT: Machine precision accuracy")
    elif max_abs_error < 1e-7:
        print("✓ EXCELLENT: Sub-nanotesla accuracy")
    elif max_abs_error < 1e-5:
        print("✓ GOOD: Acceptable for most applications")
    elif max_abs_error < 1e-3:
        print("⚠ FAIR: May need improvement for some applications")
    else:
        print("✗ POOR: Significant accuracy issues")
    
    return max_abs_error, mean_error, max_rel_error

def verify_performance():
    """Comprehensive performance verification."""
    print("\n" + "="*70)
    print("T04 VECTORIZED - PERFORMANCE VERIFICATION")
    print("="*70)
    
    # Test parameters
    parmod = np.array([5.0, -50.0, 5.0, -10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    ps = 0.2
    
    # Test different array sizes
    sizes = [1, 10, 100, 1000, 10000]
    results = []
    
    print("\nPerformance scaling test:")
    print("-" * 70)
    print(f"{'Size':>6} | {'Scalar (s)':>10} | {'Vector (s)':>10} | {'Speedup':>8} | {'Rate (pts/s)':>12}")
    print("-" * 70)
    
    for n in sizes:
        # Generate random positions
        np.random.seed(42)
        x = np.random.uniform(-10, 8, n)
        y = np.random.uniform(-8, 8, n)
        z = np.random.uniform(-5, 5, n)
        
        # Time scalar version
        if n <= 100:
            t0 = time.time()
            for i in range(n):
                t04_scalar(parmod, ps, x[i], y[i], z[i])
            scalar_time = time.time() - t0
        else:
            # Estimate from smaller sample
            n_sample = 50
            t0 = time.time()
            for i in range(n_sample):
                t04_scalar(parmod, ps, x[i], y[i], z[i])
            scalar_time = (time.time() - t0) * n / n_sample
        
        # Time vectorized version
        t0 = time.time()
        t04_vectorized(parmod, ps, x, y, z)
        vector_time = time.time() - t0
        
        # Calculate metrics
        speedup = scalar_time / vector_time
        rate = n / vector_time
        
        results.append((n, scalar_time, vector_time, speedup, rate))
        
        print(f"{n:6d} | {scalar_time:10.4f} | {vector_time:10.4f} | {speedup:8.1f}x | {rate:12.0f}")
    
    # Mixed input types test
    print("\n" + "-"*70)
    print("Mixed input types test:")
    n = 1000
    x_arr = np.random.uniform(-10, 8, n)
    y_scalar = 2.0
    z_arr = np.random.uniform(-5, 5, n)
    
    t0 = time.time()
    bx, by, bz = t04_vectorized(parmod, ps, x_arr, y_scalar, z_arr)
    mixed_time = time.time() - t0
    
    print(f"Mixed inputs (array, scalar, array): {n} points in {mixed_time:.4f}s")
    print(f"Processing rate: {n/mixed_time:.0f} points/second")
    
    # Memory efficiency test
    print("\n" + "-"*70)
    print("Memory efficiency test:")
    n = 100000
    x = np.random.uniform(-10, 8, n)
    y = np.random.uniform(-8, 8, n)
    z = np.random.uniform(-5, 5, n)
    
    t0 = time.time()
    bx, by, bz = t04_vectorized(parmod, ps, x, y, z)
    large_time = time.time() - t0
    
    print(f"Large array ({n:,} points): {large_time:.3f}s")
    print(f"Processing rate: {n/large_time:,.0f} points/second")
    
    return results

def verify_edge_cases():
    """Test edge cases and special conditions."""
    print("\n" + "="*70)
    print("T04 VECTORIZED - EDGE CASE VERIFICATION")
    print("="*70)
    
    parmod = np.array([2.0, -30.0, 5.0, -5.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    ps = 0.2
    
    test_cases = [
        # Special positions
        ((0.0, 0.0, 0.0), "Origin"),
        ((1e-10, 1e-10, 1e-10), "Near origin"),
        ((100.0, 0.0, 0.0), "Far dayside"),
        ((-14.99, 0.0, 0.0), "Just inside boundary"),
        ((0.0, 50.0, 0.0), "Far dusk"),
        ((0.0, 0.0, 50.0), "Far north"),
        
        # Arrays with special values
        (np.array([0.0, 1.0, -1.0]), "Array with zero"),
        (np.array([1e-10, 1e10, -1e10]), "Array with extremes"),
    ]
    
    all_passed = True
    
    for test_input, label in test_cases:
        try:
            if isinstance(test_input, tuple):
                x, y, z = test_input
                bx_s, by_s, bz_s = t04_scalar(parmod, ps, x, y, z)
                bx_v, by_v, bz_v = t04_vectorized(parmod, ps, x, y, z)
                
                # Check for NaN consistency
                if np.isnan(bx_s):
                    if np.isnan(bx_v):
                        print(f"{label:20s}: Both return NaN ✓")
                    else:
                        print(f"{label:20s}: Scalar NaN, vector {bx_v:.2e} ✗")
                        all_passed = False
                else:
                    error = np.sqrt((bx_v-bx_s)**2 + (by_v-by_s)**2 + (bz_v-bz_s)**2)
                    if error < 1e-6:
                        print(f"{label:20s}: Error = {error:.2e} ✓")
                    else:
                        print(f"{label:20s}: Error = {error:.2e} ✗")
                        all_passed = False
            else:
                # Array test
                x = test_input
                y = np.zeros_like(x)
                z = np.zeros_like(x)
                bx_v, by_v, bz_v = t04_vectorized(parmod, ps, x, y, z)
                print(f"{label:20s}: Returned shape {bx_v.shape} ✓")
                
        except Exception as e:
            print(f"{label:20s}: ERROR - {str(e)} ✗")
            all_passed = False
    
    return all_passed

def main():
    """Run all verification tests."""
    print("\nT04 VECTORIZED IMPLEMENTATION - COMPREHENSIVE VERIFICATION")
    print("="*70)
    print(f"Date: {datetime.now()}")
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # Run accuracy verification
    max_error, mean_error, max_rel_error = verify_accuracy_detailed()
    
    # Run performance verification
    performance_results = verify_performance()
    
    # Run edge case verification
    edge_cases_passed = verify_edge_cases()
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL VERIFICATION SUMMARY")
    print("="*70)
    
    print("\nAccuracy:")
    print(f"  Maximum absolute error: {max_error:.2e} nT")
    print(f"  Mean absolute error: {mean_error:.2e} nT")
    print(f"  Maximum relative error: {max_rel_error:.2e}")
    
    print("\nPerformance:")
    for n, st, vt, sp, rate in performance_results[-3:]:
        if n >= 1000:
            print(f"  {n:,} points: {sp:.1f}x speedup, {rate:,.0f} points/s")
    
    print("\nEdge cases:", "✓ All passed" if edge_cases_passed else "✗ Some failed")
    
    print("\n" + "-"*70)
    if max_error < 1e-7 and edge_cases_passed:
        print("✓ T04 VECTORIZED IMPLEMENTATION VERIFIED SUCCESSFULLY")
    else:
        print("⚠ VERIFICATION COMPLETED WITH WARNINGS")
    
    print(f"\nCompleted: {datetime.now()}")

if __name__ == "__main__":
    main()