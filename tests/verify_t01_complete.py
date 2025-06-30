"""
Comprehensive verification of T01 vectorized implementation.
Tests performance and accuracy under various conditions.
"""

import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopack import t01
from geopack.t01_vectorized import t01_vectorized, calculate_parameters, iterate_sigma_vectorized_full


def classify_point(x, y, z, parmod, ps):
    """Classify which magnetospheric region a point is in."""
    # Model coefficients
    a = np.array([
        1.00000, 2.47341, 0.40791, 0.30429, -0.10637, -0.89108, 3.29350,
        -0.05413, -0.00696, 1.07869, -0.02314, -0.66173, -0.68018, -0.03246,
        0.02681, 0.28062, 0.16535, -0.02939, 0.02639, -0.24891, -0.08063,
        0.08900, -0.02475, 0.05887, 0.57691, 0.65256, -0.03230, 2.24733,
        4.10546, 1.13665, 0.05506, 0.97669, 0.21164, 0.64594, 1.12556, 0.01389,
        1.02978, 0.02968, 0.15821, 9.00519, 28.17582, 1.35285, 0.42279
    ])
    
    params = calculate_parameters(parmod, ps, a, 1)
    xappa = params.xappa
    
    # Calculate sigma
    sps = np.sin(ps)
    xss, zss = iterate_sigma_vectorized_full(
        np.array([x]), np.array([y]), np.array([z]), sps, params.rh0, -5.2
    )
    
    # Constants
    a0_a, a0_s0, a0_x0 = 34.586, 1.1960, 3.4397
    dsig = 0.003
    
    x0 = a0_x0 / xappa
    am = a0_a / xappa
    s0 = a0_s0
    
    rho2 = y**2 + zss[0]**2
    asq = am**2
    xmxm = max(am + xss[0] - x0, 0)
    axx0 = xmxm**2
    aro = asq + rho2
    
    discriminant = max((aro + axx0)**2 - 4.0 * asq * axx0, 0)
    sigma = np.sqrt((aro + axx0 + np.sqrt(discriminant)) / (2.0 * asq))
    
    if sigma < s0 - dsig:
        return "INSIDE"
    elif sigma < s0 + dsig:
        return "BOUNDARY"
    else:
        return "OUTSIDE"


def verify_accuracy():
    """Verify accuracy under various parameter conditions."""
    print("T01 VECTORIZATION ACCURACY VERIFICATION")
    print("=" * 80)
    
    # Test parameter sets covering different magnetospheric conditions
    test_configs = [
        # (pdyn, dst, byimf, bzimf, g1, g2, ps, description)
        # Typical conditions
        (2.0, -20.0, 0.0, 2.0, 0.5, 0.5, 0.0, "Quiet time"),
        (4.0, -50.0, 2.0, -2.0, 1.0, 1.0, -0.05, "Moderate activity"),
        (10.0, -100.0, 3.0, -5.0, 2.0, 1.5, -0.1, "Storm time"),
        (15.0, -200.0, 5.0, -8.0, 3.0, 2.0, -0.15, "Intense storm"),
        
        # Extreme conditions
        (0.5, -10.0, 0.0, 5.0, 0.2, 0.2, 0.1, "Very quiet, northward IMF"),
        (25.0, -300.0, -8.0, -10.0, 4.0, 3.0, -0.2, "Extreme storm"),
        
        # Special cases
        (5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "No activity"),
        (8.0, -60.0, -5.0, 3.0, 1.5, 1.0, 0.15, "Dawn-dusk asymmetry"),
    ]
    
    # Representative test points
    test_points = [
        (-6.6, 0.0, 0.0, "Geosynchronous"),
        (-10.0, 0.0, 0.0, "Near tail"),
        (-15.0, 0.0, 2.0, "Plasma sheet"),
        (-4.0, 2.0, 1.0, "Ring current"),
        (8.0, 0.0, 0.0, "Dayside"),
        (-8.0, 5.0, 0.0, "Dawn flank"),
        (-8.0, -5.0, 0.0, "Dusk flank"),
        (5.0, 5.0, 5.0, "High latitude"),
        (-12.0, 0.0, -3.0, "Tail lobe"),
        (0.0, -6.6, 0.0, "Noon meridian"),
    ]
    
    # Accuracy statistics
    all_errors = []
    region_errors = {"INSIDE": [], "BOUNDARY": [], "OUTSIDE": []}
    
    print("\nDETAILED RESULTS:")
    print("-" * 80)
    
    for config_idx, config in enumerate(test_configs):
        pdyn, dst, byimf, bzimf, g1, g2, ps, desc = config
        parmod = np.array([pdyn, dst, byimf, bzimf, g1, g2])
        
        print(f"\nConfig {config_idx+1}: {desc}")
        print(f"  Parameters: pdyn={pdyn:.1f}, dst={dst:.0f}, IMF=({byimf:.1f},{bzimf:.1f})")
        
        config_errors = []
        
        for x, y, z, loc_desc in test_points:
            # Skip invalid points
            if x < -15:
                continue
                
            # Classify region
            region = classify_point(x, y, z, parmod, ps)
            
            # Calculate fields
            try:
                bx_s, by_s, bz_s = t01.t01(parmod, ps, x, y, z)
                bx_v, by_v, bz_v = t01_vectorized(parmod, ps, x, y, z)
                
                # Calculate error
                error = np.sqrt((bx_v - bx_s)**2 + (by_v - by_s)**2 + (bz_v - bz_s)**2)
                rel_error = error / np.sqrt(bx_s**2 + by_s**2 + bz_s**2) * 100
                
                config_errors.append(error)
                all_errors.append(error)
                region_errors[region].append(error)
                
                if error > 1.0:  # Only show larger errors
                    print(f"    {loc_desc:15s} ({region:8s}): {error:6.3f} nT ({rel_error:5.1f}%)")
                    
            except Exception as e:
                print(f"    {loc_desc:15s}: Error - {str(e)}")
        
        if config_errors:
            print(f"  Config summary: mean={np.mean(config_errors):.3f} nT, max={np.max(config_errors):.3f} nT")
    
    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL ACCURACY STATISTICS")
    print("-" * 40)
    print(f"Total test cases: {len(all_errors)}")
    print(f"Mean error: {np.mean(all_errors):.3f} nT")
    print(f"Median error: {np.median(all_errors):.3f} nT")
    print(f"95th percentile: {np.percentile(all_errors, 95):.3f} nT")
    print(f"Max error: {np.max(all_errors):.3f} nT")
    
    print("\nError distribution:")
    print(f"  < 0.1 nT: {sum(e < 0.1 for e in all_errors)} ({sum(e < 0.1 for e in all_errors)/len(all_errors)*100:.1f}%)")
    print(f"  < 0.5 nT: {sum(e < 0.5 for e in all_errors)} ({sum(e < 0.5 for e in all_errors)/len(all_errors)*100:.1f}%)")
    print(f"  < 1.0 nT: {sum(e < 1.0 for e in all_errors)} ({sum(e < 1.0 for e in all_errors)/len(all_errors)*100:.1f}%)")
    print(f"  < 5.0 nT: {sum(e < 5.0 for e in all_errors)} ({sum(e < 5.0 for e in all_errors)/len(all_errors)*100:.1f}%)")
    
    print("\nBy region:")
    for region in ["INSIDE", "BOUNDARY", "OUTSIDE"]:
        if region_errors[region]:
            print(f"  {region}: mean={np.mean(region_errors[region]):.3f} nT, "
                  f"max={np.max(region_errors[region]):.3f} nT, n={len(region_errors[region])}")


def verify_performance():
    """Verify performance improvements."""
    print("\n" + "=" * 80)
    print("PERFORMANCE VERIFICATION")
    print("-" * 40)
    
    # Standard parameters
    parmod = np.array([5.0, -50.0, 2.0, -3.0, 1.0, 1.0])
    ps = -0.1
    
    # Test different array sizes
    sizes = [1, 10, 100, 1000, 10000]
    
    print("Array Size   Scalar Time   Vector Time   Speedup   Points/sec")
    print("-" * 60)
    
    for n in sizes:
        # Generate random points
        np.random.seed(42)
        x = np.random.uniform(-15, 10, n)
        y = np.random.uniform(-10, 10, n)
        z = np.random.uniform(-5, 5, n)
        
        # Time scalar (sample for large arrays)
        n_scalar = min(n, 100)
        start = time.time()
        for i in range(n_scalar):
            t01.t01(parmod, ps, x[i], y[i], z[i])
        scalar_time = (time.time() - start) * n / n_scalar
        
        # Time vectorized
        start = time.time()
        t01_vectorized(parmod, ps, x, y, z)
        vector_time = time.time() - start
        
        # Calculate metrics
        speedup = scalar_time / vector_time
        points_per_sec = n / vector_time
        
        print(f"{n:10d}   {scalar_time:10.3f}s   {vector_time:10.3f}s   "
              f"{speedup:7.1f}x   {points_per_sec:10.0f}")


def verify_edge_cases():
    """Test edge cases and special conditions."""
    print("\n" + "=" * 80)
    print("EDGE CASE VERIFICATION")
    print("-" * 40)
    
    parmod = np.array([5.0, -50.0, 2.0, -3.0, 1.0, 1.0])
    ps = -0.1
    
    # Test cases
    edge_cases = [
        ("Origin", 0.0, 0.0, 0.0),
        ("Very near Earth", -1.0, 0.0, 0.0),
        ("Boundary limit x=-15", -15.0, 0.0, 0.0),
        ("High Z", -5.0, 0.0, 10.0),
        ("Large Y", -5.0, 20.0, 0.0),
        ("All components large", -10.0, 15.0, 8.0),
    ]
    
    print("Test Case              Scalar Result              Vectorized Result         Error")
    print("-" * 80)
    
    for desc, x, y, z in edge_cases:
        try:
            bx_s, by_s, bz_s = t01.t01(parmod, ps, x, y, z)
            bx_v, by_v, bz_v = t01_vectorized(parmod, ps, x, y, z)
            
            error = np.sqrt((bx_v - bx_s)**2 + (by_v - by_s)**2 + (bz_v - bz_s)**2)
            
            print(f"{desc:20s}   ({bx_s:7.2f},{by_s:7.2f},{bz_s:7.2f})   "
                  f"({bx_v:7.2f},{by_v:7.2f},{bz_v:7.2f})   {error:7.4f}")
        except Exception as e:
            print(f"{desc:20s}   Error: {str(e)}")


def main():
    """Run all verification tests."""
    verify_accuracy()
    verify_performance()
    verify_edge_cases()
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()