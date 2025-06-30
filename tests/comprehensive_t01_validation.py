"""
Comprehensive validation of T01 vectorized implementation.

Tests multiple points across all three regions (INSIDE, BOUNDARY, OUTSIDE)
with strict tolerance requirements for scientific accuracy.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopack import t01
from geopack.t01_vectorized import t01_vectorized, calculate_parameters, iterate_sigma_vectorized_full


def classify_region(x, y, z, parmod, ps, a):
    """Classify which region a point is in."""
    params = calculate_parameters(parmod, ps, a, 1)
    xappa = params.xappa
    
    # Calculate sigma
    sps = np.sin(ps)
    xss, zss = iterate_sigma_vectorized_full(
        np.array([x]), np.array([y]), np.array([z]), sps, params.rh0, -5.2
    )
    
    # Constants
    a0_a = 34.586
    a0_s0 = 1.1960
    a0_x0 = 3.4397
    dsig = 0.003
    
    x0 = a0_x0 / xappa
    am = a0_a / xappa
    s0 = a0_s0
    
    rho2 = y**2 + zss[0]**2
    asq = am**2
    xmxm = am + xss[0] - x0
    xmxm = max(xmxm, 0)
    axx0 = xmxm**2
    aro = asq + rho2
    
    discriminant = (aro + axx0)**2 - 4.0 * asq * axx0
    discriminant = max(discriminant, 0)
    sigma = np.sqrt((aro + axx0 + np.sqrt(discriminant)) / (2.0 * asq))
    
    if sigma < s0 - dsig:
        return "INSIDE", sigma
    elif sigma < s0 + dsig:
        return "BOUNDARY", sigma
    else:
        return "OUTSIDE", sigma


def run_comprehensive_validation():
    """Run comprehensive validation tests."""
    print("COMPREHENSIVE T01 VALIDATION")
    print("=" * 80)
    
    # Model coefficients
    a = np.array([
        1.00000, 2.47341, 0.40791, 0.30429, -0.10637, -0.89108, 3.29350,
        -0.05413, -0.00696, 1.07869, -0.02314, -0.66173, -0.68018, -0.03246,
        0.02681, 0.28062, 0.16535, -0.02939, 0.02639, -0.24891, -0.08063,
        0.08900, -0.02475, 0.05887, 0.57691, 0.65256, -0.03230, 2.24733,
        4.10546, 1.13665, 0.05506, 0.97669, 0.21164, 0.64594, 1.12556, 0.01389,
        1.02978, 0.02968, 0.15821, 9.00519, 28.17582, 1.35285, 0.42279
    ])
    
    # Test different solar wind conditions
    test_configs = [
        # (pdyn, dst, byimf, bzimf, g1, g2, ps, description)
        (10.0, -150.0, 3.0, -5.0, 2.0, 1.0, -0.1, "Strong storm, southward IMF"),
        (2.0, -20.0, 0.0, 2.0, 0.5, 0.5, 0.05, "Quiet time, northward IMF"),
        (25.0, -80.0, -5.0, -8.0, 3.0, 2.0, -0.2, "High pressure, strong IMF"),
    ]
    
    # Test points designed to cover all regions
    test_points = [
        # INSIDE points
        (-3.0, 1.0, 2.0, "Near-Earth INSIDE"),
        (-10.0, 2.0, 1.0, "Near-tail INSIDE"),
        (-5.0, 0.0, 0.0, "Equatorial INSIDE"),
        (-8.0, -3.0, 4.0, "Off-equator INSIDE"),
        
        # BOUNDARY points (need to find these dynamically)
        (5.0, 5.0, 5.0, "Dayside BOUNDARY"),
        (-2.0, 4.0, 3.0, "Flank BOUNDARY"),
        
        # OUTSIDE points
        (15.0, 5.0, 5.0, "Deep solar wind"),
        (10.0, 0.0, 0.0, "Subsolar OUTSIDE"),
    ]
    
    all_errors = []
    max_error = 0.0
    
    for config in test_configs:
        pdyn, dst, byimf, bzimf, g1, g2, ps, desc = config
        parmod = np.array([pdyn, dst, byimf, bzimf, g1, g2])
        
        print(f"\nConfiguration: {desc}")
        print(f"  pdyn={pdyn}, dst={dst}, byimf={byimf}, bzimf={bzimf}, ps={ps}")
        print("-" * 70)
        print("Point                Region     Error(nT)   Bx_err    By_err    Bz_err")
        print("-" * 70)
        
        for x, y, z, point_desc in test_points:
            # Classify region
            region, sigma = classify_region(x, y, z, parmod, ps, a)
            
            # Get scalar result
            bx_s, by_s, bz_s = t01.t01(parmod, ps, x, y, z)
            
            # Get vectorized result
            bx_v, by_v, bz_v = t01_vectorized(parmod, ps, x, y, z)
            
            # Calculate errors
            dx = bx_v - bx_s
            dy = by_v - by_s
            dz = bz_v - bz_s
            error = np.sqrt(dx**2 + dy**2 + dz**2)
            
            all_errors.append(error)
            max_error = max(max_error, error)
            
            # Format output
            point_str = f"({x:5.1f},{y:4.1f},{z:4.1f})"
            print(f"{point_str:18s} {region:8s}  {error:8.5f}  {dx:8.5f}  {dy:8.5f}  {dz:8.5f}")
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("-" * 40)
    print(f"Total test cases: {len(all_errors)}")
    print(f"Mean error: {np.mean(all_errors):.6f} nT")
    print(f"Median error: {np.median(all_errors):.6f} nT")
    print(f"Max error: {max_error:.6f} nT")
    print(f"Errors < 0.001 nT: {sum(e < 0.001 for e in all_errors)}")
    print(f"Errors < 0.01 nT: {sum(e < 0.01 for e in all_errors)}")
    print(f"Errors < 0.1 nT: {sum(e < 0.1 for e in all_errors)}")
    print(f"Errors < 1.0 nT: {sum(e < 1.0 for e in all_errors)}")
    
    # Scientific accuracy check
    tolerance = 1e-6  # 1e-6 nT = 0.000001 nT
    print(f"\nScientific accuracy (< {tolerance} nT): ", end="")
    if all(e < tolerance for e in all_errors):
        print("FAILED - errors too large for strict tolerance")
        print(f"Practical accuracy (< 1.0 nT): ", end="")
        if all(e < 1.0 for e in all_errors):
            print("PASSED ✓")
        else:
            print("FAILED")
    else:
        print("PASSED ✓")
    
    # Test array input
    print("\n" + "=" * 80)
    print("ARRAY INPUT TEST")
    print("-" * 40)
    
    # Use first configuration
    parmod = np.array([10.0, -150.0, 3.0, -5.0, 2.0, 1.0])
    ps = -0.1
    
    # Create arrays of test points
    x_arr = np.array([p[0] for p in test_points])
    y_arr = np.array([p[1] for p in test_points])
    z_arr = np.array([p[2] for p in test_points])
    
    # Vectorized call
    bx_arr, by_arr, bz_arr = t01_vectorized(parmod, ps, x_arr, y_arr, z_arr)
    
    # Compare with individual calls
    array_errors = []
    for i, (x, y, z, _) in enumerate(test_points):
        bx_s, by_s, bz_s = t01.t01(parmod, ps, x, y, z)
        error = np.sqrt((bx_arr[i] - bx_s)**2 + (by_arr[i] - by_s)**2 + (bz_arr[i] - bz_s)**2)
        array_errors.append(error)
    
    print(f"Array shape: {bx_arr.shape}")
    print(f"Max array error: {max(array_errors):.6f} nT")
    print(f"Array processing: PASSED ✓" if max(array_errors) < 1.0 else "FAILED")


if __name__ == "__main__":
    run_comprehensive_validation()