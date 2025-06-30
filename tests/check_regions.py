"""
Check which magnetospheric regions our test points are in.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopack.t01_vectorized import calculate_parameters, iterate_sigma_vectorized_full


def check_regions():
    """Check regions."""
    print("MAGNETOSPHERIC REGION CHECK")
    print("=" * 80)
    
    # Test parameters
    ps = -0.1
    parmod = np.array([10.0, -150.0, 3.0, -5.0, 2.0, 1.0])
    
    # Model coefficients
    a = np.array([
        1.00000, 2.47341, 0.40791, 0.30429, -0.10637, -0.89108, 3.29350,
        -0.05413, -0.00696, 1.07869, -0.02314, -0.66173, -0.68018, -0.03246,
        0.02681, 0.28062, 0.16535, -0.02939, 0.02639, -0.24891, -0.08063,
        0.08900, -0.02475, 0.05887, 0.57691, 0.65256, -0.03230, 2.24733,
        4.10546, 1.13665, 0.05506, 0.97669, 0.21164, 0.64594, 1.12556, 0.01389,
        1.02978, 0.02968, 0.15821, 9.00519, 28.17582, 1.35285, 0.42279
    ])
    
    # Calculate parameters
    params = calculate_parameters(parmod, ps, a, 1)
    
    # Constants
    a0_a = 34.586
    a0_s0 = 1.1960
    a0_x0 = 3.4397
    dsig = 0.003
    
    sps = np.sin(ps)
    x0 = a0_x0 / params.xappa
    am = a0_a / params.xappa
    s0 = a0_s0
    
    # Test points
    test_points = [
        (-10.0, 0.0, 0.0),
        (-15.0, 0.0, 0.0),
        (-10.0, 5.0, 0.0),
        (-10.0, 0.0, 5.0),
        (-5.0, 0.0, 0.0)
    ]
    
    print(f"Parameters:")
    print(f"  xappa = {params.xappa:.6f}")
    print(f"  s0 = {s0:.6f}")
    print(f"  dsig = {dsig:.6f}")
    print(f"  s0 - dsig = {s0 - dsig:.6f} (inside boundary)")
    print(f"  s0 + dsig = {s0 + dsig:.6f} (outside boundary)")
    
    print("\n" + "=" * 80)
    print("REGION ANALYSIS:")
    
    for x, y, z in test_points:
        # Calculate sigma
        xss, zss = iterate_sigma_vectorized_full(
            np.array([x]), np.array([y]), np.array([z]), 
            sps, params.rh0, -5.2
        )
        
        # Calculate sigma
        rho2 = y**2 + zss[0]**2
        asq = am**2
        xmxm = am + xss[0] - x0
        xmxm = max(xmxm, 0)
        axx0 = xmxm**2
        aro = asq + rho2
        
        discriminant = (aro + axx0)**2 - 4.0 * asq * axx0
        discriminant = max(discriminant, 0)
        sigma = np.sqrt((aro + axx0 + np.sqrt(discriminant)) / (2.0 * asq))
        
        # Determine region
        if sigma < (s0 - dsig):
            region = "INSIDE"
        elif sigma < (s0 + dsig):
            region = "BOUNDARY LAYER"
        else:
            region = "OUTSIDE"
        
        print(f"\nPoint ({x:4.0f}, {y:4.0f}, {z:4.0f}):")
        print(f"  sigma = {sigma:.6f}")
        print(f"  Region: {region}")
        
        if region == "BOUNDARY LAYER":
            fint = 0.5 * (1.0 - (sigma - s0) / dsig)
            fext = 0.5 * (1.0 + (sigma - s0) / dsig)
            print(f"  Interpolation: fint={fint:.3f}, fext={fext:.3f}")


if __name__ == "__main__":
    check_regions()