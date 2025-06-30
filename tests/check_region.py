"""
Check which region our test point is in.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopack.t01_vectorized import calculate_parameters, iterate_sigma_vectorized_full


def check_region():
    """Check which region the test point is in."""
    # Parameters
    parmod = np.array([25.0, -300.0, -8.0, -10.0, 4.0, 3.0])
    ps = -0.2
    x, y, z = 0.0, -6.6, 0.0
    
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
    xappa = params.xappa
    
    # Constants for sigma calculation
    a0_a = 34.586
    a0_s0 = 1.1960
    a0_x0 = 3.4397
    dsig = 0.003
    
    x0 = a0_x0 / xappa
    am = a0_a / xappa
    s0 = a0_s0
    
    # Calculate sigma
    sps = np.sin(ps)
    xss, zss = iterate_sigma_vectorized_full(
        np.array([x]), np.array([y]), np.array([z]), sps, params.rh0, -5.2
    )
    
    rho2 = y**2 + zss[0]**2
    asq = am**2
    xmxm = am + xss[0] - x0
    xmxm = max(xmxm, 0)
    axx0 = xmxm**2
    aro = asq + rho2
    
    discriminant = (aro + axx0)**2 - 4.0 * asq * axx0
    discriminant = max(discriminant, 0)
    sigma = np.sqrt((aro + axx0 + np.sqrt(discriminant)) / (2.0 * asq))
    
    print("REGION DETERMINATION")
    print("=" * 50)
    print(f"Test point: ({x}, {y}, {z})")
    print(f"xappa = {xappa:.6f}")
    print(f"xss = {xss[0]:.6f}, zss = {zss[0]:.6f}")
    print(f"sigma = {sigma:.6f}")
    print(f"s0 = {s0:.6f}")
    print(f"dsig = {dsig:.6f}")
    print(f"\nRegion boundaries:")
    print(f"  INSIDE:   sigma < {s0 - dsig:.6f}")
    print(f"  BOUNDARY: {s0 - dsig:.6f} <= sigma < {s0 + dsig:.6f}")
    print(f"  OUTSIDE:  sigma >= {s0 + dsig:.6f}")
    
    if sigma < s0 - dsig:
        region = "INSIDE"
    elif sigma < s0 + dsig:
        region = "BOUNDARY"
    else:
        region = "OUTSIDE"
    
    print(f"\nThis point is in the {region} region")
    
    # If it's in the boundary layer, calculate the interpolation factors
    if region == "BOUNDARY":
        fint = 0.5 * (1.0 - (sigma - s0) / dsig)
        fext = 0.5 * (1.0 + (sigma - s0) / dsig)
        print(f"\nInterpolation factors:")
        print(f"  fint = {fint:.6f} (weight for internal field)")
        print(f"  fext = {fext:.6f} (weight for external field)")


if __name__ == "__main__":
    check_region()