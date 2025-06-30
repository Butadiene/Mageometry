"""
Check the final field calculation in extall_vectorized.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopack import t01
from geopack.t01_vectorized import (extall_vectorized, calculate_parameters, 
                                    iterate_sigma_vectorized_full)


def check_final_calculation():
    """Check final field calculation."""
    print("FINAL FIELD CALCULATION CHECK")
    print("=" * 80)
    
    # Test parameters
    parmod = np.array([10.0, -150.0, 3.0, -5.0, 2.0, 1.0])
    ps = -0.1
    x = -10.0
    y = 0.0
    z = 0.0
    
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
    
    # Check region determination
    print("REGION DETERMINATION:")
    print("-" * 40)
    
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
    x_arr = np.array([x])
    y_arr = np.array([y])
    z_arr = np.array([z])
    
    xss, zss = iterate_sigma_vectorized_full(x_arr, y_arr, z_arr, sps, params.rh0, -5.2)
    
    rho2 = y**2 + zss[0]**2
    asq = am**2
    xmxm = am + xss[0] - x0
    xmxm = max(xmxm, 0)
    axx0 = xmxm**2
    aro = asq + rho2
    
    discriminant = (aro + axx0)**2 - 4.0 * asq * axx0
    discriminant = max(discriminant, 0)
    sigma = np.sqrt((aro + axx0 + np.sqrt(discriminant)) / (2.0 * asq))
    
    print(f"Test point: x={x}, y={y}, z={z}")
    print(f"xss={xss[0]:.6f}, zss={zss[0]:.6f}")
    print(f"sigma={sigma:.6f}, s0={s0:.6f}")
    print(f"s0-dsig={s0-dsig:.6f}, s0+dsig={s0+dsig:.6f}")
    
    if sigma < (s0 - dsig):
        region = "INSIDE"
        print(f"Region: INSIDE (sigma < s0-dsig)")
        print("Should return external field only")
    elif sigma < (s0 + dsig):
        region = "BOUNDARY"
        print(f"Region: BOUNDARY LAYER")
        print("Should interpolate between external+dipole and IMF")
    else:
        region = "OUTSIDE"
        print(f"Region: OUTSIDE (sigma >= s0+dsig)")
        print("Should return IMF - dipole")
    
    # Check what the scalar T01 returns
    print("\n" + "=" * 80)
    print("SCALAR T01 BEHAVIOR:")
    print("-" * 40)
    
    bx_s, by_s, bz_s = t01.t01(parmod, ps, x, y, z)
    print(f"Scalar T01: Bz={bz_s:.6f}")
    
    # Get dipole field
    qx, qy, qz = t01.dipole(ps, x, y, z)
    print(f"Dipole field: Bz={qz:.6f}")
    
    if region == "INSIDE":
        print(f"\nFor INSIDE region:")
        print(f"  T01 should return external field only")
        print(f"  External field = T01 output = {bz_s:.6f}")
        print(f"  Total field = T01 + dipole = {bz_s + qz:.6f}")
    
    # Check vectorized behavior
    print("\n" + "=" * 80)
    print("VECTORIZED BEHAVIOR:")
    print("-" * 40)
    
    from geopack.t01_vectorized import t01_vectorized
    bx_v, by_v, bz_v = t01_vectorized(parmod, ps, x, y, z)
    print(f"Vectorized T01: Bz={bz_v:.6f}")
    
    print(f"\nComparison:")
    print(f"  Scalar: Bz={bz_s:.6f}")
    print(f"  Vectorized: Bz={bz_v:.6f}")
    print(f"  Error: ΔBz={bz_v - bz_s:.6f}")
    
    # The issue might be that scalar and vectorized handle regions differently
    print("\n" + "=" * 80)
    print("HYPOTHESIS:")
    print("Both implementations should return the same thing for INSIDE region:")
    print("External field only (no dipole)")
    print(f"But we see a {bz_v - bz_s:.6f} nT difference")
    
    # Check the individual external field components
    print("\n" + "=" * 80)
    print("EXTERNAL FIELD COMPONENTS:")
    
    xx = x * xappa
    yy = y * xappa
    zz = z * xappa
    xx_arr = np.array([xx])
    yy_arr = np.array([yy])
    zz_arr = np.array([zz])
    
    pdyn = parmod[0]
    dst = parmod[1]
    dst_ast = dst * 0.8 - 13.0 * np.sqrt(pdyn)
    
    _, _, bz_ext = extall_vectorized(0, 0, 0, 0, a, 43, pdyn, dst_ast,
                                     parmod[2], parmod[3], parmod[4], parmod[5],
                                     ps, xx_arr, yy_arr, zz_arr, params)
    
    print(f"Vectorized external field (all components): Bz={bz_ext[0]:.6f}")
    print(f"This should match the vectorized T01 output for INSIDE region")
    print(f"Difference: {bz_ext[0] - bz_v:.6f}")


if __name__ == "__main__":
    check_final_calculation()