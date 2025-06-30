"""
Compare scalar and vectorized tail calculations directly.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopack import t01
from geopack.t01_vectorized import deformed_vectorized, calculate_parameters


def compare_tail():
    """Compare tail calculations."""
    print("SCALAR VS VECTORIZED TAIL COMPARISON")
    print("=" * 80)
    
    # Test parameters
    ps = -0.1
    x = -10.0
    y = 0.0
    z = 0.0
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
    
    # Calculate xappa
    pdyn = parmod[0]
    xappa = (pdyn/2.)**a[38]
    
    # Scaled coordinates
    xx = x * xappa
    yy = y * xappa
    zz = z * xappa
    
    print(f"Coordinates:")
    print(f"  Unscaled: ({x}, {y}, {z})")
    print(f"  Scaled: ({xx:.6f}, {yy:.6f}, {zz:.6f})")
    print(f"  xappa = {xappa:.6f}")
    
    # Call scalar to set globals
    _ = t01.t01(parmod, ps, x, y, z)
    
    # Call scalar deformed with scaled coords (as scalar extall does)
    bx1_s, by1_s, bz1_s, bx2_s, by2_s, bz2_s = t01.deformed(0, ps, xx, yy, zz)
    
    print(f"\nScalar deformed (with scaled coords):")
    print(f"  Mode 1: Bz={bz1_s:.6f}")
    print(f"  Mode 2: Bz={bz2_s:.6f}")
    
    # Call vectorized deformed
    params = calculate_parameters(parmod, ps, a, 1)
    bx1_v, by1_v, bz1_v, bx2_v, by2_v, bz2_v = deformed_vectorized(
        0, ps, np.array([xx]), np.array([yy]), np.array([zz]),
        params.dxshift1, params.dxshift2, params.d, params.deltady,
        params.g, params.rh0
    )
    
    print(f"\nVectorized deformed (with scaled coords):")
    print(f"  Mode 1: Bz={bz1_v[0]:.6f}")
    print(f"  Mode 2: Bz={bz2_v[0]:.6f}")
    
    print(f"\nDifferences:")
    print(f"  Mode 1: {bz1_v[0] - bz1_s:.6f}")
    print(f"  Mode 2: {bz2_v[0] - bz2_s:.6f}")
    
    # Apply amplitudes
    dst = parmod[1]
    dlp1 = (pdyn/2)**a[41]
    dlp2 = (pdyn/2)**a[42]
    tamp1 = a[1] + a[2]*dlp1 + a[3]*parmod[4] + a[4]*dst
    tamp2 = a[5] + a[6]*dlp2 + a[7]*parmod[4] + a[8]*dst
    
    print(f"\nAmplitudes:")
    print(f"  tamp1 = {tamp1:.6f}")
    print(f"  tamp2 = {tamp2:.6f}")
    
    bz_tail_scalar = tamp1 * bz1_s + tamp2 * bz2_s
    bz_tail_vec = tamp1 * bz1_v[0] + tamp2 * bz2_v[0]
    
    print(f"\nTotal tail field:")
    print(f"  Scalar: {bz_tail_scalar:.6f}")
    print(f"  Vectorized: {bz_tail_vec:.6f}")
    print(f"  Difference: {bz_tail_vec - bz_tail_scalar:.6f}")


if __name__ == "__main__":
    compare_tail()