"""
Test ONLY the tail current component.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopack import t01, geopack
from geopack.t01_vectorized import calculate_parameters


def test_tail_only():
    """Test only tail component."""
    print("TAIL COMPONENT ISOLATION TEST")
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
    
    # Call scalar version to set globals
    _ = t01.t01(parmod, ps, x, y, z)
    
    # Get scalar tail field directly
    pdyn = parmod[0]
    dst = parmod[1]
    xappa = (pdyn/2.)**a[38]
    xx = x * xappa
    yy = y * xappa
    zz = z * xappa
    
    # Get tail modes from scalar
    bxt1, byt1, bzt1, bxt2, byt2, bzt2 = t01.deformed(0, ps, xx, yy, zz)
    
    # Calculate amplitudes
    dlp1 = (pdyn/2)**a[41]
    dlp2 = (pdyn/2)**a[42]
    tamp1 = a[1] + a[2]*dlp1 + a[3]*parmod[4] + a[4]*dst
    tamp2 = a[5] + a[6]*dlp2 + a[7]*parmod[4] + a[8]*dst
    
    # Scalar tail field
    bx_tail_scalar = tamp1*bxt1 + tamp2*bxt2
    by_tail_scalar = tamp1*byt1 + tamp2*byt2
    bz_tail_scalar = tamp1*bzt1 + tamp2*bzt2
    
    print(f"Scalar tail field:")
    print(f"  Bx={bx_tail_scalar:.6f}, By={by_tail_scalar:.6f}, Bz={bz_tail_scalar:.6f}")
    
    # Get vectorized tail field
    from geopack.t01_vectorized import deformed_vectorized
    params = calculate_parameters(parmod, ps, a, 1)
    
    bxt1_v, byt1_v, bzt1_v, bxt2_v, byt2_v, bzt2_v = deformed_vectorized(
        0, ps, np.array([xx]), np.array([yy]), np.array([zz]),
        params.dxshift1, params.dxshift2, params.d, params.deltady,
        params.g, params.rh0
    )
    
    # Vectorized tail field with same amplitudes
    bx_tail_vec = tamp1*bxt1_v[0] + tamp2*bxt2_v[0]
    by_tail_vec = tamp1*byt1_v[0] + tamp2*byt2_v[0]
    bz_tail_vec = tamp1*bzt1_v[0] + tamp2*bzt2_v[0]
    
    print(f"\nVectorized tail field:")
    print(f"  Bx={bx_tail_vec:.6f}, By={by_tail_vec:.6f}, Bz={bz_tail_vec:.6f}")
    
    print(f"\nDifferences:")
    print(f"  ΔBx={bx_tail_vec-bx_tail_scalar:.6f}")
    print(f"  ΔBy={by_tail_vec-by_tail_scalar:.6f}")
    print(f"  ΔBz={bz_tail_vec-bz_tail_scalar:.6f}")
    
    print(f"\nConclusion:")
    if abs(bz_tail_vec-bz_tail_scalar) < 0.001:
        print("  Tail component matches! The error is NOT in the tail field.")
    else:
        print("  Tail component has significant error. The bug is in deformed/warped/unwarped chain.")


if __name__ == "__main__":
    test_tail_only()