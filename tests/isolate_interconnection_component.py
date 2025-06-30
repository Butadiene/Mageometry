"""
Test ONLY the interconnection field component.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopack import t01
from geopack.t01_vectorized import calculate_parameters


def test_interconnection_only():
    """Test only interconnection component."""
    print("INTERCONNECTION COMPONENT ISOLATION TEST")
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
    
    # Get interconnection field parameters
    byimf = parmod[2]
    bzimf = parmod[3]
    
    # Calculate theta and sthetah
    if byimf == 0 and bzimf == 0:
        theta = 0.0
    else:
        theta = np.arctan2(byimf, bzimf)
        if theta <= 0:
            theta += 2 * np.pi
    
    sthetah = np.sin(theta / 2.0) ** 2
    
    # IMF components
    hximf = 0.0
    hyimf = byimf
    hzimf = bzimf
    
    # Scalar interconnection field
    bx_int_scalar = a[23] * hximf + a[24] * hximf * sthetah
    by_int_scalar = a[23] * hyimf + a[24] * hyimf * sthetah
    bz_int_scalar = a[23] * hzimf + a[24] * hzimf * sthetah
    
    print(f"Scalar interconnection field:")
    print(f"  Bx={bx_int_scalar:.6f}, By={by_int_scalar:.6f}, Bz={bz_int_scalar:.6f}")
    
    # The vectorized version should be identical
    print(f"\nVectorized interconnection field calculation:")
    print(f"  Should use same formula: a[23]*hzimf + a[24]*hzimf*sthetah")
    print(f"  Where hzimf = bzimf = {bzimf}")
    print(f"  sthetah = {sthetah:.6f}")
    print(f"  Result = {a[23]}*{bzimf} + {a[24]}*{bzimf}*{sthetah:.6f}")
    print(f"        = {bz_int_scalar:.6f}")
    
    # Get the actual vectorized value from extall_vectorized with iopgen=5
    from geopack.t01_vectorized import extall_vectorized
    params = calculate_parameters(parmod, ps, a, 1)
    pdyn = parmod[0]
    dst = parmod[1]
    dst_ast = dst * 0.8 - 13.0 * np.sqrt(pdyn)
    xappa = params.xappa
    xx = x * xappa
    yy = y * xappa
    zz = z * xappa
    
    _, _, bz_int_vec = extall_vectorized(5, 0, 0, 0, a, 43, pdyn, dst_ast,
                                         parmod[2], parmod[3], parmod[4], parmod[5],
                                         ps, np.array([xx]), np.array([yy]), np.array([zz]), params)
    
    print(f"\nVectorized interconnection field (from extall iopgen=5):")
    print(f"  Bz={bz_int_vec[0]:.6f}")
    
    print(f"\nDifference:")
    print(f"  ΔBz={bz_int_vec[0]-bz_int_scalar:.6f}")
    
    print(f"\nConclusion:")
    if abs(bz_int_vec[0]-bz_int_scalar) < 0.001:
        print("  Interconnection component matches!")
    else:
        print("  Interconnection component has error.")


if __name__ == "__main__":
    test_interconnection_only()