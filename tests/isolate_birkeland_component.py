"""
Test ONLY the Birkeland current component.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopack import t01
from geopack.t01_vectorized import calculate_parameters, birk_tot_vectorized


def test_birkeland_only():
    """Test only Birkeland component."""
    print("BIRKELAND COMPONENT ISOLATION TEST")
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
    
    # Get scalar Birkeland field
    pdyn = parmod[0]
    xappa = (pdyn/2.)**a[38]
    xx = x * xappa
    yy = y * xappa
    zz = z * xappa
    
    # Get Birkeland from scalar
    bxr11, byr11, bzr11, bxr12, byr12, bzr12, bxr21, byr21, bzr21, bxr22, byr22, bzr22 = \
        t01.birk_tot(0, ps, xx, yy, zz)
    
    # Calculate amplitudes
    g2 = parmod[5]
    a_r11 = a[15] + a[16]*g2
    a_r12 = a[17] + a[18]*g2
    a_r21 = a[19] + a[20]*g2
    a_r22 = a[21] + a[22]*g2
    
    # Scalar Birkeland field
    bx_birk_scalar = a_r11*bxr11 + a_r12*bxr12 + a_r21*bxr21 + a_r22*bxr22
    by_birk_scalar = a_r11*byr11 + a_r12*byr12 + a_r21*byr21 + a_r22*byr22
    bz_birk_scalar = a_r11*bzr11 + a_r12*bzr12 + a_r21*bzr21 + a_r22*bzr22
    
    print(f"Scalar Birkeland field:")
    print(f"  Bx={bx_birk_scalar:.6f}, By={by_birk_scalar:.6f}, Bz={bz_birk_scalar:.6f}")
    
    # Get vectorized Birkeland field
    params = calculate_parameters(parmod, ps, a, 1)
    
    bxr11_v, byr11_v, bzr11_v, bxr12_v, byr12_v, bzr12_v, \
    bxr21_v, byr21_v, bzr21_v, bxr22_v, byr22_v, bzr22_v = birk_tot_vectorized(
        0, ps, np.array([xx]), np.array([yy]), np.array([zz]),
        params.xkappa1, params.xkappa2
    )
    
    # Vectorized Birkeland field with same amplitudes
    bx_birk_vec = a_r11*bxr11_v[0] + a_r12*bxr12_v[0] + a_r21*bxr21_v[0] + a_r22*bxr22_v[0]
    by_birk_vec = a_r11*byr11_v[0] + a_r12*byr12_v[0] + a_r21*byr21_v[0] + a_r22*byr22_v[0]
    bz_birk_vec = a_r11*bzr11_v[0] + a_r12*bzr12_v[0] + a_r21*bzr21_v[0] + a_r22*bzr22_v[0]
    
    print(f"\nVectorized Birkeland field:")
    print(f"  Bx={bx_birk_vec:.6f}, By={by_birk_vec:.6f}, Bz={bz_birk_vec:.6f}")
    
    print(f"\nDifferences:")
    print(f"  ΔBx={bx_birk_vec-bx_birk_scalar:.6f}")
    print(f"  ΔBy={by_birk_vec-by_birk_scalar:.6f}")
    print(f"  ΔBz={bz_birk_vec-bz_birk_scalar:.6f}")
    
    # Check parameters
    print(f"\nBirkeland parameters:")
    print(f"  Scalar xkappa1={t01.xkappa1:.6f}, xkappa2={t01.xkappa2:.6f}")
    print(f"  Vector xkappa1={params.xkappa1:.6f}, xkappa2={params.xkappa2:.6f}")
    
    print(f"\nConclusion:")
    if abs(bz_birk_vec-bz_birk_scalar) < 0.001:
        print("  Birkeland component matches! The error is NOT in the Birkeland field.")
    else:
        print("  Birkeland component has significant error. The bug is in birk_tot_vectorized chain.")


if __name__ == "__main__":
    test_birkeland_only()