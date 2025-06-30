"""
Test ONLY the ring current component.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopack import t01
from geopack.t01_vectorized import calculate_parameters, full_rc_vectorized


def test_ring_current_only():
    """Test only ring current component."""
    print("RING CURRENT COMPONENT ISOLATION TEST")
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
    
    # Get scalar ring current field
    pdyn = parmod[0]
    dst = parmod[1]
    xappa = (pdyn/2.)**a[38]
    xx = x * xappa
    yy = y * xappa
    zz = z * xappa
    
    # Get ring current from scalar
    bxsrc, bysrc, bzsrc, bxprc, byprc, bzprc = t01.full_rc(0, ps, xx, yy, zz)
    
    # Calculate amplitudes
    a_src = a[9] + a[10]*dst + a[11]*np.sqrt(pdyn)
    a_prc = a[12] + a[13]*dst + a[14]*np.sqrt(pdyn)
    
    # Scalar ring current field
    bx_rc_scalar = a_src*bxsrc + a_prc*bxprc
    by_rc_scalar = a_src*bysrc + a_prc*byprc
    bz_rc_scalar = a_src*bzsrc + a_prc*bzprc
    
    print(f"Scalar ring current field:")
    print(f"  Bx={bx_rc_scalar:.6f}, By={by_rc_scalar:.6f}, Bz={bz_rc_scalar:.6f}")
    
    # Get vectorized ring current field
    params = calculate_parameters(parmod, ps, a, 1)
    
    bxsrc_v, bysrc_v, bzsrc_v, bxprc_v, byprc_v, bzprc_v = full_rc_vectorized(
        0, ps, np.array([xx]), np.array([yy]), np.array([zz]),
        params.sc_sy, params.sc_pr, params.phi
    )
    
    # Vectorized ring current field with same amplitudes
    bx_rc_vec = a_src*bxsrc_v[0] + a_prc*bxprc_v[0]
    by_rc_vec = a_src*bysrc_v[0] + a_prc*byprc_v[0]
    bz_rc_vec = a_src*bzsrc_v[0] + a_prc*bzprc_v[0]
    
    print(f"\nVectorized ring current field:")
    print(f"  Bx={bx_rc_vec:.6f}, By={by_rc_vec:.6f}, Bz={bz_rc_vec:.6f}")
    
    print(f"\nDifferences:")
    print(f"  ΔBx={bx_rc_vec-bx_rc_scalar:.6f}")
    print(f"  ΔBy={by_rc_vec-by_rc_scalar:.6f}")
    print(f"  ΔBz={bz_rc_vec-bz_rc_scalar:.6f}")
    
    # Also check the parameters
    print(f"\nRing current parameters:")
    print(f"  Scalar sc_sy={t01.sc_sy:.6f}, sc_pr={t01.sc_pr:.6f}, phi={t01.phi:.6f}")
    print(f"  Vector sc_sy={params.sc_sy:.6f}, sc_pr={params.sc_pr:.6f}, phi={params.phi:.6f}")
    
    print(f"\nConclusion:")
    if abs(bz_rc_vec-bz_rc_scalar) < 0.001:
        print("  Ring current component matches! The error is NOT in the ring current.")
    else:
        print("  Ring current component has significant error. The bug is in full_rc_vectorized chain.")


if __name__ == "__main__":
    test_ring_current_only()