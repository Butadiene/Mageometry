"""
Test ONLY the ring current field by modifying both scalar and vectorized versions.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopack import t01
from geopack.t01_vectorized import extall_vectorized, calculate_parameters


def test_ring_current_only():
    """Test only ring current contribution."""
    print("RING CURRENT ONLY TEST")
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
    
    # Calculate parameters for scalar
    pdyn = parmod[0]
    dst = parmod[1]
    xappa = (pdyn/2.0)**a[38]
    xx = x * xappa
    yy = y * xappa
    zz = z * xappa
    
    # Calculate ring current parameters
    phi = 1.5707963 * np.tanh(np.abs(dst) / a[33])
    znam = np.abs(dst)
    if znam < 20:
        znam = 20
    sc_sy = a[29] * (20/znam)**a[30] * xappa
    sc_pr = a[31] * (20/znam)**a[32] * xappa
    
    # Set global variables for scalar version
    t01.sc_sy = sc_sy
    t01.sc_pr = sc_pr
    t01.phi = phi
    
    # Get scalar ring current
    bxsrc, bysrc, bzsrc, bxprc, byprc, bzprc = t01.full_rc(0, ps, xx, yy, zz)
    
    # Calculate amplitudes
    a_src = a[9] + a[10] * dst + a[11] * np.sqrt(pdyn)
    a_prc = a[12] + a[13] * dst + a[14] * np.sqrt(pdyn)
    
    # Total scalar ring current
    bx_scalar = a_src * bxsrc + a_prc * bxprc
    by_scalar = a_src * bysrc + a_prc * byprc
    bz_scalar = a_src * bzsrc + a_prc * bzprc
    
    print(f"Scalar ring current (total):")
    print(f"  Bx={bx_scalar:.6f}, By={by_scalar:.6f}, Bz={bz_scalar:.6f}")
    print(f"  Components: SRC=({bxsrc:.6f}, {bysrc:.6f}, {bzsrc:.6f})")
    print(f"             PRC=({bxprc:.6f}, {byprc:.6f}, {bzprc:.6f})")
    print(f"  Amplitudes: a_src={a_src:.6f}, a_prc={a_prc:.6f}")
    
    # Get vectorized ring current
    from geopack.ring_current_vectorized import full_rc_vectorized
    params = calculate_parameters(parmod, ps, a, 1)
    
    bxsrc_v, bysrc_v, bzsrc_v, bxprc_v, byprc_v, bzprc_v = full_rc_vectorized(
        0, ps, np.array([xx]), np.array([yy]), np.array([zz]),
        params.sc_sy, params.sc_pr, params.phi
    )
    
    # Total vectorized ring current
    bx_vec = a_src * bxsrc_v[0] + a_prc * bxprc_v[0]
    by_vec = a_src * bysrc_v[0] + a_prc * byprc_v[0]
    bz_vec = a_src * bzsrc_v[0] + a_prc * bzprc_v[0]
    
    print(f"\nVectorized ring current (total):")
    print(f"  Bx={bx_vec:.6f}, By={by_vec:.6f}, Bz={bz_vec:.6f}")
    print(f"  Components: SRC=({bxsrc_v[0]:.6f}, {bysrc_v[0]:.6f}, {bzsrc_v[0]:.6f})")
    print(f"             PRC=({bxprc_v[0]:.6f}, {byprc_v[0]:.6f}, {bzprc_v[0]:.6f})")
    
    print(f"\nDifferences:")
    print(f"  ΔBx={bx_vec-bx_scalar:.6f}")
    print(f"  ΔBy={by_vec-by_scalar:.6f}")
    print(f"  ΔBz={bz_vec-bz_scalar:.6f}")
    
    # Check components separately
    print(f"\nComponent differences:")
    print(f"  SRC: ΔBx={bxsrc_v[0]-bxsrc:.6f}, ΔBy={bysrc_v[0]-bysrc:.6f}, ΔBz={bzsrc_v[0]-bzsrc:.6f}")
    print(f"  PRC: ΔBx={bxprc_v[0]-bxprc:.6f}, ΔBy={byprc_v[0]-byprc:.6f}, ΔBz={bzprc_v[0]-bzprc:.6f}")


if __name__ == "__main__":
    test_ring_current_only()