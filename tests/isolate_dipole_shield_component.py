"""
Test ONLY the dipole shielding component.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopack import t01
from geopack.t01_vectorized import shlcar3x3_vectorized_partial


def test_dipole_shield_only():
    """Test only dipole shield component."""
    print("DIPOLE SHIELD COMPONENT ISOLATION TEST")
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
    
    # Get scalar dipole shield field
    pdyn = parmod[0]
    xappa = (pdyn/2.)**a[38]
    xx = x * xappa
    yy = y * xappa
    zz = z * xappa
    
    # Get dipole shield from scalar
    bxcf, bycf, bzcf = t01.shlcar3x3(xx, yy, zz, ps)
    
    # Apply amplitude (a[0] = 1.0)
    bx_ds_scalar = a[0] * bxcf
    by_ds_scalar = a[0] * bycf
    bz_ds_scalar = a[0] * bzcf
    
    print(f"Scalar dipole shield field:")
    print(f"  Bx={bx_ds_scalar:.6f}, By={by_ds_scalar:.6f}, Bz={bz_ds_scalar:.6f}")
    
    # Get vectorized dipole shield field
    bxcf_v, bycf_v, bzcf_v = shlcar3x3_vectorized_partial(
        np.array([xx]), np.array([yy]), np.array([zz]), ps
    )
    
    # Apply amplitude
    bx_ds_vec = a[0] * bxcf_v[0]
    by_ds_vec = a[0] * bycf_v[0]
    bz_ds_vec = a[0] * bzcf_v[0]
    
    print(f"\nVectorized dipole shield field:")
    print(f"  Bx={bx_ds_vec:.6f}, By={by_ds_vec:.6f}, Bz={bz_ds_vec:.6f}")
    
    print(f"\nDifferences:")
    print(f"  ΔBx={bx_ds_vec-bx_ds_scalar:.6f}")
    print(f"  ΔBy={by_ds_vec-by_ds_scalar:.6f}")
    print(f"  ΔBz={bz_ds_vec-bz_ds_scalar:.6f}")
    
    print(f"\nConclusion:")
    if abs(bz_ds_vec-bz_ds_scalar) < 0.001:
        print("  Dipole shield component matches! The error is NOT in the dipole shield.")
    else:
        print("  Dipole shield component has significant error. The bug is in shlcar3x3_vectorized_partial.")


if __name__ == "__main__":
    test_dipole_shield_only()