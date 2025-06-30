"""
Verify that the missing dipole field explains the Bz error.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopack import t01
from geopack.t01_vectorized import dipole_vectorized, t01_vectorized


def verify_dipole_issue():
    """Verify the dipole field issue."""
    print("DIPOLE FIELD VERIFICATION")
    print("=" * 80)
    
    # Test parameters
    parmod = np.array([10.0, -150.0, 3.0, -5.0, 2.0, 1.0])
    ps = -0.1
    x = -10.0
    y = 0.0
    z = 0.0
    
    print(f"Test point: x={x}, y={y}, z={z}")
    
    # Get T01 fields
    bx_t01_s, by_t01_s, bz_t01_s = t01.t01(parmod, ps, x, y, z)
    bx_t01_v, by_t01_v, bz_t01_v = t01_vectorized(parmod, ps, x, y, z)
    
    print(f"\nT01 Scalar: Bx={bx_t01_s:.6f}, By={by_t01_s:.6f}, Bz={bz_t01_s:.6f}")
    print(f"T01 Vector: Bx={bx_t01_v:.6f}, By={by_t01_v:.6f}, Bz={bz_t01_v:.6f}")
    print(f"Difference: ΔBx={bx_t01_v-bx_t01_s:.6f}, ΔBy={by_t01_v-by_t01_s:.6f}, ΔBz={bz_t01_v-bz_t01_s:.6f}")
    
    # Get dipole field
    x_arr = np.array([x])
    y_arr = np.array([y])
    z_arr = np.array([z])
    qx, qy, qz = dipole_vectorized(ps, x_arr, y_arr, z_arr)
    
    print(f"\nDipole field: Bx={qx[0]:.6f}, By={qy[0]:.6f}, Bz={qz[0]:.6f}")
    
    # Check if adding dipole to vectorized result matches scalar
    bx_corrected = bx_t01_v + qx[0]
    by_corrected = by_t01_v + qy[0]
    bz_corrected = bz_t01_v + qz[0]
    
    print(f"\nVectorized + Dipole: Bx={bx_corrected:.6f}, By={by_corrected:.6f}, Bz={bz_corrected:.6f}")
    print(f"Remaining error: ΔBx={bx_corrected-bx_t01_s:.6f}, ΔBy={by_corrected-by_t01_s:.6f}, ΔBz={bz_corrected-bz_t01_s:.6f}")
    
    # Also check with scalar dipole function
    qx_s, qy_s, qz_s = t01.dipole(ps, x, y, z)
    print(f"\nScalar dipole: Bx={qx_s:.6f}, By={qy_s:.6f}, Bz={qz_s:.6f}")
    print(f"Dipole difference: ΔBx={qx[0]-qx_s:.6f}, ΔBy={qy[0]-qy_s:.6f}, ΔBz={qz[0]-qz_s:.6f}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print(f"1. T01 vectorized is missing the dipole field")
    print(f"2. The missing dipole Bz component is: {qz[0]:.6f} nT")
    print(f"3. The T01 Bz error is: {bz_t01_v-bz_t01_s:.6f} nT")
    print(f"4. After adding dipole, remaining error is: {bz_corrected-bz_t01_s:.6f} nT")
    
    if abs(bz_corrected - bz_t01_s) < 1.0:
        print("\n✓ CONFIRMED: The missing dipole field explains the Bz error!")
    else:
        print("\n✗ There's still a significant error even after adding dipole.")


if __name__ == "__main__":
    verify_dipole_issue()