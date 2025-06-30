"""
Check if scalar T01 includes dipole field.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopack import t01


def check_scalar_t01():
    """Check scalar T01."""
    print("SCALAR T01 DIPOLE CHECK")
    print("=" * 80)
    
    # Test parameters
    ps = -0.1
    x = -10.0
    y = 0.0
    z = 0.0
    parmod = np.array([10.0, -150.0, 3.0, -5.0, 2.0, 1.0])
    
    # Get T01 field
    bx_t01, by_t01, bz_t01 = t01.t01(parmod, ps, x, y, z)
    print(f"T01 field: Bx={bx_t01:.6f}, By={by_t01:.6f}, Bz={bz_t01:.6f}")
    
    # Get dipole field
    bx_dip, by_dip, bz_dip = t01.dipole(ps, x, y, z)
    print(f"Dipole field: Bx={bx_dip:.6f}, By={by_dip:.6f}, Bz={bz_dip:.6f}")
    
    # Calculate external field
    bx_ext = bx_t01 - bx_dip
    by_ext = by_t01 - by_dip
    bz_ext = bz_t01 - bz_dip
    print(f"\nT01 - Dipole = External field:")
    print(f"  Bx={bx_ext:.6f}, By={by_ext:.6f}, Bz={bz_ext:.6f}")
    
    # The vectorized returns:
    print(f"\nVectorized T01 returns: Bx=4.547807, By=1.798416, Bz=8.882379")
    
    # Check if they match
    print(f"\nDo they match?")
    print(f"  ΔBx = {4.547807 - bx_ext:.6f}")
    print(f"  ΔBy = {1.798416 - by_ext:.6f}")
    print(f"  ΔBz = {8.882379 - bz_ext:.6f}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("The scalar T01 returns the TOTAL field (external + dipole)")
    print("But the vectorized T01 returns only the EXTERNAL field")
    print("This is the mismatch!")


if __name__ == "__main__":
    check_scalar_t01()