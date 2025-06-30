"""
Final check for the source of the 6 nT discrepancy.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopack import t01
from geopack.t01_vectorized import t01_vectorized


def final_check():
    """Final discrepancy check."""
    print("FINAL DISCREPANCY CHECK")
    print("=" * 80)
    
    # Test multiple points to see if the error is consistent
    test_cases = [
        ("Nightside mid", -10.0, 0.0, 0.0),
        ("Nightside far", -15.0, 0.0, 0.0),
        ("Dawnside", -10.0, 5.0, 0.0),
        ("North", -10.0, 0.0, 5.0),
        ("Dayside", -5.0, 0.0, 0.0),
    ]
    
    parmod = np.array([10.0, -150.0, 3.0, -5.0, 2.0, 1.0])
    ps = -0.1
    
    print("Point               Scalar Bz    Vector Bz    Difference   % Error")
    print("-" * 70)
    
    for name, x, y, z in test_cases:
        bx_s, by_s, bz_s = t01.t01(parmod, ps, x, y, z)
        bx_v, by_v, bz_v = t01_vectorized(parmod, ps, x, y, z)
        
        diff = bz_v - bz_s
        if bz_s != 0:
            pct_error = abs(diff / bz_s) * 100
        else:
            pct_error = float('inf')
        
        print(f"{name:15} {bz_s:12.6f} {bz_v:12.6f} {diff:12.6f} {pct_error:8.2f}%")
    
    print("\n" + "=" * 80)
    print("OBSERVATIONS:")
    print("1. The error is not constant but scales with field strength")
    print("2. The error is always in the same direction (vectorized more negative)")
    print("3. This suggests a systematic scaling or calculation difference")
    
    # Let's check if it's related to a specific parameter
    print("\n" + "=" * 80)
    print("PARAMETER SENSITIVITY:")
    
    # Try with different Dst
    parmod2 = parmod.copy()
    parmod2[1] = -50.0  # Less disturbed
    
    x, y, z = -10.0, 0.0, 0.0
    bx_s, by_s, bz_s = t01.t01(parmod2, ps, x, y, z)
    bx_v, by_v, bz_v = t01_vectorized(parmod2, ps, x, y, z)
    
    print(f"\nWith Dst=-50 (vs -150):")
    print(f"  Scalar: Bz={bz_s:.6f}")
    print(f"  Vector: Bz={bz_v:.6f}")
    print(f"  Difference: {bz_v - bz_s:.6f}")
    
    # Try with different pdyn
    parmod3 = parmod.copy()
    parmod3[0] = 5.0  # Lower pressure
    
    bx_s, by_s, bz_s = t01.t01(parmod3, ps, x, y, z)
    bx_v, by_v, bz_v = t01_vectorized(parmod3, ps, x, y, z)
    
    print(f"\nWith Pdyn=5 (vs 10):")
    print(f"  Scalar: Bz={bz_s:.6f}")
    print(f"  Vector: Bz={bz_v:.6f}")
    print(f"  Difference: {bz_v - bz_s:.6f}")


if __name__ == "__main__":
    final_check()