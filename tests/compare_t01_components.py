#!/usr/bin/env python3
"""
Compare T01 scalar and vectorized implementations component by component.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from geopack import geopack
from geopack.t01 import t01
from geopack.t01_vectorized import t01_vectorized


def compare_components():
    """Compare individual field components between scalar and vectorized."""
    
    # Set up test case
    ut = 86400.0
    ps = geopack.recalc(ut)
    
    # Problem case: strong IMF in ring current region
    parmod = np.array([25.0, -50.0, 9.0, -3.0, 1.0, 1.0])
    x, y, z = -4.0, 2.0, 1.0
    
    print("Comparing T01 implementations for problematic case")
    print("="*70)
    print(f"Position: ({x}, {y}, {z})")
    print(f"Parameters: pdyn={parmod[0]}, dst={parmod[1]}, byimf={parmod[2]}, bzimf={parmod[3]}")
    print()
    
    # Total field
    bx_s, by_s, bz_s = t01(parmod, ps, x, y, z)
    bx_v, by_v, bz_v = t01_vectorized(parmod, ps, x, y, z)
    
    print("Total field comparison:")
    print(f"  Scalar: Bx={bx_s:8.3f}, By={by_s:8.3f}, Bz={bz_s:8.3f}")
    print(f"  Vector: Bx={bx_v:8.3f}, By={by_v:8.3f}, Bz={bz_v:8.3f}")
    print(f"  Diff:   ΔBx={bx_v-bx_s:7.3f}, ΔBy={by_v-by_s:7.3f}, ΔBz={bz_v-bz_s:7.3f}")
    print(f"  Total difference: {np.sqrt((bx_v-bx_s)**2 + (by_v-by_s)**2 + (bz_v-bz_s)**2):.3f} nT")
    
    # Let's also test a simpler case to see if the issue is specific to strong IMF
    print("\n" + "="*70)
    print("Testing with zero IMF for comparison:")
    print("="*70)
    
    parmod_no_imf = np.array([25.0, -50.0, 0.0, 0.0, 1.0, 1.0])
    
    bx_s2, by_s2, bz_s2 = t01(parmod_no_imf, ps, x, y, z)
    bx_v2, by_v2, bz_v2 = t01_vectorized(parmod_no_imf, ps, x, y, z)
    
    print("Zero IMF comparison:")
    print(f"  Scalar: Bx={bx_s2:8.3f}, By={by_s2:8.3f}, Bz={bz_s2:8.3f}")
    print(f"  Vector: Bx={bx_v2:8.3f}, By={by_v2:8.3f}, Bz={bz_v2:8.3f}")
    print(f"  Diff:   ΔBx={bx_v2-bx_s2:7.3f}, ΔBy={by_v2-by_s2:7.3f}, ΔBz={bz_v2-bz_s2:7.3f}")
    print(f"  Total difference: {np.sqrt((bx_v2-bx_s2)**2 + (by_v2-by_s2)**2 + (bz_v2-bz_s2)**2):.3f} nT")
    
    # Calculate IMF contribution
    print("\n" + "="*70)
    print("Estimated IMF contribution:")
    print("="*70)
    
    # Difference between with and without IMF
    print("Scalar IMF effect:")
    print(f"  ΔBx={bx_s - bx_s2:7.3f}, ΔBy={by_s - by_s2:7.3f}, ΔBz={bz_s - bz_s2:7.3f}")
    
    print("Vector IMF effect:")
    print(f"  ΔBx={bx_v - bx_v2:7.3f}, ΔBy={by_v - by_v2:7.3f}, ΔBz={bz_v - bz_v2:7.3f}")
    
    # The difference in IMF effect
    imf_diff_x = (bx_v - bx_v2) - (bx_s - bx_s2)
    imf_diff_y = (by_v - by_v2) - (by_s - by_s2)
    imf_diff_z = (bz_v - bz_v2) - (bz_s - bz_s2)
    
    print("\nDifference in IMF effect (vector - scalar):")
    print(f"  ΔΔBx={imf_diff_x:7.3f}, ΔΔBy={imf_diff_y:7.3f}, ΔΔBz={imf_diff_z:7.3f}")
    
    # Test more positions to see pattern
    print("\n" + "="*70)
    print("Testing multiple positions with strong IMF:")
    print("="*70)
    
    test_positions = [
        (6.0, 0.0, 0.0, "Subsolar"),
        (0.0, 8.0, 0.0, "Dusk flank"),
        (-10.0, 0.0, 0.0, "Tail"),
        (-4.0, 2.0, 1.0, "Ring current"),
        (4.0, 4.0, 4.0, "High latitude"),
    ]
    
    for x, y, z, desc in test_positions:
        bx_s, by_s, bz_s = t01(parmod, ps, x, y, z)
        bx_v, by_v, bz_v = t01_vectorized(parmod, ps, x, y, z)
        diff = np.sqrt((bx_v-bx_s)**2 + (by_v-by_s)**2 + (bz_v-bz_s)**2)
        print(f"{desc:15} ({x:5.1f},{y:5.1f},{z:5.1f}): diff = {diff:6.3f} nT")


if __name__ == "__main__":
    compare_components()