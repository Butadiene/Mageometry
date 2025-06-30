#!/usr/bin/env python3
"""
Check T01 symmetry and coordinate handling.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from geopack import geopack
from geopack.t01 import t01
from geopack.t01_vectorized import t01_vectorized


def check_symmetry():
    """Check if the discrepancy has a symmetry pattern."""
    
    # Set up
    ut = 86400.0
    ps = geopack.recalc(ut)
    
    parmod = np.array([25.0, -50.0, 0.0, 0.0, 1.0, 1.0])
    
    print("Checking T01 symmetry patterns")
    print("="*70)
    print(f"Parameters: pdyn={parmod[0]}, dst={parmod[1]}, no IMF")
    print(f"Dipole tilt ps: {ps:.6f} rad = {np.degrees(ps):.2f} deg")
    print()
    
    # Test symmetric positions
    test_points = [
        # Dawn-dusk symmetry
        (0.0, 8.0, 0.0, "Dusk flank"),
        (0.0, -8.0, 0.0, "Dawn flank"),
        
        # North-south positions
        (0.0, 0.0, 8.0, "North"),
        (0.0, 0.0, -8.0, "South"),
        
        # Diagonal positions
        (0.0, 5.657, 5.657, "Y-Z diagonal +"),
        (0.0, -5.657, -5.657, "Y-Z diagonal -"),
        (0.0, 5.657, -5.657, "Y-Z diagonal +-"),
        (0.0, -5.657, 5.657, "Y-Z diagonal -+"),
        
        # Different X positions
        (8.0, 0.0, 0.0, "Sunward"),
        (-8.0, 0.0, 0.0, "Tailward"),
        
        # Ring current region
        (-4.0, 2.0, 1.0, "Ring current"),
        (-4.0, -2.0, -1.0, "RC symmetric"),
    ]
    
    print("Position-by-position comparison:")
    print("-"*100)
    print(f"{'Position':20} {'X':>6} {'Y':>6} {'Z':>6} | {'Scalar Bx':>10} {'By':>10} {'Bz':>10} | {'Diff':>8}")
    print("-"*100)
    
    max_diff = 0
    max_diff_point = None
    
    for x, y, z, desc in test_points:
        # Scalar
        bx_s, by_s, bz_s = t01(parmod, ps, x, y, z)
        
        # Vectorized
        bx_v, by_v, bz_v = t01_vectorized(parmod, ps, x, y, z)
        
        # Difference
        diff = np.sqrt((bx_v-bx_s)**2 + (by_v-by_s)**2 + (bz_v-bz_s)**2)
        
        if diff > max_diff:
            max_diff = diff
            max_diff_point = (x, y, z, desc)
        
        print(f"{desc:20} {x:6.1f} {y:6.1f} {z:6.1f} | "
              f"{bx_s:10.3f} {by_s:10.3f} {bz_s:10.3f} | {diff:8.3f}")
    
    print("-"*100)
    print(f"\nMaximum difference: {max_diff:.3f} nT at {max_diff_point[3]}")
    
    # Check specific pattern for dawn-dusk
    print("\n" + "="*70)
    print("Detailed dawn-dusk comparison:")
    print("="*70)
    
    # Dusk
    x, y, z = 0.0, 8.0, 0.0
    bx_s_dusk, by_s_dusk, bz_s_dusk = t01(parmod, ps, x, y, z)
    bx_v_dusk, by_v_dusk, bz_v_dusk = t01_vectorized(parmod, ps, x, y, z)
    
    # Dawn
    x, y, z = 0.0, -8.0, 0.0
    bx_s_dawn, by_s_dawn, bz_s_dawn = t01(parmod, ps, x, y, z)
    bx_v_dawn, by_v_dawn, bz_v_dawn = t01_vectorized(parmod, ps, x, y, z)
    
    print("Dusk (Y=+8):")
    print(f"  Scalar: Bx={bx_s_dusk:8.3f}, By={by_s_dusk:8.3f}, Bz={bz_s_dusk:8.3f}")
    print(f"  Vector: Bx={bx_v_dusk:8.3f}, By={by_v_dusk:8.3f}, Bz={bz_v_dusk:8.3f}")
    print(f"  Diff:   ΔBx={bx_v_dusk-bx_s_dusk:7.3f}, ΔBy={by_v_dusk-by_s_dusk:7.3f}, ΔBz={bz_v_dusk-bz_s_dusk:7.3f}")
    
    print("\nDawn (Y=-8):")
    print(f"  Scalar: Bx={bx_s_dawn:8.3f}, By={by_s_dawn:8.3f}, Bz={bz_s_dawn:8.3f}")
    print(f"  Vector: Bx={bx_v_dawn:8.3f}, By={by_v_dawn:8.3f}, Bz={bz_v_dawn:8.3f}")
    print(f"  Diff:   ΔBx={bx_v_dawn-bx_s_dawn:7.3f}, ΔBy={by_v_dawn-by_s_dawn:7.3f}, ΔBz={bz_v_dawn-bz_s_dawn:7.3f}")
    
    # Check if errors have symmetric pattern
    print("\nSymmetry check:")
    print(f"  Scalar By symmetry: {by_s_dusk:.3f} vs {-by_s_dawn:.3f} (should be equal)")
    print(f"  Vector By symmetry: {by_v_dusk:.3f} vs {-by_v_dawn:.3f} (should be equal)")
    
    # Check error pattern
    print("\nError pattern:")
    print(f"  ΔBx symmetry: {bx_v_dusk-bx_s_dusk:.3f} vs {bx_v_dawn-bx_s_dawn:.3f}")
    print(f"  ΔBy symmetry: {by_v_dusk-by_s_dusk:.3f} vs {-(by_v_dawn-by_s_dawn):.3f}")
    
    # Test with different tilt angles
    print("\n" + "="*70)
    print("Testing with zero tilt:")
    print("="*70)
    
    # Force ps = 0
    ps_zero = 0.0
    x, y, z = 0.0, 8.0, 0.0
    
    # We need to modify the scalar version to accept ps as parameter
    # For now, let's just note this limitation
    print("Note: Cannot easily test with different ps values in scalar version")
    print("(scalar version reads ps from global state via recalc)")


if __name__ == "__main__":
    check_symmetry()