#!/usr/bin/env python3
"""
Check how T01 handles points near the origin.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from geopack import geopack
from geopack.t01 import t01
from geopack.t01_vectorized import t01_vectorized


def check_origin_handling():
    """Check field calculation near origin."""
    
    # Set up
    ut = 86400.0
    ps = geopack.recalc(ut)
    
    parmod = np.array([25.0, -50.0, 0.0, 0.0, 1.0, 1.0])
    
    print("Checking T01 behavior near origin")
    print("="*70)
    print(f"Parameters: pdyn={parmod[0]}, dst={parmod[1]}")
    print()
    
    # Test points approaching origin
    print("Field along Z-axis approaching origin:")
    print("-"*70)
    print(f"{'Z':>6} | {'Scalar Bx':>10} {'By':>10} {'Bz':>10} | {'Vector Bx':>10} {'By':>10} {'Bz':>10} | {'Error':>8}")
    print("-"*70)
    
    z_values = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001]
    
    for z in z_values:
        x, y = 0.0, 0.0
        
        # Suppress warnings for this test
        with np.errstate(all='ignore'):
            bx_s, by_s, bz_s = t01(parmod, ps, x, y, z)
            bx_v, by_v, bz_v = t01_vectorized(parmod, ps, x, y, z)
        
        error = np.sqrt((bx_v-bx_s)**2 + (by_v-by_s)**2 + (bz_v-bz_s)**2)
        
        print(f"{z:6.3f} | {bx_s:10.3f} {by_s:10.3f} {bz_s:10.3f} | "
              f"{bx_v:10.3f} {by_v:10.3f} {bz_v:10.3f} | {error:8.3f}")
    
    # Test exactly at origin
    print("\nAt origin (0,0,0):")
    with np.errstate(all='ignore'):
        bx_s, by_s, bz_s = t01(parmod, ps, 0.0, 0.0, 0.0)
        bx_v, by_v, bz_v = t01_vectorized(parmod, ps, 0.0, 0.0, 0.0)
    
    print(f"  Scalar: Bx={bx_s}, By={by_s}, Bz={bz_s}")
    print(f"  Vector: Bx={bx_v}, By={by_v}, Bz={bz_v}")
    
    # Test small offsets from origin
    print("\n" + "="*70)
    print("Field at small offsets from origin:")
    print("="*70)
    
    offsets = [
        (0.1, 0.0, 0.0, "X offset"),
        (0.0, 0.1, 0.0, "Y offset"),
        (0.0, 0.0, 0.1, "Z offset"),
        (0.1, 0.1, 0.0, "XY offset"),
        (0.1, 0.0, 0.1, "XZ offset"),
        (0.0, 0.1, 0.1, "YZ offset"),
    ]
    
    for x, y, z, desc in offsets:
        with np.errstate(all='ignore'):
            bx_s, by_s, bz_s = t01(parmod, ps, x, y, z)
            bx_v, by_v, bz_v = t01_vectorized(parmod, ps, x, y, z)
        
        error = np.sqrt((bx_v-bx_s)**2 + (by_v-by_s)**2 + (bz_v-bz_s)**2)
        print(f"{desc:10}: error = {error:6.3f} nT")
    
    # Check if vectorized version has special handling for origin
    print("\n" + "="*70)
    print("Checking vectorized code for origin handling:")
    print("="*70)
    
    # Look at the code
    with open(os.path.join(os.path.dirname(__file__), '..', 'geopack', 't01_vectorized.py'), 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if 'origin' in line.lower() and ('mask' in line or 'r < ' in line):
            print(f"Line {i+1}: {line.strip()}")
    
    # Test problematic point (0,0,1)
    print("\n" + "="*70)
    print("Detailed check at problematic point (0,0,1):")
    print("="*70)
    
    x, y, z = 0.0, 0.0, 1.0
    with np.errstate(all='ignore'):
        bx_s, by_s, bz_s = t01(parmod, ps, x, y, z)
        bx_v, by_v, bz_v = t01_vectorized(parmod, ps, x, y, z, debug=True)
    
    print(f"\nScalar: Bx={bx_s:.3f}, By={by_s:.3f}, Bz={bz_s:.3f}")
    print(f"Vector: Bx={bx_v:.3f}, By={by_v:.3f}, Bz={bz_v:.3f}")
    print(f"Error:  ΔBy={by_v-by_s:.3f} nT")


if __name__ == "__main__":
    check_origin_handling()