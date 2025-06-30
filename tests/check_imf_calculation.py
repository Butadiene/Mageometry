#!/usr/bin/env python3
"""
Check IMF calculation details in T01.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from geopack import geopack


def check_imf_contribution():
    """Check how IMF contribution is calculated."""
    
    # Test parameters
    byimf = 9.0
    bzimf = -3.0
    
    # Model coefficient array (from T01)
    a = np.array([
        1.00000, 2.47341, 0.40791, 0.30429, -0.10637, -0.89108, 3.29350,
        -0.05413, -0.00696, 1.07869, -0.02314, -0.66173, -0.68018, -0.03246,
        0.02681, 0.28062, 0.16535, -0.02939, 0.02639, -0.24891, -0.08063,
        0.08900, -0.02475, 0.05887, 0.57691, 0.65256, -0.03230, 2.24733,
        4.10546, 1.13665, 0.05506, 0.97669, 0.21164, 0.64594, 1.12556, 0.01389,
        1.02978, 0.02968, 0.15821, 9.00519, 28.17582, 1.35285, 0.42279
    ])
    
    # Calculate IMF angle
    theta = np.arctan2(byimf, bzimf)
    if theta <= 0:
        theta += 2 * np.pi
    
    sthetah = np.sin(theta / 2.0) ** 2
    
    print(f"IMF components: By={byimf}, Bz={bzimf}")
    print(f"IMF angle theta: {np.degrees(theta):.1f} degrees")
    print(f"sin²(theta/2): {sthetah:.6f}")
    print()
    
    # Calculate factimf
    factimf = a[23] + a[24] * sthetah
    print(f"a[23] = {a[23]:.6f}")
    print(f"a[24] = {a[24]:.6f}")
    print(f"factimf = a[23] + a[24]*sthetah = {factimf:.6f}")
    print()
    
    # Compare different ways of calculating IMF contribution
    
    # Method 1: Direct multiplication (what document says Fortran does)
    contrib1_y = byimf * factimf
    contrib1_z = bzimf * factimf
    print(f"Method 1 (BYIMF * FACTIMF):")
    print(f"  By contribution: {contrib1_y:.6f}")
    print(f"  Bz contribution: {contrib1_z:.6f}")
    print()
    
    # Method 2: Separate terms (what current vectorized code does)
    contrib2_y = a[23] * byimf + a[24] * byimf * sthetah
    contrib2_z = a[23] * bzimf + a[24] * bzimf * sthetah
    print(f"Method 2 (a[23]*BYIMF + a[24]*BYIMF*sthetah):")
    print(f"  By contribution: {contrib2_y:.6f}")
    print(f"  Bz contribution: {contrib2_z:.6f}")
    print()
    
    # They should be identical
    print(f"Methods are identical: {np.allclose([contrib1_y, contrib1_z], [contrib2_y, contrib2_z])}")
    
    # Check what happens with different IMF orientations
    print("\n" + "="*50)
    print("Testing different IMF orientations:")
    print("="*50)
    
    test_cases = [
        (0.0, -5.0, "Pure southward"),
        (5.0, 0.0, "Pure duskward"),
        (-5.0, 0.0, "Pure dawnward"),
        (0.0, 5.0, "Pure northward"),
        (9.0, -3.0, "Original case"),
    ]
    
    for by, bz, desc in test_cases:
        if by == 0 and bz == 0:
            theta = 0.0
        else:
            theta = np.arctan2(by, bz)
            if theta <= 0:
                theta += 2 * np.pi
        
        sthetah = np.sin(theta / 2.0) ** 2
        factimf = a[23] + a[24] * sthetah
        
        print(f"\n{desc}: By={by}, Bz={bz}")
        print(f"  theta={np.degrees(theta):.1f}°, factimf={factimf:.4f}")
        print(f"  Contributions: By={by*factimf:.4f}, Bz={bz*factimf:.4f}")


if __name__ == "__main__":
    check_imf_contribution()