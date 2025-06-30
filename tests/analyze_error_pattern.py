#!/usr/bin/env python3
"""
Analyze the error pattern in T01 vectorized vs scalar.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from geopack import geopack
from geopack.t01 import t01
from geopack.t01_vectorized import t01_vectorized


def analyze_error_pattern():
    """Analyze error pattern systematically."""
    
    # Set up
    ut = 86400.0
    ps = geopack.recalc(ut)
    
    parmod = np.array([25.0, -50.0, 0.0, 0.0, 1.0, 1.0])
    
    print("Analyzing T01 error patterns")
    print("="*70)
    print(f"Parameters: pdyn={parmod[0]}, dst={parmod[1]}")
    print(f"Dipole tilt ps: {ps:.6f} rad = {np.degrees(ps):.2f} deg")
    print()
    
    # Test in Y-Z plane at X=0
    print("Error map in Y-Z plane at X=0:")
    print("-"*70)
    
    y_values = np.linspace(-10, 10, 11)
    z_values = np.linspace(-5, 5, 11)
    
    max_error = 0
    max_error_pos = None
    
    print(f"{'Y/Z':>6}", end='')
    for z in z_values:
        print(f"{z:8.1f}", end='')
    print("\n" + "-"*96)
    
    error_map = np.zeros((len(y_values), len(z_values)))
    
    for i, y in enumerate(y_values):
        print(f"{y:6.1f}", end='')
        for j, z in enumerate(z_values):
            x = 0.0
            
            # Calculate fields
            bx_s, by_s, bz_s = t01(parmod, ps, x, y, z)
            bx_v, by_v, bz_v = t01_vectorized(parmod, ps, x, y, z)
            
            # Total error
            error = np.sqrt((bx_v-bx_s)**2 + (by_v-by_s)**2 + (bz_v-bz_s)**2)
            error_map[i, j] = error
            
            if error > max_error:
                max_error = error
                max_error_pos = (x, y, z)
            
            # Color code the output
            if error < 1.0:
                print(f"{error:8.1f}", end='')
            elif error < 5.0:
                print(f"{error:8.1f}", end='')
            elif error < 10.0:
                print(f"{error:8.1f}", end='')
            else:
                print(f"{error:8.1f}", end='')
        print()
    
    print("-"*96)
    print(f"\nMaximum error: {max_error:.1f} nT at {max_error_pos}")
    
    # Analyze error components at maximum error position
    if max_error_pos:
        print("\n" + "="*70)
        print("Detailed analysis at maximum error position:")
        print("="*70)
        
        x, y, z = max_error_pos
        bx_s, by_s, bz_s = t01(parmod, ps, x, y, z)
        bx_v, by_v, bz_v = t01_vectorized(parmod, ps, x, y, z)
        
        print(f"Position: ({x}, {y}, {z})")
        print(f"Scalar: Bx={bx_s:8.3f}, By={by_s:8.3f}, Bz={bz_s:8.3f}")
        print(f"Vector: Bx={bx_v:8.3f}, By={by_v:8.3f}, Bz={bz_v:8.3f}")
        print(f"Error:  ΔBx={bx_v-bx_s:7.3f}, ΔBy={by_v-by_s:7.3f}, ΔBz={bz_v-bz_s:7.3f}")
        
        # Check if it's a scaling issue
        b_mag_s = np.sqrt(bx_s**2 + by_s**2 + bz_s**2)
        b_mag_v = np.sqrt(bx_v**2 + by_v**2 + bz_v**2)
        print(f"\nMagnitudes: Scalar={b_mag_s:.3f}, Vector={b_mag_v:.3f}, Ratio={b_mag_v/b_mag_s:.4f}")
    
    # Test radial dependence
    print("\n" + "="*70)
    print("Radial dependence of error:")
    print("="*70)
    
    r_values = np.arange(2, 12, 2)
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    
    print(f"{'R':>4} ", end='')
    for angle in angles:
        print(f"{angle:>7}°", end='')
    print("\n" + "-"*68)
    
    for r in r_values:
        print(f"{r:4.0f} ", end='')
        for angle in angles:
            theta = np.radians(angle)
            y = r * np.cos(theta)
            z = r * np.sin(theta)
            x = 0.0
            
            bx_s, by_s, bz_s = t01(parmod, ps, x, y, z)
            bx_v, by_v, bz_v = t01_vectorized(parmod, ps, x, y, z)
            
            error = np.sqrt((bx_v-bx_s)**2 + (by_v-by_s)**2 + (bz_v-bz_s)**2)
            print(f"{error:7.1f}", end='')
        print()
    
    # Check specific patterns
    print("\n" + "="*70)
    print("Checking for systematic patterns:")
    print("="*70)
    
    # Test at Y=8 with varying Z
    print("\nError vs Z at Y=8 (dusk flank):")
    z_test = np.linspace(-5, 5, 21)
    errors = []
    for z in z_test:
        bx_s, by_s, bz_s = t01(parmod, ps, 0.0, 8.0, z)
        bx_v, by_v, bz_v = t01_vectorized(parmod, ps, 0.0, 8.0, z)
        error = np.sqrt((bx_v-bx_s)**2 + (by_v-by_s)**2 + (bz_v-bz_s)**2)
        errors.append(error)
    
    # Find pattern
    max_idx = np.argmax(errors)
    min_idx = np.argmin(errors)
    print(f"  Max error: {errors[max_idx]:.1f} nT at Z={z_test[max_idx]:.1f}")
    print(f"  Min error: {errors[min_idx]:.1f} nT at Z={z_test[min_idx]:.1f}")
    print(f"  Error range: {min(errors):.1f} - {max(errors):.1f} nT")


if __name__ == "__main__":
    analyze_error_pattern()