#!/usr/bin/env python3
"""
Isolate T01 components to find source of discrepancy.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from geopack import geopack
from geopack.t01 import t01, extall
from geopack.t01_vectorized import t01_vectorized, extall_vectorized, calculate_parameters


def test_individual_components():
    """Test each T01 component individually."""
    
    # Set up
    ut = 86400.0
    ps = geopack.recalc(ut)
    
    # Test case showing large discrepancy
    parmod = np.array([25.0, -50.0, 0.0, 0.0, 1.0, 1.0])  # No IMF
    x, y, z = 0.0, 8.0, 0.0  # Dusk flank - largest discrepancy
    
    print("Testing individual T01 components")
    print("="*70)
    print(f"Position: ({x}, {y}, {z}) - Dusk flank")
    print(f"Parameters: pdyn={parmod[0]}, dst={parmod[1]}, no IMF")
    print()
    
    # Model coefficients
    a = np.array([
        1.00000, 2.47341, 0.40791, 0.30429, -0.10637, -0.89108, 3.29350,
        -0.05413, -0.00696, 1.07869, -0.02314, -0.66173, -0.68018, -0.03246,
        0.02681, 0.28062, 0.16535, -0.02939, 0.02639, -0.24891, -0.08063,
        0.08900, -0.02475, 0.05887, 0.57691, 0.65256, -0.03230, 2.24733,
        4.10546, 1.13665, 0.05506, 0.97669, 0.21164, 0.64594, 1.12556, 0.01389,
        1.02978, 0.02968, 0.15821, 9.00519, 28.17582, 1.35285, 0.42279
    ])
    
    # Calculate parameters
    pdyn = parmod[0]
    dst = parmod[1]
    byimf = parmod[2]
    bzimf = parmod[3]
    g1 = parmod[4]
    g2 = parmod[5]
    dst_ast = dst * 0.8 - 13.0 * np.sqrt(pdyn)
    
    params = calculate_parameters(parmod, ps, a, 1)
    
    # Scale coordinates
    xx = x * params.xappa
    yy = y * params.xappa
    zz = z * params.xappa
    
    print(f"Scaling factor xappa: {params.xappa:.6f}")
    print(f"Scaled coordinates: ({xx:.3f}, {yy:.3f}, {zz:.3f})")
    print()
    
    # Test each component individually
    components = [
        (1, "Dipole shielding"),
        (2, "Tail field"),
        (3, "Birkeland field"),
        (4, "Ring current field"),
        (5, "Interconnection field"),
        (0, "Total field"),
    ]
    
    print("Component-by-component comparison:")
    print("-"*70)
    
    for iopgen, name in components:
        try:
            # Scalar calculation
            bx_s, by_s, bz_s = extall(iopgen, 0, 0, 0, a, 43, pdyn, dst_ast,
                                     byimf, bzimf, g1, g2, ps, xx, yy, zz)
            
            # Vectorized calculation
            bx_v, by_v, bz_v = extall_vectorized(iopgen, 0, 0, 0, a, 43, pdyn, dst_ast,
                                                byimf, bzimf, g1, g2, ps, xx, yy, zz, params,
                                                x_unscaled=x, y_unscaled=y, z_unscaled=z)
            
            # Calculate differences
            diff_x = bx_v - bx_s
            diff_y = by_v - by_s
            diff_z = bz_v - bz_s
            diff_total = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
            
            print(f"\n{name} (iopgen={iopgen}):")
            print(f"  Scalar: Bx={bx_s:10.4f}, By={by_s:10.4f}, Bz={bz_s:10.4f}")
            print(f"  Vector: Bx={bx_v:10.4f}, By={by_v:10.4f}, Bz={bz_v:10.4f}")
            print(f"  Diff:   ΔBx={diff_x:9.4f}, ΔBy={diff_y:9.4f}, ΔBz={diff_z:9.4f}")
            print(f"  Total difference: {diff_total:.4f} nT")
            
            if diff_total > 1.0:
                print(f"  *** LARGE DISCREPANCY ***")
                
        except Exception as e:
            print(f"\n{name} (iopgen={iopgen}): ERROR - {e}")
    
    # Also test at the ring current region
    print("\n" + "="*70)
    print("Testing at ring current region:")
    print("="*70)
    
    x2, y2, z2 = -4.0, 2.0, 1.0
    xx2 = x2 * params.xappa
    yy2 = y2 * params.xappa
    zz2 = z2 * params.xappa
    
    print(f"Position: ({x2}, {y2}, {z2})")
    print(f"Scaled: ({xx2:.3f}, {yy2:.3f}, {zz2:.3f})")
    
    for iopgen, name in components:
        try:
            # Scalar calculation
            bx_s, by_s, bz_s = extall(iopgen, 0, 0, 0, a, 43, pdyn, dst_ast,
                                     byimf, bzimf, g1, g2, ps, xx2, yy2, zz2)
            
            # Vectorized calculation
            bx_v, by_v, bz_v = extall_vectorized(iopgen, 0, 0, 0, a, 43, pdyn, dst_ast,
                                                byimf, bzimf, g1, g2, ps, xx2, yy2, zz2, params,
                                                x_unscaled=x2, y_unscaled=y2, z_unscaled=z2)
            
            # Calculate differences
            diff_total = np.sqrt((bx_v-bx_s)**2 + (by_v-by_s)**2 + (bz_v-bz_s)**2)
            
            if diff_total > 0.1:
                print(f"\n{name}: diff = {diff_total:.4f} nT")
                if diff_total > 1.0:
                    print(f"  Scalar: Bx={bx_s:8.3f}, By={by_s:8.3f}, Bz={bz_s:8.3f}")
                    print(f"  Vector: Bx={bx_v:8.3f}, By={by_v:8.3f}, Bz={bz_v:8.3f}")
                    
        except Exception as e:
            pass


if __name__ == "__main__":
    test_individual_components()