#!/usr/bin/env python3
"""
Trace T01 discrepancy by examining internal calculations.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from geopack import geopack
from geopack.t01 import t01
from geopack.t01_vectorized import t01_vectorized, calculate_parameters, iterate_sigma_vectorized_full


def trace_calculations():
    """Trace through calculations to find discrepancy source."""
    
    # Set up
    ut = 86400.0
    ps = geopack.recalc(ut)
    
    # Test case with large discrepancy
    parmod = np.array([25.0, -50.0, 0.0, 0.0, 1.0, 1.0])
    x, y, z = 0.0, 8.0, 0.0  # Dusk flank
    
    print("Tracing T01 calculations for dusk flank position")
    print("="*70)
    print(f"Position: ({x}, {y}, {z})")
    print(f"Parameters: pdyn={parmod[0]}, dst={parmod[1]}")
    print(f"Dipole tilt ps: {ps:.6f} rad = {np.degrees(ps):.2f} deg")
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
    
    # Calculate key parameters
    pdyn = parmod[0]
    xappa = (pdyn / 2.0) ** a[38]
    print(f"Pressure scaling xappa: {xappa:.6f}")
    
    # Scaled coordinates
    xx = x * xappa
    yy = y * xappa
    zz = z * xappa
    print(f"Scaled coordinates: ({xx:.3f}, {yy:.3f}, {zz:.3f})")
    
    # Calculate sigma (magnetopause distance parameter)
    sps = np.sin(ps)
    params = calculate_parameters(parmod, ps, a, 1)
    
    # Unwarped coordinates
    xss, zss = iterate_sigma_vectorized_full(x, y, z, sps, params.rh0, -5.2)
    print(f"\nUnwarped coordinates: xss={xss[0]:.6f}, zss={zss[0]:.6f}")
    
    # Magnetopause parameters
    a0_a = 34.586
    a0_s0 = 1.1960
    a0_x0 = 3.4397
    dsig = 0.003
    
    x0 = a0_x0 / xappa
    am = a0_a / xappa
    s0 = a0_s0
    
    print(f"\nMagnetopause parameters:")
    print(f"  x0 = {x0:.6f}")
    print(f"  am = {am:.6f}")
    print(f"  s0 = {s0:.6f}")
    
    # Calculate sigma
    rho2 = y**2 + zss[0]**2
    asq = am**2
    xmxm = am + xss[0] - x0
    xmxm = max(xmxm, 0)
    axx0 = xmxm**2
    aro = asq + rho2
    
    discriminant = (aro + axx0)**2 - 4.0 * asq * axx0
    discriminant = max(discriminant, 0)
    sigma = np.sqrt((aro + axx0 + np.sqrt(discriminant)) / (2.0 * asq))
    
    print(f"\nSigma calculation:")
    print(f"  rho2 = {rho2:.6f}")
    print(f"  xmxm = {xmxm:.6f}")
    print(f"  sigma = {sigma:.6f}")
    
    # Determine region
    if sigma < (s0 - dsig):
        region = "Inside magnetosphere"
        region_num = 1
    elif sigma < (s0 + dsig):
        region = "Boundary layer"
        region_num = 2
    else:
        region = "Outside magnetosphere"
        region_num = 3
    
    print(f"\nRegion: {region} (region {region_num})")
    print(f"  s0 - dsig = {s0 - dsig:.6f}")
    print(f"  s0 + dsig = {s0 + dsig:.6f}")
    
    # Get total field
    print("\n" + "="*70)
    print("Total field comparison:")
    print("="*70)
    
    bx_s, by_s, bz_s = t01(parmod, ps, x, y, z)
    bx_v, by_v, bz_v = t01_vectorized(parmod, ps, x, y, z)
    
    print(f"Scalar: Bx={bx_s:10.4f}, By={by_s:10.4f}, Bz={bz_s:10.4f}")
    print(f"Vector: Bx={bx_v:10.4f}, By={by_v:10.4f}, Bz={bz_v:10.4f}")
    print(f"Diff:   ΔBx={bx_v-bx_s:9.4f}, ΔBy={by_v-by_s:9.4f}, ΔBz={bz_v-bz_s:9.4f}")
    print(f"Total difference: {np.sqrt((bx_v-bx_s)**2 + (by_v-by_s)**2 + (bz_v-bz_s)**2):.4f} nT")
    
    # Test nearby points to see if it's a boundary issue
    print("\n" + "="*70)
    print("Testing nearby points:")
    print("="*70)
    
    offsets = [
        (0.0, 0.0, 0.0, "Original"),
        (0.1, 0.0, 0.0, "+0.1 in X"),
        (-0.1, 0.0, 0.0, "-0.1 in X"),
        (0.0, 0.1, 0.0, "+0.1 in Y"),
        (0.0, -0.1, 0.0, "-0.1 in Y"),
        (0.0, 0.0, 0.1, "+0.1 in Z"),
        (0.0, 0.0, -0.1, "-0.1 in Z"),
    ]
    
    for dx, dy, dz, desc in offsets:
        x_test = x + dx
        y_test = y + dy
        z_test = z + dz
        
        bx_s, by_s, bz_s = t01(parmod, ps, x_test, y_test, z_test)
        bx_v, by_v, bz_v = t01_vectorized(parmod, ps, x_test, y_test, z_test)
        
        diff = np.sqrt((bx_v-bx_s)**2 + (by_v-by_s)**2 + (bz_v-bz_s)**2)
        
        # Calculate sigma for this point
        xss_test, zss_test = iterate_sigma_vectorized_full(x_test, y_test, z_test, sps, params.rh0, -5.2)
        rho2_test = y_test**2 + zss_test[0]**2
        xmxm_test = am + xss_test[0] - x0
        xmxm_test = max(xmxm_test, 0)
        axx0_test = xmxm_test**2
        aro_test = asq + rho2_test
        disc_test = (aro_test + axx0_test)**2 - 4.0 * asq * axx0_test
        disc_test = max(disc_test, 0)
        sigma_test = np.sqrt((aro_test + axx0_test + np.sqrt(disc_test)) / (2.0 * asq))
        
        print(f"{desc:12} sigma={sigma_test:.6f}, diff={diff:8.4f} nT")
    
    # Check if it's related to the boundary layer interpolation
    print("\n" + "="*70)
    print("Analysis:")
    print("="*70)
    
    if region_num == 2:
        print("Point is in the boundary layer where interpolation occurs.")
        print("This could explain the large discrepancy if the interpolation")
        print("is implemented differently between scalar and vectorized versions.")
    elif region_num == 3:
        print("Point is outside the magnetosphere.")
        print("Field should be IMF - dipole, which should be consistent.")
    else:
        print("Point is inside the magnetosphere.")
        print("All field components should be calculated.")


if __name__ == "__main__":
    trace_calculations()