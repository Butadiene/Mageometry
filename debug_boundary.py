#!/usr/bin/env python
"""
Debug script to trace a single field line and compare scalar vs vectorized boundary detection.
Focuses on the tail region case that shows large differences.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/skipjack/Documents/geopack-vectorize')

import geopack
from geopack.models import t89
from geopack.geopack import trace
from geopack.trace_vectorized import trace_vectorized

def debug_single_trace():
    """Debug trace from (-10.0, 0.0, 2.0) to see boundary detection differences."""
    
    # Set up parameters
    ut = 0.0  # Unix time
    ps = geopack.recalc(ut)
    
    # Starting point in tail region
    x0, y0, z0 = -10.0, 0.0, 2.0
    
    print("Debug Field Line Trace from (-10.0, 0.0, 2.0)")
    print("=" * 80)
    
    # Common parameters
    dir = 1  # North to south
    rlim = 15.0  # Outer boundary
    r0 = 1.0  # Inner boundary
    parmod = 3  # Kp index for T89
    exname = 't89'
    inname = 'igrf'
    maxloop = 10000
    
    print(f"\nParameters:")
    print(f"  Direction: {dir} (north to south)")
    print(f"  Outer boundary: {rlim} Re")
    print(f"  Inner boundary: {r0} Re")
    print(f"  Model: {exname} with Kp={parmod}")
    print(f"  Max steps: {maxloop}")
    
    # Trace with scalar version
    print("\n\nSCALAR VERSION:")
    print("-" * 40)
    
    xf_scalar, yf_scalar, zf_scalar, xx_scalar, yy_scalar, zz_scalar = trace(
        x0, y0, z0, dir, rlim=rlim, r0=r0, parmod=parmod, 
        exname=exname, inname=inname, maxloop=maxloop
    )
    
    r_final_scalar = np.sqrt(xf_scalar**2 + yf_scalar**2 + zf_scalar**2)
    
    print(f"Final position: ({xf_scalar:.6f}, {yf_scalar:.6f}, {zf_scalar:.6f})")
    print(f"Final radius: {r_final_scalar:.6f}")
    nstep_scalar = len([x for x in xx_scalar if x != 0])  # Count non-zero steps
    print(f"Total steps: {nstep_scalar}")
    
    # Show last few steps
    if nstep_scalar > 0:
        print("\nLast 10 steps of scalar trace:")
        start_idx = max(0, nstep_scalar - 10)
        for i in range(start_idx, nstep_scalar):
            x, y, z = xx_scalar[i], yy_scalar[i], zz_scalar[i]
            r = np.sqrt(x**2 + y**2 + z**2)
            print(f"  Step {i}: ({x:.6f}, {y:.6f}, {z:.6f}), r = {r:.6f}")
    
    # Trace with vectorized version
    print("\n\nVECTORIZED VERSION:")
    print("-" * 40)
    
    # Trace with full path returned
    result_vec = trace_vectorized(
        x0, y0, z0, dir=dir, rlim=rlim, r0=r0, parmod=parmod,
        exname=exname, inname=inname, maxloop=maxloop,
        return_full_path=True
    )
    
    # Handle different return formats
    if isinstance(result_vec, tuple) and len(result_vec) == 7:
        # Full path returned
        xf_vec, yf_vec, zf_vec, xx_vec, yy_vec, zz_vec, nstep_vec = result_vec
    else:
        # Try without full path
        result_vec = trace_vectorized(
            x0, y0, z0, dir=dir, rlim=rlim, r0=r0, parmod=parmod,
            exname=exname, inname=inname, maxloop=maxloop,
            return_full_path=False
        )
        xf_vec, yf_vec, zf_vec, nstep_vec = result_vec
        xx_vec = yy_vec = zz_vec = None
    
    r_final_vec = np.sqrt(xf_vec**2 + yf_vec**2 + zf_vec**2)
    
    print(f"Final position: ({xf_vec:.6f}, {yf_vec:.6f}, {zf_vec:.6f})")
    print(f"Final radius: {r_final_vec:.6f}")
    print(f"Total steps: {nstep_vec}")
    
    # Show last few steps if available
    if xx_vec is not None and nstep_vec > 0:
        print("\nLast 10 steps of vectorized trace:")
        start_idx = max(0, nstep_vec - 10)
        for i in range(start_idx, nstep_vec):
            x, y, z = xx_vec[i], yy_vec[i], zz_vec[i]
            r = np.sqrt(x**2 + y**2 + z**2)
            print(f"  Step {i}: ({x:.6f}, {y:.6f}, {z:.6f}), r = {r:.6f}")
    
    # Compare results
    print("\n\nCOMPARISON:")
    print("-" * 40)
    diff_x = xf_vec - xf_scalar
    diff_y = yf_vec - yf_scalar
    diff_z = zf_vec - zf_scalar
    diff_dist = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
    
    print(f"Position difference: ({diff_x:.6f}, {diff_y:.6f}, {diff_z:.6f})")
    print(f"Distance difference: {diff_dist:.6f}")
    print(f"Radius difference: {r_final_vec - r_final_scalar:.6f}")
    print(f"Step count difference: {nstep_vec - nstep_scalar}")
    
    # Check field at both final positions
    print("\nField at final positions:")
    bx_s, by_s, bz_s = t89(parmod, ps, xf_scalar, yf_scalar, zf_scalar)
    b_mag_s = np.sqrt(bx_s**2 + by_s**2 + bz_s**2)
    print(f"Scalar: B = ({bx_s:.6f}, {by_s:.6f}, {bz_s:.6f}), |B| = {b_mag_s:.6f}")
    
    bx_v, by_v, bz_v = t89(parmod, ps, xf_vec, yf_vec, zf_vec)
    b_mag_v = np.sqrt(bx_v**2 + by_v**2 + bz_v**2)
    print(f"Vectorized: B = ({bx_v:.6f}, {by_v:.6f}, {bz_v:.6f}), |B| = {b_mag_v:.6f}")
    
    # Check where paths diverge if we have full paths
    if xx_vec is not None and nstep_scalar > 0 and nstep_vec > 0:
        print("\n\nPATH DIVERGENCE ANALYSIS:")
        print("-" * 40)
        min_steps = min(nstep_scalar, nstep_vec)
        
        divergence_found = False
        for i in range(min_steps):
            x_s, y_s, z_s = xx_scalar[i], yy_scalar[i], zz_scalar[i]
            x_v, y_v, z_v = xx_vec[i], yy_vec[i], zz_vec[i]
            dist = np.sqrt((x_v - x_s)**2 + (y_v - y_s)**2 + (z_v - z_s)**2)
            
            if dist > 0.01 and not divergence_found:  # Threshold for significant divergence
                divergence_found = True
                print(f"Paths diverge significantly at step {i}:")
                print(f"  Scalar: ({x_s:.6f}, {y_s:.6f}, {z_s:.6f})")
                print(f"  Vectorized: ({x_v:.6f}, {y_v:.6f}, {z_v:.6f})")
                print(f"  Distance: {dist:.6f}")
                
                # Show a few steps before and after divergence
                print("\nSteps around divergence:")
                for j in range(max(0, i-3), min(min_steps, i+4)):
                    x_s, y_s, z_s = xx_scalar[j], yy_scalar[j], zz_scalar[j]
                    x_v, y_v, z_v = xx_vec[j], yy_vec[j], zz_vec[j]
                    d = np.sqrt((x_v - x_s)**2 + (y_v - y_s)**2 + (z_v - z_s)**2)
                    print(f"  Step {j}: distance = {d:.6f}")
        
        if not divergence_found:
            print("Paths remain close throughout the trace (< 0.01 Re)")
    
    # Test with different parameters
    print("\n\nTESTING WITH DIFFERENT BOUNDARY:")
    print("-" * 40)
    
    # Try with smaller boundary to see if that's the issue
    rlim_test = 12.0
    print(f"\nTesting with rlim = {rlim_test} Re")
    
    xf_s2, yf_s2, zf_s2, xx_s2, yy_s2, zz_s2 = trace(
        x0, y0, z0, dir, rlim=rlim_test, r0=r0, parmod=parmod,
        exname=exname, inname=inname, maxloop=maxloop
    )
    ns2 = len([x for x in xx_s2 if x != 0])
    
    result_v2 = trace_vectorized(
        x0, y0, z0, dir=dir, rlim=rlim_test, r0=r0, parmod=parmod,
        exname=exname, inname=inname, maxloop=maxloop,
        return_full_path=False
    )
    xf_v2, yf_v2, zf_v2, nv2 = result_v2
    
    r_s2 = np.sqrt(xf_s2**2 + yf_s2**2 + zf_s2**2)
    r_v2 = np.sqrt(xf_v2**2 + yf_v2**2 + zf_v2**2)
    diff2 = np.sqrt((xf_v2 - xf_s2)**2 + (yf_v2 - yf_s2)**2 + (zf_v2 - zf_s2)**2)
    
    print(f"Scalar: r = {r_s2:.6f}, steps = {ns2}")
    print(f"Vectorized: r = {r_v2:.6f}, steps = {nv2}")
    print(f"Difference: {diff2:.6f}")


if __name__ == "__main__":
    debug_single_trace()