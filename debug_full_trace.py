#!/usr/bin/env python
"""
Debug script to trace the full path step by step in the vectorized version.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/skipjack/Documents/geopack-vectorize')

import geopack
from geopack.models import t89
from geopack.trace_vectorized import rhand_vectorized, step_vectorized

def debug_full_vectorized_trace():
    """Manually implement the vectorized trace loop to debug."""
    
    # Set up parameters
    ut = 0.0  # Unix time
    ps = geopack.recalc(ut)
    
    # Starting point in tail region
    xi = np.array([-10.0])
    yi = np.array([0.0])
    zi = np.array([2.0])
    
    # Parameters
    dir = 1
    rlim = 15.0
    r0 = 1.0
    parmod = 3
    exname = 't89'
    inname = 'igrf'
    maxloop = 20  # Limit for debugging
    
    print("Manual Vectorized Trace Debug")
    print("=" * 80)
    print(f"Starting point: ({xi[0]}, {yi[0]}, {zi[0]})")
    print(f"Starting radius: {np.sqrt(xi[0]**2 + yi[0]**2 + zi[0]**2):.6f}")
    print(f"Boundary: {rlim} Re")
    
    # Initialize
    n_traces = 1
    x = xi.copy()
    y = yi.copy()
    z = zi.copy()
    
    # Active mask and status
    active_mask = np.ones(n_traces, dtype=bool)
    status = np.zeros(n_traces, dtype=np.int32)
    
    # Step sizes
    ds_array = np.full(n_traces, 0.5 * dir, dtype=np.float64)
    
    # Initial direction check
    ds3 = -0.5 * dir / 3.0
    r1, r2, r3 = rhand_vectorized(xi, yi, zi, parmod, exname, inname, ds3)
    br = (xi * r1 + yi * r2 + zi * r3)
    
    # Set initial ad
    ad = np.where(br < 0, -0.01, 0.01)
    if dir < 0:
        ad = -ad
    
    # Previous radial distances
    rr = np.sqrt(xi**2 + yi**2 + zi**2) + ad
    
    # Storage for path
    xx = [x[0]]
    yy = [y[0]]
    zz = [z[0]]
    
    print(f"\nInitial setup:")
    print(f"  ds_array: {ds_array[0]}")
    print(f"  ad: {ad[0]}")
    print(f"  rr: {rr[0]:.6f}")
    
    # Main integration loop
    for step in range(1, maxloop):
        if not np.any(active_mask):
            print(f"\nStep {step}: All traces inactive, stopping")
            break
        
        print(f"\n\nStep {step}:")
        print("-" * 40)
        
        # Store previous positions
        xr = x.copy()
        yr = y.copy()
        zr = z.copy()
        
        # Calculate current radial distances before step
        r_before = np.sqrt(x**2 + y**2 + z**2)
        print(f"  Position before: ({x[0]:.6f}, {y[0]:.6f}, {z[0]:.6f})")
        print(f"  Radius before: {r_before[0]:.6f}")
        print(f"  Previous radius (rr): {rr[0]:.6f}")
        
        # Store previous radial distances
        rr[active_mask] = r_before[active_mask]
        
        # Adjust step sizes based on radial distance
        # From adjust_step_sizes function
        if r_before[0] < 3:
            fc = 0.05 if (r_before[0] - r0) < 0.05 else 0.2
            al = fc * (r_before[0] - r0 + 0.2)
            ds_array[0] = dir * al
            print(f"  Adjusted step size (r<3): {ds_array[0]}")
        elif 3 <= r_before[0] < 5:
            ds_array[0] = dir
            print(f"  Adjusted step size (3<=r<5): {ds_array[0]}")
        else:
            # Keep current step size
            print(f"  Step size unchanged (r>=5): {ds_array[0]}")
        
        # Perform integration step
        iteration_count = np.zeros(n_traces, dtype=np.int32)
        errin = 0.001
        
        x, y, z = step_vectorized(x, y, z, ds_array, errin, parmod,
                                 exname, inname, active_mask, status,
                                 iteration_count)
        
        # Calculate radial distances after step
        r2 = x**2 + y**2 + z**2
        ryz = y**2 + z**2
        r = np.sqrt(r2)
        
        print(f"  Position after: ({x[0]:.6f}, {y[0]:.6f}, {z[0]:.6f})")
        print(f"  Radius after: {r[0]:.6f}")
        print(f"  Iterations: {iteration_count[0]}")
        
        # Check outer boundary conditions
        print(f"\n  Boundary checks:")
        print(f"    r >= rlim: {r[0] >= rlim} ({r[0]:.6f} >= {rlim})")
        print(f"    ryz >= 1600: {ryz[0] >= 1600} ({ryz[0]:.6f} >= 1600)")
        print(f"    x >= 20: {x[0] >= 20} ({x[0]:.6f} >= 20)")
        
        mask_outer = ((r >= rlim) | (ryz >= 1600) | (x >= 20)) & active_mask
        if np.any(mask_outer):
            print(f"  OUTER BOUNDARY HIT!")
            # Check for radial crossing with interpolation
            mask_r_cross = (r >= rlim) & (rr < rlim) & active_mask
            if np.any(mask_r_cross):
                print(f"  Radial crossing detected (rr={rr[0]:.6f} < {rlim} <= r={r[0]:.6f})")
                # Interpolate to exact boundary crossing
                r1_interp = (rlim - rr[mask_r_cross]) / (r[mask_r_cross] - rr[mask_r_cross])
                x[mask_r_cross] = xr[mask_r_cross] + (x[mask_r_cross] - xr[mask_r_cross]) * r1_interp
                y[mask_r_cross] = yr[mask_r_cross] + (y[mask_r_cross] - yr[mask_r_cross]) * r1_interp
                z[mask_r_cross] = zr[mask_r_cross] + (z[mask_r_cross] - zr[mask_r_cross]) * r1_interp
                print(f"  Interpolation factor: {r1_interp[0]:.6f}")
                print(f"  Interpolated position: ({x[0]:.6f}, {y[0]:.6f}, {z[0]:.6f})")
                print(f"  Interpolated radius: {np.sqrt(x[0]**2 + y[0]**2 + z[0]**2):.6f}")
            
            status[mask_outer] = 1
            active_mask[mask_outer] = False
            xx.append(x[0])
            yy.append(y[0])
            zz.append(z[0])
            break
        
        # Check inner boundary
        mask_inner_cross = (r < r0) & (rr > r) & active_mask
        if np.any(mask_inner_cross):
            print(f"  INNER BOUNDARY HIT!")
            status[mask_inner_cross] = 0
            active_mask[mask_inner_cross] = False
            break
        
        # Store position
        xx.append(x[0])
        yy.append(y[0])
        zz.append(z[0])
    
    print(f"\n\nFinal Summary:")
    print("-" * 40)
    print(f"Total steps: {len(xx)}")
    print(f"Final position: ({x[0]:.6f}, {y[0]:.6f}, {z[0]:.6f})")
    print(f"Final radius: {np.sqrt(x[0]**2 + y[0]**2 + z[0]**2):.6f}")
    print(f"Status: {status[0]}")


if __name__ == "__main__":
    debug_full_vectorized_trace()