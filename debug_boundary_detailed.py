#!/usr/bin/env python
"""
Detailed debug script to understand why vectorized trace takes only one step.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/skipjack/Documents/geopack-vectorize')

import geopack
from geopack.models import t89
from geopack.vectorized.t89_vectorized import t89_vectorized

def debug_step_by_step():
    """Debug the vectorized trace step by step."""
    
    # Set up parameters
    ut = 0.0  # Unix time
    ps = geopack.recalc(ut)
    
    # Starting point in tail region
    x0, y0, z0 = -10.0, 0.0, 2.0
    r0_start = np.sqrt(x0**2 + y0**2 + z0**2)
    
    print("Debug Vectorized Field Line Trace")
    print("=" * 80)
    print(f"Starting point: ({x0}, {y0}, {z0})")
    print(f"Starting radius: {r0_start:.6f}")
    
    # Parameters
    dir = 1  # North to south
    rlim = 15.0  # Outer boundary
    r0 = 1.0  # Inner boundary
    parmod = 3  # Kp index for T89
    
    print(f"\nParameters:")
    print(f"  Direction: {dir} (north to south)")
    print(f"  Outer boundary: {rlim} Re")
    print(f"  Inner boundary: {r0} Re")
    print(f"  Model: T89 with Kp={parmod}")
    
    # Get initial field
    print("\nInitial field:")
    bx0, by0, bz0 = t89(parmod, ps, x0, y0, z0)
    b_mag0 = np.sqrt(bx0**2 + by0**2 + bz0**2)
    print(f"  B = ({bx0:.6f}, {by0:.6f}, {bz0:.6f})")
    print(f"  |B| = {b_mag0:.6f}")
    
    # Calculate step direction
    bx_norm = bx0 / b_mag0
    by_norm = by0 / b_mag0
    bz_norm = bz0 / b_mag0
    
    # Initial step size (from trace_vectorized.py)
    ds = 0.5 * dir
    print(f"\nInitial step size: {ds}")
    
    # Take a simple Euler step to see where we'd go
    x1_euler = x0 + ds * bx_norm * dir
    y1_euler = y0 + ds * by_norm * dir
    z1_euler = z0 + ds * bz_norm * dir
    r1_euler = np.sqrt(x1_euler**2 + y1_euler**2 + z1_euler**2)
    
    print(f"\nSimple Euler step would give:")
    print(f"  Position: ({x1_euler:.6f}, {y1_euler:.6f}, {z1_euler:.6f})")
    print(f"  Radius: {r1_euler:.6f}")
    print(f"  Would hit boundary: {r1_euler >= rlim}")
    
    # Now let's manually trace using the actual RK method
    print("\n\nManual RK5 Integration:")
    print("-" * 40)
    
    # From trace_vectorized.py - the RK5 coefficients
    a = [0., 0.2, 0.3, 0.6, 1., 0.875]
    b = [[0.],
         [0.2],
         [3./40., 9./40.],
         [0.3, -0.9, 1.2],
         [-11./54., 2.5, -70./27., 35./27.],
         [1631./55296., 175./512., 575./13824., 44275./110592., 253./4096.]]
    c = [37./378., 0., 250./621., 125./594., 0., 512./1771.]
    
    # Current position
    x, y, z = x0, y0, z0
    ds = 0.5  # Initial step size
    
    for step in range(5):
        print(f"\nStep {step}:")
        print(f"  Starting position: ({x:.6f}, {y:.6f}, {z:.6f})")
        print(f"  Starting radius: {np.sqrt(x**2 + y**2 + z**2):.6f}")
        
        # RK5 k values
        k = np.zeros((6, 3))
        
        # k1
        bx, by, bz = t89_vectorized(parmod, ps, x, y, z)
        b_mag = np.sqrt(bx**2 + by**2 + bz**2)
        k[0] = [bx/b_mag * dir, by/b_mag * dir, bz/b_mag * dir]
        
        # k2 through k6
        for i in range(1, 6):
            xt = x + ds * sum(b[i][j] * k[j][0] for j in range(i))
            yt = y + ds * sum(b[i][j] * k[j][1] for j in range(i))
            zt = z + ds * sum(b[i][j] * k[j][2] for j in range(i))
            
            bx, by, bz = t89_vectorized(parmod, ps, xt, yt, zt)
            b_mag = np.sqrt(bx**2 + by**2 + bz**2)
            k[i] = [bx/b_mag * dir, by/b_mag * dir, bz/b_mag * dir]
        
        # Update position
        dx = ds * sum(c[i] * k[i][0] for i in range(6))
        dy = ds * sum(c[i] * k[i][1] for i in range(6))
        dz = ds * sum(c[i] * k[i][2] for i in range(6))
        
        x_new = x + dx
        y_new = y + dy
        z_new = z + dz
        r_new = np.sqrt(x_new**2 + y_new**2 + z_new**2)
        
        print(f"  Step increment: ({dx:.6f}, {dy:.6f}, {dz:.6f})")
        print(f"  New position: ({x_new:.6f}, {y_new:.6f}, {z_new:.6f})")
        print(f"  New radius: {r_new:.6f}")
        print(f"  Would hit boundary: {r_new >= rlim}")
        
        if r_new >= rlim:
            print(f"  BOUNDARY HIT! Need to interpolate.")
            # Interpolate to boundary
            r_current = np.sqrt(x**2 + y**2 + z**2)
            t = (rlim - r_current) / (r_new - r_current)
            x_boundary = x + t * (x_new - x)
            y_boundary = y + t * (y_new - y)
            z_boundary = z + t * (z_new - z)
            r_boundary = np.sqrt(x_boundary**2 + y_boundary**2 + z_boundary**2)
            print(f"  Interpolated position: ({x_boundary:.6f}, {y_boundary:.6f}, {z_boundary:.6f})")
            print(f"  Interpolated radius: {r_boundary:.6f}")
            break
        
        x, y, z = x_new, y_new, z_new
    
    # Check what's happening with the step size adjustment
    print("\n\nStep Size Analysis:")
    print("-" * 40)
    
    # From adjust_step_sizes in trace_vectorized.py
    r = np.sqrt(x0**2 + y0**2 + z0**2)
    print(f"Initial radius: {r:.6f}")
    
    # The adjustment logic
    if r < 3.:
        ds_new = dir * 0.06
    elif r < 6.:
        ds_new = dir * 0.3
    elif r < 10.:
        ds_new = dir * 0.75
    elif r < 15.:
        ds_new = dir * 1.5
    elif r < 25.:
        ds_new = dir * 2.5
    elif r < 40.:
        ds_new = dir * 4.0
    elif r < 60.:
        ds_new = dir * 5.5
    else:
        ds_new = dir * 7.25
    
    print(f"Adjusted step size would be: {ds_new}")
    
    # Test with this step size
    print(f"\nWith adjusted step size {ds_new}:")
    x1_adj = x0 + ds_new * bx_norm
    y1_adj = y0 + ds_new * by_norm
    z1_adj = z0 + ds_new * bz_norm
    r1_adj = np.sqrt(x1_adj**2 + y1_adj**2 + z1_adj**2)
    print(f"  New position: ({x1_adj:.6f}, {y1_adj:.6f}, {z1_adj:.6f})")
    print(f"  New radius: {r1_adj:.6f}")
    print(f"  Would hit boundary: {r1_adj >= rlim}")


if __name__ == "__main__":
    debug_step_by_step()