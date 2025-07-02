#!/usr/bin/env python
"""
Debug script to understand the step size issue in vectorized trace.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/skipjack/Documents/geopack-vectorize')

import geopack
from geopack.models import t89
from geopack.trace_vectorized import trace_vectorized, step_vectorized, rhand_vectorized

def debug_first_step():
    """Debug what happens in the first step of vectorized trace."""
    
    # Set up parameters
    ut = 0.0  # Unix time
    ps = geopack.recalc(ut)
    
    # Starting point in tail region
    x0 = np.array([-10.0])
    y0 = np.array([0.0])
    z0 = np.array([2.0])
    
    # Parameters
    dir = 1
    parmod = 3
    exname = 't89'
    inname = 'igrf'
    
    print("Debug First Step of Vectorized Trace")
    print("=" * 80)
    print(f"Starting point: ({x0[0]}, {y0[0]}, {z0[0]})")
    print(f"Starting radius: {np.sqrt(x0[0]**2 + y0[0]**2 + z0[0]**2):.6f}")
    
    # Initial setup from trace_vectorized
    n_traces = 1
    x = x0.copy()
    y = y0.copy()
    z = z0.copy()
    
    # Active mask and status
    active_mask = np.ones(n_traces, dtype=bool)
    status = np.zeros(n_traces, dtype=np.int32)
    
    # Initial step size
    ds_array = np.full(n_traces, 0.5 * dir, dtype=np.float64)
    print(f"\nInitial step size: {ds_array[0]}")
    
    # Check initial field direction
    ds3 = -0.5 * dir / 3.0
    r1, r2, r3 = rhand_vectorized(x, y, z, parmod, exname, inname, ds3)
    br = (x * r1 + y * r2 + z * r3)
    print(f"Initial Br (radial component): {br[0]:.6f}")
    
    # Initial ad
    ad = np.where(br < 0, -0.01, 0.01)
    if dir < 0:
        ad = -ad
    print(f"Initial ad: {ad[0]}")
    
    # Previous radial distance
    rr = np.sqrt(x**2 + y**2 + z**2) + ad
    print(f"Initial rr: {rr[0]:.6f}")
    
    # Now perform the first step
    print("\n\nFirst Integration Step:")
    print("-" * 40)
    
    # Store previous positions
    xr = x.copy()
    yr = y.copy()
    zr = z.copy()
    
    # Calculate current radial distance
    r_before = np.sqrt(x**2 + y**2 + z**2)
    print(f"Radius before step: {r_before[0]:.6f}")
    
    # Perform step
    iteration_count = np.zeros(n_traces, dtype=np.int32)
    errin = 0.001
    
    print(f"\nCalling step_vectorized with:")
    print(f"  Position: ({x[0]:.6f}, {y[0]:.6f}, {z[0]:.6f})")
    print(f"  Step size: {ds_array[0]}")
    print(f"  Error tolerance: {errin}")
    
    x_new, y_new, z_new = step_vectorized(x, y, z, ds_array, errin, parmod,
                                          exname, inname, active_mask, status,
                                          iteration_count)
    
    print(f"\nAfter step_vectorized:")
    print(f"  New position: ({x_new[0]:.6f}, {y_new[0]:.6f}, {z_new[0]:.6f})")
    print(f"  Iteration count: {iteration_count[0]}")
    
    # Calculate new radius
    r_new = np.sqrt(x_new[0]**2 + y_new[0]**2 + z_new[0]**2)
    print(f"  New radius: {r_new:.6f}")
    
    # Check boundary conditions
    rlim = 15.0
    print(f"\nBoundary check:")
    print(f"  r >= rlim: {r_new >= rlim} ({r_new:.6f} >= {rlim})")
    print(f"  rr < rlim: {rr[0] < rlim} ({rr[0]:.6f} < {rlim})")
    
    # Show the interpolation that would happen
    if r_new >= rlim and rr[0] < rlim:
        r1 = (rlim - rr[0]) / (r_new - rr[0])
        x_interp = xr[0] + (x_new[0] - xr[0]) * r1
        y_interp = yr[0] + (y_new[0] - yr[0]) * r1
        z_interp = zr[0] + (z_new[0] - zr[0]) * r1
        r_interp = np.sqrt(x_interp**2 + y_interp**2 + z_interp**2)
        print(f"\nInterpolation would give:")
        print(f"  t = {r1:.6f}")
        print(f"  Position: ({x_interp:.6f}, {y_interp:.6f}, {z_interp:.6f})")
        print(f"  Radius: {r_interp:.6f}")
    
    # Compare with scalar trace for same starting point
    print("\n\nComparison with Scalar Trace:")
    print("-" * 40)
    
    from geopack.geopack import trace
    xf, yf, zf, xx, yy, zz = trace(
        x0[0], y0[0], z0[0], dir, rlim=15.0, r0=1.0, 
        parmod=parmod, exname=exname, inname=inname, maxloop=10
    )
    
    print(f"Scalar trace first few steps:")
    for i in range(min(5, len(xx))):
        r = np.sqrt(xx[i]**2 + yy[i]**2 + zz[i]**2)
        print(f"  Step {i}: ({xx[i]:.6f}, {yy[i]:.6f}, {zz[i]:.6f}), r = {r:.6f}")


if __name__ == "__main__":
    debug_first_step()