#!/usr/bin/env python3
"""
Example demonstrating magnetic field line geometry analysis.

This script shows how to calculate the Frenet-Serret frame,
curvature, and torsion of magnetic field lines.
"""

import numpy as np
import matplotlib.pyplot as plt
import geopack
from geopack import (
    t96_vectorized,
    field_line_tangent_vectorized,
    field_line_curvature_vectorized,
    field_line_geometry_complete_vectorized
)


def main():
    """Demonstrate field line geometry calculations."""
    
    # Initialize geopack with a specific time
    ut = 1600000000  # Unix timestamp
    ps = geopack.recalc(ut)
    
    # Set up T96 model parameters
    # [Pdyn, Dst, ByIMF, BzIMF, ...]
    parmod = [2.0, -20.0, 0.5, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    print("Magnetic Field Line Geometry Analysis")
    print("=" * 40)
    print(f"Model: T96")
    print(f"Pdyn = {parmod[0]} nPa, Dst = {parmod[1]} nT")
    print(f"ByIMF = {parmod[2]} nT, BzIMF = {parmod[3]} nT")
    print(f"Dipole tilt = {np.degrees(ps):.2f}°\n")
    
    # Example 1: Single point analysis
    print("Example 1: Single Point Analysis")
    print("-" * 40)
    x, y, z = 5.0, 2.0, 1.0  # Position in Re
    
    # Get tangent vector
    tx, ty, tz = field_line_tangent_vectorized(t96_vectorized, parmod, ps, x, y, z)
    print(f"Position: ({x}, {y}, {z}) Re")
    print(f"Tangent vector: T = ({tx:.4f}, {ty:.4f}, {tz:.4f})")
    
    # Get curvature
    curvature = field_line_curvature_vectorized(t96_vectorized, parmod, ps, x, y, z)
    print(f"Curvature: κ = {curvature:.4f} Re⁻¹")
    print(f"Radius of curvature: R = {1/curvature:.2f} Re")
    
    # Example 2: Complete geometry at multiple points
    print("\nExample 2: Geometry Along a Meridian")
    print("-" * 40)
    
    # Points along X-axis at different distances
    r = np.linspace(3, 10, 8)
    x_arr = r
    y_arr = np.zeros_like(r)
    z_arr = np.zeros_like(r)
    
    # Calculate complete geometry
    result = field_line_geometry_complete_vectorized(
        t96_vectorized, parmod, ps, x_arr, y_arr, z_arr
    )
    tx, ty, tz, nx, ny, nz, bx, by, bz, curvature, torsion = result
    
    print("Distance  Curvature   Torsion    |B|")
    print("  (Re)     (Re⁻¹)     (Re⁻¹)    (nT)")
    print("-" * 40)
    
    for i in range(len(r)):
        # Calculate field strength
        bx_field, by_field, bz_field = t96_vectorized(parmod, ps, x_arr[i], y_arr[i], z_arr[i])
        b_mag = np.sqrt(bx_field**2 + by_field**2 + bz_field**2)
        
        print(f"{r[i]:6.1f}   {curvature[i]:9.4f}  {torsion[i]:9.4f}  {b_mag:7.1f}")
    
    # Example 3: 2D visualization of curvature
    print("\nExample 3: Curvature Map in Equatorial Plane")
    print("-" * 40)
    
    # Create grid
    x_grid = np.linspace(-10, 10, 41)
    y_grid = np.linspace(-10, 10, 41)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    
    # Flatten for vectorized calculation
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    
    # Calculate curvature
    curvature_flat = field_line_curvature_vectorized(
        t96_vectorized, parmod, ps, x_flat, y_flat, z_flat
    )
    curvature_grid = curvature_flat.reshape(X.shape)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot curvature
    levels = np.logspace(-2, 1, 20)
    cs = plt.contourf(X, Y, curvature_grid, levels=levels, cmap='viridis', extend='both')
    plt.colorbar(cs, label='Curvature (Re⁻¹)')
    
    # Add Earth
    earth = plt.Circle((0, 0), 1, color='white', zorder=10)
    plt.gca().add_patch(earth)
    
    # Add contour lines
    plt.contour(X, Y, curvature_grid, levels=[0.1, 0.3, 1.0, 3.0], 
                colors='white', linewidths=0.5, alpha=0.5)
    
    plt.xlabel('X GSM (Re)')
    plt.ylabel('Y GSM (Re)')
    plt.title('Field Line Curvature in Equatorial Plane (T96 Model)')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    
    plt.tight_layout()
    plt.savefig('field_line_curvature_map.png', dpi=300, bbox_inches='tight')
    print("Curvature map saved as 'field_line_curvature_map.png'")
    
    # Example 4: Field line properties along a trace
    print("\nExample 4: Properties Along a Field Line")
    print("-" * 40)
    
    # Trace along field line starting from (5, 0, 0)
    start_x, start_y, start_z = 5.0, 0.0, 0.0
    steps = 20
    ds = 0.2  # Step size in Re
    
    # Arrays to store path
    path_x = np.zeros(steps)
    path_y = np.zeros(steps)
    path_z = np.zeros(steps)
    path_curvature = np.zeros(steps)
    path_torsion = np.zeros(steps)
    
    # Initial position
    path_x[0] = start_x
    path_y[0] = start_y
    path_z[0] = start_z
    
    # Trace field line
    for i in range(steps):
        # Get geometry at current position
        tx, ty, tz, _, _, _, _, _, _, curv, tors = field_line_geometry_complete_vectorized(
            t96_vectorized, parmod, ps, path_x[i], path_y[i], path_z[i]
        )
        
        path_curvature[i] = curv
        path_torsion[i] = tors
        
        # Step along field line
        if i < steps - 1:
            path_x[i+1] = path_x[i] + ds * tx
            path_y[i+1] = path_y[i] + ds * ty
            path_z[i+1] = path_z[i] + ds * tz
    
    # Calculate arc length
    arc_length = np.cumsum(np.sqrt(np.diff(path_x)**2 + np.diff(path_y)**2 + np.diff(path_z)**2))
    arc_length = np.insert(arc_length, 0, 0)
    
    print("Arc Length  Position (Re)         Curvature  Torsion")
    print("   (Re)      X      Y      Z      (Re⁻¹)    (Re⁻¹)")
    print("-" * 55)
    
    for i in range(0, steps, 2):  # Print every other point
        print(f"{arc_length[i]:7.2f}  {path_x[i]:6.2f} {path_y[i]:6.2f} {path_z[i]:6.2f}  "
              f"{path_curvature[i]:9.4f} {path_torsion[i]:9.4f}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()