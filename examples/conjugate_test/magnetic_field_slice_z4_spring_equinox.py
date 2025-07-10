#!/usr/bin/env python
"""
Magnetic field visualization at z=4 Re slice for Spring Equinox at 8:00 AM UTC.
Creates detailed plots of the magnetic field in the XY plane at z=4 Re.
Uses T89 magnetospheric model with SM (Solar Magnetic) coordinates.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os

# Add parent directory to path to import geopack
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geopack
from geopack.vectorized import t89_vectorized
from geopack.coordinates_vectorized import smgsm_vectorized
from geopack.igrf_vectorized import igrf_gsm_vectorized


def compute_field_slice(extent=20, resolution=50, z_level=4.0, ut=None, parmod=None):
    """
    Compute magnetic field in XY plane at specified z level.
    """
    # Update geopack parameters
    ps = geopack.recalc(ut)
    
    # Create grid
    x = np.linspace(-extent, extent, resolution)
    y = np.linspace(-extent, extent, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, z_level)
    
    # Flatten
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    
    # Convert SM to GSM
    x_gsm, y_gsm, z_gsm = smgsm_vectorized(x_flat, y_flat, z_flat, 1)
    
    # External field (T89)
    iopt = int(parmod[0])
    bx_ext_gsm, by_ext_gsm, bz_ext_gsm = t89_vectorized(iopt, ps, x_gsm, y_gsm, z_gsm)
    
    # Internal field (IGRF)
    bx_int_gsm, by_int_gsm, bz_int_gsm = igrf_gsm_vectorized(x_gsm, y_gsm, z_gsm)
    
    # Total field in GSM
    bx_tot_gsm = bx_ext_gsm + bx_int_gsm
    by_tot_gsm = by_ext_gsm + by_int_gsm
    bz_tot_gsm = bz_ext_gsm + bz_int_gsm
    
    # Convert back to SM
    bx_sm, by_sm, bz_sm = smgsm_vectorized(bx_tot_gsm, by_tot_gsm, bz_tot_gsm, -1)
    
    # Reshape
    BX = bx_sm.reshape(resolution, resolution)
    BY = by_sm.reshape(resolution, resolution)
    BZ = bz_sm.reshape(resolution, resolution)
    
    # Calculate magnitude
    B_MAG = np.sqrt(BX**2 + BY**2 + BZ**2)
    
    return X, Y, BX, BY, BZ, B_MAG


def create_slice_plots(ut, parmod, z_level=4.0):
    """
    Create comprehensive plots for specified z level slice.
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Get dipole tilt
    ps = geopack.recalc(ut)
    dipole_tilt = np.degrees(ps)
    
    # Compute field at specified z
    X, Y, BX, BY, BZ, B_MAG = compute_field_slice(extent=20, resolution=60, z_level=z_level, 
                                                  ut=ut, parmod=parmod)
    
    # 1. Streamlines with magnitude background
    ax1 = plt.subplot(2, 2, 1)
    
    # Background: field magnitude
    im1 = ax1.contourf(X, Y, np.log10(B_MAG + 1), levels=20, cmap='viridis', alpha=0.7)
    
    # Streamlines
    strm = ax1.streamplot(X[0,:], Y[:,0], BX.T, BY.T, 
                         density=2, color='white', linewidth=1.5, 
                         arrowsize=1.5, arrowstyle='->')
    
    # Add contour lines for magnitude
    contours = ax1.contour(X, Y, B_MAG, levels=[10, 50, 100, 200, 500], 
                          colors='black', linewidths=0.5, alpha=0.5)
    ax1.clabel(contours, inline=True, fontsize=8, fmt='%d nT')
    
    ax1.set_xlim(-20, 20)
    ax1.set_ylim(-20, 20)
    ax1.set_xlabel('X_SM (Re)')
    ax1.set_ylabel('Y_SM (Re)')
    ax1.set_title(f'Magnetic Field at Z={z_level} Re\nStreamlines with |B| background')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('log₁₀(|B|+1) [nT]')
    
    # 2. Vector field (quiver plot)
    ax2 = plt.subplot(2, 2, 2)
    
    # Subsample for quiver
    skip = 3
    x_sub = X[::skip, ::skip]
    y_sub = Y[::skip, ::skip]
    bx_sub = BX[::skip, ::skip]
    by_sub = BY[::skip, ::skip]
    b_mag_sub = B_MAG[::skip, ::skip]
    
    # Normalize vectors
    b_mag_safe = np.where(b_mag_sub > 0, b_mag_sub, 1.0)
    bx_norm = bx_sub / b_mag_safe
    by_norm = by_sub / b_mag_safe
    
    # Quiver plot
    Q = ax2.quiver(x_sub, y_sub, bx_norm, by_norm, b_mag_sub,
                   scale=20, scale_units='xy', cmap='plasma',
                   norm=plt.Normalize(vmin=0, vmax=200))
    
    ax2.set_xlim(-20, 20)
    ax2.set_ylim(-20, 20)
    ax2.set_xlabel('X_SM (Re)')
    ax2.set_ylabel('Y_SM (Re)')
    ax2.set_title(f'Vector Field at Z={z_level} Re')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    cbar2 = plt.colorbar(Q, ax=ax2)
    cbar2.set_label('|B| [nT]')
    
    # 3. Vertical component (Bz)
    ax3 = plt.subplot(2, 2, 3)
    
    # Plot Bz component
    im3 = ax3.contourf(X, Y, BZ, levels=20, cmap='RdBu_r', 
                      vmin=-np.max(np.abs(BZ)), vmax=np.max(np.abs(BZ)))
    
    # Add contour lines
    contours_bz = ax3.contour(X, Y, BZ, levels=[-100, -50, -20, 0, 20, 50, 100], 
                             colors='black', linewidths=0.5)
    ax3.clabel(contours_bz, inline=True, fontsize=8, fmt='%d nT')
    
    ax3.set_xlim(-20, 20)
    ax3.set_ylim(-20, 20)
    ax3.set_xlabel('X_SM (Re)')
    ax3.set_ylabel('Y_SM (Re)')
    ax3.set_title(f'Vertical Component B_z at Z={z_level} Re')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label('B_z [nT]')
    
    # 4. Field line mapping projection
    ax4 = plt.subplot(2, 2, 4)
    
    # Calculate field line footprint mapping
    # Approximate by projecting along field direction
    r = np.sqrt(X**2 + Y**2 + 16)  # z=4, so z²=16
    
    # Plot radial distance from z-axis
    r_cyl = np.sqrt(X**2 + Y**2)
    im4 = ax4.contourf(X, Y, r_cyl, levels=20, cmap='rainbow')
    
    # Add field magnitude contours
    contours4 = ax4.contour(X, Y, B_MAG, levels=[10, 50, 100, 200], 
                           colors='white', linewidths=1, alpha=0.7)
    
    # Add radial lines from origin
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    for angle in angles:
        ax4.plot([0, 20*np.cos(angle)], [0, 20*np.sin(angle)], 
                'k--', alpha=0.3, linewidth=0.5)
    
    ax4.set_xlim(-20, 20)
    ax4.set_ylim(-20, 20)
    ax4.set_xlabel('X_SM (Re)')
    ax4.set_ylabel('Y_SM (Re)')
    ax4.set_title(f'Cylindrical Radius at Z={z_level} Re')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    
    cbar4 = plt.colorbar(im4, ax=ax4)
    cbar4.set_label('Cylindrical Radius (Re)')
    
    # Overall title
    fig.suptitle(f'Magnetic Field Analysis at Z={z_level} Re (SM Coordinates)\n' +
                 f'Spring Equinox (March 20, 2024) at 08:00 UTC\n' +
                 f'T89 Model (Kp=0), Dipole Tilt: {dipole_tilt:.1f}°', 
                 fontsize=16)
    
    plt.tight_layout()
    return fig


def create_3d_context_plot(ut, parmod, z_level=4.0):
    """
    Create a 3D plot showing where the z slice is located.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create the z plane
    x = np.linspace(-20, 20, 20)
    y = np.linspace(-20, 20, 20)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, z_level)
    
    # Plot the plane
    ax.plot_surface(X, Y, Z, alpha=0.3, color='cyan', edgecolor='none')
    
    # Add Earth sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_earth = np.outer(np.cos(u), np.sin(v))
    y_earth = np.outer(np.sin(u), np.sin(v))
    z_earth = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x_earth, y_earth, z_earth, color='blue', alpha=0.6)
    
    # Add coordinate axes
    ax.plot([0, 10], [0, 0], [0, 0], 'r-', linewidth=2, label='X_SM')
    ax.plot([0, 0], [0, 10], [0, 0], 'g-', linewidth=2, label='Y_SM')
    ax.plot([0, 0], [0, 0], [0, 10], 'b-', linewidth=2, label='Z_SM')
    
    # Add some reference field lines
    ps = geopack.recalc(ut)
    
    # Trace a few field lines from the z plane
    start_points = [
        (5, 0, z_level), (-5, 0, z_level), (0, 5, z_level), (0, -5, z_level),
        (10, 0, z_level), (-10, 0, z_level), (0, 10, z_level), (0, -10, z_level)
    ]
    
    for x0, y0, z0 in start_points:
        # Simple dipole approximation for illustration
        r0 = np.sqrt(x0**2 + y0**2 + z0**2)
        lat0 = np.arcsin(z0/r0)
        lon0 = np.arctan2(y0, x0)
        
        # Create field line path
        s = np.linspace(0, 2, 50)
        r = r0 * np.exp(-s)
        x_line = r * np.cos(lat0) * np.cos(lon0)
        y_line = r * np.cos(lat0) * np.sin(lon0)
        z_line = r * np.sin(lat0)
        
        # Only plot points outside Earth
        mask = r > 1.0
        ax.plot(x_line[mask], y_line[mask], z_line[mask], 
               'gray', alpha=0.5, linewidth=1)
    
    # Add text label for the plane
    ax.text(0, 0, z_level, f'Z = {z_level} Re', fontsize=14, ha='center', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-5, max(15, z_level + 5))
    ax.set_xlabel('X_SM (Re)')
    ax.set_ylabel('Y_SM (Re)')
    ax.set_zlabel('Z_SM (Re)')
    ax.set_title(f'Location of Z={z_level} Re Analysis Plane', fontsize=14)
    ax.view_init(elev=20, azim=45)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig


def main():
    """Main function."""
    # Set time to Spring Equinox (March 20, 2024) at 8:00 AM UTC
    spring_equinox = datetime(2024, 3, 20, 8, 0, 0)
    ut = spring_equinox.timestamp()
    ps = geopack.recalc(ut)
    
    print(f"Using date: {spring_equinox}")
    print(f"Dipole tilt angle: {np.degrees(ps):.1f}°")
    
    # Set T89 model parameters (quiet conditions, Kp=0)
    parmod = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Process both z=4 and z=8
    z_levels = [4.0, 8.0]
    
    for z_level in z_levels:
        # Create slice analysis
        print(f"\nCreating magnetic field analysis at Z={z_level} Re...")
        fig_slice = create_slice_plots(ut, parmod, z_level=z_level)
        
        # Save slice plots
        output_file_slice = os.path.join(os.path.dirname(__file__), 
                                        f'magnetic_field_z{int(z_level)}_slice_spring_equinox.png')
        fig_slice.savefig(output_file_slice, dpi=300, bbox_inches='tight')
        print(f"Saved {output_file_slice}")
        
        # Create 3D context plot
        print(f"\nCreating 3D context visualization for Z={z_level} Re...")
        fig_3d = create_3d_context_plot(ut, parmod, z_level=z_level)
        
        # Save 3D context
        output_file_3d = os.path.join(os.path.dirname(__file__), 
                                     f'magnetic_field_z{int(z_level)}_context_spring_equinox.png')
        fig_3d.savefig(output_file_3d, dpi=300, bbox_inches='tight')
        print(f"Saved {output_file_3d}")
        
        # Print some statistics
        print(f"\nField statistics at Z={z_level} Re:")
        X, Y, BX, BY, BZ, B_MAG = compute_field_slice(extent=20, resolution=60, 
                                                      z_level=z_level, ut=ut, parmod=parmod)
        print(f"  Max field magnitude: {np.max(B_MAG):.1f} nT")
        print(f"  Min field magnitude: {np.min(B_MAG):.1f} nT")
        print(f"  Mean field magnitude: {np.mean(B_MAG):.1f} nT")
        print(f"  Max Bz component: {np.max(BZ):.1f} nT")
        print(f"  Min Bz component: {np.min(BZ):.1f} nT")


if __name__ == '__main__':
    main()