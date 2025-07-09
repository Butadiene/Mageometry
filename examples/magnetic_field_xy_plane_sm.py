#!/usr/bin/env python
"""
Plot magnetic field vectors on the XY plane (Z=0) of the SM coordinate system.
Shows how the magnetic field pattern looks in the equatorial plane of SM coordinates.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os

# Add parent directory to path to import geopack
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geopack
from geopack.vectorized import t96_vectorized
from geopack.coordinates_vectorized import smgsm_vectorized


def calculate_field_xy_plane_sm(x_sm, y_sm, z_sm, parmod, ps):
    """
    Calculate magnetic field at given points in SM coordinates.
    Returns field components in SM coordinates.
    """
    # Convert SM to GSM for field calculation
    x_gsm, y_gsm, z_gsm = smgsm_vectorized(x_sm, y_sm, z_sm, 1)
    
    # Calculate field in GSM
    bx_gsm, by_gsm, bz_gsm = t96_vectorized(parmod, ps, x_gsm, y_gsm, z_gsm)
    
    # Convert field back to SM
    bx_sm, by_sm, bz_sm = smgsm_vectorized(bx_gsm, by_gsm, bz_gsm, -1)
    
    return bx_sm, by_sm, bz_sm


def plot_field_xy_plane(date, parmod, extent=10, grid_density=20):
    """
    Plot magnetic field vectors on the XY plane (Z=0) in SM coordinates.
    """
    # Set up time
    ut = date.timestamp()
    ps = geopack.recalc(ut)
    
    # Create grid in SM coordinates
    x_range = np.linspace(-extent, extent, grid_density)
    y_range = np.linspace(-extent, extent, grid_density)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)  # XY plane at Z=0
    
    # Flatten for vectorized calculation
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    
    # Calculate field
    print(f"Calculating field at {len(x_flat)} points...")
    bx_sm, by_sm, bz_sm = calculate_field_xy_plane_sm(x_flat, y_flat, z_flat, parmod, ps)
    
    # Reshape back to grid
    BX = bx_sm.reshape(X.shape)
    BY = by_sm.reshape(Y.shape)
    BZ = bz_sm.reshape(Z.shape)
    
    # Calculate field magnitude
    B_mag = np.sqrt(BX**2 + BY**2 + BZ**2)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Field vectors in XY plane with magnitude coloring
    ax1.set_aspect('equal')
    
    # Create streamplot for field lines
    strm = ax1.streamplot(X, Y, BX, BY, 
                         color=B_mag, 
                         cmap='viridis',
                         density=1.5,
                         linewidth=1.5,
                         arrowsize=1.5)
    
    # Add colorbar
    cbar1 = plt.colorbar(strm.lines, ax=ax1)
    cbar1.set_label('|B| (nT)', fontsize=12)
    
    # Add Earth
    earth = plt.Circle((0, 0), 1.0, color='blue', alpha=0.3, label='Earth')
    ax1.add_patch(earth)
    
    # Add coordinate axes
    ax1.arrow(0, 0, extent*0.8, 0, head_width=0.5, head_length=0.5, 
             fc='red', ec='red', linewidth=2, alpha=0.7)
    ax1.arrow(0, 0, 0, extent*0.8, head_width=0.5, head_length=0.5, 
             fc='green', ec='green', linewidth=2, alpha=0.7)
    
    
    # Add SM longitude labels
    sm_angles_deg = [0, 90, 180, 270]  # SM longitudes
    sm_labels = ['0°', '90°', '180°', '270°']
    for angle_deg, label in zip(sm_angles_deg, sm_labels):
        angle_rad = angle_deg * np.pi / 180
        x_pos = extent * 0.9 * np.cos(angle_rad)
        y_pos = extent * 0.9 * np.sin(angle_rad)
        ax1.text(x_pos, y_pos, label, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                fontsize=10)
    
    ax1.set_xlim(-extent, extent)
    ax1.set_ylim(-extent, extent)
    ax1.set_xlabel('X_SM (Re)', fontsize=12)
    ax1.set_ylabel('Y_SM (Re)', fontsize=12)
    ax1.set_title(f'Magnetic Field in SM XY Plane (Z=0)\n{date.strftime("%Y-%m-%d")}, ps={np.degrees(ps):.1f}°', 
                 fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Bz component (out-of-plane)
    ax2.set_aspect('equal')
    
    # Create contour plot of Bz
    levels = np.linspace(-np.max(np.abs(BZ)), np.max(np.abs(BZ)), 21)
    if len(levels) > 1:
        cf = ax2.contourf(X, Y, BZ, levels=levels, cmap='RdBu_r', extend='both')
        cbar2 = plt.colorbar(cf, ax=ax2)
        cbar2.set_label('B_Z (nT)', fontsize=12)
        
        # Add contour lines
        cs = ax2.contour(X, Y, BZ, levels=[0], colors='black', linewidths=2)
        ax2.clabel(cs, inline=True, fontsize=10)
    
    # Add quiver plot for in-plane components
    skip = 2  # Skip some points for clarity
    ax2.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
              BX[::skip, ::skip], BY[::skip, ::skip],
              scale=100, width=0.002, alpha=0.7)
    
    # Add Earth
    earth2 = plt.Circle((0, 0), 1.0, color='black', fill=False, linewidth=2)
    ax2.add_patch(earth2)
    
    # Add SM longitude labels  
    for angle_deg, label in zip(sm_angles_deg, sm_labels):
        angle_rad = angle_deg * np.pi / 180
        x_pos = extent * 0.9 * np.cos(angle_rad)
        y_pos = extent * 0.9 * np.sin(angle_rad)
        ax2.text(x_pos, y_pos, label, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                fontsize=10)
    
    ax2.set_xlim(-extent, extent)
    ax2.set_ylim(-extent, extent)
    ax2.set_xlabel('X_SM (Re)', fontsize=12)
    ax2.set_ylabel('Y_SM (Re)', fontsize=12)
    ax2.set_title(f'B_Z Component (out of plane)\nwith in-plane vectors', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Main function."""
    # Set date
    date = datetime(2024, 3, 20, 12, 0, 0)  # Spring Equinox
    
    print(f"Plotting magnetic field for: {date}")
    
    # Set T96 model parameters (extreme storm conditions)
    parmod = np.array([10.0, 100.0, 0.0, -30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    print("T96 Parameters:")
    print(f"  Pdyn = {parmod[0]} nPa")
    print(f"  Dst = {parmod[1]} nT")
    print(f"  ByIMF = {parmod[2]} nT")
    print(f"  BzIMF = {parmod[3]} nT")
    
    # Create plots for different extents
    extents = [5, 10, 20]
    
    for extent in extents:
        print(f"\nCreating plot with extent ±{extent} Re...")
        fig = plot_field_xy_plane(date, parmod, extent=extent, grid_density=25)
        
        # Save figure
        filename = f'magnetic_field_xy_sm_extent{extent}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved {filename}")
    
    # Also create seasonal comparison
    print("\nCreating seasonal comparison...")
    dates = [
        ("Spring Equinox", datetime(2024, 3, 20, 12, 0, 0)),
        ("Summer Solstice", datetime(2024, 6, 21, 12, 0, 0))
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for idx, (season, date) in enumerate(dates):
        ax = axes[idx]
        ut = date.timestamp()
        ps = geopack.recalc(ut)
        
        # Create grid
        extent = 10
        grid_density = 20
        x_range = np.linspace(-extent, extent, grid_density)
        y_range = np.linspace(-extent, extent, grid_density)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)
        
        # Calculate field
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        bx_sm, by_sm, bz_sm = calculate_field_xy_plane_sm(x_flat, y_flat, z_flat, parmod, ps)
        
        # Reshape
        BX = bx_sm.reshape(X.shape)
        BY = by_sm.reshape(Y.shape)
        B_mag = np.sqrt(BX**2 + BY**2 + bz_sm.reshape(X.shape)**2)
        
        # Plot
        ax.set_aspect('equal')
        strm = ax.streamplot(X, Y, BX, BY, 
                           color=B_mag, 
                           cmap='viridis',
                           density=1.5,
                           linewidth=1.5)
        
        # Add Earth
        earth = plt.Circle((0, 0), 1.0, color='blue', alpha=0.3)
        ax.add_patch(earth)
        
        # Add SM longitude labels
        sm_angles_deg = [0, 90, 180, 270]
        sm_labels = ['0°', '90°', '180°', '270°']
        for angle_deg, label in zip(sm_angles_deg, sm_labels):
            angle_rad = angle_deg * np.pi / 180
            x_pos = extent * 0.9 * np.cos(angle_rad)
            y_pos = extent * 0.9 * np.sin(angle_rad)
            ax.text(x_pos, y_pos, label, ha='center', va='center',
                   bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', alpha=0.7),
                   fontsize=10)
        
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_xlabel('X_SM (Re)', fontsize=12)
        ax.set_ylabel('Y_SM (Re)', fontsize=12)
        ax.set_title(f'{season} ({date.strftime("%Y-%m-%d")})\nDipole tilt = {np.degrees(ps):.1f}°', 
                    fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(strm.lines, ax=ax)
        cbar.set_label('|B| (nT)', fontsize=10)
    
    fig.suptitle('Magnetic Field in SM XY Plane - Seasonal Comparison\nT96 Model', fontsize=16)
    plt.tight_layout()
    plt.savefig('magnetic_field_xy_sm_seasonal.png', dpi=300, bbox_inches='tight')
    print("Saved magnetic_field_xy_sm_seasonal.png")
    
    plt.show()


if __name__ == '__main__':
    main()