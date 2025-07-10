#!/usr/bin/env python
"""
3D Visualization of magnetic field vectors for Spring Equinox at 8:00 AM UTC.
Creates a 3D quiver plot showing magnetic field vectors at various points in space.
Uses T89 magnetospheric model with SM (Solar Magnetic) coordinates.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as mcolors
from datetime import datetime
import sys
import os

# Add parent directory to path to import geopack
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geopack
from geopack.vectorized import t89_vectorized
from geopack.coordinates_vectorized import smgsm_vectorized


def create_3d_grid(xrange=(-15, 15), yrange=(-15, 15), zrange=(-10, 10), 
                   nx=8, ny=8, nz=6):
    """
    Create a 3D grid of points for vector field visualization.
    """
    x = np.linspace(xrange[0], xrange[1], nx)
    y = np.linspace(yrange[0], yrange[1], ny)
    z = np.linspace(zrange[0], zrange[1], nz)
    
    # Create 3D meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Flatten for processing
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    
    # Remove points inside Earth (r < 1 Re)
    r = np.sqrt(x_flat**2 + y_flat**2 + z_flat**2)
    mask = r > 1.5  # Keep points outside 1.5 Re for clarity
    
    return x_flat[mask], y_flat[mask], z_flat[mask]


def compute_magnetic_field(x_sm, y_sm, z_sm, ut, parmod):
    """
    Compute magnetic field at given points in SM coordinates.
    """
    # Update geopack parameters
    ps = geopack.recalc(ut)
    
    # Convert SM to GSM for field calculation
    x_gsm, y_gsm, z_gsm = smgsm_vectorized(x_sm, y_sm, z_sm, 1)
    
    # Calculate total field (external + internal)
    # External field (T89)
    # For T89, first parameter is iopt (Kp index), convert to int
    iopt = int(parmod[0])
    bx_ext_gsm, by_ext_gsm, bz_ext_gsm = t89_vectorized(iopt, ps, x_gsm, y_gsm, z_gsm)
    
    # Internal field (IGRF)
    from geopack.igrf_vectorized import igrf_gsm_vectorized
    bx_int_gsm, by_int_gsm, bz_int_gsm = igrf_gsm_vectorized(x_gsm, y_gsm, z_gsm)
    
    # Total field in GSM
    bx_tot_gsm = bx_ext_gsm + bx_int_gsm
    by_tot_gsm = by_ext_gsm + by_int_gsm
    bz_tot_gsm = bz_ext_gsm + bz_int_gsm
    
    # Convert back to SM
    bx_sm, by_sm, bz_sm = smgsm_vectorized(bx_tot_gsm, by_tot_gsm, bz_tot_gsm, -1)
    
    return bx_sm, by_sm, bz_sm


def create_vector_field_plot(x_sm, y_sm, z_sm, bx_sm, by_sm, bz_sm, ut, parmod):
    """
    Create 3D vector field visualization.
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate field magnitude for coloring
    b_mag = np.sqrt(bx_sm**2 + by_sm**2 + bz_sm**2)
    
    # Normalize vectors for better visualization
    # Use different scaling for near-Earth vs far regions
    r = np.sqrt(x_sm**2 + y_sm**2 + z_sm**2)
    scale_factor = np.where(r < 5, 0.5, 2.0)
    
    # Create normalized vectors
    b_mag_safe = np.where(b_mag > 0, b_mag, 1.0)
    bx_norm = bx_sm / b_mag_safe * scale_factor
    by_norm = by_sm / b_mag_safe * scale_factor
    bz_norm = bz_sm / b_mag_safe * scale_factor
    
    # Apply log scale to colors for better visualization
    colors = np.log10(b_mag + 1)
    
    # Create quiver plot
    quiver = ax.quiver(x_sm, y_sm, z_sm, bx_norm, by_norm, bz_norm,
                       cmap='viridis', 
                       length=1.0, normalize=False,
                       arrow_length_ratio=0.3, alpha=0.8)
    
    # Add Earth sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_earth = np.outer(np.cos(u), np.sin(v))
    y_earth = np.outer(np.sin(u), np.sin(v))
    z_earth = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x_earth, y_earth, z_earth, color='blue', alpha=0.4, shade=True)
    
    # Add coordinate axes
    axis_length = 5
    ax.plot([0, axis_length], [0, 0], [0, 0], 'r-', linewidth=2, label='X_SM')
    ax.plot([0, 0], [0, axis_length], [0, 0], 'g-', linewidth=2, label='Y_SM')
    ax.plot([0, 0], [0, 0], [0, axis_length], 'b-', linewidth=2, label='Z_SM')
    
    # Add some reference field lines in specific planes
    # Noon-midnight meridian plane (Y=0)
    x_meridian = np.linspace(-10, 10, 20)
    z_meridian = np.linspace(-8, 8, 15)
    X_mer, Z_mer = np.meshgrid(x_meridian, z_meridian)
    Y_mer = np.zeros_like(X_mer)
    
    # Flatten
    x_mer_flat = X_mer.flatten()
    y_mer_flat = Y_mer.flatten()
    z_mer_flat = Z_mer.flatten()
    
    # Remove points inside Earth
    r_mer = np.sqrt(x_mer_flat**2 + y_mer_flat**2 + z_mer_flat**2)
    mask_mer = r_mer > 1.5
    
    if np.any(mask_mer):
        # Compute field
        bx_mer, by_mer, bz_mer = compute_magnetic_field(
            x_mer_flat[mask_mer], y_mer_flat[mask_mer], z_mer_flat[mask_mer], ut, parmod
        )
        
        # Add streamlines in meridian plane
        ax.scatter(x_mer_flat[mask_mer], y_mer_flat[mask_mer], z_mer_flat[mask_mer],
                  c='red', s=10, alpha=0.3, label='Meridian plane')
    
    # Set labels and title
    ax.set_xlabel('X_SM (Re)', fontsize=12)
    ax.set_ylabel('Y_SM (Re)', fontsize=12)
    ax.set_zlabel('Z_SM (Re)', fontsize=12)
    
    # Get dipole tilt
    ps = geopack.recalc(ut)
    dipole_tilt = np.degrees(ps)
    
    ax.set_title(f'3D Magnetic Field Vectors - T89 Model (Kp=0)\n' + 
                 f'Spring Equinox (March 20, 2024) at 08:00 UTC\n' +
                 f'Dipole Tilt: {dipole_tilt:.1f}°', 
                 fontsize=14, pad=20)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Set axis limits
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_zlim(-10, 10)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(quiver, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('log₁₀(|B| + 1) [nT]', fontsize=10)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=10)
    
    return fig, ax


def create_slice_plots(ut, parmod):
    """
    Create 2D slices of the magnetic field in different planes.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Common parameters
    extent = 20
    resolution = 25
    
    # Get dipole tilt
    ps = geopack.recalc(ut)
    dipole_tilt = np.degrees(ps)
    
    # 1. XY plane (Z=0, equatorial plane)
    ax1 = axes[0, 0]
    x = np.linspace(-extent, extent, resolution)
    y = np.linspace(-extent, extent, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Flatten and remove Earth
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    r_flat = np.sqrt(x_flat**2 + y_flat**2 + z_flat**2)
    mask = r_flat > 1.5
    
    # Compute field
    bx, by, bz = compute_magnetic_field(x_flat[mask], y_flat[mask], z_flat[mask], ut, parmod)
    
    # Create full arrays with NaN inside Earth
    BX = np.full_like(X, np.nan)
    BY = np.full_like(Y, np.nan)
    BZ = np.full_like(Z, np.nan)
    
    # Fill valid points
    idx = 0
    for i in range(resolution):
        for j in range(resolution):
            if r_flat[i*resolution + j] > 1.5:
                BX[i, j] = bx[idx]
                BY[i, j] = by[idx]
                BZ[i, j] = bz[idx]
                idx += 1
    
    # Plot
    ax1.streamplot(x, y, BX.T, BY.T, density=1.5, color='blue', linewidth=1)
    ax1.add_patch(plt.Circle((0, 0), 1, color='black', fill=True))
    ax1.set_xlim(-extent, extent)
    ax1.set_ylim(-extent, extent)
    ax1.set_xlabel('X_SM (Re)')
    ax1.set_ylabel('Y_SM (Re)')
    ax1.set_title('XY Plane (Z=0, Equatorial)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # 2. XZ plane (Y=0, noon-midnight meridian)
    ax2 = axes[0, 1]
    x = np.linspace(-extent, extent, resolution)
    z = np.linspace(-15, 15, resolution)
    X, Z = np.meshgrid(x, z)
    Y = np.zeros_like(X)
    
    # Flatten and remove Earth
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    r_flat = np.sqrt(x_flat**2 + y_flat**2 + z_flat**2)
    mask = r_flat > 1.5
    
    # Compute field
    bx, by, bz = compute_magnetic_field(x_flat[mask], y_flat[mask], z_flat[mask], ut, parmod)
    
    # Create full arrays
    BX = np.full_like(X, np.nan)
    BZ = np.full_like(Z, np.nan)
    
    # Fill valid points
    idx = 0
    for i in range(resolution):
        for j in range(resolution):
            if r_flat[i*resolution + j] > 1.5:
                BX[i, j] = bx[idx]
                BZ[i, j] = bz[idx]
                idx += 1
    
    # Plot
    ax2.streamplot(x, z, BX.T, BZ.T, density=1.5, color='red', linewidth=1)
    ax2.add_patch(plt.Circle((0, 0), 1, color='black', fill=True))
    ax2.set_xlim(-extent, extent)
    ax2.set_ylim(-15, 15)
    ax2.set_xlabel('X_SM (Re)')
    ax2.set_ylabel('Z_SM (Re)')
    ax2.set_title('XZ Plane (Y=0, Noon-Midnight Meridian)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # 3. YZ plane (X=5, dusk side)
    ax3 = axes[1, 0]
    y = np.linspace(-extent, extent, resolution)
    z = np.linspace(-15, 15, resolution)
    Y, Z = np.meshgrid(y, z)
    X = np.full_like(Y, 5.0)  # Fixed X=5 Re
    
    # Flatten and remove Earth
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    r_flat = np.sqrt(x_flat**2 + y_flat**2 + z_flat**2)
    mask = r_flat > 1.5
    
    # Compute field
    bx, by, bz = compute_magnetic_field(x_flat[mask], y_flat[mask], z_flat[mask], ut, parmod)
    
    # Create full arrays
    BY = np.full_like(Y, np.nan)
    BZ = np.full_like(Z, np.nan)
    
    # Fill valid points
    idx = 0
    for i in range(resolution):
        for j in range(resolution):
            if r_flat[i*resolution + j] > 1.5:
                BY[i, j] = by[idx]
                BZ[i, j] = bz[idx]
                idx += 1
    
    # Plot
    ax3.streamplot(y, z, BY.T, BZ.T, density=1.5, color='green', linewidth=1)
    ax3.set_xlim(-extent, extent)
    ax3.set_ylim(-15, 15)
    ax3.set_xlabel('Y_SM (Re)')
    ax3.set_ylabel('Z_SM (Re)')
    ax3.set_title('YZ Plane (X=5 Re, Dusk Side)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # 4. Field magnitude in XY plane
    ax4 = axes[1, 1]
    x = np.linspace(-extent, extent, resolution*2)
    y = np.linspace(-extent, extent, resolution*2)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Flatten and remove Earth
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    r_flat = np.sqrt(x_flat**2 + y_flat**2 + z_flat**2)
    mask = r_flat > 1.0
    
    # Compute field
    bx, by, bz = compute_magnetic_field(x_flat[mask], y_flat[mask], z_flat[mask], ut, parmod)
    b_mag = np.sqrt(bx**2 + by**2 + bz**2)
    
    # Create full array
    B_MAG = np.full(X.shape, np.nan)
    
    # Fill valid points
    idx = 0
    for i in range(resolution*2):
        for j in range(resolution*2):
            if r_flat[i*resolution*2 + j] > 1.0:
                B_MAG[i, j] = b_mag[idx]
                idx += 1
    
    # Plot
    im = ax4.contourf(X, Y, np.log10(B_MAG + 1), levels=20, cmap='plasma')
    ax4.add_patch(plt.Circle((0, 0), 1, color='white', fill=True))
    ax4.set_xlim(-extent, extent)
    ax4.set_ylim(-extent, extent)
    ax4.set_xlabel('X_SM (Re)')
    ax4.set_ylabel('Y_SM (Re)')
    ax4.set_title('Field Magnitude in XY Plane (log₁₀(|B|+1))')
    ax4.set_aspect('equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('log₁₀(|B|+1) [nT]')
    
    # Overall title
    fig.suptitle(f'Magnetic Field Slices - T89 Model (Kp=0)\n' + 
                 f'Spring Equinox at 08:00 UTC, Dipole Tilt: {dipole_tilt:.1f}°', 
                 fontsize=16)
    
    plt.tight_layout()
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
    
    # Create 3D grid
    print("\nCreating 3D grid for vector field...")
    x_sm, y_sm, z_sm = create_3d_grid(
        xrange=(-15, 15), yrange=(-15, 15), zrange=(-10, 10),
        nx=8, ny=8, nz=6
    )
    
    print(f"Grid points: {len(x_sm)}")
    
    # Compute magnetic field
    print("Computing magnetic field vectors...")
    bx_sm, by_sm, bz_sm = compute_magnetic_field(x_sm, y_sm, z_sm, ut, parmod)
    
    # Create 3D vector field plot
    print("\nCreating 3D vector field visualization...")
    fig_3d, ax_3d = create_vector_field_plot(x_sm, y_sm, z_sm, bx_sm, by_sm, bz_sm, ut, parmod)
    
    # Save 3D plot
    output_file_3d = os.path.join(os.path.dirname(__file__), 'magnetic_field_vectors_3d_spring_equinox.png')
    plt.savefig(output_file_3d, dpi=300, bbox_inches='tight')
    print(f"Saved {output_file_3d}")
    
    # Create slice plots
    print("\nCreating 2D slice plots...")
    fig_slices = create_slice_plots(ut, parmod)
    
    # Save slice plots
    output_file_slices = os.path.join(os.path.dirname(__file__), 'magnetic_field_slices_spring_equinox.png')
    fig_slices.savefig(output_file_slices, dpi=300, bbox_inches='tight')
    print(f"Saved {output_file_slices}")
    
    # plt.show()  # Comment out to avoid blocking


if __name__ == '__main__':
    main()