#!/usr/bin/env python3
"""
Magnetic Field Vector Plot - XZ Plane at Y=0
Shows vector field (Bx, Bz) with field strength in background color
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import geopack
from geopack import t96_vectorized

# Create output directory
output_dir = "figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*80)
print("MAGNETIC FIELD VECTOR PLOT")
print("XZ Plane (Y=0) - T96 Model")
print("="*80)

# Initialize geopack for current conditions
ut = 1584662400  # March 20, 2020 (equinox)
ps = geopack.recalc(ut)

print(f"\nModel conditions:")
print(f"Dipole tilt: {np.degrees(ps):.2f}°")

# Model parameters - standard parameters from previous analysis
parmod = [3.0, -30.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
print(f"\nT96 Parameters:")
print(f"Pdyn = {parmod[0]} nPa")
print(f"Dst = {parmod[1]} nT")
print(f"By_IMF = {parmod[2]} nT")
print(f"Bz_IMF = {parmod[3]} nT")

def create_vector_field_plot():
    """Create magnetic field vector plot with field strength background"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create grid for background color (field strength)
    x_bg = np.linspace(-20, 10, 151)
    z_bg = np.linspace(-10, 10, 101)
    X_bg, Z_bg = np.meshgrid(x_bg, z_bg)
    Y_bg = np.zeros_like(X_bg)
    
    # Calculate field for background
    print("\nCalculating magnetic field strength...")
    x_flat = X_bg.flatten()
    y_flat = Y_bg.flatten()
    z_flat = Z_bg.flatten()
    
    bx_bg, by_bg, bz_bg = t96_vectorized(parmod, ps, x_flat, y_flat, z_flat)
    B_magnitude = np.sqrt(bx_bg**2 + by_bg**2 + bz_bg**2)
    B_grid = B_magnitude.reshape(X_bg.shape)
    
    # Plot field strength as background
    im = ax.contourf(X_bg, Z_bg, B_grid, levels=50, cmap='viridis', 
                     norm=LogNorm(vmin=5, vmax=500))
    
    # Create coarser grid for vectors
    x_vec = np.linspace(-20, 10, 31)
    z_vec = np.linspace(-10, 10, 21)
    X_vec, Z_vec = np.meshgrid(x_vec, z_vec)
    Y_vec = np.zeros_like(X_vec)
    
    # Calculate field vectors
    print("Calculating field vectors...")
    x_vec_flat = X_vec.flatten()
    y_vec_flat = Y_vec.flatten()
    z_vec_flat = Z_vec.flatten()
    
    bx_vec, by_vec, bz_vec = t96_vectorized(parmod, ps, x_vec_flat, y_vec_flat, z_vec_flat)
    
    # Reshape vectors
    Bx_grid = bx_vec.reshape(X_vec.shape)
    Bz_grid = bz_vec.reshape(Z_vec.shape)
    
    # Normalize vectors for display
    B_vec_mag = np.sqrt(Bx_grid**2 + Bz_grid**2)
    Bx_norm = np.divide(Bx_grid, B_vec_mag, out=np.zeros_like(Bx_grid), where=B_vec_mag>0)
    Bz_norm = np.divide(Bz_grid, B_vec_mag, out=np.zeros_like(Bz_grid), where=B_vec_mag>0)
    
    # Scale vectors by log of field strength for better visualization
    scale_factor = np.log10(B_vec_mag + 1) * 0.3
    
    # Plot vectors
    ax.quiver(X_vec, Z_vec, Bx_norm * scale_factor, Bz_norm * scale_factor,
              color='white', alpha=0.8, width=0.003, headwidth=3, headlength=4,
              scale=15, scale_units='xy')
    
    # Add field lines (streamlines)
    # Start points for field lines
    seed_x = np.concatenate([
        np.full(5, -15),  # Tail field lines
        np.full(5, -10),
        np.full(5, -5),
        np.full(7, -2),   # Near-Earth field lines
        np.full(7, 0),
        np.full(7, 2)
    ])
    seed_z = np.concatenate([
        np.linspace(-3, 3, 5),
        np.linspace(-4, 4, 5),
        np.linspace(-5, 5, 5),
        np.linspace(-6, 6, 7),
        np.linspace(-7, 7, 7),
        np.linspace(-5, 5, 7)
    ])
    
    # Create interpolation functions for streamplot
    from scipy.interpolate import RegularGridInterpolator
    
    bx_interp = RegularGridInterpolator((z_bg, x_bg), 
                                       bx_bg.reshape(len(z_bg), len(x_bg)), 
                                       bounds_error=False, fill_value=0)
    bz_interp = RegularGridInterpolator((z_bg, x_bg), 
                                       bz_bg.reshape(len(z_bg), len(x_bg)), 
                                       bounds_error=False, fill_value=0)
    
    # Plot streamlines
    strm = ax.streamplot(x_bg, z_bg, 
                        bx_bg.reshape(len(z_bg), len(x_bg)), 
                        bz_bg.reshape(len(z_bg), len(x_bg)),
                        color='red', linewidth=1, density=0.5,
                        start_points=np.column_stack([seed_x, seed_z]))
    
    # Set alpha for streamlines
    strm.lines.set_alpha(0.6)
    
    # Add Earth
    earth = plt.Circle((0, 0), 1, color='white', zorder=10)
    ax.add_patch(earth)
    ax.text(0, 0, 'E', ha='center', va='center', fontsize=12, weight='bold')
    
    # Add magnetopause boundary (approximate)
    theta = np.linspace(0, np.pi, 100)
    r_mp = 10 * (2 / (1 + np.cos(theta)))**0.5  # Simple magnetopause model
    x_mp = r_mp * np.cos(theta)
    z_mp = r_mp * np.sin(theta)
    ax.plot(x_mp, z_mp, 'w--', linewidth=2, alpha=0.5, label='Magnetopause')
    ax.plot(x_mp, -z_mp, 'w--', linewidth=2, alpha=0.5)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Magnetic Field Strength |B| (nT)', fontsize=12)
    
    # Labels and formatting
    ax.set_xlabel('X GSM (Re)', fontsize=12)
    ax.set_ylabel('Z GSM (Re)', fontsize=12)
    ax.set_title(f'T96 Magnetic Field: Vectors and Strength in XZ Plane (Y=0)\n' +
                f'Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT, IMF By={parmod[2]} nT, Bz={parmod[3]} nT',
                fontsize=14, weight='bold')
    
    ax.set_xlim(-20, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, color='gray')
    
    # Add text annotations
    ax.text(-18, 8, 'Magnetotail', fontsize=11, color='white', weight='bold')
    ax.text(5, 8, 'Dayside', fontsize=11, color='white', weight='bold')
    ax.text(-10, -8, 'Vectors: Field direction (scaled by log|B|)', 
            fontsize=9, color='white', style='italic')
    ax.text(-10, -9, 'Red lines: Magnetic field lines', 
            fontsize=9, color='red', style='italic')
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, 'fig03_magnetic_field_vectors_xz.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nFigure saved: {output_file}")
    
    # Print some statistics
    print(f"\nField strength statistics:")
    print(f"  Min |B|: {np.min(B_magnitude):.1f} nT")
    print(f"  Max |B|: {np.max(B_magnitude):.1f} nT")
    print(f"  Median |B|: {np.median(B_magnitude):.1f} nT")

def create_separate_component_plots():
    """Create separate plots for Bx and Bz components"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Create grid
    x_grid = np.linspace(-20, 10, 151)
    z_grid = np.linspace(-10, 10, 101)
    X, Z = np.meshgrid(x_grid, z_grid)
    Y = np.zeros_like(X)
    
    # Calculate field
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    
    bx, by, bz = t96_vectorized(parmod, ps, x_flat, y_flat, z_flat)
    Bx_grid = bx.reshape(X.shape)
    Bz_grid = bz.reshape(Z.shape)
    
    # Plot Bx component
    vmax = max(abs(np.min(Bx_grid)), abs(np.max(Bx_grid)))
    im1 = ax1.contourf(X, Z, Bx_grid, levels=50, cmap='RdBu_r', 
                       vmin=-vmax, vmax=vmax)
    ax1.contour(X, Z, Bx_grid, levels=[0], colors='black', linewidths=2)
    
    # Add Earth
    earth1 = plt.Circle((0, 0), 1, color='gray', zorder=10)
    ax1.add_patch(earth1)
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Bx (nT)', fontsize=12)
    
    ax1.set_xlabel('X GSM (Re)')
    ax1.set_ylabel('Z GSM (Re)')
    ax1.set_title('Bx Component (Sunward/Tailward)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-20, 10)
    ax1.set_ylim(-10, 10)
    
    # Plot Bz component
    vmax = max(abs(np.min(Bz_grid)), abs(np.max(Bz_grid)))
    im2 = ax2.contourf(X, Z, Bz_grid, levels=50, cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax)
    ax2.contour(X, Z, Bz_grid, levels=[0], colors='black', linewidths=2)
    
    # Add Earth
    earth2 = plt.Circle((0, 0), 1, color='gray', zorder=10)
    ax2.add_patch(earth2)
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Bz (nT)', fontsize=12)
    
    ax2.set_xlabel('X GSM (Re)')
    ax2.set_ylabel('Z GSM (Re)')
    ax2.set_title('Bz Component (Northward/Southward)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-20, 10)
    ax2.set_ylim(-10, 10)
    
    plt.suptitle(f'T96 Magnetic Field Components in XZ Plane (Y=0)\n' +
                f'Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT',
                fontsize=14, weight='bold')
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'fig04_magnetic_field_components_xz.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Component plot saved: {output_file}")

if __name__ == "__main__":
    # Create main vector field plot
    create_vector_field_plot()
    
    # Create component plots
    print("\nCreating component plots...")
    create_separate_component_plots()
    
    print("\n" + "="*80)
    print("Magnetic field vector plots complete!")
    print("="*80)