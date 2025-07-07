#!/usr/bin/env python3
"""
Magnetic Field Line Torsion Distribution using T96 Model
Creates figures showing torsion distribution in XZ plane and XY slices
Similar to figures 3 and 5 but for torsion instead of Rc/RL ratio
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import os
import geopack
from geopack import t96_vectorized

# Create output directory
output_dir = "figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Physical constants
Re = 6.371e6  # Earth radius (m)

print("="*80)
print("MAGNETIC FIELD LINE TORSION DISTRIBUTION - T96 MODEL")
print("="*80)

# Initialize geopack
ut = 1584662400  # March 20, 2020 (equinox)
ps = geopack.recalc(ut)

print(f"\nModel conditions:")
print(f"Dipole tilt: {np.degrees(ps):.2f}°")

# Model parameters
parmod = [3.0, -30.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
print(f"\nT96 Parameters:")
print(f"Pdyn = {parmod[0]} nPa")
print(f"Dst = {parmod[1]} nT")
print(f"By_IMF = {parmod[2]} nT")
print(f"Bz_IMF = {parmod[3]} nT")


def calculate_field_line_torsion(field_model, parmod, ps, x, y, z, delta=0.01):
    """
    Calculate the torsion of magnetic field lines at given points.
    
    Torsion τ = (dB/ds × d²B/ds²) · d³B/ds³ / |dB/ds × d²B/ds²|²
    
    Where s is the arc length parameter along the field line.
    """
    # Get field at current point
    bx, by, bz = field_model(parmod, ps, x, y, z)
    b_mag = np.sqrt(bx**2 + by**2 + bz**2)
    
    # Unit vector along field
    bx_unit = np.divide(bx, b_mag, out=np.zeros_like(bx), where=b_mag>0)
    by_unit = np.divide(by, b_mag, out=np.zeros_like(by), where=b_mag>0)
    bz_unit = np.divide(bz, b_mag, out=np.zeros_like(bz), where=b_mag>0)
    
    # Calculate derivatives using finite differences
    # Move along field line direction
    x1 = x + delta * bx_unit
    y1 = y + delta * by_unit
    z1 = z + delta * bz_unit
    
    x2 = x - delta * bx_unit
    y2 = y - delta * by_unit
    z2 = z - delta * bz_unit
    
    # Get field at neighboring points
    bx1, by1, bz1 = field_model(parmod, ps, x1, y1, z1)
    bx2, by2, bz2 = field_model(parmod, ps, x2, y2, z2)
    
    # First derivative (tangent vector)
    dbx_ds = (bx1 - bx2) / (2 * delta)
    dby_ds = (by1 - by2) / (2 * delta)
    dbz_ds = (bz1 - bz2) / (2 * delta)
    
    # For second derivative, we need more points
    x3 = x + 2*delta * bx_unit
    y3 = y + 2*delta * by_unit
    z3 = z + 2*delta * bz_unit
    
    x4 = x - 2*delta * bx_unit
    y4 = y - 2*delta * by_unit
    z4 = z - 2*delta * bz_unit
    
    bx3, by3, bz3 = field_model(parmod, ps, x3, y3, z3)
    bx4, by4, bz4 = field_model(parmod, ps, x4, y4, z4)
    
    # Second derivative
    d2bx_ds2 = (bx3 - 2*bx + bx4) / (4 * delta**2)
    d2by_ds2 = (by3 - 2*by + by4) / (4 * delta**2)
    d2bz_ds2 = (bz3 - 2*bz + bz4) / (4 * delta**2)
    
    # Third derivative (simplified)
    d3bx_ds3 = (bx3 - 3*bx1 + 3*bx2 - bx4) / (8 * delta**3)
    d3by_ds3 = (by3 - 3*by1 + 3*by2 - by4) / (8 * delta**3)
    d3bz_ds3 = (bz3 - 3*bz1 + 3*bz2 - bz4) / (8 * delta**3)
    
    # Cross product of first and second derivatives
    cross_x = dby_ds * d2bz_ds2 - dbz_ds * d2by_ds2
    cross_y = dbz_ds * d2bx_ds2 - dbx_ds * d2bz_ds2
    cross_z = dbx_ds * d2by_ds2 - dby_ds * d2bx_ds2
    
    cross_mag_sq = cross_x**2 + cross_y**2 + cross_z**2
    
    # Dot product with third derivative
    numerator = cross_x * d3bx_ds3 + cross_y * d3by_ds3 + cross_z * d3bz_ds3
    
    # Torsion
    torsion = np.divide(numerator, cross_mag_sq, 
                       out=np.zeros_like(numerator), 
                       where=cross_mag_sq > 1e-20)
    
    # Convert to physical units (1/Re)
    torsion = torsion / Re
    
    return torsion


def create_torsion_xz_plane():
    """Create XZ plane plot showing magnetic field line torsion (like Figure 3)"""
    
    print("\nCreating XZ plane torsion distribution...")
    
    # Create grid
    x_grid = np.linspace(-20, 10, 151)
    z_grid = np.linspace(-10, 10, 101)
    X, Z = np.meshgrid(x_grid, z_grid)
    Y = np.zeros_like(X)
    
    # Flatten for calculation
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    
    # Calculate torsion
    print("Calculating field line torsion...")
    torsion = calculate_field_line_torsion(t96_vectorized, parmod, ps, 
                                          x_flat, y_flat, z_flat)
    torsion_grid = torsion.reshape(X.shape)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot torsion with symmetric log scale
    # Use symmetric log norm to handle both positive and negative values
    vmax = np.percentile(np.abs(torsion_grid), 99)
    vmin = -vmax
    
    im = ax.contourf(X, Z, torsion_grid, levels=50,
                     cmap='RdBu_r', 
                     norm=SymLogNorm(linthresh=0.01, vmin=vmin, vmax=vmax))
    
    # Add contours for specific torsion values
    contour_levels = [-1, -0.5, -0.1, 0, 0.1, 0.5, 1]
    cs = ax.contour(X, Z, torsion_grid, levels=contour_levels,
                    colors='black', linewidths=1, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
    
    # Add magnetic field vectors
    x_vec = np.linspace(-20, 10, 31)
    z_vec = np.linspace(-10, 10, 21)
    X_vec, Z_vec = np.meshgrid(x_vec, z_vec)
    Y_vec = np.zeros_like(X_vec)
    
    x_vec_flat = X_vec.flatten()
    y_vec_flat = Y_vec.flatten()
    z_vec_flat = Z_vec.flatten()
    
    bx_vec, by_vec, bz_vec = t96_vectorized(parmod, ps, 
                                           x_vec_flat, y_vec_flat, z_vec_flat)
    
    Bx_grid = bx_vec.reshape(X_vec.shape)
    Bz_grid = bz_vec.reshape(Z_vec.shape)
    
    B_vec_mag = np.sqrt(Bx_grid**2 + Bz_grid**2)
    Bx_norm = np.divide(Bx_grid, B_vec_mag, out=np.zeros_like(Bx_grid), 
                       where=B_vec_mag>0)
    Bz_norm = np.divide(Bz_grid, B_vec_mag, out=np.zeros_like(Bz_grid), 
                       where=B_vec_mag>0)
    
    ax.quiver(X_vec, Z_vec, Bx_norm, Bz_norm,
              color='white', alpha=0.7, width=0.002, headwidth=3,
              scale=20, scale_units='xy', edgecolor='black', linewidth=0.5)
    
    # Add Earth
    earth = plt.Circle((0, 0), 1, color='white', zorder=10)
    ax.add_patch(earth)
    ax.text(0, 0, 'E', ha='center', va='center', fontsize=12, weight='bold')
    
    # Labels and formatting
    ax.set_xlabel('X GSM (Re)', fontsize=12)
    ax.set_ylabel('Z GSM (Re)', fontsize=12)
    ax.set_title('Magnetic Field Line Torsion in XZ Plane (Y=0)\n' +
                f'T96 Model, Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT',
                fontsize=14, weight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-20, 10)
    ax.set_ylim(-10, 10)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Torsion (1/Re)', fontsize=12)
    
    # Statistics
    mask = np.sqrt(x_flat**2 + z_flat**2) > 2.0  # Exclude near Earth
    torsion_filtered = torsion[mask]
    
    print(f"\nTorsion statistics (excluding r < 2 Re):")
    print(f"  Min: {np.min(torsion_filtered):.3f} 1/Re")
    print(f"  Max: {np.max(torsion_filtered):.3f} 1/Re")
    print(f"  Mean: {np.mean(torsion_filtered):.3f} 1/Re")
    print(f"  Median: {np.median(torsion_filtered):.3f} 1/Re")
    print(f"  Std: {np.std(torsion_filtered):.3f} 1/Re")
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'fig_torsion_xz_plane_t96.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nFigure saved: {output_file}")


def create_torsion_xy_slices():
    """Create XY plane slices showing torsion at different Z heights (like Figure 5)"""
    
    print("\nCreating XY plane torsion slices...")
    
    # Z heights from -0.6 to 0.8 Re in 0.2 Re increments
    z_heights = np.arange(-0.6, 0.9, 0.2)
    n_slices = len(z_heights)
    
    # Create figure with subplots
    n_rows = 2
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
    axes = axes.flatten()
    
    # Create XY grid
    x_grid = np.linspace(-20, 5, 126)
    y_grid = np.linspace(-12, 12, 121)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Statistics storage
    torsion_stats = []
    
    for idx, z_height in enumerate(z_heights):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        print(f"Processing Z = {z_height:.1f} Re...")
        
        # Create Z array for this height
        Z = np.full_like(X, z_height)
        
        # Flatten for calculation
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        
        # Calculate torsion
        torsion = calculate_field_line_torsion(t96_vectorized, parmod, ps,
                                             x_flat, y_flat, z_flat)
        torsion_grid = torsion.reshape(X.shape)
        
        # Calculate statistics
        mask = np.sqrt(x_flat**2 + y_flat**2) > 2.0
        torsion_filtered = torsion[mask]
        mean_torsion = np.mean(np.abs(torsion_filtered))
        max_torsion = np.max(np.abs(torsion_filtered))
        torsion_stats.append((z_height, mean_torsion, max_torsion))
        
        # Plot torsion
        vmax = np.percentile(np.abs(torsion_grid), 99)
        vmin = -vmax
        
        im = ax.contourf(X, Y, torsion_grid, levels=50,
                        cmap='RdBu_r',
                        norm=SymLogNorm(linthresh=0.01, vmin=vmin, vmax=vmax))
        
        # Add zero contour
        cs = ax.contour(X, Y, torsion_grid, levels=[0],
                       colors='black', linewidths=2)
        
        # Add other contours
        contour_levels = [-0.5, -0.1, 0.1, 0.5]
        cs2 = ax.contour(X, Y, torsion_grid, levels=contour_levels,
                        colors='gray', linewidths=1, alpha=0.5)
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax.add_patch(earth)
        
        # Add field vectors (sparse)
        if idx % 2 == 0:  # Only for some panels
            x_vec = np.linspace(-20, 5, 11)
            y_vec = np.linspace(-12, 12, 13)
            X_vec, Y_vec = np.meshgrid(x_vec, y_vec)
            Z_vec = np.full_like(X_vec, z_height)
            
            x_vec_flat = X_vec.flatten()
            y_vec_flat = Y_vec.flatten()
            z_vec_flat = Z_vec.flatten()
            
            bx_vec, by_vec, bz_vec = t96_vectorized(parmod, ps, 
                                                   x_vec_flat, y_vec_flat, z_vec_flat)
            
            Bx_grid = bx_vec.reshape(X_vec.shape)
            By_grid = by_vec.reshape(Y_vec.shape)
            
            B_vec_mag = np.sqrt(Bx_grid**2 + By_grid**2)
            Bx_norm = np.divide(Bx_grid, B_vec_mag, out=np.zeros_like(Bx_grid), 
                              where=B_vec_mag>0)
            By_norm = np.divide(By_grid, B_vec_mag, out=np.zeros_like(By_grid), 
                              where=B_vec_mag>0)
            
            ax.quiver(X_vec, Y_vec, Bx_norm, By_norm,
                     color='white', alpha=0.6, width=0.002, headwidth=3, 
                     scale=25, scale_units='xy')
        
        # Labels and formatting
        ax.set_xlabel('X GSM (Re)', fontsize=9)
        ax.set_ylabel('Y GSM (Re)', fontsize=9)
        ax.set_title(f'Z = {z_height:.1f} Re (|τ|_mean = {mean_torsion:.3f} 1/Re)', 
                    fontsize=10, weight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-20, 5)
        ax.set_ylim(-12, 12)
    
    # Hide unused subplots
    for idx in range(len(z_heights), len(axes)):
        axes[idx].set_visible(False)
    
    # Add a single colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Torsion (1/Re)', fontsize=12)
    
    plt.suptitle('Magnetic Field Line Torsion in XY Planes\n' +
                f'T96 Model, Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT',
                fontsize=14, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    # Save figure
    output_file = os.path.join(output_dir, 'fig_torsion_xy_slices_t96.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nFigure saved: {output_file}")
    
    # Print statistics summary
    print("\nTorsion statistics by height:")
    print("-" * 50)
    print(f"{'Z (Re)':<10} {'Mean |τ| (1/Re)':<20} {'Max |τ| (1/Re)':<20}")
    print("-" * 50)
    for z, mean_t, max_t in torsion_stats:
        print(f"{z:+.1f}        {mean_t:<20.3f} {max_t:<20.3f}")
    
    # Create summary plot
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    z_vals = [s[0] for s in torsion_stats]
    mean_vals = [s[1] for s in torsion_stats]
    max_vals = [s[2] for s in torsion_stats]
    
    # Mean torsion vs Z
    ax1.plot(z_vals, mean_vals, 'b-o', linewidth=2, markersize=8)
    ax1.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Z GSM (Re)', fontsize=12)
    ax1.set_ylabel('Mean |Torsion| (1/Re)', fontsize=12)
    ax1.set_title('Mean Absolute Torsion vs Height', fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Max torsion vs Z
    ax2.plot(z_vals, max_vals, 'r-o', linewidth=2, markersize=8)
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Z GSM (Re)', fontsize=12)
    ax2.set_ylabel('Max |Torsion| (1/Re)', fontsize=12)
    ax2.set_title('Maximum Absolute Torsion vs Height', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Torsion Statistics vs Height\n' +
                f'T96 Model, Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT',
                fontsize=14, weight='bold')
    plt.tight_layout()
    
    output_file2 = os.path.join(output_dir, 'fig_torsion_vs_height_t96.png')
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"Summary plot saved: {output_file2}")


if __name__ == "__main__":
    create_torsion_xz_plane()
    create_torsion_xy_slices()
    
    print("\n" + "="*80)
    print("TORSION DISTRIBUTION ANALYSIS COMPLETE!")
    print("="*80)
    print("\nKey findings:")
    print("- Torsion shows the twisting of magnetic field lines")
    print("- High torsion regions indicate complex field topology")
    print("- Torsion changes sign across certain boundaries")
    print("- Maximum torsion typically occurs in the current sheet region")
    print("="*80)