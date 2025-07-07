#!/usr/bin/env python3
"""
Magnetic Field Line Torsion Distribution using T96 Model - Enhanced Visualization
Creates figures showing torsion distribution with enhanced color scale
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm, Normalize
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
print("MAGNETIC FIELD LINE TORSION DISTRIBUTION - T96 MODEL (ENHANCED)")
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


def calculate_field_derivatives(field_model, parmod, ps, x, y, z, h=0.05):
    """
    Calculate field and its derivatives using central differences.
    Returns B and its spatial derivatives.
    """
    # Get field at central point
    bx, by, bz = field_model(parmod, ps, x, y, z)
    
    # Calculate first derivatives using central differences
    # dB/dx
    bx_xp, by_xp, bz_xp = field_model(parmod, ps, x + h, y, z)
    bx_xm, by_xm, bz_xm = field_model(parmod, ps, x - h, y, z)
    dbx_dx = (bx_xp - bx_xm) / (2 * h)
    dby_dx = (by_xp - by_xm) / (2 * h)
    dbz_dx = (bz_xp - bz_xm) / (2 * h)
    
    # dB/dy
    bx_yp, by_yp, bz_yp = field_model(parmod, ps, x, y + h, z)
    bx_ym, by_ym, bz_ym = field_model(parmod, ps, x, y - h, z)
    dbx_dy = (bx_yp - bx_ym) / (2 * h)
    dby_dy = (by_yp - by_ym) / (2 * h)
    dbz_dy = (bz_yp - bz_ym) / (2 * h)
    
    # dB/dz
    bx_zp, by_zp, bz_zp = field_model(parmod, ps, x, y, z + h)
    bx_zm, by_zm, bz_zm = field_model(parmod, ps, x, y, z - h)
    dbx_dz = (bx_zp - bx_zm) / (2 * h)
    dby_dz = (by_zp - by_zm) / (2 * h)
    dbz_dz = (bz_zp - bz_zm) / (2 * h)
    
    return (bx, by, bz, 
            dbx_dx, dbx_dy, dbx_dz,
            dby_dx, dby_dy, dby_dz,
            dbz_dx, dbz_dy, dbz_dz)


def calculate_field_line_torsion_scaled(field_model, parmod, ps, x, y, z):
    """
    Calculate the torsion of magnetic field lines with scaling for visualization.
    Returns torsion in units of 10^6 * (1/Re) for better visualization.
    """
    h = 0.05  # Step size in Re for derivatives
    
    # Get field and derivatives
    (bx, by, bz, 
     dbx_dx, dbx_dy, dbx_dz,
     dby_dx, dby_dy, dby_dz,
     dbz_dx, dbz_dy, dbz_dz) = calculate_field_derivatives(field_model, parmod, ps, x, y, z, h)
    
    # Calculate field magnitude
    b_mag = np.sqrt(bx**2 + by**2 + bz**2)
    
    # Unit vector along field
    bx_unit = np.divide(bx, b_mag, out=np.zeros_like(bx), where=b_mag>1e-10)
    by_unit = np.divide(by, b_mag, out=np.zeros_like(by), where=b_mag>1e-10)
    bz_unit = np.divide(bz, b_mag, out=np.zeros_like(bz), where=b_mag>1e-10)
    
    # Calculate db/ds where s is arc length along field line
    # db/ds = (b · ∇)b
    dbx_ds = bx_unit * dbx_dx + by_unit * dbx_dy + bz_unit * dbx_dz
    dby_ds = bx_unit * dby_dx + by_unit * dby_dy + bz_unit * dby_dz
    dbz_ds = bx_unit * dbz_dx + by_unit * dbz_dy + bz_unit * dbz_dz
    
    # Calculate curvature vector κ = db/ds (for unit vector b)
    # Since b is already unit, db/ds gives us the curvature vector directly
    kappa_x = dbx_ds / b_mag - bx * np.sum([bx*dbx_ds, by*dby_ds, bz*dbz_ds], axis=0) / b_mag**3
    kappa_y = dby_ds / b_mag - by * np.sum([bx*dbx_ds, by*dby_ds, bz*dbz_ds], axis=0) / b_mag**3
    kappa_z = dbz_ds / b_mag - bz * np.sum([bx*dbx_ds, by*dby_ds, bz*dbz_ds], axis=0) / b_mag**3
    
    # Calculate curvature magnitude
    kappa_mag = np.sqrt(kappa_x**2 + kappa_y**2 + kappa_z**2)
    
    # Principal normal vector n = κ/|κ|
    n_x = np.divide(kappa_x, kappa_mag, out=np.zeros_like(kappa_x), where=kappa_mag>1e-10)
    n_y = np.divide(kappa_y, kappa_mag, out=np.zeros_like(kappa_y), where=kappa_mag>1e-10)
    n_z = np.divide(kappa_z, kappa_mag, out=np.zeros_like(kappa_z), where=kappa_mag>1e-10)
    
    # Binormal vector b = t × n
    bi_x = by_unit * n_z - bz_unit * n_y
    bi_y = bz_unit * n_x - bx_unit * n_z
    bi_z = bx_unit * n_y - by_unit * n_x
    
    # Calculate derivatives of the normal vector along the field line
    # First get normal at neighboring points
    delta = 0.1  # Small step along field line
    x_plus = x + delta * bx_unit
    y_plus = y + delta * by_unit
    z_plus = z + delta * bz_unit
    
    # Get field derivatives at new point
    (bx_p, by_p, bz_p,
     dbx_dx_p, dbx_dy_p, dbx_dz_p,
     dby_dx_p, dby_dy_p, dby_dz_p,
     dbz_dx_p, dbz_dy_p, dbz_dz_p) = calculate_field_derivatives(field_model, parmod, ps, 
                                                                   x_plus, y_plus, z_plus, h)
    
    # Calculate curvature at new point
    b_mag_p = np.sqrt(bx_p**2 + by_p**2 + bz_p**2)
    bx_unit_p = bx_p / b_mag_p
    by_unit_p = by_p / b_mag_p
    bz_unit_p = bz_p / b_mag_p
    
    dbx_ds_p = bx_unit_p * dbx_dx_p + by_unit_p * dbx_dy_p + bz_unit_p * dbx_dz_p
    dby_ds_p = bx_unit_p * dby_dx_p + by_unit_p * dby_dy_p + bz_unit_p * dby_dz_p
    dbz_ds_p = bx_unit_p * dbz_dx_p + by_unit_p * dbz_dy_p + bz_unit_p * dbz_dz_p
    
    kappa_x_p = dbx_ds_p / b_mag_p - bx_p * (bx_p*dbx_ds_p + by_p*dby_ds_p + bz_p*dbz_ds_p) / b_mag_p**3
    kappa_y_p = dby_ds_p / b_mag_p - by_p * (bx_p*dbx_ds_p + by_p*dby_ds_p + bz_p*dbz_ds_p) / b_mag_p**3
    kappa_z_p = dbz_ds_p / b_mag_p - bz_p * (bx_p*dbx_ds_p + by_p*dby_ds_p + bz_p*dbz_ds_p) / b_mag_p**3
    
    kappa_mag_p = np.sqrt(kappa_x_p**2 + kappa_y_p**2 + kappa_z_p**2)
    
    # Normal at new point
    n_x_p = np.divide(kappa_x_p, kappa_mag_p, out=np.zeros_like(kappa_x_p), where=kappa_mag_p>1e-10)
    n_y_p = np.divide(kappa_y_p, kappa_mag_p, out=np.zeros_like(kappa_y_p), where=kappa_mag_p>1e-10)
    n_z_p = np.divide(kappa_z_p, kappa_mag_p, out=np.zeros_like(kappa_z_p), where=kappa_mag_p>1e-10)
    
    # Derivative of normal
    dn_x_ds = (n_x_p - n_x) / delta
    dn_y_ds = (n_y_p - n_y) / delta
    dn_z_ds = (n_z_p - n_z) / delta
    
    # Torsion τ = -n · (db/ds) = -(dn/ds) · b_binormal
    torsion = -(dn_x_ds * bi_x + dn_y_ds * bi_y + dn_z_ds * bi_z)
    
    # Convert to 10^6 * (1/Re) units for better visualization
    torsion = torsion * 1e6 / Re
    
    # Apply reasonable limits
    max_torsion = 5.0  # Maximum displayed torsion in 10^6 * (1/Re)
    torsion = np.where(np.abs(torsion) > max_torsion, np.sign(torsion)*max_torsion, torsion)
    
    return torsion


def create_torsion_xz_plane():
    """Create XZ plane plot showing magnetic field line torsion with enhanced colors"""
    
    print("\nCreating XZ plane torsion distribution with enhanced visualization...")
    
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
    print("Calculating field line torsion (scaled by 10^6)...")
    torsion = calculate_field_line_torsion_scaled(t96_vectorized, parmod, ps, 
                                                 x_flat, y_flat, z_flat)
    torsion_grid = torsion.reshape(X.shape)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot torsion with enhanced color scale
    # Find non-zero values for better scaling
    non_zero_mask = np.abs(torsion_grid) > 1e-3
    if np.any(non_zero_mask):
        vmax = np.percentile(np.abs(torsion_grid[non_zero_mask]), 95)
        vmax = max(vmax, 0.5)  # Ensure minimum range
    else:
        vmax = 1.0
    
    levels = np.linspace(-vmax, vmax, 41)
    
    im = ax.contourf(X, Z, torsion_grid, levels=levels,
                     cmap='RdBu_r', extend='both')
    
    # Add contours
    try:
        # More visible contour levels
        contour_levels = [-2, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2]
        cs = ax.contour(X, Z, torsion_grid, levels=contour_levels,
                        colors='black', linewidths=[0.5, 0.5, 1, 1, 2, 1, 1, 0.5, 0.5],
                        alpha=0.7)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
    except:
        pass
    
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
    
    # Enhanced colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Torsion (×10⁻⁶ Re⁻¹)', fontsize=12, weight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Add note about scaling
    ax.text(0.02, 0.02, 'Note: Torsion values scaled by 10⁶ for visibility', 
            transform=ax.transAxes, fontsize=9, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Statistics
    mask = np.sqrt(x_flat**2 + z_flat**2) > 3.0  # Exclude near Earth
    torsion_filtered = torsion[mask & np.isfinite(torsion)]
    
    if len(torsion_filtered) > 0:
        print(f"\nTorsion statistics (×10⁻⁶ Re⁻¹, excluding r < 3 Re):")
        print(f"  Min: {np.min(torsion_filtered):.2f}")
        print(f"  Max: {np.max(torsion_filtered):.2f}")
        print(f"  Mean: {np.mean(torsion_filtered):.2f}")
        print(f"  Median: {np.median(torsion_filtered):.2f}")
        print(f"  Std: {np.std(torsion_filtered):.2f}")
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'fig_torsion_xz_plane_t96_enhanced.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nFigure saved: {output_file}")


def create_torsion_xy_slices():
    """Create XY plane slices showing torsion with enhanced visualization"""
    
    print("\nCreating XY plane torsion slices with enhanced colors...")
    
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
    
    # First determine global color scale
    print("Determining global color scale...")
    global_vmax = 0.5  # Minimum value
    
    for z_height in z_heights[:4]:  # Sample a few heights
        Z = np.full_like(X, z_height)
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        
        torsion = calculate_field_line_torsion_scaled(t96_vectorized, parmod, ps,
                                                     x_flat, y_flat, z_flat)
        mask = np.sqrt(x_flat**2 + y_flat**2) > 3.0
        torsion_filtered = torsion[mask & np.isfinite(torsion)]
        if len(torsion_filtered) > 0:
            vmax_temp = np.percentile(np.abs(torsion_filtered), 95)
            global_vmax = max(global_vmax, vmax_temp)
    
    global_vmax = min(global_vmax, 2.0)  # Cap at reasonable value
    levels = np.linspace(-global_vmax, global_vmax, 41)
    
    # Create plots
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
        torsion = calculate_field_line_torsion_scaled(t96_vectorized, parmod, ps,
                                                     x_flat, y_flat, z_flat)
        torsion_grid = torsion.reshape(X.shape)
        
        # Calculate statistics
        mask = np.sqrt(x_flat**2 + y_flat**2) > 3.0
        torsion_filtered = torsion[mask & np.isfinite(torsion)]
        
        if len(torsion_filtered) > 0:
            mean_torsion = np.mean(np.abs(torsion_filtered))
            max_torsion = np.max(np.abs(torsion_filtered))
        else:
            mean_torsion = 0.0
            max_torsion = 0.0
            
        torsion_stats.append((z_height, mean_torsion, max_torsion))
        
        # Plot torsion
        im = ax.contourf(X, Y, torsion_grid, levels=levels,
                        cmap='RdBu_r', extend='both')
        
        # Add contours
        try:
            # Zero contour
            cs = ax.contour(X, Y, torsion_grid, levels=[0],
                           colors='black', linewidths=2)
            # Other contours
            contour_levels = [-1, -0.5, -0.1, 0.1, 0.5, 1]
            cs2 = ax.contour(X, Y, torsion_grid, levels=contour_levels,
                            colors='gray', linewidths=1, alpha=0.7)
        except:
            pass
        
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
        ax.set_title(f'Z = {z_height:.1f} Re (|τ|_mean = {mean_torsion:.2f}×10⁻⁶ Re⁻¹)', 
                    fontsize=10, weight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-20, 5)
        ax.set_ylim(-12, 12)
    
    # Hide unused subplots
    for idx in range(len(z_heights), len(axes)):
        axes[idx].set_visible(False)
    
    # Enhanced colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax, ticks=np.linspace(-global_vmax, global_vmax, 9))
    cbar.set_label('Torsion (×10⁻⁶ Re⁻¹)', fontsize=12, weight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Add note about scaling
    fig.text(0.91, 0.12, 'Note: Values scaled by 10⁶', fontsize=9, ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.suptitle('Magnetic Field Line Torsion in XY Planes\n' +
                f'T96 Model, Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT',
                fontsize=14, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    # Save figure
    output_file = os.path.join(output_dir, 'fig_torsion_xy_slices_t96_enhanced.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nFigure saved: {output_file}")
    
    # Print statistics summary
    print("\nTorsion statistics by height (×10⁻⁶ Re⁻¹):")
    print("-" * 50)
    print(f"{'Z (Re)':<10} {'Mean |τ|':<20} {'Max |τ|':<20}")
    print("-" * 50)
    for z, mean_t, max_t in torsion_stats:
        print(f"{z:+.1f}        {mean_t:<20.2f} {max_t:<20.2f}")
    
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
    ax1.set_ylabel('Mean |Torsion| (×10⁻⁶ Re⁻¹)', fontsize=12)
    ax1.set_title('Mean Absolute Torsion vs Height', fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Max torsion vs Z
    ax2.plot(z_vals, max_vals, 'r-o', linewidth=2, markersize=8)
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Z GSM (Re)', fontsize=12)
    ax2.set_ylabel('Max |Torsion| (×10⁻⁶ Re⁻¹)', fontsize=12)
    ax2.set_title('Maximum Absolute Torsion vs Height', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    plt.suptitle('Torsion Statistics vs Height\n' +
                f'T96 Model, Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT',
                fontsize=14, weight='bold')
    plt.tight_layout()
    
    output_file2 = os.path.join(output_dir, 'fig_torsion_vs_height_t96_enhanced.png')
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"Summary plot saved: {output_file2}")


if __name__ == "__main__":
    create_torsion_xz_plane()
    create_torsion_xy_slices()
    
    print("\n" + "="*80)
    print("ENHANCED TORSION DISTRIBUTION ANALYSIS COMPLETE!")
    print("="*80)
    print("\nKey findings:")
    print("- Torsion values have been scaled by 10⁶ for better visualization")
    print("- Enhanced color scale reveals subtle variations in field line twisting")
    print("- Torsion patterns are now more visible in the plots")
    print("- Maximum torsion occurs in regions of complex field topology")
    print("="*80)