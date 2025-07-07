#!/usr/bin/env python3
"""
Curvature Radius and Larmor Radius in XY Plane Slices for 100 keV Electrons
Shows XY plane slices from Z=-0.6 to Z=0.8 Re in 0.2 Re increments
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import geopack
from geopack import t96_vectorized, field_line_curvature_vectorized

# Create output directory
output_dir = "figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Physical constants
c = 2.99792458e8  # Speed of light (m/s)
me = 9.10938356e-31  # Electron mass (kg)
e = 1.602176634e-19  # Elementary charge (C)
Re = 6.371e6  # Earth radius (m)

# Critical threshold
CRITICAL_RATIO = 8.0

print("="*80)
print("CURVATURE AND LARMOR RADIUS IN XY PLANE SLICES - 100 keV ELECTRONS")
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


def calculate_larmor_radius(energy_keV, B_nT, pitch_angle_deg=90):
    """Calculate the Larmor radius for an electron."""
    E_k = energy_keV * 1000 * e
    B = B_nT * 1e-9
    
    gamma = 1 + E_k / (me * c**2)
    beta = np.sqrt(1 - 1/gamma**2)
    v = beta * c
    
    alpha = np.radians(pitch_angle_deg)
    v_perp = v * np.sin(alpha)
    
    RL = gamma * me * v_perp / (e * B)
    return RL


def create_curvature_radius_slices():
    """Create XY plane slices showing curvature radius"""
    
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
    
    print(f"\nCalculating curvature radius...")
    
    # Statistics storage
    rc_stats = []
    
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
        
        # Calculate curvature
        kappa = field_line_curvature_vectorized(t96_vectorized, parmod, ps, 
                                               x_flat, y_flat, z_flat)
        Rc_Re = np.where(kappa > 1e-10, 1.0 / kappa, 1e3)
        Rc_grid = Rc_Re.reshape(X.shape)
        
        # Calculate statistics
        median_rc = np.median(Rc_Re)
        min_rc = np.min(Rc_Re)
        rc_stats.append((z_height, median_rc, min_rc))
        
        # Plot curvature radius
        im = ax.contourf(X, Y, Rc_grid, levels=50, 
                        cmap='plasma', norm=LogNorm(vmin=0.1, vmax=100))
        
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
                     color='white', alpha=0.7, width=0.002, headwidth=3, 
                     scale=25, scale_units='xy')
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax.add_patch(earth)
        
        # Labels and formatting
        ax.set_xlabel('X GSM (Re)', fontsize=9)
        ax.set_ylabel('Y GSM (Re)', fontsize=9)
        ax.set_title(f'Z = {z_height:.1f} Re (Med: {median_rc:.1f} Re)', 
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
    cbar.set_label('Radius of Curvature (Re)', fontsize=12)
    
    plt.suptitle(f'Field Line Curvature Radius in XY Planes\n' +
                f'T96 Model, Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT',
                fontsize=14, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    # Save figure
    output_file = os.path.join(output_dir, 'fig07_curvature_radius_xy_slices.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nCurvature radius figure saved: {output_file}")
    
    return rc_stats


def create_larmor_radius_slices():
    """Create XY plane slices showing Larmor radius for 100 keV"""
    
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
    
    # Fixed energy
    energy = 100  # keV
    
    print(f"\nCalculating Larmor radius for {energy} keV electrons...")
    
    # Statistics storage
    rl_stats = []
    
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
        
        # Calculate magnetic field
        bx, by, bz = t96_vectorized(parmod, ps, x_flat, y_flat, z_flat)
        B_magnitude = np.sqrt(bx**2 + by**2 + bz**2)
        
        # Calculate Larmor radius
        RL_m = calculate_larmor_radius(energy, B_magnitude)
        RL_Re = RL_m / Re
        RL_grid = RL_Re.reshape(X.shape)
        
        # Calculate statistics
        median_rl = np.median(RL_Re)
        max_rl = np.max(RL_Re)
        rl_stats.append((z_height, median_rl, max_rl))
        
        # Plot Larmor radius
        im = ax.contourf(X, Y, RL_grid, levels=50, 
                        cmap='viridis', norm=LogNorm(vmin=0.001, vmax=1))
        
        # Add contours for specific values
        cs = ax.contour(X, Y, RL_grid, levels=[0.01, 0.1, 0.5], 
                       colors='white', linewidths=1, alpha=0.5)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f Re')
        
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
                     color='white', alpha=0.7, width=0.002, headwidth=3, 
                     scale=25, scale_units='xy')
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax.add_patch(earth)
        
        # Labels and formatting
        ax.set_xlabel('X GSM (Re)', fontsize=9)
        ax.set_ylabel('Y GSM (Re)', fontsize=9)
        ax.set_title(f'Z = {z_height:.1f} Re (Med: {median_rl*6371:.0f} km)', 
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
    cbar.set_label('Larmor Radius (Re)', fontsize=12)
    
    plt.suptitle(f'Electron Larmor Radius in XY Planes: 100 keV\n' +
                f'T96 Model, Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT',
                fontsize=14, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    # Save figure
    output_file = os.path.join(output_dir, 'fig08_larmor_radius_xy_slices_100keV.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Larmor radius figure saved: {output_file}")
    
    return rl_stats


def create_combined_statistics_plot(rc_stats, rl_stats):
    """Create a summary plot showing how Rc and RL vary with height"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract data
    z_vals = [s[0] for s in rc_stats]
    rc_medians = [s[1] for s in rc_stats]
    rc_mins = [s[2] for s in rc_stats]
    
    rl_medians = [s[1] for s in rl_stats]
    rl_maxs = [s[2] for s in rl_stats]
    
    # Plot 1: Curvature radius vs height
    ax1.plot(z_vals, rc_medians, 'b-o', linewidth=2, markersize=8, label='Median Rc')
    ax1.plot(z_vals, rc_mins, 'r--s', linewidth=1.5, markersize=6, label='Min Rc')
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Z GSM (Re)', fontsize=12)
    ax1.set_ylabel('Radius of Curvature (Re)', fontsize=12)
    ax1.set_title('Field Line Curvature vs Height', fontsize=14, weight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(-0.8, 1.0)
    ax1.set_ylim(0.01, 100)
    
    # Plot 2: Larmor radius vs height
    ax2.plot(z_vals, np.array(rl_medians)*6371, 'g-o', linewidth=2, markersize=8, label='Median RL')
    ax2.plot(z_vals, np.array(rl_maxs)*6371, 'm--s', linewidth=1.5, markersize=6, label='Max RL')
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Z GSM (Re)', fontsize=12)
    ax2.set_ylabel('Larmor Radius (km)', fontsize=12)
    ax2.set_title('100 keV Electron Larmor Radius vs Height', fontsize=14, weight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(-0.8, 1.0)
    ax2.set_ylim(10, 10000)
    
    plt.suptitle('Variation of Rc and RL with Height in XY Planes', 
                fontsize=16, weight='bold')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'fig09_rc_rl_vs_height_statistics.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Statistics plot saved: {output_file}")


if __name__ == "__main__":
    # Create curvature radius slices
    rc_stats = create_curvature_radius_slices()
    
    # Create Larmor radius slices
    rl_stats = create_larmor_radius_slices()
    
    # Create combined statistics plot
    create_combined_statistics_plot(rc_stats, rl_stats)
    
    print("\n" + "="*80)
    print("Curvature and Larmor radius XY plane slice analysis complete!")
    print("="*80)