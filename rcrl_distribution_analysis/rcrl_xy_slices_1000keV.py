#!/usr/bin/env python3
"""
Rc/RL Ratio in XY Plane Slices for 1000 keV Electrons
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
print("Rc/RL RATIO IN XY PLANE SLICES - 1000 keV ELECTRONS")
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


def create_xy_slices():
    """Create XY plane slices at different Z heights"""
    
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
    energy = 1000  # keV
    
    # Color levels - same as Figure 2
    levels = np.logspace(-1, 3, 20)
    
    print(f"\nCalculating Rc/RL ratio for {energy} keV electrons...")
    
    # Statistics storage
    scatter_stats = []
    
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
        
        # Calculate curvature
        kappa = field_line_curvature_vectorized(t96_vectorized, parmod, ps, 
                                               x_flat, y_flat, z_flat)
        Rc_Re = np.where(kappa > 1e-10, 1.0 / kappa, 1e3)
        Rc_m = Rc_Re * Re
        
        # Calculate Larmor radius
        RL_m = calculate_larmor_radius(energy, B_magnitude)
        
        # Calculate Rc/RL ratio
        ratio = Rc_m / RL_m
        ratio = np.where(ratio > 1000, 1000, ratio)
        ratio = np.where(ratio < 0.1, 0.1, ratio)
        ratio_grid = ratio.reshape(X.shape)
        
        # Calculate statistics
        scatter_frac = np.sum(ratio < CRITICAL_RATIO) / len(ratio) * 100
        scatter_stats.append((z_height, scatter_frac))
        
        # Plot Rc/RL ratio
        im = ax.contourf(X, Y, ratio_grid, levels=levels, 
                        cmap='RdBu_r', norm=LogNorm(vmin=0.1, vmax=1000))
        
        # Add critical contour
        cs = ax.contour(X, Y, ratio_grid, levels=[CRITICAL_RATIO], 
                       colors='black', linewidths=2)
        ax.clabel(cs, inline=True, fontsize=8, fmt='8')
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax.add_patch(earth)
        
        # Add field vectors (sparse)
        if idx % 2 == 0:  # Only for some panels to avoid clutter
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
        ax.set_title(f'Z = {z_height:.1f} Re ({scatter_frac:.1f}% < 8)', 
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
    cbar = plt.colorbar(im, cax=cbar_ax, 
                       ticks=[0.1, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000])
    cbar.set_label('Rc/RL Ratio', fontsize=12)
    cbar.ax.axhline(y=8, color='black', linewidth=3)
    
    # Add text annotation for the critical line
    cbar.ax.text(1.3, 8, 'Critical', fontsize=9, va='center')
    
    plt.suptitle(f'Rc/RL Ratio in XY Planes: 1000 keV Electrons\n' +
                f'T96 Model, Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT',
                fontsize=14, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    # Save figure
    output_file = os.path.join(output_dir, 'fig05_rcrl_xy_slices_1000keV.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nFigure saved: {output_file}")
    
    # Print statistics summary
    print("\nScattering statistics (% with Rc/RL < 8):")
    for z, frac in scatter_stats:
        print(f"  Z = {z:+.1f} Re: {frac:5.1f}%")
    
    # Create a summary plot of scattering vs Z
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    z_vals = [s[0] for s in scatter_stats]
    scatter_fracs = [s[1] for s in scatter_stats]
    
    ax2.plot(z_vals, scatter_fracs, 'b-o', linewidth=2, markersize=8)
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Z GSM (Re)', fontsize=12)
    ax2.set_ylabel('Scattering Fraction (%)', fontsize=12)
    ax2.set_title('Rc/RL < 8 Fraction vs Height: 1000 keV Electrons', 
                 fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.8, 1.0)
    ax2.set_ylim(0, max(scatter_fracs)*1.2)
    
    # Add annotation
    max_idx = np.argmax(scatter_fracs)
    ax2.annotate(f'Peak: {scatter_fracs[max_idx]:.1f}% at Z={z_vals[max_idx]:.1f} Re',
                xy=(z_vals[max_idx], scatter_fracs[max_idx]),
                xytext=(0.2, scatter_fracs[max_idx]+1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, ha='left')
    
    plt.tight_layout()
    
    output_file2 = os.path.join(output_dir, 'fig06_rcrl_vs_height_1000keV.png')
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"Summary plot saved: {output_file2}")


if __name__ == "__main__":
    create_xy_slices()
    
    print("\n" + "="*80)
    print("XY plane slice analysis complete!")
    print("="*80)