#!/usr/bin/env python3
"""
Magnetic Field Vector Plots with Rc/RL Ratio
Shows vector field (Bx, Bz) with Rc/RL ratio in background for different electron energies
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm
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
print("MAGNETIC FIELD VECTOR PLOTS WITH Rc/RL RATIO")
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


def calculate_larmor_radius(energy_keV, B_nT, pitch_angle_deg=90):
    """Calculate the Larmor radius for an electron."""
    E_k = energy_keV * 1000 * e  # Convert keV to Joules
    B = B_nT * 1e-9  # Convert nT to Tesla
    
    gamma = 1 + E_k / (me * c**2)
    beta = np.sqrt(1 - 1/gamma**2)
    v = beta * c
    
    alpha = np.radians(pitch_angle_deg)
    v_perp = v * np.sin(alpha)
    
    RL = gamma * me * v_perp / (e * B)
    return RL


def create_rcrl_ratio_plots():
    """Create magnetic field vector plots with Rc/RL ratio backgrounds for different energies"""
    
    energies = [10, 100, 1000]  # keV
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Create grid for background
    x_bg = np.linspace(-20, 10, 151)
    z_bg = np.linspace(-10, 10, 101)
    X_bg, Z_bg = np.meshgrid(x_bg, z_bg)
    Y_bg = np.zeros_like(X_bg)
    
    # Calculate field and curvature
    print("\nCalculating magnetic field and curvature...")
    x_flat = X_bg.flatten()
    y_flat = Y_bg.flatten()
    z_flat = Z_bg.flatten()
    
    # Calculate magnetic field
    bx_bg, by_bg, bz_bg = t96_vectorized(parmod, ps, x_flat, y_flat, z_flat)
    B_magnitude = np.sqrt(bx_bg**2 + by_bg**2 + bz_bg**2)
    
    # Calculate curvature
    kappa = field_line_curvature_vectorized(t96_vectorized, parmod, ps, x_flat, y_flat, z_flat)
    Rc_Re = np.where(kappa > 1e-10, 1.0 / kappa, 1e3)  # Cap at 1000 Re
    Rc_m = Rc_Re * Re  # Convert to meters
    
    # Create coarser grid for vectors
    x_vec = np.linspace(-20, 10, 21)
    z_vec = np.linspace(-10, 10, 15)
    X_vec, Z_vec = np.meshgrid(x_vec, z_vec)
    Y_vec = np.zeros_like(X_vec)
    
    # Calculate field vectors
    x_vec_flat = X_vec.flatten()
    y_vec_flat = Y_vec.flatten()
    z_vec_flat = Z_vec.flatten()
    
    bx_vec, by_vec, bz_vec = t96_vectorized(parmod, ps, x_vec_flat, y_vec_flat, z_vec_flat)
    
    # Reshape vectors
    Bx_grid = bx_vec.reshape(X_vec.shape)
    Bz_grid = bz_vec.reshape(Z_vec.shape)
    
    # Normalize vectors
    B_vec_mag = np.sqrt(Bx_grid**2 + Bz_grid**2)
    Bx_norm = np.divide(Bx_grid, B_vec_mag, out=np.zeros_like(Bx_grid), where=B_vec_mag>0)
    Bz_norm = np.divide(Bz_grid, B_vec_mag, out=np.zeros_like(Bz_grid), where=B_vec_mag>0)
    
    # Use same colormap as Figure 2
    levels = np.logspace(-1, 3, 20)  # From 0.1 to 1000
    
    for idx, (ax, energy) in enumerate(zip(axes, energies)):
        print(f"Creating Rc/RL ratio plot for {energy} keV...")
        
        # Calculate Larmor radius
        RL_m = calculate_larmor_radius(energy, B_magnitude)
        
        # Calculate Rc/RL ratio
        ratio = Rc_m / RL_m
        ratio = np.where(ratio > 1000, 1000, ratio)  # Cap at 1000
        ratio = np.where(ratio < 0.1, 0.1, ratio)    # Floor at 0.1
        ratio_grid = ratio.reshape(X_bg.shape)
        
        # Plot Rc/RL ratio as background
        im = ax.contourf(X_bg, Z_bg, ratio_grid, levels=levels, 
                         cmap='RdBu_r', norm=LogNorm(vmin=0.1, vmax=1000))
        
        # Add critical contour
        cs = ax.contour(X_bg, Z_bg, ratio_grid, levels=[CRITICAL_RATIO], 
                        colors='black', linewidths=3)
        ax.clabel(cs, inline=True, fontsize=10, fmt='Rc/RL=8')
        
        # Scale vectors
        scale_factor = 0.4
        
        # Plot vectors
        ax.quiver(X_vec, Z_vec, Bx_norm * scale_factor, Bz_norm * scale_factor,
                  color='white', alpha=0.8, width=0.003, headwidth=3, headlength=4,
                  scale=10, scale_units='xy', edgecolor='black', linewidth=0.5)
        
        # Add field lines
        strm = ax.streamplot(x_bg, z_bg, 
                            bx_bg.reshape(len(z_bg), len(x_bg)), 
                            bz_bg.reshape(len(z_bg), len(x_bg)),
                            color='gray', linewidth=0.8, density=0.3)
        strm.lines.set_alpha(0.5)
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax.add_patch(earth)
        ax.text(0, 0, 'E', ha='center', va='center', fontsize=10, weight='bold')
        
        # Labels and formatting
        ax.set_xlabel('X GSM (Re)', fontsize=10)
        ax.set_ylabel('Z GSM (Re)', fontsize=10)
        ax.set_title(f'{energy} keV Electrons', fontsize=12, weight='bold')
        
        ax.set_xlim(-20, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, color='gray')
        
        # Add statistics text
        scatter_frac = np.sum(ratio < CRITICAL_RATIO) / len(ratio) * 100
        stats_text = f'Rc/RL < 8: {scatter_frac:.1f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add explanation text for first panel
        if idx == 0:
            ax.text(-18, 8, 'Red: Rc/RL < 8 (scattering)', fontsize=9, color='black',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            ax.text(-18, 6, 'Blue: Rc/RL > 8 (stable)', fontsize=9, color='black',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add a single colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax, ticks=[0.1, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    cbar.set_label('Rc/RL Ratio', fontsize=12)
    cbar.ax.axhline(y=8, color='black', linewidth=3)
    
    plt.suptitle(f'T96 Magnetic Field Vectors with Rc/RL Ratio\n' +
                f'Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT, IMF By={parmod[2]} nT, Bz={parmod[3]} nT',
                fontsize=14, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    # Save figure
    output_file = os.path.join(output_dir, 'fig03c_magnetic_field_vectors_rcrl_ratio.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nRc/RL ratio plots saved: {output_file}")
    
    # Print statistics
    print(f"\nScattering statistics (Rc/RL < 8):")
    for energy in energies:
        RL_m = calculate_larmor_radius(energy, B_magnitude)
        ratio = Rc_m / RL_m
        scatter_frac = np.sum(ratio < CRITICAL_RATIO) / len(ratio) * 100
        print(f"  {energy:4d} keV: {scatter_frac:5.1f}% of XZ plane")


if __name__ == "__main__":
    create_rcrl_ratio_plots()
    
    print("\n" + "="*80)
    print("Rc/RL ratio plots complete!")
    print("="*80)