#!/usr/bin/env python3
"""
Rc/RL Distribution Analysis - XY Plane at Z=0
Analyzes the distribution in the equatorial plane where scattering is more likely
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm
import matplotlib.patches as patches
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
print("Rc/RL DISTRIBUTION ANALYSIS")
print("XY Plane (Z=0) at Equinox Conditions")
print("="*80)

# Initialize geopack for equinox
ut_equinox = 1584662400  # March 20, 2020
ps_equinox = geopack.recalc(ut_equinox)

print(f"\nEquinox conditions:")
print(f"Dipole tilt: {np.degrees(ps_equinox):.2f}°")

# Model parameters
parmod = [3.0, -30.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

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

def create_xy_distribution():
    """Create distribution plot for XY plane at Z=0"""
    
    fig = plt.figure(figsize=(20, 15))
    
    # Create XY grid - focus on regions where scattering is likely
    x_grid = np.linspace(-15, 5, 101)
    y_grid = np.linspace(-12, 12, 121)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)  # Z = 0 (equatorial plane)
    
    # Try multiple energies
    energies = [10, 30, 100]
    
    for idx, energy in enumerate(energies):
        print(f"\nAnalyzing {energy} keV electrons...")
        
        # Calculate field
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        
        # Calculate curvature
        kappa = field_line_curvature_vectorized(t96_vectorized, parmod, ps_equinox, 
                                               x_flat, y_flat, z_flat)
        Rc_Re = np.where(kappa > 1e-10, 1.0 / kappa, 1e10)
        Rc_m = Rc_Re * Re
        
        # Calculate B field
        bx, by, bz = t96_vectorized(parmod, ps_equinox, x_flat, y_flat, z_flat)
        B_nT = np.sqrt(bx**2 + by**2 + bz**2)
        
        # Calculate Larmor radius
        RL_m = calculate_larmor_radius(energy, B_nT)
        
        # Calculate ratio
        ratio = Rc_m / RL_m
        ratio = np.where(ratio > 1000, 1000, ratio)
        ratio = np.where(ratio < 0.1, 0.1, ratio)
        ratio_grid = ratio.reshape(X.shape)
        
        # Calculate statistics
        scatter_frac = np.sum(ratio < CRITICAL_RATIO) / len(ratio) * 100
        
        # Plot 1: Spatial distribution
        ax1 = plt.subplot(3, 3, idx*3 + 1)
        
        levels = np.logspace(-1, 3, 20)
        im = ax1.contourf(X, Y, ratio_grid, levels=levels, 
                         cmap='RdBu_r', norm=LogNorm(vmin=0.1, vmax=1000))
        
        cs = ax1.contour(X, Y, ratio_grid, levels=[CRITICAL_RATIO], 
                         colors='black', linewidths=2)
        ax1.clabel(cs, inline=True, fontsize=10, fmt='8')
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax1.add_patch(earth)
        
        ax1.set_xlabel('X GSM (Re)')
        ax1.set_ylabel('Y GSM (Re)')
        ax1.set_title(f'{energy} keV: Rc/RL Distribution\n{scatter_frac:.1f}% < 8')
        ax1.set_aspect('equal')
        ax1.set_xlim(-15, 5)
        ax1.set_ylim(-12, 12)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Histogram
        ax2 = plt.subplot(3, 3, idx*3 + 2)
        
        bins = np.logspace(-1, 3, 40)
        ax2.hist(ratio, bins=bins, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(CRITICAL_RATIO, color='red', linewidth=2, linestyle='--')
        
        ax2.set_xscale('log')
        ax2.set_xlabel('Rc/RL Ratio')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Distribution ({energy} keV)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Radial profile
        ax3 = plt.subplot(3, 3, idx*3 + 3)
        
        R = np.sqrt(X**2 + Y**2)
        r_bins = np.linspace(2, 15, 27)
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2
        
        scatter_radial = []
        for i in range(len(r_bins)-1):
            mask = (R.flatten() >= r_bins[i]) & (R.flatten() < r_bins[i+1])
            if np.any(mask):
                scatter_radial.append(np.sum(ratio[mask] < CRITICAL_RATIO) / np.sum(mask) * 100)
            else:
                scatter_radial.append(0)
        
        ax3.plot(r_centers, scatter_radial, 'b-', linewidth=2, marker='o')
        ax3.set_xlabel('Radial Distance (Re)')
        ax3.set_ylabel('Scattering %')
        ax3.set_title(f'Radial Profile ({energy} keV)')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(2, 15)
        ax3.set_ylim(0, max(scatter_radial)*1.2 if max(scatter_radial) > 0 else 10)
        
        print(f"  Scattering fraction: {scatter_frac:.2f}%")
        print(f"  Median Rc/RL: {np.median(ratio):.1f}")
    
    plt.suptitle('Rc/RL Distribution Analysis - XY Plane (Z=0) at Equinox\n' +
                 f'T96 Model, Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT',
                 fontsize=16, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_file = os.path.join(output_dir, 'fig02_rcrl_distribution_xy_equinox.png')
    plt.savefig(output_file)
    plt.close(fig)
    
    print(f"\nFigure saved: {output_file}")

if __name__ == "__main__":
    create_xy_distribution()