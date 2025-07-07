#!/usr/bin/env python3
"""
Regenerate fig11 with extended X range - Simple version with lower resolution
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import geopack
from geopack import t96_vectorized, field_line_curvature_vectorized

# Create output directory
output_dir = "figures"

# Physical constants
c = 2.99792458e8  # Speed of light (m/s)
me = 9.10938356e-31  # Electron mass (kg)
e = 1.602176634e-19  # Elementary charge (C)
Re = 6.371e6  # Earth radius (m)

# Critical threshold
CRITICAL_RATIO = 8.0

print("="*80)
print("REGENERATING FIGURE 11 - SIMPLE VERSION")
print("="*80)

# Initialize geopack
ut = 1584662400  # March 20, 2020 (equinox)
ps = geopack.recalc(ut)

print(f"Dipole tilt: {np.degrees(ps):.2f}°")


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


# Define parameter sets
param_sets = [
    ([2.0, 0.0, 0.0, 0.0], "Quiet"),
    ([3.0, -30.0, 1.0, -3.0], "Moderate"),
    ([5.0, -100.0, 5.0, -10.0], "Storm"),
    ([1.0, -200.0, 10.0, -20.0], "Extreme")
]

# Z heights to analyze
z_heights = [-0.4, -0.2, 0.0, 0.2]

fig, axes = plt.subplots(4, 4, figsize=(20, 20))

# Create XY grid with very low resolution for fast computation
x_grid = np.linspace(-20, 5, 26)  # Only 26 points
y_grid = np.linspace(-12, 12, 25)  # Only 25 points
X, Y = np.meshgrid(x_grid, y_grid)

# Fixed energy
energy = 100  # keV

# Color levels
levels = np.logspace(-1, 3, 20)

print("\nProcessing parameter variations...")

for row_idx, (params, condition) in enumerate(param_sets):
    parmod = params + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    print(f"  {condition}...", end='', flush=True)
    
    for col_idx, z_height in enumerate(z_heights):
        ax = axes[row_idx, col_idx]
        
        # Create Z array
        Z = np.full_like(X, z_height)
        
        # Flatten for calculation
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        
        # Calculate field
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
        
        # Plot Rc/RL ratio
        im = ax.contourf(X, Y, ratio_grid, levels=levels, 
                        cmap='RdBu_r', norm=LogNorm(vmin=0.1, vmax=1000))
        
        # Add critical contour
        try:
            cs = ax.contour(X, Y, ratio_grid, levels=[CRITICAL_RATIO], 
                           colors='black', linewidths=1.5)
            ax.clabel(cs, inline=True, fontsize=6, fmt='8')
        except:
            pass
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax.add_patch(earth)
        
        # Labels
        if col_idx == 0:
            ax.set_ylabel(f'{condition}\nY GSM (Re)', fontsize=9)
        
        if row_idx == 3:
            ax.set_xlabel('X GSM (Re)', fontsize=9)
        
        if row_idx == 0:
            ax.set_title(f'Z = {z_height} Re', fontsize=10, weight='bold')
        
        # Add scattering percentage
        color = 'red' if scatter_frac > 5 else 'black'
        ax.text(0.95, 0.95, f'{scatter_frac:.1f}%', 
               transform=ax.transAxes, fontsize=7,
               ha='right', va='top', color=color, weight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        ax.set_aspect('equal')
        ax.set_xlim(-20, 5)
        ax.set_ylim(-12, 12)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([-20, -15, -10, -5, 0, 5])
        ax.set_yticks([-10, -5, 0, 5, 10])
        ax.tick_params(labelsize=7)
    
    print(" done")

# Add colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_label('Rc/RL Ratio', fontsize=12)
cbar.ax.axhline(y=8, color='black', linewidth=2)
cbar.ax.tick_params(labelsize=9)

plt.suptitle('Rc/RL Ratio in XY Planes: Parameter and Height Variations\n' +
            '100 keV Electrons', fontsize=16, weight='bold')

plt.tight_layout(rect=[0, 0, 0.91, 0.96])

output_file = os.path.join(output_dir, 'fig11_rcrl_xy_parameter_variations.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"\nFigure saved: {output_file}")
print("Figure 11 regenerated with extended X range (-20 to 5 Re)!")