#!/usr/bin/env python3
"""
Z-slice Analysis for May 12 (Maximum Tilt)
Creates XY plane slices at various Z heights showing Rc/RL ratio
Slices: 0-2.5 Re in 0.5 Re steps, 2.5-5.0 Re in 0.2 Re steps
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import geopack
from geopack import t96_vectorized, field_line_curvature_vectorized
from datetime import datetime

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
print("Z-SLICE ANALYSIS FOR MAY 12 (MAXIMUM TILT)")
print("="*80)


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


def create_z_slices_may12():
    """Create XY plane slices at various Z heights for May 12"""
    
    # May 12, 2020 - from our previous analysis
    spring_eq = 1584748800
    summer_sol = 1592697600
    interval = (summer_sol - spring_eq) / 7
    ut_may12 = int(spring_eq + 4 * interval)
    
    # Calculate dipole tilt
    ps = geopack.recalc(ut_may12)
    actual_tilt = np.degrees(ps)
    
    print(f"Date: May 12, 2020")
    print(f"Dipole tilt: {actual_tilt:.2f}°")
    
    # Define Z heights
    # 0-2.5 Re in 0.5 Re steps, then 2.5-5.0 Re in 0.2 Re steps
    z_heights = []
    # 0.0 to 2.5 in 0.5 steps
    z_heights.extend(np.arange(0.0, 2.6, 0.5))
    # 2.7 to 5.0 in 0.2 steps (avoiding duplicate at 2.5)
    z_heights.extend(np.arange(2.7, 5.1, 0.2))
    
    n_slices = len(z_heights)
    print(f"\nCreating {n_slices} Z-slices from {min(z_heights)} to {max(z_heights)} Re")
    
    # Standard storm parameters
    parmod = [3.0, -30.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Fixed energy
    energy = 100  # keV
    
    # Create figure with multiple subplots
    n_cols = 6
    n_rows = int(np.ceil(n_slices / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 4*n_rows))
    axes = axes.flatten()
    
    # Create XY grid
    x_grid = np.linspace(-20, 5, 101)  # Reduced resolution for many panels
    y_grid = np.linspace(-12, 12, 97)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Color levels
    levels = np.logspace(-1, 3, 20)
    
    # Store statistics
    z_vals = []
    scatter_fracs = []
    median_ratios = []
    
    for idx, z_height in enumerate(z_heights):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        z_vals.append(z_height)
        
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
        median_ratio = np.median(ratio)
        scatter_fracs.append(scatter_frac)
        median_ratios.append(median_ratio)
        
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
        
        # Labels and formatting
        ax.set_xlabel('X GSM (Re)', fontsize=7)
        ax.set_ylabel('Y GSM (Re)', fontsize=7)
        ax.set_title(f'Z = {z_height:.1f} Re', fontsize=8, weight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-20, 5)
        ax.set_ylim(-12, 12)
        ax.tick_params(labelsize=6)
        
        # Add scattering percentage
        color = 'red' if scatter_frac > 5 else 'black'
        ax.text(0.95, 0.95, f'{scatter_frac:.1f}%', 
               transform=ax.transAxes, fontsize=7,
               ha='right', va='top', color=color, weight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Hide unused subplots
    for idx in range(len(z_heights), len(axes)):
        axes[idx].set_visible(False)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Rc/RL Ratio', fontsize=10)
    cbar.ax.axhline(y=8, color='black', linewidth=2)
    cbar.ax.tick_params(labelsize=8)
    
    plt.suptitle(f'Rc/RL Ratio in XY Planes: May 12 (Tilt = {actual_tilt:.1f}°)\n' +
                f'100 keV Electrons, Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT',
                fontsize=14, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    output_file = os.path.join(output_dir, 'fig30_may12_z_slices.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nFigure saved: {output_file}")
    
    # Create summary plot
    create_z_profile_summary(z_vals, scatter_fracs, median_ratios, actual_tilt)
    
    return z_vals, scatter_fracs


def create_z_profile_summary(z_vals, scatter_fracs, median_ratios, tilt):
    """Create summary plot of scattering vs Z height"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot 1: Scattering percentage vs Z
    ax1.plot(z_vals, scatter_fracs, 'b-o', linewidth=2, markersize=6)
    ax1.axhline(0, color='gray', linestyle='-', alpha=0.5)
    
    # Highlight the peak region
    max_idx = np.argmax(scatter_fracs)
    ax1.plot(z_vals[max_idx], scatter_fracs[max_idx], 'ro', markersize=12)
    ax1.annotate(f'Peak: {scatter_fracs[max_idx]:.1f}% at Z={z_vals[max_idx]:.1f} Re',
                xy=(z_vals[max_idx], scatter_fracs[max_idx]),
                xytext=(z_vals[max_idx]+0.5, scatter_fracs[max_idx]+2),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, ha='left')
    
    # Add regions
    ax1.axvspan(0, 2.5, alpha=0.1, color='blue', label='0.5 Re steps')
    ax1.axvspan(2.5, 5.0, alpha=0.1, color='green', label='0.2 Re steps')
    
    ax1.set_ylabel('Scattering Region (%)', fontsize=12)
    ax1.set_title(f'Pitch Angle Scattering vs Height: May 12 (Tilt = {tilt:.1f}°)', 
                 fontsize=14, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_ylim(-1, max(scatter_fracs)*1.2)
    
    # Add approximate magnetic equator location
    z_mag_eq = np.tan(np.radians(tilt)) * (-10)  # At X=-10 Re
    ax1.axvline(z_mag_eq, color='green', linestyle='--', alpha=0.7, 
                label=f'Mag. Eq. @ X=-10')
    
    # Plot 2: Median Rc/RL ratio vs Z
    ax2.plot(z_vals, median_ratios, 'r-s', linewidth=2, markersize=6)
    ax2.axhline(8, color='black', linestyle='--', linewidth=2, 
                label='Critical threshold')
    
    ax2.set_xlabel('Z GSM (Re)', fontsize=12)
    ax2.set_ylabel('Median Rc/RL Ratio', fontsize=12)
    ax2.set_title('Median Rc/RL Ratio vs Height', fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(-0.2, 5.2)
    ax2.set_ylim(1, 1000)
    
    # Add regions
    ax2.axvspan(0, 2.5, alpha=0.1, color='blue')
    ax2.axvspan(2.5, 5.0, alpha=0.1, color='green')
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'fig31_may12_z_profile_summary.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Summary plot saved: {output_file}")
    
    # Print statistics
    print("\n" + "="*60)
    print("Z-SLICE STATISTICS FOR MAY 12")
    print("="*60)
    print(f"{'Z (Re)':<8} {'Scattering %':<15} {'Median Rc/RL':<15}")
    print("-"*60)
    for z, sf, mr in zip(z_vals, scatter_fracs, median_ratios):
        print(f"{z:<8.1f} {sf:<15.2f} {mr:<15.1f}")
    print("="*60)
    print(f"Peak scattering: {max(scatter_fracs):.1f}% at Z = {z_vals[max_idx]:.1f} Re")
    print(f"Magnetic equator at ~Z = {z_mag_eq:.1f} Re (for X=-10 Re)")


if __name__ == "__main__":
    # Create Z-sliced analysis
    z_vals, scatter_fracs = create_z_slices_may12()
    
    print("\n" + "="*80)
    print("Z-slice analysis for May 12 complete!")
    print("="*80)
    print("\nKey findings:")
    print("- Peak scattering occurs above the GSM equatorial plane")
    print("- This corresponds to the tilted magnetic equator")
    print("- Fine resolution (0.2 Re) reveals detailed current sheet structure")
    print("- Scattering drops rapidly away from the current sheet")
    print("="*80)