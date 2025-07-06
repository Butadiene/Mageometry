#!/usr/bin/env python3
"""
Rc/RL Distribution Analysis
Analyzes the distribution of Rc/RL values in the XZ plane (Y=0) at equinox conditions

This analysis creates distribution plots showing how Rc/RL values are distributed
spatially and statistically, with emphasis on the critical threshold Rc/RL = 8.

Author: Generated Analysis
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm
import matplotlib.patches as patches
import os
from datetime import datetime
import geopack
from geopack import t96_vectorized, field_line_curvature_vectorized

# Create output directory
output_dir = "figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set up plotting
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Physical constants
c = 2.99792458e8  # Speed of light (m/s)
me = 9.10938356e-31  # Electron mass (kg)
e = 1.602176634e-19  # Elementary charge (C)
me_c2_keV = 511.0  # Electron rest energy (keV)
Re = 6.371e6  # Earth radius (m)

# Critical threshold
CRITICAL_RATIO = 8.0

print("="*80)
print("Rc/RL DISTRIBUTION ANALYSIS")
print("XZ Plane (Y=0) at Equinox Conditions")
print("="*80)

# Initialize geopack for equinox (March 20 or September 22)
# Using Spring equinox 2020: March 20, 2020
ut_equinox = 1584662400  # March 20, 2020 00:00:00 UTC
ps_equinox = geopack.recalc(ut_equinox)

print(f"\nEquinox conditions:")
print(f"Date: March 20, 2020")
print(f"Dipole tilt: {np.degrees(ps_equinox):.2f}° (should be near 0)")

# Model parameters - moderate storm conditions
parmod = [3.0, -30.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
print(f"\nModel parameters (T96):")
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


def calculate_curvature_radius(x, y, z):
    """Calculate radius of curvature and magnetic field strength."""
    kappa = field_line_curvature_vectorized(t96_vectorized, parmod, ps_equinox, x, y, z)
    Rc = np.where(kappa > 1e-10, 1.0 / kappa, 1e10)
    
    bx, by, bz = t96_vectorized(parmod, ps_equinox, x, y, z)
    B_nT = np.sqrt(bx**2 + by**2 + bz**2)
    
    return Rc, B_nT


def create_distribution_plot():
    """
    Create a comprehensive distribution plot of Rc/RL values in the XZ plane at Y=0
    """
    print("\n" + "="*60)
    print("Creating Rc/RL Distribution Plot")
    print("="*60)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Define grid for XZ plane (Y=0)
    # Focus on the tail region where scattering is more likely
    x_grid = np.linspace(-20, 5, 126)
    z_grid = np.linspace(-5, 5, 101)
    X, Z = np.meshgrid(x_grid, z_grid)
    Y = np.zeros_like(X)  # Y = 0 plane
    
    # Flatten for calculation
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    
    # Calculate Rc and B
    print("Calculating field line curvature...")
    Rc_Re, B_nT = calculate_curvature_radius(x_flat, y_flat, z_flat)
    Rc_m = Rc_Re * Re
    
    # Calculate for 30 keV electrons (lower energy = larger Larmor radius = more scattering)
    energy = 30  # keV
    print(f"Calculating Larmor radius for {energy} keV electrons...")
    RL_m = calculate_larmor_radius(energy, B_nT, pitch_angle_deg=90)
    
    # Calculate Rc/RL ratio
    ratio = Rc_m / RL_m
    
    # Clean up extreme values
    ratio = np.where(ratio > 1000, 1000, ratio)
    ratio = np.where(ratio < 0.1, 0.1, ratio)
    
    # Reshape for plotting
    ratio_grid = ratio.reshape(X.shape)
    
    # Subplot 1: Spatial distribution with contours
    ax1 = plt.subplot(2, 3, 1)
    
    # Create custom colormap
    levels = np.array([0.1, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    colors = ['darkred', 'red', 'orange', 'yellow', 'lightgreen', 'green', 
              'cyan', 'blue', 'darkblue', 'purple', 'magenta', 'pink']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, cmap.N)
    
    im1 = ax1.contourf(X, Z, ratio_grid, levels=levels, cmap=cmap, norm=norm, extend='both')
    
    # Add critical contour
    cs = ax1.contour(X, Z, ratio_grid, levels=[CRITICAL_RATIO], 
                     colors='black', linewidths=3)
    ax1.clabel(cs, inline=True, fontsize=12, fmt='Rc/RL=8')
    
    # Add Earth
    earth = plt.Circle((0, 0), 1, color='lightgray', zorder=10)
    ax1.add_patch(earth)
    
    ax1.set_xlabel('X GSM (Re)')
    ax1.set_ylabel('Z GSM (Re)')
    ax1.set_title(f'Rc/RL Spatial Distribution\n{energy} keV, Equinox (PS={np.degrees(ps_equinox):.1f}°)')
    ax1.set_aspect('equal')
    ax1.set_xlim(-15, 5)
    ax1.set_ylim(-10, 10)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Histogram of Rc/RL values
    ax2 = plt.subplot(2, 3, 2)
    
    # Create histogram
    bins = np.logspace(-1, 3, 50)
    hist, edges = np.histogram(ratio, bins=bins)
    
    # Plot histogram
    ax2.stairs(hist, edges, fill=True, alpha=0.7, color='blue')
    ax2.axvline(CRITICAL_RATIO, color='red', linewidth=2, linestyle='--', 
                label=f'Rc/RL = {CRITICAL_RATIO}')
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Rc/RL Ratio')
    ax2.set_ylabel('Number of Grid Points')
    ax2.set_title('Distribution of Rc/RL Values')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Calculate statistics
    scatter_fraction = np.sum(ratio < CRITICAL_RATIO) / len(ratio) * 100
    median_ratio = np.median(ratio)
    mean_ratio = np.mean(ratio)
    
    stats_text = f'Rc/RL < 8: {scatter_fraction:.1f}%\nMedian: {median_ratio:.1f}\nMean: {mean_ratio:.1f}'
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Subplot 3: Cumulative distribution
    ax3 = plt.subplot(2, 3, 3)
    
    # Sort values for cumulative plot
    sorted_ratio = np.sort(ratio)
    cumulative = np.arange(1, len(sorted_ratio) + 1) / len(sorted_ratio) * 100
    
    ax3.plot(sorted_ratio, cumulative, 'b-', linewidth=2)
    ax3.axvline(CRITICAL_RATIO, color='red', linewidth=2, linestyle='--', 
                label=f'Rc/RL = {CRITICAL_RATIO}')
    ax3.axhline(scatter_fraction, color='green', linewidth=1, linestyle=':',
                label=f'{scatter_fraction:.1f}% < 8')
    
    ax3.set_xscale('log')
    ax3.set_xlabel('Rc/RL Ratio')
    ax3.set_ylabel('Cumulative Percentage (%)')
    ax3.set_title('Cumulative Distribution')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(0.1, 1000)
    ax3.set_ylim(0, 100)
    
    # Subplot 4: Radial dependence
    ax4 = plt.subplot(2, 3, 4)
    
    # Calculate radial distance
    R = np.sqrt(X**2 + Z**2)
    r_bins = np.linspace(0, 15, 31)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    
    # Calculate mean Rc/RL in radial bins
    mean_ratio_radial = []
    scatter_frac_radial = []
    
    for i in range(len(r_bins)-1):
        mask = (R.flatten() >= r_bins[i]) & (R.flatten() < r_bins[i+1])
        if np.any(mask):
            mean_ratio_radial.append(np.mean(ratio[mask]))
            scatter_frac_radial.append(np.sum(ratio[mask] < CRITICAL_RATIO) / np.sum(mask) * 100)
        else:
            mean_ratio_radial.append(np.nan)
            scatter_frac_radial.append(0)
    
    # Plot radial profiles
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(r_centers, mean_ratio_radial, 'b-', linewidth=2, 
                     marker='o', label='Mean Rc/RL')
    ax4.axhline(CRITICAL_RATIO, color='red', linewidth=2, linestyle='--', alpha=0.5)
    ax4.set_yscale('log')
    ax4.set_xlabel('Radial Distance (Re)')
    ax4.set_ylabel('Mean Rc/RL Ratio', color='b')
    ax4.tick_params(axis='y', labelcolor='b')
    ax4.grid(True, alpha=0.3)
    
    line2 = ax4_twin.plot(r_centers, scatter_frac_radial, 'g-', linewidth=2, 
                          marker='s', label='% with Rc/RL < 8')
    ax4_twin.set_ylabel('Scattering Percentage (%)', color='g')
    ax4_twin.tick_params(axis='y', labelcolor='g')
    ax4_twin.set_ylim(0, 100)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper right')
    ax4.set_title('Radial Profiles')
    
    # Subplot 5: Z-dependence at different X
    ax5 = plt.subplot(2, 3, 5)
    
    x_slices = [-10, -7, -4, -2, 0]
    colors_slice = ['purple', 'blue', 'green', 'orange', 'red']
    
    for x_val, color in zip(x_slices, colors_slice):
        # Find closest x index
        x_idx = np.argmin(np.abs(x_grid - x_val))
        ratio_slice = ratio_grid[:, x_idx]
        
        ax5.plot(z_grid, ratio_slice, color=color, linewidth=2, 
                label=f'X = {x_val} Re')
    
    ax5.axhline(CRITICAL_RATIO, color='black', linewidth=2, linestyle='--', alpha=0.5)
    ax5.set_yscale('log')
    ax5.set_xlabel('Z GSM (Re)')
    ax5.set_ylabel('Rc/RL Ratio')
    ax5.set_title('Z-profiles at Different X')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_xlim(-10, 10)
    ax5.set_ylim(0.1, 1000)
    
    # Subplot 6: Scatter region map
    ax6 = plt.subplot(2, 3, 6)
    
    scatter_mask = ratio_grid < CRITICAL_RATIO
    
    # Create three categories
    categories = np.zeros_like(ratio_grid)
    categories[ratio_grid < 4] = 3  # Strong scattering
    categories[(ratio_grid >= 4) & (ratio_grid < 8)] = 2  # Moderate scattering
    categories[(ratio_grid >= 8) & (ratio_grid < 16)] = 1  # Weak scattering
    
    colors_cat = ['white', 'lightblue', 'orange', 'red']
    cmap_cat = ListedColormap(colors_cat)
    
    im6 = ax6.contourf(X, Z, categories, levels=[0, 1, 2, 3, 4], 
                       cmap=cmap_cat, extend='neither')
    
    # Add Earth
    earth = plt.Circle((0, 0), 1, color='gray', zorder=10)
    ax6.add_patch(earth)
    
    ax6.set_xlabel('X GSM (Re)')
    ax6.set_ylabel('Z GSM (Re)')
    ax6.set_title('Scattering Intensity Regions')
    ax6.set_aspect('equal')
    ax6.set_xlim(-15, 5)
    ax6.set_ylim(-10, 10)
    ax6.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [patches.Patch(color='red', label='Strong (Rc/RL < 4)'),
                      patches.Patch(color='orange', label='Moderate (4-8)'),
                      patches.Patch(color='lightblue', label='Weak (8-16)'),
                      patches.Patch(color='white', label='Negligible (> 16)')]
    ax6.legend(handles=legend_elements, loc='upper right')
    
    # Add colorbar for first subplot
    cbar_ax = fig.add_axes([0.92, 0.55, 0.02, 0.35])
    cbar = plt.colorbar(im1, cax=cbar_ax, ticks=levels)
    cbar.set_label('Rc/RL Ratio', fontsize=12)
    
    # Overall title
    fig.suptitle(f'Rc/RL Distribution Analysis - XZ Plane (Y=0) at Equinox\n' +
                 f'T96 Model, Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT',
                 fontsize=16, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    # Save figure
    output_file = os.path.join(output_dir, 'fig01_rcrl_distribution_equinox.png')
    plt.savefig(output_file)
    plt.close(fig)
    
    print(f"\nFigure saved: {output_file}")
    print(f"\nKey Statistics:")
    print(f"- {scatter_fraction:.1f}% of region has Rc/RL < 8")
    print(f"- Median Rc/RL: {median_ratio:.1f}")
    print(f"- Mean Rc/RL: {mean_ratio:.1f}")
    
    return scatter_fraction, median_ratio, mean_ratio


def main():
    """Main analysis function"""
    # Create distribution plot
    scatter_frac, median_ratio, mean_ratio = create_distribution_plot()
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nSummary for equinox conditions (PS ≈ 0°):")
    print(f"- Scattering regions (Rc/RL < 8) cover {scatter_frac:.1f}% of the XZ plane")
    print(f"- Strong pitch angle scattering expected in these regions")
    print(f"- Maximum scattering near current sheet (Z ≈ 0)")
    print("="*80)


if __name__ == "__main__":
    main()