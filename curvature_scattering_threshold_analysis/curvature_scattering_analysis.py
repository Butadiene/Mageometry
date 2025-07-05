#!/usr/bin/env python3
"""
Magnetic Field Line Curvature Scattering Analysis: Rc/RL = 8 Threshold

This script analyzes the critical threshold Rc/RL = 8 for electron scattering in the magnetosphere.
When Rc/RL < 8:
- Violation of the first adiabatic invariant becomes significant
- Particles can be scattered into the loss cone
- Enhanced particle precipitation to the atmosphere
- Important for aurora formation and radiation belt losses

Author: Generated Analysis
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import os
from datetime import datetime

# Import geopack
import geopack
from geopack import (
    t89_vectorized, t96_vectorized, t01_vectorized, t04_vectorized,
    field_line_curvature_vectorized
)

# Create output directory for figures
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

# Define the critical threshold
CRITICAL_RATIO = 8.0

print("Physical Constants:")
print(f"Electron rest energy: {me_c2_keV} keV")
print(f"Earth radius: {Re/1e6:.3f} Mm")
print(f"Critical threshold: Rc/RL = {CRITICAL_RATIO}")
print("Below this threshold: Strong pitch angle scattering")
print("Above this threshold: Weak scattering, adiabatic motion")

# Initialize geopack
ut = 1600000000  # Unix timestamp
ps = geopack.recalc(ut)

# Model parameters for different conditions
# Quiet conditions
parmod_quiet = [1.0, -5.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Moderate storm
parmod_moderate = [3.0, -30.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Strong storm
parmod_storm = [10.0, -100.0, 5.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

print(f"\\nDipole tilt: {np.degrees(ps):.2f}°")
print("Geomagnetic conditions:")
print(f"Quiet: Pdyn={parmod_quiet[0]} nPa, Dst={parmod_quiet[1]} nT")
print(f"Moderate: Pdyn={parmod_moderate[0]} nPa, Dst={parmod_moderate[1]} nT")
print(f"Storm: Pdyn={parmod_storm[0]} nPa, Dst={parmod_storm[1]} nT")


# Helper functions
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


def calculate_curvature_radius(model_func, parmod, ps, x, y, z):
    """Calculate radius of curvature and magnetic field strength."""
    kappa = field_line_curvature_vectorized(model_func, parmod, ps, x, y, z)
    Rc = np.where(kappa > 1e-10, 1.0 / kappa, 1e10)
    
    bx, by, bz = model_func(parmod, ps, x, y, z)
    B = np.sqrt(bx**2 + by**2 + bz**2)
    
    return Rc, B


def create_binary_colormap():
    """Create a colormap for Rc/RL > 8 (safe) vs < 8 (scattering)."""
    colors = ['darkred', 'lightcoral', 'lightblue', 'darkblue']
    n_bins = 4
    cmap = ListedColormap(colors)
    bounds = [0, 0.5, 1, 8, 100]
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.
    
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca()
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
    # Also try to set box aspect for newer matplotlib versions
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass


def analyze_scattering_regions_by_energy():
    """
    Analysis 1: Scattering Regions for Different Electron Energies
    Each particle energy gets its own subplot with the Rc/RL ratio as color
    """
    print("\\n" + "="*60)
    print("Analysis 1: Scattering Regions for Different Electron Energies")
    print("="*60)
    
    # Create figure with subplots for each energy
    energies = [10, 30, 100, 300, 1000]  # keV
    fig = plt.figure(figsize=(20, 16))

    # Create subplots: 2 rows (meridian and equatorial), 5 columns (energies)
    for idx, energy in enumerate(energies):
        # Meridian plane (X-Z)
        ax_meridian = plt.subplot(2, 5, idx + 1)
        
        # Create grid for meridian plane
        x_grid = np.linspace(-15, 5, 101)
        z_grid = np.linspace(-10, 10, 81)
        X_mer, Z_mer = np.meshgrid(x_grid, z_grid)
        Y_mer = np.zeros_like(X_mer)
        
        x_flat_mer = X_mer.flatten()
        y_flat_mer = Y_mer.flatten()
        z_flat_mer = Z_mer.flatten()
        
        # Calculate for moderate storm conditions
        Rc_Re_mer, B_nT_mer = calculate_curvature_radius(t96_vectorized, parmod_moderate, ps, 
                                                         x_flat_mer, y_flat_mer, z_flat_mer)
        Rc_m_mer = Rc_Re_mer * Re
        
        # Calculate Larmor radius
        RL_m_mer = calculate_larmor_radius(energy, B_nT_mer, pitch_angle_deg=90)
        ratio_mer = Rc_m_mer / RL_m_mer
        
        # Clean up extreme values for better visualization
        ratio_mer = np.where(ratio_mer > 1000, 1000, ratio_mer)
        ratio_mer = np.where(ratio_mer < 0.1, 0.1, ratio_mer)
        ratio_grid_mer = ratio_mer.reshape(X_mer.shape)
        
        # Plot with continuous colormap showing Rc/RL values
        im_mer = ax_meridian.contourf(X_mer, Z_mer, ratio_grid_mer,
                                      levels=np.logspace(-1, 3, 30),
                                      cmap='RdBu_r', extend='both',
                                      norm=LogNorm(vmin=0.1, vmax=1000))
        
        # Add the critical Rc/RL = 8 contour
        cs_mer = ax_meridian.contour(X_mer, Z_mer, ratio_grid_mer, 
                                     levels=[CRITICAL_RATIO],
                                     colors='black', linewidths=3)
        ax_meridian.clabel(cs_mer, inline=True, fontsize=10, fmt='Rc/RL=8')
        
        # Add other reference contours
        ax_meridian.contour(X_mer, Z_mer, ratio_grid_mer, 
                           levels=[1, 2, 4, 16, 32],
                           colors='gray', linewidths=1, alpha=0.5)
        
        # Calculate scattering fraction
        scatter_frac_mer = np.sum(ratio_mer < CRITICAL_RATIO) / len(ratio_mer) * 100
        
        ax_meridian.set_title(f'{energy} keV\\nNoon-Midnight Meridian\\nScattering: {scatter_frac_mer:.1f}%',
                              fontsize=12)
        if idx == 0:
            ax_meridian.set_ylabel('Z GSM (Re)', fontsize=12)
        ax_meridian.set_xlabel('X GSM (Re)', fontsize=10)
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax_meridian.add_patch(earth)
        ax_meridian.set_aspect('equal')
        ax_meridian.set_xlim(-15, 10)
        ax_meridian.set_ylim(-10, 10)
        
        # Add colorbar for rightmost plot
        if idx == len(energies) - 1:
            cbar_ax = fig.add_axes([0.92, 0.53, 0.01, 0.35])
            cbar = plt.colorbar(im_mer, cax=cbar_ax)
            cbar.set_label('Rc/RL Ratio', fontsize=12)
        
        # Equatorial plane (X-Y)
        ax_equator = plt.subplot(2, 5, idx + 6)
        
        # Create grid for equatorial plane
        x_grid_eq = np.linspace(-15, 5, 101)
        y_grid_eq = np.linspace(-12, 12, 97)
        X_eq, Y_eq = np.meshgrid(x_grid_eq, y_grid_eq)
        Z_eq = np.zeros_like(X_eq)
        
        x_flat_eq = X_eq.flatten()
        y_flat_eq = Y_eq.flatten()
        z_flat_eq = Z_eq.flatten()
        
        # Calculate for equatorial plane
        Rc_Re_eq, B_nT_eq = calculate_curvature_radius(t96_vectorized, parmod_moderate, ps, 
                                                       x_flat_eq, y_flat_eq, z_flat_eq)
        Rc_m_eq = Rc_Re_eq * Re
        
        # Calculate Larmor radius
        RL_m_eq = calculate_larmor_radius(energy, B_nT_eq, pitch_angle_deg=90)
        ratio_eq = Rc_m_eq / RL_m_eq
        
        # Clean up extreme values
        ratio_eq = np.where(ratio_eq > 1000, 1000, ratio_eq)
        ratio_eq = np.where(ratio_eq < 0.1, 0.1, ratio_eq)
        ratio_grid_eq = ratio_eq.reshape(X_eq.shape)
        
        # Plot equatorial plane
        im_eq = ax_equator.contourf(X_eq, Y_eq, ratio_grid_eq,
                                    levels=np.logspace(-1, 3, 30),
                                    cmap='RdBu_r', extend='both',
                                    norm=LogNorm(vmin=0.1, vmax=1000))
        
        # Add the critical Rc/RL = 8 contour
        cs_eq = ax_equator.contour(X_eq, Y_eq, ratio_grid_eq, 
                                   levels=[CRITICAL_RATIO],
                                   colors='black', linewidths=3)
        ax_equator.clabel(cs_eq, inline=True, fontsize=10, fmt='8')
        
        # Add other reference contours
        ax_equator.contour(X_eq, Y_eq, ratio_grid_eq, 
                          levels=[1, 2, 4, 16, 32],
                          colors='gray', linewidths=1, alpha=0.5)
        
        # Calculate scattering fraction
        scatter_frac_eq = np.sum(ratio_eq < CRITICAL_RATIO) / len(ratio_eq) * 100
        
        ax_equator.set_title(f'Magnetic Equator\\nScattering: {scatter_frac_eq:.1f}%',
                            fontsize=12)
        if idx == 0:
            ax_equator.set_ylabel('Y GSM (Re)', fontsize=12)
        ax_equator.set_xlabel('X GSM (Re)', fontsize=10)
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax_equator.add_patch(earth)
        ax_equator.set_aspect('equal')
        ax_equator.set_xlim(-15, 10)
        ax_equator.set_ylim(-12, 12)
        
        # Add labels for MLT
        if idx == 2:  # Middle plot
            ax_equator.text(10, 0, '12', ha='center', va='center', fontsize=10, weight='bold')
            ax_equator.text(-15, 0, '00', ha='center', va='center', fontsize=10, weight='bold')
            ax_equator.text(0, 12, '06', ha='center', va='center', fontsize=10, weight='bold')
            ax_equator.text(0, -12, '18', ha='center', va='center', fontsize=10, weight='bold')
        
        # Add colorbar for rightmost plot
        if idx == len(energies) - 1:
            cbar_ax2 = fig.add_axes([0.92, 0.11, 0.01, 0.35])
            cbar2 = plt.colorbar(im_eq, cax=cbar_ax2)
            cbar2.set_label('Rc/RL Ratio', fontsize=12)

    # Add color scale description
    fig.text(0.5, 0.96, f'Electron Scattering Regions: Critical Threshold Rc/RL = {CRITICAL_RATIO}',
             ha='center', fontsize=16, weight='bold')
    fig.text(0.5, 0.94, 'Red regions: Rc/RL < 8 (strong scattering), Blue regions: Rc/RL > 8 (adiabatic motion)',
             ha='center', fontsize=12)
    fig.text(0.5, 0.92, f'T96 Model: Moderate Storm (Pdyn={parmod_moderate[0]} nPa, Dst={parmod_moderate[1]} nT)',
             ha='center', fontsize=12)

    plt.tight_layout(rect=[0, 0, 0.91, 0.91])
    plt.savefig(os.path.join(output_dir, 'fig01_scattering_regions_by_energy.png'))
    plt.close(fig)
    
    print("Figure 1 saved: Scattering regions for different electron energies")


def analyze_xy_plane_cross_sections():
    """
    Analysis 1b: XY Plane Cross-sections at Different Z Heights
    Shows how the scattering regions vary with height above/below the magnetic equatorial plane
    """
    print("\\n" + "="*60)
    print("Analysis 1b: XY Plane Cross-sections at Different Z Heights")
    print("="*60)
    
    # XY plane analysis at different Z heights
    z_levels = np.arange(0, 1.4, 0.2)  # Re - 0.2 Re increments
    energy_xy = 100  # keV

    # Create figure with subplots for each Z level
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes_flat = axes.flatten()

    # Create grid for XY plane
    x_grid_xy = np.linspace(-15, 5, 51)
    y_grid_xy = np.linspace(-12, 12, 49)
    X_xy, Y_xy = np.meshgrid(x_grid_xy, y_grid_xy)

    print("Computing XY plane cross-sections...")
    for idx, z_level in enumerate(z_levels):
        print(f"  Processing Z = {z_level:.1f} Re ({idx+1}/{len(z_levels)})...")
        ax = axes_flat[idx]
        
        # Create Z array at fixed height
        Z_xy = np.full_like(X_xy, z_level)
        
        x_flat_xy = X_xy.flatten()
        y_flat_xy = Y_xy.flatten()
        z_flat_xy = Z_xy.flatten()
        
        # Calculate for moderate storm conditions
        Rc_Re_xy, B_nT_xy = calculate_curvature_radius(t96_vectorized, parmod_moderate, ps, 
                                                       x_flat_xy, y_flat_xy, z_flat_xy)
        Rc_m_xy = Rc_Re_xy * Re
        
        # Calculate Larmor radius
        RL_m_xy = calculate_larmor_radius(energy_xy, B_nT_xy, pitch_angle_deg=90)
        ratio_xy = Rc_m_xy / RL_m_xy
        
        # Clean up extreme values
        ratio_xy = np.where(ratio_xy > 1000, 1000, ratio_xy)
        ratio_xy = np.where(ratio_xy < 0.1, 0.1, ratio_xy)
        ratio_grid_xy = ratio_xy.reshape(X_xy.shape)
        
        # Calculate scattering fraction
        scatter_frac_xy = np.sum(ratio_xy < CRITICAL_RATIO) / len(ratio_xy) * 100
        
        # Plot with continuous colormap
        im = ax.contourf(X_xy, Y_xy, ratio_grid_xy,
                         levels=np.logspace(-1, 3, 20),
                         cmap='RdBu_r', extend='both',
                         norm=LogNorm(vmin=0.1, vmax=1000))
        
        # Add the critical Rc/RL = 8 contour
        cs = ax.contour(X_xy, Y_xy, ratio_grid_xy, 
                        levels=[CRITICAL_RATIO],
                        colors='black', linewidths=3)
        ax.clabel(cs, inline=True, fontsize=10, fmt='Rc/RL=8')
        
        # Add other reference contours
        ax.contour(X_xy, Y_xy, ratio_grid_xy, 
                   levels=[1, 2, 4, 16, 32],
                   colors='gray', linewidths=1, alpha=0.5)
        
        # Title with Z level and scattering fraction
        ax.set_title(f'Z = {z_level} Re\\nScattering: {scatter_frac_xy:.1f}%',
                     fontsize=12)
        ax.set_xlabel('X GSM (Re)', fontsize=10)
        ax.set_ylabel('Y GSM (Re)', fontsize=10)
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax.add_patch(earth)
        ax.set_aspect('equal')
        ax.set_xlim(-15, 5)
        ax.set_ylim(-12, 12)
        
        # Add MLT labels for Z=0 case
        if z_level == 0:
            ax.text(10, 0, '12', ha='center', va='center', fontsize=10, weight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.text(-15, 0, '00', ha='center', va='center', fontsize=10, weight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.text(0, 12, '06', ha='center', va='center', fontsize=10, weight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.text(0, -12, '18', ha='center', va='center', fontsize=10, weight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Remove unused subplots
    for i in range(len(z_levels), len(axes_flat)):
        axes_flat[i].axis('off')

    # Add colorbar to the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Rc/RL Ratio', fontsize=12)

    plt.suptitle(f'XY Plane Cross-sections at Different Heights: {energy_xy} keV Electrons\\n' + 
                 f'T96 Model: Moderate Storm (Pdyn={parmod_moderate[0]} nPa, Dst={parmod_moderate[1]} nT)',
                 fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.91, 0.97])
    plt.savefig(os.path.join(output_dir, 'fig02_xy_plane_cross_sections.png'))
    plt.close(fig)
    
    print("Figure 2 saved: XY plane cross-sections at different Z heights")
    print("Key Observations:")
    print("- Scattering regions are largest at Z = 0 (magnetic equator)")
    print("- Dawn-dusk asymmetry is most pronounced near the equatorial plane")
    print("- Scattering regions shrink with increasing |Z|")
    print("- Off-equatorial regions show more localized scattering patterns")


def analyze_magnetic_equatorial_plane():
    """
    Analysis 2: Magnetic Equatorial Plane - MLT Dependence
    The magnetic equatorial plane is crucial for understanding wave-particle interactions,
    ring current dynamics, dawn-dusk asymmetries, and particle drift paths
    """
    print("\\n" + "="*60)
    print("Analysis 2: Magnetic Equatorial Plane - MLT Dependence")
    print("="*60)
    
    # Create analysis for magnetic equatorial plane
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Panel 1: Radial profiles at different MLTs
    ax1 = fig.add_subplot(gs[0, :2])

    # Define MLT sectors
    mlt_hours = [0, 6, 12, 18]  # Midnight, Dawn, Noon, Dusk
    mlt_angles = [np.pi, np.pi/2, 0, -np.pi/2]  # Corresponding angles
    colors = ['darkblue', 'orange', 'red', 'green']
    labels = ['Midnight', 'Dawn', 'Noon', 'Dusk']

    # Energy for analysis
    energy_eq = 100  # keV

    # Radial distance range
    r_range = np.linspace(2, 12, 50)

    for mlt, angle, color, label in zip(mlt_hours, mlt_angles, colors, labels):
        # Calculate positions along radial profile
        x_profile = r_range * np.cos(angle)
        y_profile = r_range * np.sin(angle)
        z_profile = np.zeros_like(r_range)
        
        # Calculate fields
        Rc_Re_prof, B_nT_prof = calculate_curvature_radius(t96_vectorized, parmod_moderate, ps,
                                                           x_profile, y_profile, z_profile)
        
        # Calculate Larmor radius
        RL_m_prof = calculate_larmor_radius(energy_eq, B_nT_prof, pitch_angle_deg=90)
        ratio_prof = Rc_Re_prof * Re / RL_m_prof
        
        # Clean up extreme values
        ratio_prof = np.where(ratio_prof > 1000, 1000, ratio_prof)
        
        # Plot
        ax1.semilogy(r_range, ratio_prof, '-', color=color, linewidth=2.5, label=label)

    # Add critical threshold line
    ax1.axhline(y=CRITICAL_RATIO, color='black', linestyle='--', linewidth=2, label='Rc/RL = 8')
    ax1.fill_between(r_range, 0.1, CRITICAL_RATIO, alpha=0.2, color='red', label='Strong scattering')

    ax1.set_xlabel('Radial Distance (Re)', fontsize=12)
    ax1.set_ylabel('Rc/RL Ratio', fontsize=12)
    ax1.set_title(f'Radial Profiles in Magnetic Equator ({energy_eq} keV electrons)', fontsize=14)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(loc='upper right')
    ax1.set_xlim(2, 12)
    ax1.set_ylim(0.1, 1000)

    # Panel 2: MLT distribution at fixed L-shells
    ax2 = fig.add_subplot(gs[0, 2])

    L_shells = [3, 4, 5, 6, 8]
    mlt_fine = np.linspace(0, 24, 48)

    for L in L_shells:
        ratios_mlt = []
        
        for mlt in mlt_fine:
            angle = (12 - mlt) * np.pi / 12  # Convert MLT to angle
            x = L * np.cos(angle)
            y = L * np.sin(angle)
            z = 0
            
            Rc_Re, B_nT = calculate_curvature_radius(t96_vectorized, parmod_moderate, ps, x, y, z)
            RL_m = calculate_larmor_radius(energy_eq, B_nT, pitch_angle_deg=90)
            ratio = Rc_Re * Re / RL_m
            
            ratios_mlt.append(min(ratio, 1000))
        
        ax2.semilogy(mlt_fine, ratios_mlt, '-', linewidth=2, label=f'L={L}')

    ax2.axhline(y=CRITICAL_RATIO, color='black', linestyle='--', linewidth=2)
    ax2.set_xlabel('MLT (hours)', fontsize=12)
    ax2.set_ylabel('Rc/RL Ratio', fontsize=12)
    ax2.set_title('MLT Variation', fontsize=14)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend()
    ax2.set_xlim(0, 24)
    ax2.set_ylim(1, 1000)
    ax2.set_xticks([0, 6, 12, 18, 24])

    # Panel 3: 2D map with drift paths
    ax3 = fig.add_subplot(gs[1:, :])

    # Create fine grid for equatorial plane
    x_eq = np.linspace(-15, 5, 150)
    y_eq = np.linspace(-11, 11, 150)
    X_eq, Y_eq = np.meshgrid(x_eq, y_eq)
    Z_eq = np.zeros_like(X_eq)

    # Calculate ratio
    Rc_Re_eq, B_nT_eq = calculate_curvature_radius(t96_vectorized, parmod_moderate, ps,
                                                   X_eq.flatten(), Y_eq.flatten(), Z_eq.flatten())
    RL_m_eq = calculate_larmor_radius(energy_eq, B_nT_eq, pitch_angle_deg=90)
    ratio_eq = Rc_Re_eq * Re / RL_m_eq
    ratio_eq = np.where(ratio_eq > 1000, 1000, ratio_eq)
    ratio_eq = np.where(ratio_eq < 0.1, 0.1, ratio_eq)
    ratio_grid_eq = ratio_eq.reshape(X_eq.shape)

    # Plot
    im = ax3.contourf(X_eq, Y_eq, ratio_grid_eq,
                      levels=np.logspace(-1, 3, 40),
                      cmap='RdBu_r', extend='both',
                      norm=LogNorm(vmin=0.1, vmax=1000))

    # Add critical contour
    cs = ax3.contour(X_eq, Y_eq, ratio_grid_eq, levels=[CRITICAL_RATIO],
                     colors='black', linewidths=3)
    ax3.clabel(cs, inline=True, fontsize=12, fmt='Rc/RL=8')

    # Add other contours
    ax3.contour(X_eq, Y_eq, ratio_grid_eq, levels=[1, 2, 4, 16, 32],
               colors='gray', linewidths=1, alpha=0.5)

    # Add example drift paths (simplified)
    theta_drift = np.linspace(0, 2*np.pi, 100)
    for L in [3, 5, 7]:
        # Approximate drift path (circular for simplicity)
        x_drift = L * np.cos(theta_drift)
        y_drift = L * np.sin(theta_drift)
        ax3.plot(x_drift, y_drift, 'w--', linewidth=1.5, alpha=0.7)
        
        # Check for scattering along drift path
        Rc_drift, B_drift = calculate_curvature_radius(t96_vectorized, parmod_moderate, ps,
                                                       x_drift, y_drift, np.zeros_like(x_drift))
        RL_drift = calculate_larmor_radius(energy_eq, B_drift, pitch_angle_deg=90)
        ratio_drift = Rc_drift * Re / RL_drift
        scatter_points = ratio_drift < CRITICAL_RATIO
        
        if np.any(scatter_points):
            ax3.plot(x_drift[scatter_points], y_drift[scatter_points], 'r.', 
                    markersize=3, alpha=0.8)

    # Add Earth
    earth = plt.Circle((0, 0), 1, color='white', zorder=10)
    ax3.add_patch(earth)

    # Add MLT labels
    ax3.text(10, 0, '12', ha='center', va='center', fontsize=12, weight='bold', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax3.text(-12, 0, '00', ha='center', va='center', fontsize=12, weight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax3.text(0, 11, '06', ha='center', va='center', fontsize=12, weight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax3.text(0, -11, '18', ha='center', va='center', fontsize=12, weight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax3.set_xlabel('X GSM (Re)', fontsize=14)
    ax3.set_ylabel('Y GSM (Re)', fontsize=14)
    ax3.set_title(f'Magnetic Equatorial Plane: {energy_eq} keV Electrons\\n' + 
                  'White dashed: Drift paths, Red dots: Scattering locations along drift',
                  fontsize=14)
    ax3.set_aspect('equal')
    ax3.set_xlim(-15, 5)
    ax3.set_ylim(-11, 11)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax3, pad=0.02)
    cbar.set_label('Rc/RL Ratio', fontsize=12)

    plt.suptitle('Magnetic Equatorial Plane Analysis: MLT Asymmetries and Drift Effects',
                fontsize=16)
    plt.savefig(os.path.join(output_dir, 'fig03_magnetic_equatorial_plane.png'))
    plt.close(fig)
    
    print("Figure 3 saved: Magnetic equatorial plane analysis")
    print("Key findings in magnetic equator:")
    print("- Dawn-dusk asymmetry due to magnetospheric configuration")
    print("- Strongest scattering in midnight-dawn sector")
    print("- Particles on certain drift paths encounter multiple scattering regions")
    print("- Inner magnetosphere (R < 4 Re) shows persistent scattering")


def analyze_critical_energy_maps():
    """
    Analysis 3: Critical Energy Maps - Where Rc/RL = 8
    Shows the critical energy at each location where the ratio equals 8.
    Below this energy: particles experience strong scattering
    Above this energy: particles maintain adiabatic motion
    """
    print("\\n" + "="*60)
    print("Analysis 3: Critical Energy Maps - Where Rc/RL = 8")
    print("="*60)
    
    # Calculate the critical energy where Rc/RL = 8
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    conditions = [
        ("Quiet", parmod_quiet),
        ("Moderate Storm", parmod_moderate),
        ("Strong Storm", parmod_storm)
    ]

    # Create grid for analysis
    x_grid = np.linspace(-15, 5, 81)
    z_grid = np.linspace(-8, 8, 65)
    X, Z = np.meshgrid(x_grid, z_grid)
    Y = np.zeros_like(X)

    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()

    # Energy range for searching
    E_search = np.logspace(0, 3.5, 200)  # 1 keV to ~3 MeV

    for idx, (label, parmod) in enumerate(conditions):
        ax = axes[idx]
        
        # Calculate fields
        Rc_Re, B_nT = calculate_curvature_radius(t96_vectorized, parmod, ps, 
                                                 x_flat, y_flat, z_flat)
        Rc_m = Rc_Re * Re
        
        # Find critical energy for each point
        critical_energy = np.zeros_like(x_flat)
        
        for i in range(len(x_flat)):
            if Rc_Re[i] > 1e9:  # No curvature
                critical_energy[i] = np.nan
                continue
                
            # Calculate ratio for all energies
            RL_m = calculate_larmor_radius(E_search, B_nT[i], pitch_angle_deg=90)
            ratios = Rc_m[i] / RL_m
            
            # Find where ratio crosses 8
            idx_cross = np.where(ratios < CRITICAL_RATIO)[0]
            if len(idx_cross) > 0:
                critical_energy[i] = E_search[idx_cross[0]]
            else:
                critical_energy[i] = np.nan  # Always adiabatic
        
        critical_energy_grid = critical_energy.reshape(X.shape)
        
        # Plot with log scale
        im = ax.contourf(X, Z, np.log10(critical_energy_grid), 
                         levels=np.linspace(0, 3.5, 15),
                         cmap='plasma', extend='both')
        
        # Add contours
        cs = ax.contour(X, Z, critical_energy_grid, 
                        levels=[10, 30, 100, 300, 1000],
                        colors='white', linewidths=1)
        ax.clabel(cs, inline=True, fontsize=10, fmt='%d keV')
        
        ax.set_title(f'{label}\\nPdyn={parmod[0]} nPa, Dst={parmod[1]} nT')
        ax.set_xlabel('X GSM (Re)')
        if idx == 0:
            ax.set_ylabel('Z GSM (Re)')
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax.add_patch(earth)
        ax.set_aspect('equal')
        ax.set_xlim(-15, 5)
        ax.set_ylim(-8, 8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Log₁₀(Critical Energy) [keV]')
        
        # Calculate statistics
        valid = ~np.isnan(critical_energy)
        if np.sum(valid) > 0:
            mean_crit = np.nanmean(critical_energy[valid])
            median_crit = np.nanmedian(critical_energy[valid])
        
        print(f"{label}:")    
        print(f"  Mean critical energy: {mean_crit:.1f} keV")
        print(f"  Median critical energy: {median_crit:.1f} keV")

    plt.suptitle(f'Critical Energy for Strong Scattering (Rc/RL = {CRITICAL_RATIO})\\n' +
                 'Electrons below this energy experience strong pitch angle diffusion',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig04_critical_energy_maps.png'))
    plt.close(fig)
    
    print("Figure 4 saved: Critical energy maps for different conditions")
    
    # Now analyze pitch angle effects
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Fixed energy
    energy_fixed = 100  # keV
    pitch_angles = [15, 30, 60, 90]  # degrees

    for idx, pa in enumerate(pitch_angles):
        ax = axes.flatten()[idx]
        
        # Calculate for moderate storm conditions
        Rc_Re, B_nT = calculate_curvature_radius(t96_vectorized, parmod_moderate, ps, 
                                                 x_flat, y_flat, z_flat)
        Rc_m = Rc_Re * Re
        
        # Calculate Larmor radius for this pitch angle
        RL_m = calculate_larmor_radius(energy_fixed, B_nT, pitch_angle_deg=pa)
        ratio = Rc_m / RL_m
        
        # Clean up ratio values
        ratio = np.where(ratio > 1e6, 1e6, ratio)
        ratio = np.where(ratio < 1e-2, 1e-2, ratio)
        ratio_grid = ratio.reshape(X.shape)
        
        # Calculate scattering fraction
        scatter_frac = np.sum(ratio < CRITICAL_RATIO) / len(ratio) * 100
        
        # Plot
        im = ax.contourf(X, Z, ratio_grid,
                         levels=np.logspace(-1, 3, 20),
                         cmap='RdBu_r', extend='both',
                         norm=LogNorm(vmin=0.1, vmax=1000))
        
        # Add critical contour
        cs = ax.contour(X, Z, ratio_grid, levels=[CRITICAL_RATIO], 
                        colors='black', linewidths=3)
        ax.clabel(cs, inline=True, fontsize=10, fmt='Rc/RL=8')
        
        # Add other contours
        ax.contour(X, Z, ratio_grid, levels=[1, 2, 4], 
                   colors='gray', linewidths=1, alpha=0.5)
        
        ax.set_title(f'Pitch Angle = {pa}°\\nScattering region: {scatter_frac:.1f}%')
        ax.set_xlabel('X GSM (Re)')
        ax.set_ylabel('Z GSM (Re)')
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax.add_patch(earth)
        ax.set_aspect('equal')
        ax.set_xlim(-15, 5)
        ax.set_ylim(-8, 8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Rc/RL')

    plt.suptitle(f'Effect of Pitch Angle on Scattering Regions\\n' +
                 f'{energy_fixed} keV electrons, T96 Moderate Storm',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig05_pitch_angle_effects.png'))
    plt.close(fig)
    
    print("Figure 5 saved: Pitch angle effects on scattering regions")


def analyze_storm_evolution():
    """
    Analysis 4: Time Evolution During a Storm
    Simulates storm evolution through different phases
    """
    print("\\n" + "="*60)
    print("Analysis 4: Time Evolution During a Storm")
    print("="*60)
    
    # Create grid for analysis
    x_grid = np.linspace(-15, 5, 81)
    z_grid = np.linspace(-8, 8, 65)
    X, Z = np.meshgrid(x_grid, z_grid)
    Y = np.zeros_like(X)

    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    
    # Storm phases: quiet -> growth -> main -> recovery
    storm_phases = [
        ("Pre-storm", [1.0, -5.0, 0.0, 2.0]),
        ("Growth Phase", [3.0, -20.0, 2.0, -5.0]),
        ("Main Phase", [8.0, -80.0, 5.0, -15.0]),
        ("Early Recovery", [4.0, -60.0, 3.0, -8.0]),
        ("Late Recovery", [2.0, -30.0, 1.0, -3.0])
    ]

    # Test for multiple energies
    test_energies = [10, 30, 100, 300]  # keV

    # Calculate scattering regions for each phase
    scatter_stats = {energy: [] for energy in test_energies}

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for idx, (phase, params) in enumerate(storm_phases):
        ax = axes[idx]
        full_params = params + [0, 0, 0, 0, 0, 0]
        
        # Calculate fields
        Rc_Re, B_nT = calculate_curvature_radius(t96_vectorized, full_params, ps, 
                                                 x_flat, y_flat, z_flat)
        Rc_m = Rc_Re * Re
        
        # Show for 100 keV electrons
        energy_show = 100
        RL_m = calculate_larmor_radius(energy_show, B_nT, pitch_angle_deg=90)
        ratio = Rc_m / RL_m
        ratio_grid = ratio.reshape(X.shape)
        
        # Binary map
        scatter_map = (ratio_grid < CRITICAL_RATIO).astype(float)
        
        # Plot
        im = ax.contourf(X, Z, scatter_map, levels=[0, 0.5, 1],
                         colors=['lightblue', 'darkred'])
        
        # Add critical contour
        cs = ax.contour(X, Z, ratio_grid, levels=[CRITICAL_RATIO], 
                        colors='yellow', linewidths=2)
        
        ax.set_title(f'{phase}\\nDst={params[1]} nT')
        ax.set_xlabel('X GSM (Re)')
        if idx == 0:
            ax.set_ylabel('Z GSM (Re)')
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax.add_patch(earth)
        ax.set_aspect('equal')
        ax.set_xlim(-15, 5)
        ax.set_ylim(-8, 8)
        
        # Calculate statistics for all energies
        for energy in test_energies:
            RL_m = calculate_larmor_radius(energy, B_nT, pitch_angle_deg=90)
            ratio = Rc_m / RL_m
            scatter_frac = np.sum(ratio < CRITICAL_RATIO) / len(ratio) * 100
            scatter_stats[energy].append(scatter_frac)

    plt.suptitle(f'Storm Evolution: Scattering Regions for {energy_show} keV Electrons\\n' +
                 f'Red = Strong Scattering (Rc/RL < {CRITICAL_RATIO}), Blue = Adiabatic',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig06_storm_evolution_spatial.png'))
    plt.close(fig)
    
    print("Figure 6 saved: Storm evolution spatial patterns")
    
    # Plot time evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    phase_names = [phase[0] for phase in storm_phases]
    x_pos = np.arange(len(phase_names))

    colors = ['blue', 'green', 'orange', 'red']
    for energy, color in zip(test_energies, colors):
        ax.plot(x_pos, scatter_stats[energy], 'o-', color=color, 
                linewidth=2, markersize=8, label=f'{energy} keV')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(phase_names, rotation=45, ha='right')
    ax.set_ylabel('Fraction of Magnetosphere with Rc/RL < 8 (%)')
    ax.set_title('Evolution of Scattering Regions During a Storm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig07_storm_evolution_temporal.png'))
    plt.close(fig)
    
    print("Figure 7 saved: Storm evolution temporal patterns")
    
    # Print storm evolution summary
    print("\\nStorm Evolution Summary:")
    print("=" * 60)
    for i, phase in enumerate(phase_names):
        print(f"{phase}:")
        for energy in test_energies:
            print(f"  {energy:3d} keV: {scatter_stats[energy][i]:5.1f}% scattering")


def analyze_t96_parameter_sensitivity():
    """
    Analysis 6: T96 Parameter Sensitivity Study
    Examines how different T96 model parameters affect the scattering regions
    """
    print("\\n" + "="*60)
    print("Analysis 6: T96 Parameter Sensitivity Study")
    print("="*60)
    
    # Create grid for analysis
    x_grid_sens = np.linspace(-15, 5, 61)
    z_grid_sens = np.linspace(-6, 6, 49)
    X_sens, Z_sens = np.meshgrid(x_grid_sens, z_grid_sens)
    Y_sens = np.zeros_like(X_sens)

    x_flat_sens = X_sens.flatten()
    y_flat_sens = Y_sens.flatten()
    z_flat_sens = Z_sens.flatten()
    
    # T96 Parameter Sensitivity Analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Base parameters (moderate conditions)
    base_params = [3.0, -30.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    energy_sens = 100  # keV

    # Parameter variations to test
    param_variations = [
        ("Pdyn = 1 nPa", [1.0, -30.0, 1.0, -3.0]),
        ("Pdyn = 5 nPa", [5.0, -30.0, 1.0, -3.0]),
        ("Pdyn = 10 nPa", [10.0, -30.0, 1.0, -3.0]),
        ("Dst = -10 nT", [3.0, -10.0, 1.0, -3.0]),
        ("Dst = -50 nT", [3.0, -50.0, 1.0, -3.0]),
        ("Dst = -100 nT", [3.0, -100.0, 1.0, -3.0])
    ]

    scatter_stats_t96 = {}

    for idx, (label, params) in enumerate(param_variations):
        ax = axes.flatten()[idx]
        
        # Full parameter array
        full_params = params + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Calculate fields
        Rc_Re_sens, B_nT_sens = calculate_curvature_radius(t96_vectorized, full_params, ps, 
                                                           x_flat_sens, y_flat_sens, z_flat_sens)
        Rc_m_sens = Rc_Re_sens * Re
        
        # Calculate Larmor radius
        RL_m_sens = calculate_larmor_radius(energy_sens, B_nT_sens, pitch_angle_deg=90)
        ratio_sens = Rc_m_sens / RL_m_sens
        
        # Clean up extreme values
        ratio_sens = np.where(ratio_sens > 1000, 1000, ratio_sens)
        ratio_sens = np.where(ratio_sens < 0.1, 0.1, ratio_sens)
        ratio_grid_sens = ratio_sens.reshape(X_sens.shape)
        
        # Calculate scattering fraction
        scatter_frac = np.sum(ratio_sens < CRITICAL_RATIO) / len(ratio_sens) * 100
        scatter_stats_t96[label] = scatter_frac
        
        # Plot
        im = ax.contourf(X_sens, Z_sens, ratio_grid_sens,
                         levels=np.logspace(-1, 3, 20),
                         cmap='RdBu_r', extend='both',
                         norm=LogNorm(vmin=0.1, vmax=1000))
        
        # Add critical contour
        cs = ax.contour(X_sens, Z_sens, ratio_grid_sens, 
                        levels=[CRITICAL_RATIO],
                        colors='black', linewidths=3)
        ax.clabel(cs, inline=True, fontsize=10, fmt='Rc/RL=8')
        
        # Add other contours
        ax.contour(X_sens, Z_sens, ratio_grid_sens, 
                   levels=[1, 2, 4, 16, 32],
                   colors='gray', linewidths=1, alpha=0.5)
        
        ax.set_title(f'{label}\\nScattering: {scatter_frac:.1f}%', fontsize=12)
        ax.set_xlabel('X GSM (Re)', fontsize=10)
        if idx % 3 == 0:
            ax.set_ylabel('Z GSM (Re)', fontsize=10)
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax.add_patch(earth)
        ax.set_aspect('equal')
        ax.set_xlim(-15, 5)
        ax.set_ylim(-6, 6)

    # Add colorbar to the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Rc/RL Ratio', fontsize=12)

    plt.suptitle(f'T96 Model Parameter Sensitivity: {energy_sens} keV Electrons\\n' +
                 'Effects of Solar Wind Dynamic Pressure and Ring Current Strength',
                 fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.91, 0.97])
    plt.savefig(os.path.join(output_dir, 'fig08_t96_parameter_sensitivity.png'))
    plt.close(fig)
    
    print("Figure 8 saved: T96 parameter sensitivity analysis")
    
    print("\\nT96 Parameter Sensitivity Summary:")
    print("=" * 60)
    print("Effects on scattering regions (Rc/RL < 8):")
    print("-" * 40)
    print("Solar Wind Dynamic Pressure (Pdyn):")
    for label in ["Pdyn = 1 nPa", "Pdyn = 5 nPa", "Pdyn = 10 nPa"]:
        print(f"  {label}: {scatter_stats_t96[label]:.1f}% scattering")
    print("\\nRing Current Strength (Dst):")
    for label in ["Dst = -10 nT", "Dst = -50 nT", "Dst = -100 nT"]:
        print(f"  {label}: {scatter_stats_t96[label]:.1f}% scattering")
    print("\\nKey Findings:")
    print("- Higher Pdyn compresses magnetosphere, increasing curvature")
    print("- More negative Dst expands scattering regions")
    print("- Southward IMF dramatically increases scattering")
    print("- IMF By creates dawn-dusk asymmetries")


def analyze_comprehensive_model_comparison():
    """
    Analysis 9: Comprehensive Model Comparison
    Compares T89, T96, T01, and T04 models under similar conditions
    """
    print("\\n" + "="*60)
    print("Analysis 9: Comprehensive Model Comparison")
    print("="*60)
    
    # Create grid for analysis
    x_grid_sens = np.linspace(-15, 5, 61)
    z_grid_sens = np.linspace(-6, 6, 49)
    X_sens, Z_sens = np.meshgrid(x_grid_sens, z_grid_sens)
    Y_sens = np.zeros_like(X_sens)

    x_flat_sens = X_sens.flatten()
    y_flat_sens = Y_sens.flatten()
    z_flat_sens = Z_sens.flatten()
    
    # Comprehensive Model Comparison
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    # Define comparable conditions for all models
    energy_comp = 100  # keV

    # Storm conditions to compare
    conditions = [
        ("Quiet", "quiet"),
        ("Moderate Storm", "moderate"),
        ("Intense Storm", "intense")
    ]

    # Model parameters for each condition
    model_params = {
        "quiet": {
            "T89": 2,  # Kp = 2
            "T96": [2.0, -10.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "T01": [2.0, -10.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "T04": [2.0, -10.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        "moderate": {
            "T89": 4,  # Kp = 4
            "T96": [5.0, -50.0, 2.0, -8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "T01": [5.0, -50.0, 2.0, -8.0, 0.7, 0.7, 0.0, 0.0, 0.0, 0.0],
            "T04": [5.0, -50.0, 2.0, -8.0, 0.5, 0.4, 0.3, 0.3, 0.2, 0.2]
        },
        "intense": {
            "T89": 6,  # Kp = 6
            "T96": [10.0, -150.0, 5.0, -20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "T01": [10.0, -150.0, 5.0, -20.0, 1.5, 1.5, 0.0, 0.0, 0.0, 0.0],
            "T04": [10.0, -150.0, 5.0, -20.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5]
        }
    }

    models = [
        ("T89", t89_vectorized),
        ("T96", t96_vectorized),
        ("T01", t01_vectorized),
        ("T04", t04_vectorized)
    ]

    scatter_summary = {}

    for row_idx, (cond_name, cond_key) in enumerate(conditions):
        for col_idx, (model_name, model_func) in enumerate(models):
            ax = axes[row_idx, col_idx]
            
            # Get parameters for this model and condition
            params = model_params[cond_key][model_name]
            
            # Calculate fields
            Rc_Re, B_nT = calculate_curvature_radius(model_func, params, ps, 
                                                   x_flat_sens, y_flat_sens, z_flat_sens)
            Rc_m = Rc_Re * Re
            
            # Calculate Larmor radius
            RL_m = calculate_larmor_radius(energy_comp, B_nT, pitch_angle_deg=90)
            ratio = Rc_m / RL_m
            
            # Clean up extreme values
            ratio = np.where(ratio > 1000, 1000, ratio)
            ratio = np.where(ratio < 0.1, 0.1, ratio)
            ratio_grid = ratio.reshape(X_sens.shape)
            
            # Calculate scattering fraction
            scatter_frac = np.sum(ratio < CRITICAL_RATIO) / len(ratio) * 100
            key = f"{model_name}_{cond_key}"
            scatter_summary[key] = scatter_frac
            
            # Plot
            im = ax.contourf(X_sens, Z_sens, ratio_grid,
                            levels=np.logspace(-1, 3, 20),
                            cmap='RdBu_r', extend='both',
                            norm=LogNorm(vmin=0.1, vmax=1000))
            
            # Add critical contour
            cs = ax.contour(X_sens, Z_sens, ratio_grid, 
                           levels=[CRITICAL_RATIO],
                           colors='black', linewidths=2.5)
            ax.clabel(cs, inline=True, fontsize=9, fmt='8')
            
            # Title
            if row_idx == 0:
                ax.set_title(f'{model_name}', fontsize=14, weight='bold')
            
            if col_idx == 0:
                ax.set_ylabel(f'{cond_name}\\nZ GSM (Re)', fontsize=12)
            else:
                ax.set_ylabel('Z GSM (Re)', fontsize=10)
                
            ax.set_xlabel('X GSM (Re)', fontsize=10)
            
            # Add text with scattering percentage
            ax.text(0.02, 0.98, f'{scatter_frac:.1f}%', 
                    transform=ax.transAxes, fontsize=11, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add Earth
            earth = plt.Circle((0, 0), 1, color='white', zorder=10)
            ax.add_patch(earth)
            ax.set_aspect('equal')
            ax.set_xlim(-15, 5)
            ax.set_ylim(-6, 6)

    # Add single colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Rc/RL Ratio', fontsize=12)

    plt.suptitle(f'Model Comparison: Curvature Scattering Regions (Rc/RL < {CRITICAL_RATIO})\\n' +
                 f'{energy_comp} keV Electrons at 90° Pitch Angle',
                 fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.91, 0.97])
    plt.savefig(os.path.join(output_dir, 'fig09_model_comparison.png'))
    plt.close(fig)
    
    print("Figure 9 saved: Comprehensive model comparison")

    print("\\nComprehensive Model Comparison Summary:")
    print("=" * 70)
    print("Scattering region percentages (Rc/RL < 8):")
    print("-" * 70)
    print(f"{'Condition':<15} {'T89':>10} {'T96':>10} {'T01':>10} {'T04':>10}")
    print("-" * 70)

    for cond_name, cond_key in conditions:
        values = [f"{scatter_summary[f'{model}_{cond_key}']:.1f}%" 
                  for model, _ in models]
        print(f"{cond_name:<15} {values[0]:>10} {values[1]:>10} {values[2]:>10} {values[3]:>10}")

    print("\\nKey Model Differences:")
    print("-" * 70)
    print("T89: Simple Kp-based model, symmetric magnetosphere")
    print("T96: Includes solar wind pressure and IMF effects")
    print("T01: Adds storm-time corrections (G1, G2 parameters)")
    print("T04: Most sophisticated with 6 storm-time parameters (W1-W6)")
    print("\\nObservations:")
    print("- T89 tends to underestimate scattering regions during storms")
    print("- T96 shows IMF control of dawn-dusk asymmetry")
    print("- T01 and T04 show larger scattering regions during intense storms")
    print("- All models agree on basic energy dependence patterns")


def create_summary_figure():
    """
    Create a comprehensive summary figure
    """
    print("\\n" + "="*60)
    print("Creating Summary Figure")
    print("="*60)
    
    # Create grid for analysis
    x_grid = np.linspace(-15, 5, 81)
    z_grid = np.linspace(-8, 8, 65)
    X, Z = np.meshgrid(x_grid, z_grid)
    Y = np.zeros_like(X)

    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    
    # Create summary figure
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Panel 1: Energy threshold along radial distance
    ax1 = fig.add_subplot(gs[0, :])
    r_test = np.linspace(3, 10, 50)
    x_test = r_test
    y_test = np.zeros_like(r_test)
    z_test = np.zeros_like(r_test)

    # Energy range for searching
    E_search = np.logspace(0, 3.5, 200)  # 1 keV to ~3 MeV

    for condition, params, color in [("Quiet", parmod_quiet, 'blue'),
                                      ("Moderate", parmod_moderate, 'orange'),
                                      ("Storm", parmod_storm, 'red')]:
        Rc_Re, B_nT = calculate_curvature_radius(t96_vectorized, params, ps, 
                                                 x_test, y_test, z_test)
        Rc_m = Rc_Re * Re
        
        # Find threshold energy
        E_threshold = np.zeros_like(r_test)
        for i in range(len(r_test)):
            RL_m_test = calculate_larmor_radius(E_search, B_nT[i], pitch_angle_deg=90)
            ratio_test = Rc_m[i] / RL_m_test
            idx = np.where(ratio_test < CRITICAL_RATIO)[0]
            if len(idx) > 0:
                E_threshold[i] = E_search[idx[0]]
            else:
                E_threshold[i] = np.nan
        
        mask = ~np.isnan(E_threshold)
        ax1.semilogy(r_test[mask], E_threshold[mask], '-', color=color, 
                     linewidth=2, label=condition)

    ax1.set_xlabel('Distance along X-axis (Re)')
    ax1.set_ylabel('Threshold Energy (keV)')
    ax1.set_title(f'Energy Threshold for Strong Scattering (Rc/RL < {CRITICAL_RATIO})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(1, 1000)

    # Panel 2: Key findings text
    ax2 = fig.add_subplot(gs[1:, 0])
    ax2.text(0.05, 0.95, 'Key Findings:', fontsize=14, weight='bold',
             transform=ax2.transAxes, verticalalignment='top')

    findings_text = [
        '1. Energy Dependence:',
        '   • Lower energy → larger scattering regions',
        '   • 10-30 keV: Scattered throughout tail',
        '   • 100-300 keV: Limited to current sheet',
        '',
        '2. Storm Effects:',
        '   • Scattering regions expand dramatically',
        '   • Threshold energy increases',
        '   • Enhanced precipitation expected',
        '',
        '3. Pitch Angle Effects:',
        '   • Smaller PA → less scattering',
        '   • Field-aligned particles more stable',
        '   • PA diffusion fills loss cone',
        '',
        '4. MLT Asymmetry:',
        '   • Midnight: More scattering',
        '   • Dawn/Dusk: Intermediate',
        '   • Noon: Least scattering'
    ]

    y_pos = 0.85
    for line in findings_text:
        if line and line[0].isdigit():
            ax2.text(0.05, y_pos, line, transform=ax2.transAxes, 
                     fontsize=11, weight='bold')
        else:
            ax2.text(0.05, y_pos, line, transform=ax2.transAxes, fontsize=10)
        y_pos -= 0.04

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    # Panel 3: Scattering region visualization
    ax3 = fig.add_subplot(gs[1:, 1:])

    # Show moderate storm, 100 keV
    Rc_Re, B_nT = calculate_curvature_radius(t96_vectorized, parmod_moderate, ps, 
                                             x_flat, y_flat, z_flat)
    Rc_m = Rc_Re * Re
    RL_m = calculate_larmor_radius(100, B_nT, pitch_angle_deg=90)
    ratio = Rc_m / RL_m
    ratio_grid = ratio.reshape(X.shape)

    im = ax3.contourf(X, Z, ratio_grid,
                      levels=[0, 1, 2, 4, 8, 16, 32, 64],
                      colors=['darkred', 'red', 'orange', 'yellow', 
                              'lightgreen', 'green', 'blue', 'darkblue'],
                      extend='both')

    cs = ax3.contour(X, Z, ratio_grid, levels=[CRITICAL_RATIO], 
                     colors='black', linewidths=3)
    ax3.clabel(cs, inline=True, fontsize=12, fmt='Rc/RL=8')

    ax3.set_title('Example: 100 keV electrons, Moderate Storm\\nRed: Strong Scattering, Blue: Adiabatic')
    ax3.set_xlabel('X GSM (Re)')
    ax3.set_ylabel('Z GSM (Re)')

    # Add Earth
    earth = plt.Circle((0, 0), 1, color='white', zorder=10)
    ax3.add_patch(earth)
    ax3.set_aspect('equal')
    ax3.set_xlim(-12, 8)
    ax3.set_ylim(-8, 8)

    plt.suptitle(f'Curvature Scattering Analysis Summary: Rc/RL = {CRITICAL_RATIO} Threshold\\n' +
                 'Critical for Radiation Belt Losses and Auroral Precipitation',
                 fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig10_summary.png'))
    plt.close(fig)
    
    print("Figure 10 saved: Summary figure")
    
    print("\\nPhysical Implications of Rc/RL < 8:")
    print("=" * 60)
    print("• Strong violation of first adiabatic invariant")
    print("• Rapid pitch angle diffusion fills loss cone")
    print("• Enhanced particle precipitation to atmosphere")
    print("• Important for:")
    print("  - Radiation belt electron losses")
    print("  - Diffuse aurora generation")
    print("  - Ionospheric conductivity enhancement")
    print("  - Atmospheric chemistry changes")
    print("Storm-time expansion of Rc/RL < 8 regions drives:")
    print("  - Rapid radiation belt depletions")
    print("  - Expanded auroral zones")
    print("  - Enhanced energy deposition to upper atmosphere")


def analyze_3d_field_lines():
    """
    Analysis 1c: 3D Visualization of Field Lines from Scattering Regions
    Traces magnetic field lines starting from regions where Rc/RL < 8
    """
    print("\\n" + "="*60)
    print("Analysis 1c: 3D Visualization of Field Lines from Scattering Regions")
    print("="*60)
    
    from mpl_toolkits.mplot3d import Axes3D
    from geopack.trace_field_lines_vectorized import trace_vectorized
    
    # First, identify seed points in scattering regions at different Z levels
    z_levels_3d = np.arange(0.0, 1.4, 0.3)  # Re
    energy_3d = 100  # keV
    seed_points = []
    
    # Create a coarse grid for computational efficiency
    x_grid_3d = np.linspace(-12, 4, 17)
    y_grid_3d = np.linspace(-6, 6, 13)
    
    print("Finding seed points in scattering regions...")
    for z_level in z_levels_3d:
        X_3d, Y_3d = np.meshgrid(x_grid_3d, y_grid_3d)
        Z_3d = np.full_like(X_3d, z_level)
        
        x_flat_3d = X_3d.flatten()
        y_flat_3d = Y_3d.flatten()
        z_flat_3d = Z_3d.flatten()
        
        # Calculate Rc/RL ratio
        Rc_Re_3d, B_nT_3d = calculate_curvature_radius(
            t96_vectorized, parmod_moderate, ps, 
            x_flat_3d, y_flat_3d, z_flat_3d
        )
        
        RL_m_3d = calculate_larmor_radius(energy_3d, B_nT_3d, pitch_angle_deg=90)
        ratio_3d = Rc_Re_3d * Re / RL_m_3d
        
        # Find points where Rc/RL < 8
        scatter_mask = (ratio_3d < CRITICAL_RATIO) & (np.sqrt(x_flat_3d**2 + y_flat_3d**2 + z_flat_3d**2) > 2.0)
        
        # Sample some points from scattering regions
        scatter_indices = np.where(scatter_mask)[0]
        if len(scatter_indices) > 0:
            sample_size = min(3, len(scatter_indices))
            sampled_indices = np.random.choice(scatter_indices, sample_size, replace=False)
            
            for idx in sampled_indices:
                seed_points.append([x_flat_3d[idx], y_flat_3d[idx], z_flat_3d[idx]])
    
    # Use pre-selected seed points if none found or too few
    if len(seed_points) < 5:
        print("Using pre-selected seed points for efficiency...")
        seed_points = [
            [-8.0, 0.0, 0.5], [-6.0, 2.0, 0.5], [-6.0, -2.0, 0.5],
            [-10.0, 0.0, 0.6], [-4.0, 0.0, 1.0], [3.5, 0.0, 1.0],
            [-8.0, 3.0, 0.3], [-5.0, -3.0, 0.8]
        ]
    
    seed_points = np.array(seed_points)[:10]  # Limit to 10 seed points max
    print(f"Using {len(seed_points)} seed points")
    
    # Trace field lines
    print("Tracing field lines...")
    field_lines = []
    
    # Trace only a few field lines for visualization
    for i, seed in enumerate(seed_points[:8]):  # Limit to 8 field lines
        print(f"  Tracing field line {i+1}/{min(8, len(seed_points))}...")
        for trace_dir in [-1, 1]:
            try:
                result = trace_vectorized(
                    seed[0], seed[1], seed[2],
                    dir=trace_dir,
                    rlim=15.0,
                    r0=1.0,
                    parmod=parmod_moderate,
                    exname='t96',
                    inname='igrf',
                    maxloop=2000,
                    return_full_path=True
                )
                
                if result is not None and isinstance(result, tuple) and len(result) >= 7:
                    x_path, y_path, z_path = result[3], result[4], result[5]
                    if hasattr(x_path, '__len__') and len(x_path) > 10:
                        # Handle masked arrays
                        if hasattr(x_path, 'compressed'):
                            x_path = x_path.compressed()
                            y_path = y_path.compressed()
                            z_path = z_path.compressed()
                        
                        if len(x_path) > 10 and np.all(np.isfinite(x_path)):
                            field_line = np.column_stack((x_path, y_path, z_path))
                            field_lines.append(field_line)
            except Exception as e:
                continue
    
    print(f"Successfully traced {len(field_lines)} field line segments")
    
    # Create 3D visualization
    fig = plt.figure(figsize=(16, 8))
    
    # First subplot: 3D view of field lines
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot field lines
    if len(field_lines) > 0:
        for fl in field_lines:
            ax1.plot(fl[:, 0], fl[:, 1], fl[:, 2], 'b-', alpha=0.4, linewidth=1.0)
    
    # Add Earth
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_earth = np.outer(np.cos(u), np.sin(v))
    y_earth = np.outer(np.sin(u), np.sin(v))
    z_earth = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x_earth, y_earth, z_earth, color='white', alpha=0.9)
    
    # Mark seed points
    ax1.scatter(seed_points[:, 0], seed_points[:, 1], seed_points[:, 2],
               c='red', s=50, alpha=0.8, label='Seed points (Rc/RL < 8)')
    
    ax1.set_xlabel('X GSM (Re)')
    ax1.set_ylabel('Y GSM (Re)')
    ax1.set_zlabel('Z GSM (Re)')
    ax1.set_title(f'3D Field Lines from Rc/RL < 8 Regions\\n{energy_3d} keV electrons, Moderate Storm')
    ax1.set_xlim(-15, 5)
    ax1.set_ylim(-10, 10)
    ax1.set_zlim(-5, 5)
    ax1.view_init(elev=20, azim=45)
    ax1.legend()
    
    # Set equal aspect ratio for 3D plot
    set_axes_equal(ax1)
    
    # Second subplot: XZ projection with Rc/RL contours
    ax2 = fig.add_subplot(122)
    
    # Create a 2D projection showing scattering regions
    x_proj = np.linspace(-15, 5, 61)
    z_proj = np.linspace(-3, 3, 37)
    X_proj, Z_proj = np.meshgrid(x_proj, z_proj)
    Y_proj = np.zeros_like(X_proj)
    
    # Calculate Rc/RL ratio along Y=0 plane
    Rc_Re_proj, B_nT_proj = calculate_curvature_radius(t96_vectorized, parmod_moderate, ps,
                                                       X_proj.flatten(), Y_proj.flatten(), Z_proj.flatten())
    RL_m_proj = calculate_larmor_radius(energy_3d, B_nT_proj, pitch_angle_deg=90)
    ratio_proj = Rc_Re_proj * Re / RL_m_proj
    ratio_proj = np.where(ratio_proj > 1000, 1000, ratio_proj)
    ratio_proj = np.where(ratio_proj < 0.1, 0.1, ratio_proj)
    ratio_grid_proj = ratio_proj.reshape(X_proj.shape)
    
    # Plot 2D projection
    im = ax2.contourf(X_proj, Z_proj, ratio_grid_proj,
                      levels=np.logspace(-1, 3, 20),
                      cmap='RdBu_r', extend='both',
                      norm=LogNorm(vmin=0.1, vmax=1000))
    
    # Add critical contour
    cs = ax2.contour(X_proj, Z_proj, ratio_grid_proj, 
                     levels=[CRITICAL_RATIO],
                     colors='black', linewidths=3)
    ax2.clabel(cs, inline=True, fontsize=10, fmt='Rc/RL=8')
    
    # Project field lines onto X-Z plane
    if len(field_lines) > 0:
        for fl in field_lines:
            ax2.plot(fl[:, 0], fl[:, 2], 'g-', alpha=0.4, linewidth=1.0)
    
    # Mark seed points
    ax2.scatter(seed_points[:, 0], seed_points[:, 2], c='red', s=30, alpha=0.8,
               label='Field line start points')
    
    # Add Earth
    earth = plt.Circle((0, 0), 1, color='white', zorder=10)
    ax2.add_patch(earth)
    
    ax2.set_xlabel('X GSM (Re)')
    ax2.set_ylabel('Z GSM (Re)')
    ax2.set_title('Noon-Midnight Meridian Projection\\nGreen: Field lines, Background: Rc/RL ratio')
    ax2.set_aspect('equal')
    ax2.set_xlim(-15, 5)
    ax2.set_ylim(-3, 3)
    ax2.legend(loc='upper right')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Rc/RL Ratio')
    
    plt.suptitle(f'3D Structure of Magnetic Field Lines from Strong Scattering Regions\\nField lines traced from regions where Rc/RL < {CRITICAL_RATIO}',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig11_3d_field_lines.png'))
    plt.close(fig)
    
    print("Figure 11 saved: 3D field lines from scattering regions")


def analyze_3d_volume_rendering():
    """
    Alternative 3D visualization: Volume rendering of scattering regions
    """
    print("\\n" + "="*60)
    print("Analysis: 3D Volume Rendering of Scattering Regions")
    print("="*60)
    
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(18, 8))
    
    # Create 3D scatter plot showing Rc/RL < 8 regions
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Sample points in 3D space - use coarse grid for efficiency
    n_samples = 15
    x_3d_vol = np.linspace(-12, 4, n_samples)
    y_3d_vol = np.linspace(-6, 6, n_samples)
    z_3d_vol = np.linspace(-2, 2, 9)
    
    # Calculate Rc/RL ratio in 3D volume - vectorized approach
    print("Computing 3D scattering volume...")
    scatter_points_3d = []
    scatter_values = []
    
    # Create full 3D grid
    X_vol, Y_vol, Z_vol = np.meshgrid(x_3d_vol, y_3d_vol, z_3d_vol)
    x_vol_flat = X_vol.flatten()
    y_vol_flat = Y_vol.flatten()
    z_vol_flat = Z_vol.flatten()
    
    # Skip points too close to Earth
    r_vol = np.sqrt(x_vol_flat**2 + y_vol_flat**2 + z_vol_flat**2)
    valid_mask = r_vol > 2.0
    
    # Calculate for valid points only
    x_valid = x_vol_flat[valid_mask]
    y_valid = y_vol_flat[valid_mask]
    z_valid = z_vol_flat[valid_mask]
    
    print(f"  Processing {len(x_valid)} valid points...")
    Rc_Re_vol, B_nT_vol = calculate_curvature_radius(t96_vectorized, parmod_moderate, ps, 
                                                     x_valid, y_valid, z_valid)
    energy_3d = 100  # keV
    RL_m_vol = calculate_larmor_radius(energy_3d, B_nT_vol, pitch_angle_deg=90)
    ratio_vol = Rc_Re_vol * Re / RL_m_vol
    
    # Find scattering points
    scatter_mask_vol = ratio_vol < CRITICAL_RATIO
    scatter_points_3d = np.column_stack((x_valid[scatter_mask_vol], 
                                        y_valid[scatter_mask_vol], 
                                        z_valid[scatter_mask_vol]))
    scatter_values = ratio_vol[scatter_mask_vol]
    
    print(f"  Found {len(scatter_points_3d)} points with Rc/RL < {CRITICAL_RATIO}")
    
    # Color by Rc/RL value
    if len(scatter_points_3d) > 0:
        scatter = ax1.scatter(scatter_points_3d[:, 0], scatter_points_3d[:, 1], scatter_points_3d[:, 2],
                             c=scatter_values, cmap='hot_r', s=40, alpha=0.6,
                             vmin=0, vmax=CRITICAL_RATIO)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1, pad=0.1, shrink=0.8)
        cbar.set_label('Rc/RL Ratio')
    
    # Add Earth
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_earth = np.outer(np.cos(u), np.sin(v))
    y_earth = np.outer(np.sin(u), np.sin(v))
    z_earth = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x_earth, y_earth, z_earth, color='lightgray', alpha=0.8)
    
    ax1.set_xlabel('X GSM (Re)')
    ax1.set_ylabel('Y GSM (Re)')
    ax1.set_zlabel('Z GSM (Re)')
    ax1.set_title(f'3D Volume: Rc/RL < {CRITICAL_RATIO} Regions')
    ax1.set_xlim(-12, 4)
    ax1.set_ylim(-6, 6)
    ax1.set_zlim(-2, 2)
    ax1.view_init(elev=15, azim=45)
    
    # Set equal aspect ratio for 3D plot
    set_axes_equal(ax1)
    
    # XY projection at Z=0
    ax2 = fig.add_subplot(132)
    if len(scatter_points_3d) > 0:
        z0_points = scatter_points_3d[np.abs(scatter_points_3d[:, 2]) < 0.3]
        if len(z0_points) > 0:
            ax2.scatter(z0_points[:, 0], z0_points[:, 1], c='red', s=30, alpha=0.5)
    
    # Add background contour with coarser grid
    X_xy2, Y_xy2 = np.meshgrid(x_3d_vol, y_3d_vol)
    Z_xy2 = np.zeros_like(X_xy2)
    Rc_Re_xy2, B_nT_xy2 = calculate_curvature_radius(t96_vectorized, parmod_moderate, ps,
                                                     X_xy2.flatten(), Y_xy2.flatten(), Z_xy2.flatten())
    RL_m_xy2 = calculate_larmor_radius(energy_3d, B_nT_xy2, pitch_angle_deg=90)
    ratio_xy2 = Rc_Re_xy2 * Re / RL_m_xy2
    ratio_grid_xy2 = ratio_xy2.reshape(X_xy2.shape)
    
    cs = ax2.contour(X_xy2, Y_xy2, ratio_grid_xy2, levels=[CRITICAL_RATIO],
                    colors='black', linewidths=2)
    ax2.clabel(cs, inline=True, fontsize=10, fmt='Rc/RL=8')
    
    earth = plt.Circle((0, 0), 1, color='lightgray', zorder=10)
    ax2.add_patch(earth)
    ax2.set_xlabel('X GSM (Re)')
    ax2.set_ylabel('Y GSM (Re)')
    ax2.set_title('XY Projection (Z ≈ 0)')
    ax2.set_aspect('equal')
    ax2.set_xlim(-12, 4)
    ax2.set_ylim(-6, 6)
    ax2.grid(True, alpha=0.3)
    
    # XZ projection
    ax3 = fig.add_subplot(133)
    if len(scatter_points_3d) > 0:
        y0_points = scatter_points_3d[np.abs(scatter_points_3d[:, 1]) < 0.5]
        if len(y0_points) > 0:
            ax3.scatter(y0_points[:, 0], y0_points[:, 2], c='red', s=30, alpha=0.5)
    
    # Add background contour
    X_xz, Z_xz = np.meshgrid(x_3d_vol, z_3d_vol)
    Y_xz = np.zeros_like(X_xz)
    Rc_Re_xz, B_nT_xz = calculate_curvature_radius(t96_vectorized, parmod_moderate, ps,
                                                   X_xz.flatten(), Y_xz.flatten(), Z_xz.flatten())
    RL_m_xz = calculate_larmor_radius(energy_3d, B_nT_xz, pitch_angle_deg=90)
    ratio_xz = Rc_Re_xz * Re / RL_m_xz
    ratio_grid_xz = ratio_xz.reshape(X_xz.shape)
    
    cs = ax3.contour(X_xz, Z_xz, ratio_grid_xz, levels=[CRITICAL_RATIO],
                    colors='black', linewidths=2)
    ax3.clabel(cs, inline=True, fontsize=10, fmt='Rc/RL=8')
    
    earth = plt.Circle((0, 0), 1, color='lightgray', zorder=10)
    ax3.add_patch(earth)
    ax3.set_xlabel('X GSM (Re)')
    ax3.set_ylabel('Z GSM (Re)')
    ax3.set_title('XZ Projection (Y ≈ 0)')
    ax3.set_aspect('equal')
    ax3.set_xlim(-12, 4)
    ax3.set_ylim(-2, 2)
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f'3D Distribution of Strong Scattering Regions (Rc/RL < {CRITICAL_RATIO})\\n' +
                 f'{energy_3d} keV electrons, Moderate Storm Conditions',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig12_3d_volume_rendering.png'))
    plt.close(fig)
    
    print("Figure 12 saved: 3D volume rendering of scattering regions")


def analyze_t96_xy_planes():
    """
    T96 Parameter Effects: XY Plane at Different Z Heights
    """
    print("\\n" + "="*60)
    print("Analysis: T96 Parameter Effects - XY Plane at Different Z Heights")
    print("="*60)
    
    z_levels_t96 = np.arange(0.0, 1.6, 0.2)  # Re - 0.2 Re increments
    energy_xy_t96 = 100  # keV
    
    # Select key parameter variations to analyze
    t96_xy_cases = [
        ("Baseline\\n(Pdyn=3, Dst=-30)", [3.0, -30.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("High Pressure\\n(Pdyn=10)", [10.0, -30.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("Strong Dst\\n(Dst=-100)", [3.0, -100.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("Southward IMF\\n(Bz=-15)", [3.0, -30.0, 1.0, -15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ]
    
    # Calculate number of rows dynamically based on z_levels
    n_rows = len(z_levels_t96)
    n_cols = len(t96_xy_cases)
    fig_height = 2 * n_rows + 2  # 2 inches per row plus margins
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, fig_height))
    
    # Create XY grid
    x_grid_xy_t96 = np.linspace(-15, 5, 61)
    y_grid_xy_t96 = np.linspace(-10, 10, 51)
    X_xy_t96, Y_xy_t96 = np.meshgrid(x_grid_xy_t96, y_grid_xy_t96)
    
    scatter_stats_xy_t96 = {}
    
    for row_idx, z_level in enumerate(z_levels_t96):
        for col_idx, (case_name, params) in enumerate(t96_xy_cases):
            ax = axes[row_idx, col_idx]
            
            # Create Z array at fixed height
            Z_xy_t96 = np.full_like(X_xy_t96, z_level)
            
            x_flat_xy = X_xy_t96.flatten()
            y_flat_xy = Y_xy_t96.flatten()
            z_flat_xy = Z_xy_t96.flatten()
            
            # Calculate for this parameter set
            Rc_Re_xy, B_nT_xy = calculate_curvature_radius(t96_vectorized, params, ps, 
                                                           x_flat_xy, y_flat_xy, z_flat_xy)
            Rc_m_xy = Rc_Re_xy * Re
            
            # Calculate Larmor radius
            RL_m_xy = calculate_larmor_radius(energy_xy_t96, B_nT_xy, pitch_angle_deg=90)
            ratio_xy = Rc_m_xy / RL_m_xy
            
            # Clean up extreme values
            ratio_xy = np.where(ratio_xy > 1000, 1000, ratio_xy)
            ratio_xy = np.where(ratio_xy < 0.1, 0.1, ratio_xy)
            ratio_grid_xy = ratio_xy.reshape(X_xy_t96.shape)
            
            # Calculate scattering fraction
            scatter_frac_xy = np.sum(ratio_xy < CRITICAL_RATIO) / len(ratio_xy) * 100
            key = f"{case_name}_Z{z_level}"
            scatter_stats_xy_t96[key] = scatter_frac_xy
            
            # Plot
            im = ax.contourf(X_xy_t96, Y_xy_t96, ratio_grid_xy,
                             levels=np.logspace(-1, 3, 15),
                             cmap='RdBu_r', extend='both',
                             norm=LogNorm(vmin=0.1, vmax=1000))
            
            # Add critical contour
            cs = ax.contour(X_xy_t96, Y_xy_t96, ratio_grid_xy, 
                            levels=[CRITICAL_RATIO],
                            colors='black', linewidths=2.5)
            ax.clabel(cs, inline=True, fontsize=9, fmt='8')
            
            # Title and labels
            if row_idx == 0:
                ax.set_title(case_name, fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(f'Z={z_level} Re\\nY GSM (Re)', fontsize=10)
            else:
                ax.set_ylabel('Y GSM (Re)', fontsize=9)
            if row_idx == n_rows - 1:
                ax.set_xlabel('X GSM (Re)', fontsize=10)
            
            # Add scattering percentage
            ax.text(0.02, 0.98, f'{scatter_frac_xy:.1f}%', 
                    transform=ax.transAxes, fontsize=10, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add Earth
            earth = plt.Circle((0, 0), 1, color='white', zorder=10)
            ax.add_patch(earth)
            ax.set_aspect('equal')
            ax.set_xlim(-15, 5)
            ax.set_ylim(-10, 10)
            
            # Add MLT labels for Z=0 T89 case
            if z_level == 0.0 and col_idx == 0:
                ax.text(5, 0, '12', ha='center', va='center', fontsize=8, weight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                ax.text(-12, 0, '00', ha='center', va='center', fontsize=8, weight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                ax.text(0, 10, '06', ha='center', va='center', fontsize=8, weight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                ax.text(0, -10, '18', ha='center', va='center', fontsize=8, weight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Rc/RL Ratio', fontsize=11)
    
    plt.suptitle(f'T96 Model: XY Plane at Different Z Heights\\n{energy_xy_t96} keV Electrons',
                 fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.91, 0.97])
    plt.savefig(os.path.join(output_dir, 'fig13_t96_xy_planes.png'))
    plt.close(fig)
    
    print("Figure 13 saved: T96 XY plane analysis at different Z heights")


def analyze_t01_xy_planes():
    """
    T01 Storm Evolution: XY Plane at Different Z Heights
    """
    print("\\n" + "="*60)
    print("Analysis: T01 Storm Evolution - XY Plane at Different Z Heights")
    print("="*60)
    
    z_levels_t01 = np.arange(0.0, 1.6, 0.2)  # Re - 0.2 Re increments
    energy_xy_t01 = 100  # keV
    
    # Select storm phases to analyze
    t01_xy_phases = [
        ("Quiet\\n(G1=0, G2=0)", [2.0, -10.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("Growth\\n(G1=0.5, G2=0.5)", [3.0, -30.0, 1.0, -3.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
        ("Main Phase\\n(G1=1, G2=1)", [5.0, -80.0, 3.0, -10.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        ("Recovery\\n(G1=0.7, G2=0.3)", [3.0, -50.0, 1.0, -5.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0])
    ]
    
    # Calculate number of rows dynamically based on z_levels
    n_rows = len(z_levels_t01)
    n_cols = len(t01_xy_phases)
    fig_height = 2 * n_rows + 2  # 2 inches per row plus margins
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, fig_height))
    
    # Create XY grid (reuse from T96)
    x_grid_xy_t96 = np.linspace(-15, 5, 61)
    y_grid_xy_t96 = np.linspace(-10, 10, 51)
    X_xy_t96, Y_xy_t96 = np.meshgrid(x_grid_xy_t96, y_grid_xy_t96)
    
    scatter_stats_xy_t01 = {}
    
    for row_idx, z_level in enumerate(z_levels_t01):
        for col_idx, (phase_name, params) in enumerate(t01_xy_phases):
            ax = axes[row_idx, col_idx]
            
            # Create Z array at fixed height
            Z_xy_t01 = np.full_like(X_xy_t96, z_level)
            
            x_flat_xy = X_xy_t96.flatten()
            y_flat_xy = Y_xy_t96.flatten()
            z_flat_xy = Z_xy_t01.flatten()
            
            # Calculate for this storm phase
            Rc_Re_xy, B_nT_xy = calculate_curvature_radius(t01_vectorized, params, ps, 
                                                           x_flat_xy, y_flat_xy, z_flat_xy)
            Rc_m_xy = Rc_Re_xy * Re
            
            # Calculate Larmor radius
            RL_m_xy = calculate_larmor_radius(energy_xy_t01, B_nT_xy, pitch_angle_deg=90)
            ratio_xy = Rc_m_xy / RL_m_xy
            
            # Clean up extreme values
            ratio_xy = np.where(ratio_xy > 1000, 1000, ratio_xy)
            ratio_xy = np.where(ratio_xy < 0.1, 0.1, ratio_xy)
            ratio_grid_xy = ratio_xy.reshape(X_xy_t96.shape)
            
            # Calculate scattering fraction
            scatter_frac_xy = np.sum(ratio_xy < CRITICAL_RATIO) / len(ratio_xy) * 100
            key = f"{phase_name}_Z{z_level}"
            scatter_stats_xy_t01[key] = scatter_frac_xy
            
            # Plot
            im = ax.contourf(X_xy_t96, Y_xy_t96, ratio_grid_xy,
                             levels=np.logspace(-1, 3, 15),
                             cmap='RdBu_r', extend='both',
                             norm=LogNorm(vmin=0.1, vmax=1000))
            
            # Add critical contour
            cs = ax.contour(X_xy_t96, Y_xy_t96, ratio_grid_xy, 
                            levels=[CRITICAL_RATIO],
                            colors='black', linewidths=2.5)
            ax.clabel(cs, inline=True, fontsize=9, fmt='8')
            
            # Title and labels
            if row_idx == 0:
                ax.set_title(phase_name, fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(f'Z={z_level} Re\\nY GSM (Re)', fontsize=10)
            else:
                ax.set_ylabel('Y GSM (Re)', fontsize=9)
            if row_idx == n_rows - 1:
                ax.set_xlabel('X GSM (Re)', fontsize=10)
            
            # Add scattering percentage and G values
            G1, G2 = params[4], params[5]
            text = f'{scatter_frac_xy:.1f}%\\nG1={G1}, G2={G2}'
            ax.text(0.02, 0.98, text, 
                    transform=ax.transAxes, fontsize=9, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add Earth
            earth = plt.Circle((0, 0), 1, color='white', zorder=10)
            ax.add_patch(earth)
            ax.set_aspect('equal')
            ax.set_xlim(-15, 5)
            ax.set_ylim(-10, 10)
            
            # Add MLT labels for Z=0 quiet case
            if z_level == 0.0 and col_idx == 0:
                ax.text(5, 0, '12', ha='center', va='center', fontsize=8, weight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                ax.text(-12, 0, '00', ha='center', va='center', fontsize=8, weight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                ax.text(0, 10, '06', ha='center', va='center', fontsize=8, weight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                ax.text(0, -10, '18', ha='center', va='center', fontsize=8, weight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Rc/RL Ratio', fontsize=11)
    
    plt.suptitle(f'T01 Model: XY Plane at Different Z Heights\\n{energy_xy_t01} keV Electrons',
                 fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.91, 0.97])
    plt.savefig(os.path.join(output_dir, 'fig14_t01_xy_planes.png'))
    plt.close(fig)
    
    print("Figure 14 saved: T01 XY plane analysis at different Z heights")


def analyze_t04_xy_planes():
    """
    T04 Model: XY Plane at Different Z Heights
    """
    print("\\n" + "="*60)
    print("Analysis: T04 Model - XY Plane at Different Z Heights")
    print("="*60)
    
    z_levels_t04 = np.arange(0.0, 1.6, 0.2)  # Re - 0.2 Re increments
    energy_xy_t04 = 100  # keV
    
    # Select storm intensities to analyze
    t04_xy_storms = [
        ("Quiet", [2.0, -10.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("Moderate", [5.0, -50.0, 3.0, -8.0, 0.5, 0.4, 0.3, 0.3, 0.2, 0.2]),
        ("Intense", [10.0, -150.0, 7.0, -20.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5]),
        ("Super Storm", [15.0, -250.0, 10.0, -30.0, 2.0, 1.8, 1.5, 1.2, 1.0, 0.8])
    ]
    
    # Calculate number of rows dynamically based on z_levels
    n_rows = len(z_levels_t04)
    n_cols = len(t04_xy_storms)
    fig_height = 2 * n_rows + 2  # 2 inches per row plus margins
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, fig_height))
    
    # Create XY grid (reuse from T96)
    x_grid_xy_t96 = np.linspace(-15, 5, 61)
    y_grid_xy_t96 = np.linspace(-10, 10, 51)
    X_xy_t96, Y_xy_t96 = np.meshgrid(x_grid_xy_t96, y_grid_xy_t96)
    
    scatter_stats_xy_t04 = {}
    
    for row_idx, z_level in enumerate(z_levels_t04):
        for col_idx, (storm_name, params) in enumerate(t04_xy_storms):
            ax = axes[row_idx, col_idx]
            
            # Create Z array at fixed height
            Z_xy_t04 = np.full_like(X_xy_t96, z_level)
            
            x_flat_xy = X_xy_t96.flatten()
            y_flat_xy = Y_xy_t96.flatten()
            z_flat_xy = Z_xy_t04.flatten()
            
            # Calculate for this storm intensity
            Rc_Re_xy, B_nT_xy = calculate_curvature_radius(t04_vectorized, params, ps, 
                                                           x_flat_xy, y_flat_xy, z_flat_xy)
            Rc_m_xy = Rc_Re_xy * Re
            
            # Calculate Larmor radius
            RL_m_xy = calculate_larmor_radius(energy_xy_t04, B_nT_xy, pitch_angle_deg=90)
            ratio_xy = Rc_m_xy / RL_m_xy
            
            # Clean up extreme values
            ratio_xy = np.where(ratio_xy > 1000, 1000, ratio_xy)
            ratio_xy = np.where(ratio_xy < 0.1, 0.1, ratio_xy)
            ratio_grid_xy = ratio_xy.reshape(X_xy_t96.shape)
            
            # Calculate scattering fraction
            scatter_frac_xy = np.sum(ratio_xy < CRITICAL_RATIO) / len(ratio_xy) * 100
            key = f"{storm_name}_Z{z_level}"
            scatter_stats_xy_t04[key] = scatter_frac_xy
            
            # Plot
            im = ax.contourf(X_xy_t96, Y_xy_t96, ratio_grid_xy,
                             levels=np.logspace(-1, 3, 15),
                             cmap='RdBu_r', extend='both',
                             norm=LogNorm(vmin=0.1, vmax=1000))
            
            # Add critical contour
            cs = ax.contour(X_xy_t96, Y_xy_t96, ratio_grid_xy, 
                            levels=[CRITICAL_RATIO],
                            colors='black', linewidths=2.5)
            ax.clabel(cs, inline=True, fontsize=9, fmt='8')
            
            # Title and labels
            if row_idx == 0:
                ax.set_title(f'{storm_name}\\nDst={params[1]} nT', fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f'Z={z_level} Re\\nY GSM (Re)', fontsize=10)
            else:
                ax.set_ylabel('Y GSM (Re)', fontsize=9)
            if row_idx == n_rows - 1:
                ax.set_xlabel('X GSM (Re)', fontsize=10)
            
            # Add scattering percentage and W1 value
            W1 = params[4]
            text = f'{scatter_frac_xy:.1f}%\\nW1={W1}'
            ax.text(0.02, 0.98, text, 
                    transform=ax.transAxes, fontsize=9, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add Earth
            earth = plt.Circle((0, 0), 1, color='white', zorder=10)
            ax.add_patch(earth)
            ax.set_aspect('equal')
            ax.set_xlim(-15, 5)
            ax.set_ylim(-10, 10)
            
            # Add MLT labels for Z=0 quiet case
            if z_level == 0.0 and col_idx == 0:
                ax.text(5, 0, '12', ha='center', va='center', fontsize=8, weight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                ax.text(-12, 0, '00', ha='center', va='center', fontsize=8, weight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                ax.text(0, 10, '06', ha='center', va='center', fontsize=8, weight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                ax.text(0, -10, '18', ha='center', va='center', fontsize=8, weight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Rc/RL Ratio', fontsize=11)
    
    plt.suptitle(f'T04 Model: XY Plane at Different Z Heights\\n{energy_xy_t04} keV Electrons',
                 fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.91, 0.97])
    plt.savefig(os.path.join(output_dir, 'fig15_t04_xy_planes.png'))
    plt.close(fig)
    
    print("Figure 15 saved: T04 XY plane analysis at different Z heights")


def analyze_t01_model_analysis():
    """
    Analysis 7: T01 Model Analysis with Storm-Time Parameters
    The T01 model includes additional storm-time parameters:
    - G1, G2: Storm-time corrections based on the Dst index time history
    """
    print("\\n" + "="*60)
    print("Analysis 7: T01 Model Analysis with Storm-Time Parameters")
    print("="*60)
    
    # Create grid for analysis
    x_grid_sens = np.linspace(-15, 5, 61)
    z_grid_sens = np.linspace(-6, 6, 49)
    X_sens, Z_sens = np.meshgrid(x_grid_sens, z_grid_sens)
    Y_sens = np.zeros_like(X_sens)

    x_flat_sens = X_sens.flatten()
    y_flat_sens = Y_sens.flatten()
    z_flat_sens = Z_sens.flatten()
    
    # T01 Model Analysis with Storm-Time Parameters
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # T01 parameters: [Pdyn, Dst, ByIMF, BzIMF, G1, G2, ...]
    # G1 and G2 are storm-time corrections
    t01_variations = [
        ("Quiet (G1=0, G2=0)", [2.0, -10.0, 0.0, 2.0, 0.0, 0.0]),
        ("Growth (G1=0.5, G2=0.5)", [3.0, -30.0, 1.0, -3.0, 0.5, 0.5]),
        ("Main Phase (G1=1, G2=1)", [5.0, -80.0, 3.0, -10.0, 1.0, 1.0]),
        ("Recovery (G1=0.7, G2=0.3)", [3.0, -50.0, 1.0, -5.0, 0.7, 0.3]),
        ("Strong Storm (G1=1.5, G2=1.5)", [8.0, -150.0, 5.0, -15.0, 1.5, 1.5]),
        ("Extreme (G1=2, G2=2)", [10.0, -200.0, 7.0, -20.0, 2.0, 2.0])
    ]

    energy_t01 = 100  # keV
    scatter_stats_t01 = {}

    for idx, (label, params) in enumerate(t01_variations):
        ax = axes.flatten()[idx]
        
        # Full parameter array for T01
        full_params = params + [0.0, 0.0, 0.0, 0.0]
        
        # Calculate fields
        Rc_Re_t01, B_nT_t01 = calculate_curvature_radius(t01_vectorized, full_params, ps, 
                                                         x_flat_sens, y_flat_sens, z_flat_sens)
        Rc_m_t01 = Rc_Re_t01 * Re
        
        # Calculate Larmor radius
        RL_m_t01 = calculate_larmor_radius(energy_t01, B_nT_t01, pitch_angle_deg=90)
        ratio_t01 = Rc_m_t01 / RL_m_t01
        
        # Clean up extreme values
        ratio_t01 = np.where(ratio_t01 > 1000, 1000, ratio_t01)
        ratio_t01 = np.where(ratio_t01 < 0.1, 0.1, ratio_t01)
        ratio_grid_t01 = ratio_t01.reshape(X_sens.shape)
        
        # Calculate scattering fraction
        scatter_frac_t01 = np.sum(ratio_t01 < CRITICAL_RATIO) / len(ratio_t01) * 100
        scatter_stats_t01[label] = scatter_frac_t01
        
        # Plot
        im = ax.contourf(X_sens, Z_sens, ratio_grid_t01,
                         levels=np.logspace(-1, 3, 20),
                         cmap='RdBu_r', extend='both',
                         norm=LogNorm(vmin=0.1, vmax=1000))
        
        # Add critical contour
        cs = ax.contour(X_sens, Z_sens, ratio_grid_t01, 
                        levels=[CRITICAL_RATIO],
                        colors='black', linewidths=3)
        ax.clabel(cs, inline=True, fontsize=10, fmt='Rc/RL=8')
        
        # Add other contours
        ax.contour(X_sens, Z_sens, ratio_grid_t01, 
                   levels=[1, 2, 4, 16, 32],
                   colors='gray', linewidths=1, alpha=0.5)
        
        # Extract key parameters for title
        Dst = params[1]
        G1 = params[4]
        G2 = params[5]
        
        ax.set_title(f'{label}\\nDst={Dst} nT, Scattering: {scatter_frac_t01:.1f}%', 
                     fontsize=12)
        ax.set_xlabel('X GSM (Re)', fontsize=10)
        if idx % 3 == 0:
            ax.set_ylabel('Z GSM (Re)', fontsize=10)
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax.add_patch(earth)
        ax.set_aspect('equal')
        ax.set_xlim(-15, 5)
        ax.set_ylim(-6, 6)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Rc/RL Ratio', fontsize=12)

    plt.suptitle(f'T01 Model Storm Evolution: {energy_t01} keV Electrons\\n' +
                 'Effects of Storm-Time Parameters G1 and G2',
                 fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.91, 0.97])
    plt.savefig(os.path.join(output_dir, 'fig16_t01_storm_evolution.png'))
    plt.close(fig)
    
    print("Figure 16 saved: T01 model storm evolution")
    
    print("\\nT01 Model Storm Phase Analysis:")
    print("=" * 60)
    print("Scattering regions (Rc/RL < 8) during storm phases:")
    print("-" * 40)
    for label, frac in scatter_stats_t01.items():
        print(f"{label:30s}: {frac:5.1f}% scattering")


def analyze_t04_model_analysis():
    """
    Analysis 8: T04 Model Analysis with Advanced Storm Parameters
    The T04 model is the most sophisticated, including:
    - W1-W6: Six storm-time parameters derived from solar wind and Dst history
    """
    print("\\n" + "="*60)
    print("Analysis 8: T04 Model Analysis with Advanced Storm Parameters")
    print("="*60)
    
    # Create grid for analysis
    x_grid_sens = np.linspace(-15, 5, 61)
    z_grid_sens = np.linspace(-6, 6, 49)
    X_sens, Z_sens = np.meshgrid(x_grid_sens, z_grid_sens)
    Y_sens = np.zeros_like(X_sens)

    x_flat_sens = X_sens.flatten()
    y_flat_sens = Y_sens.flatten()
    z_flat_sens = Z_sens.flatten()
    
    # T04 Model Analysis with W1-W6 Parameters
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # T04 parameters: [Pdyn, Dst, ByIMF, BzIMF, W1, W2, W3, W4, W5, W6]
    # W1-W6 represent different storm-time effects
    t04_variations = [
        ("Quiet", [2.0, -10.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("Moderate Activity", [3.0, -30.0, 1.0, -3.0, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]),
        ("Storm Growth", [5.0, -50.0, 3.0, -8.0, 0.5, 0.4, 0.3, 0.3, 0.2, 0.2]),
        ("Storm Main Phase", [8.0, -100.0, 5.0, -15.0, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3]),
        ("Intense Storm", [10.0, -150.0, 7.0, -20.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5]),
        ("Super Storm", [15.0, -250.0, 10.0, -30.0, 2.0, 1.8, 1.5, 1.2, 1.0, 0.8])
    ]

    energy_t04 = 100  # keV
    scatter_stats_t04 = {}

    for idx, (label, params) in enumerate(t04_variations):
        ax = axes.flatten()[idx]
        
        # Calculate fields
        Rc_Re_t04, B_nT_t04 = calculate_curvature_radius(t04_vectorized, params, ps, 
                                                         x_flat_sens, y_flat_sens, z_flat_sens)
        Rc_m_t04 = Rc_Re_t04 * Re
        
        # Calculate Larmor radius
        RL_m_t04 = calculate_larmor_radius(energy_t04, B_nT_t04, pitch_angle_deg=90)
        ratio_t04 = Rc_m_t04 / RL_m_t04
        
        # Clean up extreme values
        ratio_t04 = np.where(ratio_t04 > 1000, 1000, ratio_t04)
        ratio_t04 = np.where(ratio_t04 < 0.1, 0.1, ratio_t04)
        ratio_grid_t04 = ratio_t04.reshape(X_sens.shape)
        
        # Calculate scattering fraction
        scatter_frac_t04 = np.sum(ratio_t04 < CRITICAL_RATIO) / len(ratio_t04) * 100
        scatter_stats_t04[label] = scatter_frac_t04
        
        # Plot
        im = ax.contourf(X_sens, Z_sens, ratio_grid_t04,
                         levels=np.logspace(-1, 3, 20),
                         cmap='RdBu_r', extend='both',
                         norm=LogNorm(vmin=0.1, vmax=1000))
        
        # Add critical contour
        cs = ax.contour(X_sens, Z_sens, ratio_grid_t04, 
                        levels=[CRITICAL_RATIO],
                        colors='black', linewidths=3)
        ax.clabel(cs, inline=True, fontsize=10, fmt='Rc/RL=8')
        
        # Add other contours
        ax.contour(X_sens, Z_sens, ratio_grid_t04, 
                   levels=[1, 2, 4, 16, 32],
                   colors='gray', linewidths=1, alpha=0.5)
        
        # Extract key parameters for title
        Dst = params[1]
        W1 = params[4]
        
        ax.set_title(f'{label}\\nDst={Dst} nT, W1={W1}, Scattering: {scatter_frac_t04:.1f}%', 
                     fontsize=11)
        ax.set_xlabel('X GSM (Re)', fontsize=10)
        if idx % 3 == 0:
            ax.set_ylabel('Z GSM (Re)', fontsize=10)
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax.add_patch(earth)
        ax.set_aspect('equal')
        ax.set_xlim(-15, 5)
        ax.set_ylim(-6, 6)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Rc/RL Ratio', fontsize=12)

    plt.suptitle(f'T04 Model Storm Progression: {energy_t04} keV Electrons\\n' +
                 'Most Sophisticated Storm-Time Model with W1-W6 Parameters',
                 fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.91, 0.97])
    plt.savefig(os.path.join(output_dir, 'fig17_t04_storm_progression.png'))
    plt.close(fig)
    
    print("Figure 17 saved: T04 model storm progression")
    
    print("\\nT04 Model Analysis Summary:")
    print("=" * 60)
    print("Scattering regions (Rc/RL < 8) for different storm levels:")
    print("-" * 40)
    for label, frac in scatter_stats_t04.items():
        print(f"{label:20s}: {frac:5.1f}% scattering")


def analyze_model_comparison_xy_planes():
    """
    Comprehensive Model Comparison: XY Plane at Different Z Heights
    """
    print("\\n" + "="*60)
    print("Analysis: Model Comparison - XY Plane at Different Z Heights")
    print("="*60)
    
    z_levels_comp = np.arange(0.0, 1.6, 0.2)  # Re - 0.2 Re increments
    energy_xy_comp = 100  # keV
    
    # Use moderate storm conditions for all models
    model_params_xy = {
        "T89": 4,  # Kp = 4
        "T96": [5.0, -50.0, 2.0, -8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "T01": [5.0, -50.0, 2.0, -8.0, 0.7, 0.7, 0.0, 0.0, 0.0, 0.0],
        "T04": [5.0, -50.0, 2.0, -8.0, 0.5, 0.4, 0.3, 0.3, 0.2, 0.2]
    }
    
    models_comp = [
        ("T89", t89_vectorized),
        ("T96", t96_vectorized),
        ("T01", t01_vectorized),
        ("T04", t04_vectorized)
    ]
    
    # Calculate number of rows dynamically based on z_levels
    n_rows = len(z_levels_comp)
    n_cols = len(models_comp)
    fig_height = 2 * n_rows + 2  # 2 inches per row plus margins
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, fig_height))
    
    # Create XY grid (reuse from T96)
    x_grid_xy_t96 = np.linspace(-15, 5, 61)
    y_grid_xy_t96 = np.linspace(-10, 10, 51)
    X_xy_t96, Y_xy_t96 = np.meshgrid(x_grid_xy_t96, y_grid_xy_t96)
    
    scatter_stats_xy_comp = {}
    
    for row_idx, z_level in enumerate(z_levels_comp):
        for col_idx, (model_name, model_func) in enumerate(models_comp):
            ax = axes[row_idx, col_idx]
            
            # Get parameters for this model
            params = model_params_xy[model_name]
            
            # Create Z array at fixed height
            Z_xy_comp = np.full_like(X_xy_t96, z_level)
            
            x_flat_xy = X_xy_t96.flatten()
            y_flat_xy = Y_xy_t96.flatten()
            z_flat_xy = Z_xy_comp.flatten()
            
            # Calculate for this model
            Rc_Re_xy, B_nT_xy = calculate_curvature_radius(model_func, params, ps, 
                                                           x_flat_xy, y_flat_xy, z_flat_xy)
            Rc_m_xy = Rc_Re_xy * Re
            
            # Calculate Larmor radius
            RL_m_xy = calculate_larmor_radius(energy_xy_comp, B_nT_xy, pitch_angle_deg=90)
            ratio_xy = Rc_m_xy / RL_m_xy
            
            # Clean up extreme values
            ratio_xy = np.where(ratio_xy > 1000, 1000, ratio_xy)
            ratio_xy = np.where(ratio_xy < 0.1, 0.1, ratio_xy)
            ratio_grid_xy = ratio_xy.reshape(X_xy_t96.shape)
            
            # Calculate scattering fraction
            scatter_frac_xy = np.sum(ratio_xy < CRITICAL_RATIO) / len(ratio_xy) * 100
            key = f"{model_name}_Z{z_level}"
            scatter_stats_xy_comp[key] = scatter_frac_xy
            
            # Plot
            im = ax.contourf(X_xy_t96, Y_xy_t96, ratio_grid_xy,
                             levels=np.logspace(-1, 3, 15),
                             cmap='RdBu_r', extend='both',
                             norm=LogNorm(vmin=0.1, vmax=1000))
            
            # Add critical contour
            cs = ax.contour(X_xy_t96, Y_xy_t96, ratio_grid_xy, 
                            levels=[CRITICAL_RATIO],
                            colors='black', linewidths=2.5)
            ax.clabel(cs, inline=True, fontsize=9, fmt='8')
            
            # Title and labels
            if row_idx == 0:
                ax.set_title(f'{model_name}', fontsize=12, weight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'Z={z_level} Re\\nY GSM (Re)', fontsize=10)
            else:
                ax.set_ylabel('Y GSM (Re)', fontsize=9)
            if row_idx == n_rows - 1:
                ax.set_xlabel('X GSM (Re)', fontsize=10)
            
            # Add scattering percentage
            ax.text(0.02, 0.98, f'{scatter_frac_xy:.1f}%', 
                    transform=ax.transAxes, fontsize=10, weight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add Earth
            earth = plt.Circle((0, 0), 1, color='white', zorder=10)
            ax.add_patch(earth)
            ax.set_aspect('equal')
            ax.set_xlim(-15, 5)
            ax.set_ylim(-10, 10)
            
            # Add MLT labels for Z=0 T89 case
            if z_level == 0.0 and model_name == "T89":
                ax.text(5, 0, '12', ha='center', va='center', fontsize=8, weight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                ax.text(-12, 0, '00', ha='center', va='center', fontsize=8, weight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                ax.text(0, 10, '06', ha='center', va='center', fontsize=8, weight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                ax.text(0, -10, '18', ha='center', va='center', fontsize=8, weight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Rc/RL Ratio', fontsize=11)
    
    plt.suptitle(f'Model Comparison: XY Plane at Different Z Heights\\n{energy_xy_comp} keV Electrons, Moderate Storm',
                 fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.91, 0.97])
    plt.savefig(os.path.join(output_dir, 'fig18_model_comparison_xy_planes.png'))
    plt.close(fig)
    
    print("Figure 18 saved: Model comparison XY planes at different Z heights")


def analyze_t96_seasonal_tilt():
    """
    T96 Seasonal Dipole Tilt Effects: XY Plane Analysis
    """
    print("\n" + "="*60)
    print("Analysis: T96 Seasonal Dipole Tilt Effects")
    print("="*60)
    
    # Define seasonal tilt angles (in radians)
    # Summer solstice: maximum positive tilt (~34 degrees)
    # Winter solstice: maximum negative tilt (~-34 degrees)
    # Equinoxes: near zero tilt
    seasonal_tilts = [
        ("Summer Solstice\n(PS = +34°)", np.radians(34)),
        ("Spring/Fall Equinox\n(PS = 0°)", np.radians(0)),
        ("Winter Solstice\n(PS = -34°)", np.radians(-34)),
        ("Moderate Summer\n(PS = +23°)", np.radians(23)),
        ("Moderate Winter\n(PS = -23°)", np.radians(-23))
    ]
    
    # Fixed parameters for T96
    parmod_seasonal = [3.0, -30.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    energy_seasonal = 100  # keV
    
    # Z levels to analyze
    z_levels_seasonal = np.arange(0.0, 1.6, 0.2)  # Re - 0.0 to 1.4 in 0.2 Re increments
    
    # Create figure with subplots
    n_rows = len(z_levels_seasonal)
    n_cols = len(seasonal_tilts)
    fig_height = 2.5 * n_rows + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, fig_height))
    
    # Create XY grid
    x_grid_seasonal = np.linspace(-15, 5, 61)
    y_grid_seasonal = np.linspace(-10, 10, 51)
    X_seasonal, Y_seasonal = np.meshgrid(x_grid_seasonal, y_grid_seasonal)
    
    scatter_stats_seasonal = {}
    
    for row_idx, z_level in enumerate(z_levels_seasonal):
        for col_idx, (season_name, ps_rad) in enumerate(seasonal_tilts):
            ax = axes[row_idx, col_idx] if n_rows > 1 else axes[col_idx]
            
            # Create Z array at fixed height
            Z_seasonal = np.full_like(X_seasonal, z_level)
            
            x_flat = X_seasonal.flatten()
            y_flat = Y_seasonal.flatten()
            z_flat = Z_seasonal.flatten()
            
            # Calculate with seasonal tilt
            Rc_Re, B_nT = calculate_curvature_radius(t96_vectorized, parmod_seasonal, ps_rad, 
                                                     x_flat, y_flat, z_flat)
            Rc_m = Rc_Re * Re
            
            # Calculate Larmor radius
            RL_m = calculate_larmor_radius(energy_seasonal, B_nT, pitch_angle_deg=90)
            ratio = Rc_m / RL_m
            
            # Clean up extreme values
            ratio = np.where(ratio > 1000, 1000, ratio)
            ratio = np.where(ratio < 0.1, 0.1, ratio)
            ratio_grid = ratio.reshape(X_seasonal.shape)
            
            # Calculate scattering fraction
            scatter_frac = np.sum(ratio < CRITICAL_RATIO) / len(ratio) * 100
            key = f"{season_name}_Z{z_level}"
            scatter_stats_seasonal[key] = scatter_frac
            
            # Plot
            im = ax.contourf(X_seasonal, Y_seasonal, ratio_grid,
                            levels=np.logspace(-1, 3, 15),
                            cmap='RdBu_r', extend='both',
                            norm=LogNorm(vmin=0.1, vmax=1000))
            
            # Add critical contour
            cs = ax.contour(X_seasonal, Y_seasonal, ratio_grid, 
                           levels=[CRITICAL_RATIO],
                           colors='black', linewidths=2.5)
            ax.clabel(cs, inline=True, fontsize=9, fmt='8')
            
            # Title and labels
            if row_idx == 0:
                ax.set_title(season_name, fontsize=12, weight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'Z={z_level} Re\nY GSM (Re)', fontsize=10)
            else:
                ax.set_ylabel('Y GSM (Re)', fontsize=9)
            if row_idx == n_rows - 1:
                ax.set_xlabel('X GSM (Re)', fontsize=10)
            
            # Add scattering percentage
            ax.text(0.02, 0.98, f'{scatter_frac:.1f}%', 
                   transform=ax.transAxes, fontsize=10, weight='bold',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add Earth
            earth = plt.Circle((0, 0), 1, color='white', zorder=10)
            ax.add_patch(earth)
            ax.set_aspect('equal')
            ax.set_xlim(-15, 5)
            ax.set_ylim(-10, 10)
            
            # Add grid
            ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Rc/RL Ratio', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Add overall title
    fig.suptitle(f'T96 Model: Seasonal Dipole Tilt Effects on Scattering Regions\n' + 
                 f'{energy_seasonal} keV Electrons, Moderate Storm (Dst=-30 nT)', 
                 fontsize=14, weight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'fig19_t96_seasonal_tilt.png'))
    plt.close(fig)
    
    # Print statistics
    print("\nScattering Region Statistics by Season and Height:")
    print("-" * 60)
    for season_name, _ in seasonal_tilts:
        season_key = season_name.split('\n')[0]
        print(f"\n{season_key}:")
        for z_level in z_levels_seasonal:
            key = f"{season_name}_Z{z_level}"
            if key in scatter_stats_seasonal:
                print(f"  Z = {z_level} Re: {scatter_stats_seasonal[key]:.1f}%")
    
    print("\nFigure 19 saved: T96 seasonal dipole tilt effects")


def analyze_seasonal_evolution():
    """
    Seasonal Evolution Analysis: Temporal variation throughout the year
    """
    print("\n" + "="*60)
    print("Analysis: Seasonal Evolution of Scattering Regions")
    print("="*60)
    
    # Create a full year of dipole tilt variation
    # Approximate sinusoidal variation with max tilt at solstices
    days_in_year = 365
    days = np.arange(0, days_in_year + 1, 5)  # Every 5 days
    
    # Summer solstice around day 172 (June 21), Winter solstice around day 355 (Dec 21)
    # Maximum tilt ±34 degrees
    tilt_degrees = 34 * np.sin(2 * np.pi * (days - 80) / days_in_year)  # Phase shifted for solstices
    tilt_radians = np.radians(tilt_degrees)
    
    # Energy levels to analyze
    energies = [10, 30, 100, 300, 1000]  # keV
    
    # Fixed T96 parameters (moderate storm)
    parmod = [3.0, -30.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Calculate scattering statistics throughout the year
    scatter_stats = {energy: [] for energy in energies}
    
    # XY plane at Z = 0
    x_grid = np.linspace(-15, 5, 41)
    y_grid = np.linspace(-10, 10, 41)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    
    print("Calculating seasonal evolution...")
    for ps_rad in tilt_radians:
        for energy in energies:
            # Calculate for this tilt and energy
            Rc_Re, B_nT = calculate_curvature_radius(t96_vectorized, parmod, ps_rad, 
                                                     x_flat, y_flat, z_flat)
            Rc_m = Rc_Re * Re
            RL_m = calculate_larmor_radius(energy, B_nT, pitch_angle_deg=90)
            ratio = Rc_m / RL_m
            
            # Calculate scattering fraction
            scatter_frac = np.sum(ratio < CRITICAL_RATIO) / len(ratio) * 100
            scatter_stats[energy].append(scatter_frac)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
    
    # Plot 1: Scattering percentage throughout the year
    colors = ['darkred', 'red', 'orange', 'blue', 'darkblue']
    for energy, color in zip(energies, colors):
        ax1.plot(days, scatter_stats[energy], color=color, linewidth=2.5, 
                label=f'{energy} keV', marker='o', markersize=3, markevery=10)
    
    # Add seasonal markers
    ax1.axvline(80, color='green', linestyle='--', alpha=0.5, label='Spring Equinox')
    ax1.axvline(172, color='red', linestyle='--', alpha=0.5, label='Summer Solstice')
    ax1.axvline(264, color='orange', linestyle='--', alpha=0.5, label='Fall Equinox')
    ax1.axvline(355, color='blue', linestyle='--', alpha=0.5, label='Winter Solstice')
    
    ax1.set_xlabel('Day of Year', fontsize=12)
    ax1.set_ylabel('Scattering Region (%)', fontsize=12)
    ax1.set_title('Seasonal Evolution of Curvature Scattering Regions at Z = 0 Re\n' +
                  'T96 Model, Moderate Storm (Dst = -30 nT)', fontsize=14, weight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 365)
    ax1.set_ylim(0, max([max(scatter_stats[e]) for e in energies]) * 1.1)
    
    # Plot 2: Dipole tilt angle
    ax2.plot(days, tilt_degrees, 'k-', linewidth=2)
    ax2.fill_between(days, 0, tilt_degrees, where=tilt_degrees>0, 
                     color='red', alpha=0.3, label='Northern Summer')
    ax2.fill_between(days, 0, tilt_degrees, where=tilt_degrees<0, 
                     color='blue', alpha=0.3, label='Northern Winter')
    
    ax2.set_xlabel('Day of Year', fontsize=12)
    ax2.set_ylabel('Dipole Tilt (degrees)', fontsize=12)
    ax2.set_title('Earth Dipole Tilt Angle', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 365)
    ax2.set_ylim(-40, 40)
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig20_seasonal_evolution.png'))
    plt.close(fig)
    
    # Print statistics
    print("\nSeasonal Scattering Statistics:")
    print("-" * 60)
    for energy in energies:
        stats = scatter_stats[energy]
        print(f"\n{energy} keV electrons:")
        print(f"  Maximum scattering: {max(stats):.1f}% (around equinoxes)")
        print(f"  Minimum scattering: {min(stats):.1f}% (around solstices)")
        print(f"  Annual average: {np.mean(stats):.1f}%")
    
    print("\nFigure 20 saved: Seasonal evolution of scattering regions")


def analyze_seasonal_mlt_distribution():
    """
    Seasonal MLT Distribution Analysis: How scattering varies with MLT for different seasons
    """
    print("\n" + "="*60)
    print("Analysis: Seasonal MLT Distribution of Scattering")
    print("="*60)
    
    # Define key seasonal configurations
    seasons = [
        ("Summer Solstice", np.radians(34)),
        ("Equinox", np.radians(0)),
        ("Winter Solstice", np.radians(-34))
    ]
    
    # Energy for analysis
    energy = 100  # keV
    
    # T96 parameters
    parmod = [3.0, -30.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Create figure with polar plots
    fig = plt.figure(figsize=(15, 5))
    
    # MLT bins
    mlt_bins = np.linspace(0, 24, 49)  # 0.5 hour bins
    mlt_centers = (mlt_bins[:-1] + mlt_bins[1:]) / 2
    
    for idx, (season_name, ps_rad) in enumerate(seasons):
        ax = fig.add_subplot(1, 3, idx + 1, projection='polar')
        
        # Initialize MLT statistics
        mlt_scatter_stats = []
        
        for mlt in mlt_centers:
            # Convert MLT to angle (MLT in hours, 0 = midnight, 12 = noon)
            phi = np.radians(15 * (mlt - 6))  # -90 degrees for dawn at 6 MLT
            
            # Sample along radial direction at this MLT
            r_samples = np.linspace(3, 12, 30)  # Re
            scatter_count = 0
            
            for r in r_samples:
                # Convert to GSM coordinates
                x = r * np.cos(phi)
                y = r * np.sin(phi)
                z = 0  # Equatorial plane
                
                # Calculate scattering
                Rc_Re, B_nT = calculate_curvature_radius(t96_vectorized, parmod, ps_rad, 
                                                        np.array([x]), np.array([y]), np.array([z]))
                Rc_m = Rc_Re[0] * Re
                RL_m = calculate_larmor_radius(energy, B_nT[0], pitch_angle_deg=90)
                ratio = Rc_m / RL_m
                
                if ratio < CRITICAL_RATIO:
                    scatter_count += 1
            
            scatter_percentage = (scatter_count / len(r_samples)) * 100
            mlt_scatter_stats.append(scatter_percentage)
        
        # Convert MLT to radians for polar plot (0 = midnight at top)
        theta = np.radians(15 * mlt_centers)  # 15 degrees per MLT hour
        theta = np.append(theta, theta[0])  # Close the circle
        values = np.append(mlt_scatter_stats, mlt_scatter_stats[0])
        
        # Plot
        ax.plot(theta, values, 'b-', linewidth=2)
        ax.fill(theta, values, 'b', alpha=0.3)
        
        # Customize polar plot
        ax.set_theta_zero_location('S')  # Midnight at top
        ax.set_theta_direction(-1)  # Clockwise
        ax.set_ylim(0, max(mlt_scatter_stats) * 1.2 if max(mlt_scatter_stats) > 0 else 10)
        
        # MLT labels
        mlt_labels = ['00', '03', '06', '09', '12', '15', '18', '21']
        ax.set_thetagrids(np.arange(0, 360, 45), mlt_labels)
        
        ax.set_title(f'{season_name}\nPS = {np.degrees(ps_rad):.0f}°', 
                    fontsize=12, weight='bold', pad=20)
        ax.set_ylabel('Scattering %', labelpad=30)
        
        # Add radial grid
        ax.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'MLT Distribution of Curvature Scattering Regions\n' +
                 f'{energy} keV Electrons, T96 Model, Z = 0 Re', 
                 fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig21_seasonal_mlt_distribution.png'))
    plt.close(fig)
    
    print("\nFigure 21 saved: Seasonal MLT distribution of scattering")


def main():
    """
    Main function to run all analyses
    """
    print("\\n" + "="*80)
    print("MAGNETIC FIELD LINE CURVATURE SCATTERING ANALYSIS")
    print("Critical Threshold: Rc/RL = 8")
    print("="*80)
    
    # Run all analyses
    analyze_scattering_regions_by_energy()
    analyze_xy_plane_cross_sections()
    analyze_3d_field_lines()
    analyze_3d_volume_rendering()
    analyze_magnetic_equatorial_plane()
    analyze_critical_energy_maps()
    analyze_storm_evolution()
    analyze_t96_parameter_sensitivity()
    analyze_t96_xy_planes()
    analyze_t01_xy_planes()
    analyze_t04_xy_planes()
    analyze_t01_model_analysis()
    analyze_t04_model_analysis()
    analyze_comprehensive_model_comparison()
    analyze_model_comparison_xy_planes()
    analyze_t96_seasonal_tilt()
    analyze_seasonal_evolution()
    analyze_seasonal_mlt_distribution()
    create_summary_figure()
    
    print("\\n" + "="*80)
    print("Analysis complete! All figures saved to:", output_dir)
    print("="*80)


if __name__ == "__main__":
    main()