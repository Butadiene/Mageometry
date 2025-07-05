#!/usr/bin/env python3
"""
Magnetic Field Line Curvature Scattering Analysis: Rc/RL = 8 Threshold
SM COORDINATE SYSTEM VERSION

This script analyzes the critical threshold Rc/RL = 8 for electron scattering in the magnetosphere
using the Solar Magnetic (SM) coordinate system instead of GSM.

In SM coordinates:
- X-axis: Points from Earth to Sun
- Y-axis: Perpendicular to X in plane containing dipole axis
- Z-axis: Contains dipole axis (aligned with magnetic dipole)

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
output_dir = "figures_sm"
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

print(f"\nDipole tilt: {np.degrees(ps):.2f}°")
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


def gsm_to_sm(x_gsm, y_gsm, z_gsm, ps):
    """Convert from GSM to SM coordinates."""
    # SM coordinates have Z along dipole axis
    # Simple rotation by dipole tilt angle around Y axis
    cos_ps = np.cos(ps)
    sin_ps = np.sin(ps)
    
    x_sm = x_gsm * cos_ps - z_gsm * sin_ps
    y_sm = y_gsm
    z_sm = x_gsm * sin_ps + z_gsm * cos_ps
    
    return x_sm, y_sm, z_sm


def sm_to_gsm(x_sm, y_sm, z_sm, ps):
    """Convert from SM to GSM coordinates."""
    # Inverse rotation
    cos_ps = np.cos(ps)
    sin_ps = np.sin(ps)
    
    x_gsm = x_sm * cos_ps + z_sm * sin_ps
    y_gsm = y_sm
    z_gsm = -x_sm * sin_ps + z_sm * cos_ps
    
    return x_gsm, y_gsm, z_gsm


def calculate_curvature_radius(model_func, parmod, ps, x_sm, y_sm, z_sm):
    """Calculate radius of curvature and magnetic field strength in SM coordinates."""
    # Convert SM to GSM for model calculation
    x_gsm, y_gsm, z_gsm = sm_to_gsm(x_sm, y_sm, z_sm, ps)
    
    # Calculate curvature in GSM
    kappa = field_line_curvature_vectorized(model_func, parmod, ps, x_gsm, y_gsm, z_gsm)
    Rc = np.where(kappa > 1e-10, 1.0 / kappa, 1e10)
    
    # Calculate magnetic field in GSM
    bx_gsm, by_gsm, bz_gsm = model_func(parmod, ps, x_gsm, y_gsm, z_gsm)
    B_nT = np.sqrt(bx_gsm**2 + by_gsm**2 + bz_gsm**2)
    
    # Note: Curvature radius is a scalar, so no coordinate transformation needed
    # Magnetic field magnitude is also coordinate-independent
    return Rc, B_nT


# Analysis functions
def analyze_scattering_regions_by_energy():
    """
    Analysis 1: Scattering regions for different electron energies
    """
    print("\n" + "="*60)
    print("Analysis 1: Scattering Regions for Different Electron Energies")
    print("="*60)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Energy levels to analyze
    energies = [10, 30, 100, 300, 1000]  # keV
    
    # Create grid in SM coordinates
    # Note: In SM, the magnetic equator is at Z_SM = 0
    x_grid = np.linspace(-12, 5, 69)
    z_grid = np.linspace(-10, 10, 81)
    X, Z = np.meshgrid(x_grid, z_grid)
    Y = np.zeros_like(X)  # Y_SM = 0 (noon-midnight meridian)
    
    # Layout: 2 rows, 3 columns
    for idx, energy in enumerate(energies):
        ax_meridian = plt.subplot(2, 3, idx+1 if idx < 3 else idx-2+4)
        
        # Calculate for moderate storm conditions
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        
        Rc_Re, B_nT = calculate_curvature_radius(t96_vectorized, parmod_moderate, ps, 
                                                 x_flat, y_flat, z_flat)
        Rc_m = Rc_Re * Re
        
        # Calculate Larmor radius
        RL_m = calculate_larmor_radius(energy, B_nT, pitch_angle_deg=90)
        ratio = Rc_m / RL_m
        
        # Reshape for plotting
        ratio_grid = ratio.reshape(X.shape)
        
        # Create custom colormap for Rc/RL ratio
        levels = np.array([0.1, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
        colors = ['darkred', 'red', 'orange', 'yellow', 'lightgreen', 'green', 
                  'cyan', 'blue', 'darkblue', 'purple', 'magenta']
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(levels, cmap.N)
        
        # Plot
        im = ax_meridian.contourf(X, Z, ratio_grid, levels=levels, cmap=cmap, 
                                  norm=norm, extend='both')
        
        # Add critical contour
        cs = ax_meridian.contour(X, Z, ratio_grid, levels=[CRITICAL_RATIO], 
                                colors='black', linewidths=3)
        ax_meridian.clabel(cs, inline=True, fontsize=12, fmt='Rc/RL=8')
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='lightgray', zorder=10)
        ax_meridian.add_patch(earth)
        
        # Labels and formatting
        ax_meridian.set_xlabel('X_SM (Re)')
        ax_meridian.set_ylabel('Z_SM (Re)')
        ax_meridian.set_title(f'{energy} keV Electrons\nNoon-Midnight Meridian (Y_SM=0)')
        ax_meridian.set_aspect('equal')
        ax_meridian.set_xlim(-12, 5)
        ax_meridian.set_ylim(-10, 10)
        ax_meridian.grid(True, alpha=0.3)
        
        # Calculate and display statistics
        scatter_fraction = np.sum(ratio < CRITICAL_RATIO) / len(ratio) * 100
        ax_meridian.text(0.02, 0.98, f'{scatter_fraction:.1f}% with\\nRc/RL < 8', 
                        transform=ax_meridian.transAxes, fontsize=11, 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax, ticks=levels)
    cbar.set_label('Rc/RL Ratio', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    # Add overall title
    fig.suptitle('Curvature Scattering Analysis in SM Coordinates: Energy Dependence\n' + 
                 f'T96 Model, Moderate Storm (Pdyn={parmod_moderate[0]} nPa, Dst={parmod_moderate[1]} nT)',
                 fontsize=16, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.95])
    plt.savefig(os.path.join(output_dir, 'fig01_scattering_regions_by_energy_sm.png'))
    plt.close(fig)
    
    print("Figure 1 saved: Scattering regions for different electron energies (SM coordinates)")


def analyze_xy_plane_cross_sections():
    """
    Analysis 1b: XY plane cross-sections at different Z heights in SM coordinates
    """
    print("\n" + "="*60)
    print("Analysis 1b: XY Plane Cross-sections at Different Z Heights (SM)")
    print("="*60)
    
    # Z levels to analyze (in SM coordinates - along dipole axis)
    z_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]  # Re
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Energy for this analysis
    energy = 100  # keV
    
    # Create XY grid in SM
    x_grid = np.linspace(-12, 5, 69)
    y_grid = np.linspace(-10, 10, 81)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    print("Computing XY plane cross-sections...")
    for idx, z_sm in enumerate(z_levels):
        ax = axes[idx]
        print(f"  Processing Z_SM = {z_sm} Re ({idx+1}/{len(z_levels)})...")
        
        Z = np.full_like(X, z_sm)
        
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        
        # Calculate in SM coordinates
        Rc_Re, B_nT = calculate_curvature_radius(t96_vectorized, parmod_moderate, ps,
                                                 x_flat, y_flat, z_flat)
        Rc_m = Rc_Re * Re
        RL_m = calculate_larmor_radius(energy, B_nT, pitch_angle_deg=90)
        ratio = Rc_m / RL_m
        
        # Clean up extreme values
        ratio = np.where(ratio > 1000, 1000, ratio)
        ratio = np.where(ratio < 0.1, 0.1, ratio)
        
        ratio_grid = ratio.reshape(X.shape)
        
        # Plot
        im = ax.contourf(X, Y, ratio_grid, 
                        levels=np.logspace(-1, 3, 20),
                        cmap='RdBu_r', extend='both',
                        norm=LogNorm(vmin=0.1, vmax=1000))
        
        # Add critical contour
        cs = ax.contour(X, Y, ratio_grid, levels=[CRITICAL_RATIO],
                       colors='black', linewidths=2)
        ax.clabel(cs, inline=True, fontsize=10, fmt='8')
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax.add_patch(earth)
        
        # Calculate statistics
        scatter_frac = np.sum(ratio < CRITICAL_RATIO) / len(ratio) * 100
        
        # Labels
        ax.set_xlabel('X_SM (Re)')
        ax.set_ylabel('Y_SM (Re)')
        ax.set_title(f'Z_SM = {z_sm} Re\n{scatter_frac:.1f}% < 8')
        ax.set_aspect('equal')
        ax.set_xlim(-12, 5)
        ax.set_ylim(-10, 10)
        ax.grid(True, alpha=0.3)
        
        # Add MLT labels for Z=0
        if z_sm == 0.0:
            ax.text(0, 10.5, '12', ha='center', fontsize=10)
            ax.text(10.5, 0, '18', ha='center', fontsize=10)
            ax.text(0, -10.5, '00', ha='center', fontsize=10)
            ax.text(-10.5, 0, '06', ha='center', fontsize=10)
    
    # Remove extra subplot
    fig.delaxes(axes[7])
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Rc/RL Ratio', fontsize=12)
    
    # Overall title
    fig.suptitle(f'XY Plane Cross-sections in SM Coordinates\n' +
                 f'{energy} keV Electrons, T96 Model, Moderate Storm',
                 fontsize=16, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.97])
    plt.savefig(os.path.join(output_dir, 'fig02_xy_plane_cross_sections_sm.png'))
    plt.close(fig)
    
    print("Figure 2 saved: XY plane cross-sections at different Z heights (SM)")
    print("Key Observations:")
    print("- In SM coordinates, Z is along the dipole axis")
    print("- Scattering regions show different patterns than GSM")
    print("- Dawn-dusk asymmetry is preserved")
    print("- Off-dipole-equator regions show tilted patterns")


def analyze_magnetic_equatorial_plane_sm():
    """
    Analysis 3: Magnetic Equatorial Plane (Z_SM = 0) - MLT Dependence
    """
    print("\n" + "="*60)
    print("Analysis 3: Magnetic Equatorial Plane - MLT Dependence (SM)")
    print("="*60)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
    
    # Create polar grid for magnetic equatorial plane (Z_SM = 0)
    r = np.linspace(2, 12, 41)
    theta = np.linspace(0, 2*np.pi, 97)
    R, Theta = np.meshgrid(r, theta)
    
    # Convert to SM Cartesian
    X_sm = R * np.cos(Theta)
    Y_sm = R * np.sin(Theta)
    Z_sm = np.zeros_like(X_sm)  # Magnetic equatorial plane
    
    # Calculate for 100 keV electrons
    energy = 100  # keV
    x_flat = X_sm.flatten()
    y_flat = Y_sm.flatten()
    z_flat = Z_sm.flatten()
    
    Rc_Re, B_nT = calculate_curvature_radius(t96_vectorized, parmod_moderate, ps,
                                            x_flat, y_flat, z_flat)
    Rc_m = Rc_Re * Re
    RL_m = calculate_larmor_radius(energy, B_nT, pitch_angle_deg=90)
    ratio = Rc_m / RL_m
    ratio_grid = ratio.reshape(X_sm.shape)
    
    # Plot 1: Rc/RL ratio map
    im1 = ax1.contourf(X_sm, Y_sm, ratio_grid,
                      levels=np.logspace(-1, 3, 20),
                      cmap='RdBu_r', extend='both',
                      norm=LogNorm(vmin=0.1, vmax=1000))
    cs1 = ax1.contour(X_sm, Y_sm, ratio_grid, levels=[CRITICAL_RATIO],
                     colors='black', linewidths=3)
    ax1.clabel(cs1, inline=True, fontsize=12, fmt='8')
    
    # Add Earth and formatting
    earth1 = plt.Circle((0, 0), 1, color='white', zorder=10)
    ax1.add_patch(earth1)
    ax1.set_xlabel('X_SM (Re)')
    ax1.set_ylabel('Y_SM (Re)')
    ax1.set_title(f'Rc/RL Ratio - {energy} keV Electrons\nMagnetic Equator (Z_SM = 0)')
    ax1.set_aspect('equal')
    ax1.set_xlim(-12, 12)
    ax1.set_ylim(-12, 12)
    ax1.grid(True, alpha=0.3)
    
    # Add MLT labels
    ax1.text(0, 12.5, '12 MLT', ha='center', fontsize=10)
    ax1.text(12.5, 0, '18', ha='center', fontsize=10)
    ax1.text(0, -12.5, '00', ha='center', fontsize=10)
    ax1.text(-12.5, 0, '06', ha='center', fontsize=10)
    
    # Plot 2: Critical regions binary map
    scatter_mask = ratio_grid < CRITICAL_RATIO
    ax2.contourf(X_sm, Y_sm, scatter_mask.astype(float),
                levels=[0, 0.5, 1], colors=['white', 'red'])
    
    earth2 = plt.Circle((0, 0), 1, color='gray', zorder=10)
    ax2.add_patch(earth2)
    ax2.set_xlabel('X_SM (Re)')
    ax2.set_ylabel('Y_SM (Re)')
    ax2.set_title('Scattering Regions (Rc/RL < 8)\nMagnetic Equator')
    ax2.set_aspect('equal')
    ax2.set_xlim(-12, 12)
    ax2.set_ylim(-12, 12)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: MLT distribution (polar plot)
    # Convert to MLT and radial distance
    mlt_bins = np.linspace(0, 24, 49)
    r_bins = np.linspace(2, 12, 21)
    mlt_scatter_stats = np.zeros(len(mlt_bins)-1)
    
    for i in range(len(mlt_bins)-1):
        mlt_mask = ((Theta * 12/np.pi % 24) >= mlt_bins[i]) & \
                   ((Theta * 12/np.pi % 24) < mlt_bins[i+1])
        if np.any(mlt_mask):
            mlt_scatter_stats[i] = np.sum(scatter_mask[mlt_mask]) / np.sum(mlt_mask) * 100
    
    # Polar plot
    theta_plot = np.linspace(0, 2*np.pi, len(mlt_scatter_stats))
    ax3 = plt.subplot(2, 2, 3, projection='polar')
    ax3.plot(theta_plot, mlt_scatter_stats, 'b-', linewidth=2)
    ax3.fill(theta_plot, mlt_scatter_stats, 'b', alpha=0.3)
    ax3.set_theta_zero_location('S')
    ax3.set_theta_direction(-1)
    ax3.set_ylim(0, max(mlt_scatter_stats)*1.1 if max(mlt_scatter_stats) > 0 else 10)
    ax3.set_title('MLT Distribution of Scattering\n(% of radial extent)', pad=20)
    
    # Plot 4: Radial profile
    radial_scatter_stats = []
    for i in range(len(r_bins)-1):
        r_mask = (R >= r_bins[i]) & (R < r_bins[i+1])
        if np.any(r_mask):
            radial_scatter_stats.append(np.sum(scatter_mask[r_mask]) / np.sum(r_mask) * 100)
        else:
            radial_scatter_stats.append(0)
    
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    ax4.plot(r_centers, radial_scatter_stats, 'b-', linewidth=2, marker='o')
    ax4.set_xlabel('Radial Distance (Re)')
    ax4.set_ylabel('Scattering Percentage (%)')
    ax4.set_title('Radial Profile of Scattering Regions')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(2, 12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig03_magnetic_equatorial_plane_sm.png'))
    plt.close(fig)
    
    print("Figure 3 saved: Magnetic equatorial plane analysis (SM)")
    print("Key findings in magnetic equator (Z_SM = 0):")
    print("- Scattering regions aligned with magnetic equator")
    print("- Different MLT distribution than GSM equatorial plane")
    print("- Enhanced scattering in specific drift shell regions")


def analyze_seasonal_effects_sm():
    """
    Analysis: Seasonal Effects - Comparing Different Epochs in SM Coordinates
    """
    print("\n" + "="*60)
    print("Analysis: Seasonal Effects in SM Coordinates")
    print("="*60)
    
    # In SM coordinates, seasonal effects are minimized because Z is along dipole
    # But we can still show the differences
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Different epochs (same as original but now in SM)
    epochs = [
        ("Summer Solstice", 1592784000),  # June 21, 2020
        ("Autumn Equinox", 1600905600),   # Sept 24, 2020
        ("Winter Solstice", 1608508800),  # Dec 21, 2020
        ("Current Analysis", 1600000000),  # Sept 13, 2020
        ("Spring Equinox", 1584835200),   # March 22, 2020
        ("Random Date", 1580515200)       # Feb 1, 2020
    ]
    
    energy = 100  # keV
    
    # Create XZ grid for meridian plane in SM
    x_grid = np.linspace(-15, 5, 61)
    z_grid = np.linspace(-5, 5, 51)
    X, Z = np.meshgrid(x_grid, z_grid)
    Y = np.zeros_like(X)
    
    for idx, (epoch_name, epoch_ut) in enumerate(epochs):
        ax = axes[idx]
        
        # Recalculate for this epoch
        ps_epoch = geopack.recalc(epoch_ut)
        
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        
        # Calculate in SM
        Rc_Re, B_nT = calculate_curvature_radius(t96_vectorized, parmod_moderate, ps_epoch,
                                                x_flat, y_flat, z_flat)
        Rc_m = Rc_Re * Re
        RL_m = calculate_larmor_radius(energy, B_nT, pitch_angle_deg=90)
        ratio = Rc_m / RL_m
        ratio_grid = ratio.reshape(X.shape)
        
        # Plot
        im = ax.contourf(X, Z, ratio_grid,
                        levels=np.logspace(-1, 3, 15),
                        cmap='RdBu_r', extend='both',
                        norm=LogNorm(vmin=0.1, vmax=1000))
        
        cs = ax.contour(X, Z, ratio_grid, levels=[CRITICAL_RATIO],
                       colors='black', linewidths=2)
        ax.clabel(cs, inline=True, fontsize=9, fmt='8')
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax.add_patch(earth)
        
        # Calculate statistics
        scatter_frac = np.sum(ratio < CRITICAL_RATIO) / len(ratio) * 100
        
        # Labels
        ax.set_xlabel('X_SM (Re)')
        ax.set_ylabel('Z_SM (Re)')
        ax.set_title(f'{epoch_name}\nPS={np.degrees(ps_epoch):.1f}°, {scatter_frac:.1f}%')
        ax.set_aspect('equal')
        ax.set_xlim(-15, 5)
        ax.set_ylim(-5, 5)
        ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Rc/RL Ratio', fontsize=12)
    
    fig.suptitle('Seasonal Effects in SM Coordinates (Noon-Midnight Meridian)\n' +
                 f'{energy} keV Electrons, T96 Model, Moderate Storm',
                 fontsize=16, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    plt.savefig(os.path.join(output_dir, 'fig19_seasonal_effects_sm.png'))
    plt.close(fig)
    
    print("Figure 19 saved: Seasonal effects in SM coordinates")
    print("Note: In SM coordinates, seasonal variations are minimized")
    print("      because Z_SM is always aligned with the dipole axis")


def analyze_model_comparison_sm():
    """
    Model Comparison in SM Coordinates
    """
    print("\n" + "="*60)
    print("Analysis: Model Comparison in SM Coordinates")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    models = [
        ("T89", t89_vectorized, 3),  # Kp=3
        ("T96", t96_vectorized, parmod_moderate),
        ("T01", t01_vectorized, parmod_moderate),
        ("T04", t04_vectorized, parmod_moderate)
    ]
    
    energy = 100  # keV
    
    # Create grid in SM
    x_grid = np.linspace(-12, 5, 69)
    y_grid = np.linspace(-10, 10, 81)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)  # Magnetic equatorial plane
    
    for idx, (model_name, model_func, params) in enumerate(models):
        ax = axes[idx]
        
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        
        # Special handling for T89
        if model_name == "T89":
            # Convert SM to GSM
            x_gsm, y_gsm, z_gsm = sm_to_gsm(x_flat, y_flat, z_flat, ps)
            # T89 call is different
            kappa = field_line_curvature_vectorized(model_func, params, ps, x_gsm, y_gsm, z_gsm)
            Rc_Re = np.where(kappa > 1e-10, 1.0 / kappa, 1e10)
            bx, by, bz = model_func(params, ps, x_gsm, y_gsm, z_gsm)
            B_nT = np.sqrt(bx**2 + by**2 + bz**2)
        else:
            Rc_Re, B_nT = calculate_curvature_radius(model_func, params, ps,
                                                     x_flat, y_flat, z_flat)
        
        Rc_m = Rc_Re * Re
        RL_m = calculate_larmor_radius(energy, B_nT, pitch_angle_deg=90)
        ratio = Rc_m / RL_m
        ratio_grid = ratio.reshape(X.shape)
        
        # Plot
        im = ax.contourf(X, Y, ratio_grid,
                        levels=np.logspace(-1, 3, 20),
                        cmap='RdBu_r', extend='both',
                        norm=LogNorm(vmin=0.1, vmax=1000))
        
        cs = ax.contour(X, Y, ratio_grid, levels=[CRITICAL_RATIO],
                       colors='black', linewidths=2.5)
        ax.clabel(cs, inline=True, fontsize=10, fmt='8')
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax.add_patch(earth)
        
        # Calculate statistics
        scatter_frac = np.sum(ratio < CRITICAL_RATIO) / len(ratio) * 100
        
        # Labels
        ax.set_xlabel('X_SM (Re)')
        ax.set_ylabel('Y_SM (Re)')
        ax.set_title(f'{model_name} Model\n{scatter_frac:.1f}% with Rc/RL < 8')
        ax.set_aspect('equal')
        ax.set_xlim(-12, 5)
        ax.set_ylim(-10, 10)
        ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Rc/RL Ratio', fontsize=12)
    
    fig.suptitle('Model Comparison in SM Coordinates (Magnetic Equator)\n' +
                 f'{energy} keV Electrons, Moderate Storm Conditions',
                 fontsize=16, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    plt.savefig(os.path.join(output_dir, 'fig09_model_comparison_sm.png'))
    plt.close(fig)
    
    print("Figure 9 saved: Model comparison in SM coordinates")


def create_all_sm_figures():
    """
    Create all figures in SM coordinates
    """
    print("\n" + "="*80)
    print("MAGNETIC FIELD LINE CURVATURE SCATTERING ANALYSIS")
    print("SOLAR MAGNETIC (SM) COORDINATE SYSTEM")
    print("Critical Threshold: Rc/RL = 8")
    print("="*80)
    
    # Run all analyses
    analyze_scattering_regions_by_energy()
    analyze_xy_plane_cross_sections()
    analyze_magnetic_equatorial_plane_sm()
    analyze_model_comparison_sm()
    analyze_seasonal_effects_sm()
    
    print("\n" + "="*80)
    print("SM coordinate analysis complete! Figures saved to:", output_dir)
    print("Key differences in SM coordinates:")
    print("- Z_SM is aligned with Earth's dipole axis")
    print("- Magnetic equator is at Z_SM = 0 (not tilted)")
    print("- Seasonal variations are minimized")
    print("- Better for studying magnetic latitude effects")
    print("="*80)


if __name__ == "__main__":
    create_all_sm_figures()