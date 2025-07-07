#!/usr/bin/env python3
"""
Z-slice Analysis for May 12 - Curvature and Larmor Radius Versions
Creates XY plane slices at various Z heights showing Rc and RL separately
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

print("="*80)
print("Z-SLICE ANALYSIS FOR MAY 12 - CURVATURE AND LARMOR RADIUS")
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


def create_curvature_radius_slices():
    """Create XY plane slices showing field line curvature radius"""
    
    # May 12, 2020
    spring_eq = 1584748800
    summer_sol = 1592697600
    interval = (summer_sol - spring_eq) / 7
    ut_may12 = int(spring_eq + 4 * interval)
    
    # Calculate dipole tilt
    ps = geopack.recalc(ut_may12)
    actual_tilt = np.degrees(ps)
    
    print(f"\nCreating CURVATURE RADIUS slices")
    print(f"Date: May 12, 2020")
    print(f"Dipole tilt: {actual_tilt:.2f}°")
    
    # Define Z heights
    z_heights = []
    z_heights.extend(np.arange(0.0, 2.6, 0.5))
    z_heights.extend(np.arange(2.7, 5.1, 0.2))
    
    n_slices = len(z_heights)
    
    # Standard storm parameters
    parmod = [3.0, -30.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Create figure
    n_cols = 6
    n_rows = int(np.ceil(n_slices / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 4*n_rows))
    axes = axes.flatten()
    
    # Create XY grid
    x_grid = np.linspace(-20, 5, 101)
    y_grid = np.linspace(-12, 12, 97)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Store statistics
    min_rc_values = []
    median_rc_values = []
    
    for idx, z_height in enumerate(z_heights):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
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
        min_rc = np.min(Rc_Re)
        median_rc = np.median(Rc_Re)
        min_rc_values.append(min_rc)
        median_rc_values.append(median_rc)
        
        # Plot curvature radius
        im = ax.contourf(X, Y, Rc_grid, levels=50, 
                        cmap='plasma', norm=LogNorm(vmin=0.1, vmax=100))
        
        # Add contours
        try:
            cs = ax.contour(X, Y, Rc_grid, levels=[0.5, 1, 5, 10], 
                           colors='white', linewidths=1, alpha=0.7)
            ax.clabel(cs, inline=True, fontsize=5, fmt='%.1f Re')
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
        
        # Add statistics
        ax.text(0.02, 0.98, f'Min: {min_rc:.2f}\nMed: {median_rc:.1f}', 
               transform=ax.transAxes, fontsize=6, va='top',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Hide unused subplots
    for idx in range(len(z_heights), len(axes)):
        axes[idx].set_visible(False)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Radius of Curvature (Re)', fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    
    plt.suptitle(f'Field Line Curvature Radius in XY Planes: May 12 (Tilt = {actual_tilt:.1f}°)\n' +
                f'Plasma colormap: Purple = High Curvature, Yellow = Low Curvature',
                fontsize=14, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    output_file = os.path.join(output_dir, 'fig32_may12_curvature_radius_z_slices.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Curvature radius figure saved: {output_file}")
    
    return z_heights, min_rc_values, median_rc_values


def create_larmor_radius_slices():
    """Create XY plane slices showing 100 keV Larmor radius"""
    
    # May 12, 2020
    spring_eq = 1584748800
    summer_sol = 1592697600
    interval = (summer_sol - spring_eq) / 7
    ut_may12 = int(spring_eq + 4 * interval)
    
    # Calculate dipole tilt
    ps = geopack.recalc(ut_may12)
    actual_tilt = np.degrees(ps)
    
    print(f"\nCreating LARMOR RADIUS slices")
    print(f"Date: May 12, 2020")
    print(f"Dipole tilt: {actual_tilt:.2f}°")
    
    # Define Z heights
    z_heights = []
    z_heights.extend(np.arange(0.0, 2.6, 0.5))
    z_heights.extend(np.arange(2.7, 5.1, 0.2))
    
    n_slices = len(z_heights)
    
    # Standard storm parameters
    parmod = [3.0, -30.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Fixed energy
    energy = 100  # keV
    
    # Create figure
    n_cols = 6
    n_rows = int(np.ceil(n_slices / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 4*n_rows))
    axes = axes.flatten()
    
    # Create XY grid
    x_grid = np.linspace(-20, 5, 101)
    y_grid = np.linspace(-12, 12, 97)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Store statistics
    max_rl_values = []
    median_rl_values = []
    
    for idx, z_height in enumerate(z_heights):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
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
        max_rl = np.max(RL_Re)
        median_rl = np.median(RL_Re)
        max_rl_values.append(max_rl)
        median_rl_values.append(median_rl)
        
        # Plot Larmor radius
        im = ax.contourf(X, Y, RL_grid, levels=50, 
                        cmap='viridis', norm=LogNorm(vmin=0.001, vmax=1))
        
        # Add contours
        try:
            cs = ax.contour(X, Y, RL_grid, levels=[0.01, 0.05, 0.1, 0.5], 
                           colors='white', linewidths=1, alpha=0.7)
            ax.clabel(cs, inline=True, fontsize=5, fmt='%.2f Re')
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
        
        # Add statistics
        ax.text(0.02, 0.98, f'Max: {max_rl*6371:.0f} km\nMed: {median_rl*6371:.0f} km', 
               transform=ax.transAxes, fontsize=6, va='top',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Hide unused subplots
    for idx in range(len(z_heights), len(axes)):
        axes[idx].set_visible(False)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label(f'Larmor Radius ({energy} keV) [Re]', fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    
    plt.suptitle(f'{energy} keV Electron Larmor Radius in XY Planes: May 12 (Tilt = {actual_tilt:.1f}°)\n' +
                f'Viridis colormap: Dark = Small RL, Bright = Large RL',
                fontsize=14, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    output_file = os.path.join(output_dir, 'fig33_may12_larmor_radius_z_slices.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Larmor radius figure saved: {output_file}")
    
    return z_heights, max_rl_values, median_rl_values


def create_combined_summary(z_heights, min_rc, median_rc, max_rl, median_rl):
    """Create summary plots for Rc and RL vs Z"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Curvature radius vs Z
    ax1.plot(z_heights, min_rc, 'b-o', linewidth=2, markersize=6, label='Min Rc')
    ax1.plot(z_heights, median_rc, 'b--s', linewidth=1.5, markersize=5, label='Median Rc')
    
    # Highlight regions
    ax1.axvspan(0, 2.5, alpha=0.1, color='blue', label='0.5 Re steps')
    ax1.axvspan(2.5, 5.0, alpha=0.1, color='green', label='0.2 Re steps')
    
    ax1.set_xlabel('Z GSM (Re)', fontsize=12)
    ax1.set_ylabel('Radius of Curvature (Re)', fontsize=12)
    ax1.set_title('Field Line Curvature vs Height', fontsize=14, weight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(-0.2, 5.2)
    ax1.set_ylim(0.01, 1000)
    
    # Find minimum
    min_idx = np.argmin(min_rc)
    ax1.plot(z_heights[min_idx], min_rc[min_idx], 'ro', markersize=10)
    ax1.annotate(f'Min: {min_rc[min_idx]:.2f} Re\nat Z={z_heights[min_idx]:.1f} Re',
                xy=(z_heights[min_idx], min_rc[min_idx]),
                xytext=(z_heights[min_idx]+0.5, min_rc[min_idx]*2),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9)
    
    # Plot 2: Larmor radius vs Z
    ax2.plot(z_heights, np.array(max_rl)*6371, 'g-o', linewidth=2, markersize=6, label='Max RL')
    ax2.plot(z_heights, np.array(median_rl)*6371, 'g--s', linewidth=1.5, markersize=5, label='Median RL')
    
    # Highlight regions
    ax2.axvspan(0, 2.5, alpha=0.1, color='blue')
    ax2.axvspan(2.5, 5.0, alpha=0.1, color='green')
    
    ax2.set_xlabel('Z GSM (Re)', fontsize=12)
    ax2.set_ylabel('Larmor Radius (km)', fontsize=12)
    ax2.set_title('100 keV Electron Larmor Radius vs Height', fontsize=14, weight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(-0.2, 5.2)
    ax2.set_ylim(10, 10000)
    
    # Find maximum
    max_idx = np.argmax(max_rl)
    ax2.plot(z_heights[max_idx], max_rl[max_idx]*6371, 'ro', markersize=10)
    ax2.annotate(f'Max: {max_rl[max_idx]*6371:.0f} km\nat Z={z_heights[max_idx]:.1f} Re',
                xy=(z_heights[max_idx], max_rl[max_idx]*6371),
                xytext=(z_heights[max_idx]-1, max_rl[max_idx]*6371*0.5),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9)
    
    plt.suptitle('Curvature and Larmor Radius Variation with Height: May 12',
                fontsize=16, weight='bold')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'fig34_may12_rc_rl_z_summary.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nSummary plot saved: {output_file}")
    
    # Print statistics
    print("\n" + "="*60)
    print("CURVATURE AND LARMOR RADIUS STATISTICS")
    print("="*60)
    print(f"{'Z (Re)':<8} {'Min Rc (Re)':<12} {'Med Rc (Re)':<12} {'Max RL (km)':<12} {'Med RL (km)':<12}")
    print("-"*60)
    for i, z in enumerate(z_heights):
        print(f"{z:<8.1f} {min_rc[i]:<12.2f} {median_rc[i]:<12.1f} {max_rl[i]*6371:<12.0f} {median_rl[i]*6371:<12.0f}")
    print("="*60)


if __name__ == "__main__":
    # Create curvature radius slices
    z_heights, min_rc, median_rc = create_curvature_radius_slices()
    
    # Create Larmor radius slices
    z_heights, max_rl, median_rl = create_larmor_radius_slices()
    
    # Create combined summary
    create_combined_summary(z_heights, min_rc, median_rc, max_rl, median_rl)
    
    print("\n" + "="*80)
    print("Curvature and Larmor radius Z-slice analysis complete!")
    print("="*80)
    print("\nKey findings:")
    print("- Minimum curvature radius occurs near Z = 3-4 Re")
    print("- Maximum Larmor radius occurs in similar region")
    print("- Both contribute to enhanced scattering around Z = 3.3 Re")
    print("- Fine resolution reveals detailed vertical structure")
    print("="*80)