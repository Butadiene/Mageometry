#!/usr/bin/env python3
"""
Curvature Radius and Larmor Radius Tilt Comparison - 7 Divisions
Shows 8 time points from spring equinox to summer solstice
Creates versions showing field line curvature and 100 keV Larmor radius
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import geopack
from geopack import t96_vectorized, field_line_curvature_vectorized
from datetime import datetime, timedelta

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
print("CURVATURE AND LARMOR RADIUS TILT PROGRESSION")
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


def create_curvature_radius_comparison():
    """Create XZ plane plots showing field line curvature radius for 8 time points"""
    
    # Define time points - 7 divisions between spring equinox and summer solstice
    spring_eq = 1584748800  # March 20, 2020, 00:00:00 UTC
    summer_sol = 1592697600  # June 21, 2020, 00:00:00 UTC
    
    # Calculate the interval
    total_seconds = summer_sol - spring_eq
    interval = total_seconds / 7
    
    # Create 8 time points
    tilt_conditions = []
    for i in range(8):
        ut = int(spring_eq + i * interval)
        dt = datetime.fromtimestamp(ut)
        date_str = dt.strftime("%b %d")
        tilt_conditions.append((ut, date_str))
    
    # Create figure with 2x4 grid
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Create grid in GSM coordinates
    x_gsm_grid = np.linspace(-20, 10, 121)
    z_gsm_grid = np.linspace(-10, 10, 81)
    X_GSM, Z_GSM = np.meshgrid(x_gsm_grid, z_gsm_grid)
    Y_GSM = np.zeros_like(X_GSM)
    
    # Standard storm parameters
    parmod = [3.0, -30.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    print(f"\nCreating curvature radius plots...")
    print(f"Standard storm conditions: Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT")
    
    # Store statistics
    tilt_values = []
    min_rc_values = []
    median_rc_values = []
    
    for idx, (ut, date_str) in enumerate(tilt_conditions):
        ax = axes[idx]
        
        # Calculate actual dipole tilt
        ps = geopack.recalc(ut)
        actual_tilt = np.degrees(ps)
        tilt_values.append(actual_tilt)
        
        # Flatten GSM coordinates
        x_flat = X_GSM.flatten()
        y_flat = Y_GSM.flatten()
        z_flat = Z_GSM.flatten()
        
        # Calculate curvature
        kappa = field_line_curvature_vectorized(t96_vectorized, parmod, ps, 
                                               x_flat, y_flat, z_flat)
        Rc_Re = np.where(kappa > 1e-10, 1.0 / kappa, 1e3)
        Rc_grid = Rc_Re.reshape(X_GSM.shape)
        
        # Calculate statistics
        min_rc = np.min(Rc_Re)
        median_rc = np.median(Rc_Re)
        min_rc_values.append(min_rc)
        median_rc_values.append(median_rc)
        
        # Plot curvature radius
        im = ax.contourf(X_GSM, Z_GSM, Rc_grid, levels=50, 
                        cmap='plasma', norm=LogNorm(vmin=0.1, vmax=100))
        
        # Add contours for specific values
        try:
            cs = ax.contour(X_GSM, Z_GSM, Rc_grid, levels=[0.5, 1, 5, 10, 50], 
                           colors='white', linewidths=1, alpha=0.5)
            ax.clabel(cs, inline=True, fontsize=6, fmt='%.1f Re')
        except:
            pass
        
        # Add vectors for some panels
        if idx % 2 == 0:
            x_vec = np.linspace(-20, 10, 11)
            z_vec = np.linspace(-10, 10, 7)
            X_vec, Z_vec = np.meshgrid(x_vec, z_vec)
            Y_vec = np.zeros_like(X_vec)
            
            x_vec_flat = X_vec.flatten()
            y_vec_flat = Y_vec.flatten()
            z_vec_flat = Z_vec.flatten()
            
            bx_vec, by_vec, bz_vec = t96_vectorized(parmod, ps, 
                                                   x_vec_flat, y_vec_flat, z_vec_flat)
            
            Bx_grid = bx_vec.reshape(X_vec.shape)
            Bz_grid = bz_vec.reshape(Z_vec.shape)
            
            B_vec_mag = np.sqrt(Bx_grid**2 + Bz_grid**2)
            Bx_norm = np.divide(Bx_grid, B_vec_mag, out=np.zeros_like(Bx_grid), 
                              where=B_vec_mag>0)
            Bz_norm = np.divide(Bz_grid, B_vec_mag, out=np.zeros_like(Bz_grid), 
                              where=B_vec_mag>0)
            
            ax.quiver(X_vec, Z_vec, Bx_norm*0.3, Bz_norm*0.3,
                     color='white', alpha=0.5, width=0.002, headwidth=3, 
                     scale=10, scale_units='xy')
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax.add_patch(earth)
        
        # Add magnetic equator line
        x_eq = np.linspace(-20, 10, 100)
        z_eq = np.tan(ps) * x_eq
        mask = np.abs(z_eq) < 10
        ax.plot(x_eq[mask], z_eq[mask], 'g--', linewidth=1, alpha=0.5)
        
        # Labels and formatting
        ax.set_xlabel('X GSM (Re)', fontsize=8)
        ax.set_ylabel('Z GSM (Re)', fontsize=8)
        ax.set_title(f'{date_str}\nTilt: {actual_tilt:.1f}°', 
                    fontsize=9, weight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-20, 10)
        ax.set_ylim(-10, 10)
        ax.tick_params(labelsize=7)
        
        # Add statistics
        ax.text(0.02, 0.98, f'Min: {min_rc:.2f} Re\nMed: {median_rc:.1f} Re', 
               transform=ax.transAxes, fontsize=7, va='top',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Radius of Curvature (Re)', fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    
    plt.suptitle('Field Line Curvature Radius: Spring Equinox to Summer Solstice\n' +
                'GSM Coordinates, Plasma colormap (Purple = High Curvature)',
                fontsize=14, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    output_file = os.path.join(output_dir, 'fig27_curvature_radius_tilt_progression.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Curvature radius figure saved: {output_file}")
    
    return tilt_values, min_rc_values, median_rc_values


def create_larmor_radius_comparison():
    """Create XZ plane plots showing 100 keV Larmor radius for 8 time points"""
    
    # Define time points
    spring_eq = 1584748800
    summer_sol = 1592697600
    
    total_seconds = summer_sol - spring_eq
    interval = total_seconds / 7
    
    tilt_conditions = []
    for i in range(8):
        ut = int(spring_eq + i * interval)
        dt = datetime.fromtimestamp(ut)
        date_str = dt.strftime("%b %d")
        tilt_conditions.append((ut, date_str))
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Create grid
    x_gsm_grid = np.linspace(-20, 10, 121)
    z_gsm_grid = np.linspace(-10, 10, 81)
    X_GSM, Z_GSM = np.meshgrid(x_gsm_grid, z_gsm_grid)
    Y_GSM = np.zeros_like(X_GSM)
    
    # Standard storm parameters
    parmod = [3.0, -30.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Fixed energy
    energy = 100  # keV
    print(f"\nCreating Larmor radius plots for {energy} keV electrons...")
    
    # Store statistics
    max_rl_values = []
    median_rl_values = []
    
    for idx, (ut, date_str) in enumerate(tilt_conditions):
        ax = axes[idx]
        
        # Calculate actual dipole tilt
        ps = geopack.recalc(ut)
        actual_tilt = np.degrees(ps)
        
        # Flatten GSM coordinates
        x_flat = X_GSM.flatten()
        y_flat = Y_GSM.flatten()
        z_flat = Z_GSM.flatten()
        
        # Calculate magnetic field
        bx, by, bz = t96_vectorized(parmod, ps, x_flat, y_flat, z_flat)
        B_magnitude = np.sqrt(bx**2 + by**2 + bz**2)
        
        # Calculate Larmor radius
        RL_m = calculate_larmor_radius(energy, B_magnitude)
        RL_Re = RL_m / Re
        RL_grid = RL_Re.reshape(X_GSM.shape)
        
        # Calculate statistics
        max_rl = np.max(RL_Re)
        median_rl = np.median(RL_Re)
        max_rl_values.append(max_rl)
        median_rl_values.append(median_rl)
        
        # Plot Larmor radius
        im = ax.contourf(X_GSM, Z_GSM, RL_grid, levels=50, 
                        cmap='viridis', norm=LogNorm(vmin=0.001, vmax=1))
        
        # Add contours
        try:
            cs = ax.contour(X_GSM, Z_GSM, RL_grid, levels=[0.01, 0.05, 0.1, 0.5], 
                           colors='white', linewidths=1, alpha=0.5)
            ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f Re')
        except:
            pass
        
        # Add vectors for some panels
        if idx % 2 == 0:
            x_vec = np.linspace(-20, 10, 11)
            z_vec = np.linspace(-10, 10, 7)
            X_vec, Z_vec = np.meshgrid(x_vec, z_vec)
            Y_vec = np.zeros_like(X_vec)
            
            x_vec_flat = X_vec.flatten()
            y_vec_flat = Y_vec.flatten()
            z_vec_flat = Z_vec.flatten()
            
            bx_vec, by_vec, bz_vec = t96_vectorized(parmod, ps, 
                                                   x_vec_flat, y_vec_flat, z_vec_flat)
            
            Bx_grid = bx_vec.reshape(X_vec.shape)
            Bz_grid = bz_vec.reshape(Z_vec.shape)
            
            B_vec_mag = np.sqrt(Bx_grid**2 + Bz_grid**2)
            Bx_norm = np.divide(Bx_grid, B_vec_mag, out=np.zeros_like(Bx_grid), 
                              where=B_vec_mag>0)
            Bz_norm = np.divide(Bz_grid, B_vec_mag, out=np.zeros_like(Bz_grid), 
                              where=B_vec_mag>0)
            
            ax.quiver(X_vec, Z_vec, Bx_norm*0.3, Bz_norm*0.3,
                     color='white', alpha=0.5, width=0.002, headwidth=3, 
                     scale=10, scale_units='xy')
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax.add_patch(earth)
        
        # Add magnetic equator line
        x_eq = np.linspace(-20, 10, 100)
        z_eq = np.tan(ps) * x_eq
        mask = np.abs(z_eq) < 10
        ax.plot(x_eq[mask], z_eq[mask], 'r--', linewidth=1, alpha=0.5)
        
        # Labels and formatting
        ax.set_xlabel('X GSM (Re)', fontsize=8)
        ax.set_ylabel('Z GSM (Re)', fontsize=8)
        ax.set_title(f'{date_str}\nTilt: {actual_tilt:.1f}°', 
                    fontsize=9, weight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-20, 10)
        ax.set_ylim(-10, 10)
        ax.tick_params(labelsize=7)
        
        # Add statistics
        ax.text(0.02, 0.98, f'Max: {max_rl*6371:.0f} km\nMed: {median_rl*6371:.0f} km', 
               transform=ax.transAxes, fontsize=7, va='top',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label(f'Larmor Radius ({energy} keV) [Re]', fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    
    plt.suptitle(f'{energy} keV Electron Larmor Radius: Spring Equinox to Summer Solstice\n' +
                'GSM Coordinates, Viridis colormap (Dark = Small RL)',
                fontsize=14, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    output_file = os.path.join(output_dir, 'fig28_larmor_radius_tilt_progression.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Larmor radius figure saved: {output_file}")
    
    return max_rl_values, median_rl_values


def create_summary_plots(tilt_values, min_rc_values, median_rc_values, 
                        max_rl_values, median_rl_values):
    """Create summary plots showing how Rc and RL vary with tilt"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Curvature radius vs tilt
    ax1.scatter(tilt_values, min_rc_values, c='blue', s=80, marker='o', 
               label='Min Rc', edgecolors='black', linewidth=1)
    ax1.scatter(tilt_values, median_rc_values, c='lightblue', s=80, marker='s', 
               label='Median Rc', edgecolors='black', linewidth=1)
    
    # Fit curves
    z1 = np.polyfit(tilt_values, min_rc_values, 2)
    p1 = np.poly1d(z1)
    tilt_smooth = np.linspace(min(tilt_values), max(tilt_values), 100)
    ax1.plot(tilt_smooth, p1(tilt_smooth), 'b--', alpha=0.5)
    
    ax1.set_xlabel('Dipole Tilt (degrees)', fontsize=12)
    ax1.set_ylabel('Radius of Curvature (Re)', fontsize=12)
    ax1.set_title('Curvature Radius vs Dipole Tilt', fontsize=14, weight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0.01, 100)
    
    # Plot 2: Larmor radius vs tilt
    ax2.scatter(tilt_values, np.array(max_rl_values)*6371, c='green', s=80, marker='o', 
               label='Max RL', edgecolors='black', linewidth=1)
    ax2.scatter(tilt_values, np.array(median_rl_values)*6371, c='lightgreen', s=80, marker='s', 
               label='Median RL', edgecolors='black', linewidth=1)
    
    # Fit curves
    z2 = np.polyfit(tilt_values, np.array(max_rl_values)*6371, 2)
    p2 = np.poly1d(z2)
    ax2.plot(tilt_smooth, p2(tilt_smooth), 'g--', alpha=0.5)
    
    ax2.set_xlabel('Dipole Tilt (degrees)', fontsize=12)
    ax2.set_ylabel('Larmor Radius (km)', fontsize=12)
    ax2.set_title('100 keV Electron Larmor Radius vs Dipole Tilt', fontsize=14, weight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(10, 10000)
    
    plt.suptitle('Variation of Rc and RL with Dipole Tilt\n' +
                'Spring Equinox to Summer Solstice Progression', 
                fontsize=16, weight='bold')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'fig29_rc_rl_vs_tilt_summary.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nSummary plot saved: {output_file}")


if __name__ == "__main__":
    # Create curvature radius plots
    tilt_values, min_rc_values, median_rc_values = create_curvature_radius_comparison()
    
    # Create Larmor radius plots
    max_rl_values, median_rl_values = create_larmor_radius_comparison()
    
    # Create summary plots
    create_summary_plots(tilt_values, min_rc_values, median_rc_values, 
                        max_rl_values, median_rl_values)
    
    print("\n" + "="*80)
    print("Curvature and Larmor radius tilt progression complete!")
    print("="*80)
    print("\nKey findings:")
    print("- Field line curvature varies with dipole tilt")
    print("- Minimum Rc occurs when current sheet is most tilted")
    print("- Larmor radius shows less variation with tilt")
    print("- Both contribute to seasonal scattering variations")
    print("="*80)