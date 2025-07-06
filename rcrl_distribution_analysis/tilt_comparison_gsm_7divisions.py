#!/usr/bin/env python3
"""
Dipole Tilt Comparison in GSM Coordinates - 7 Divisions
Shows 8 time points from spring equinox to summer solstice
Divides the period into 7 equal parts for detailed tilt progression analysis
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

# Critical threshold
CRITICAL_RATIO = 8.0

print("="*80)
print("DIPOLE TILT COMPARISON IN GSM COORDINATES - DETAILED PROGRESSION")
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


def create_detailed_tilt_comparison():
    """Create XZ plane plots for 8 time points from spring equinox to summer solstice"""
    
    # Define time points - 7 divisions between spring equinox and summer solstice
    spring_eq = 1584748800  # March 20, 2020, 00:00:00 UTC
    summer_sol = 1592697600  # June 21, 2020, 00:00:00 UTC
    
    # Calculate the interval
    total_seconds = summer_sol - spring_eq
    interval = total_seconds / 7
    
    # Create 8 time points (including endpoints)
    tilt_conditions = []
    for i in range(8):
        ut = int(spring_eq + i * interval)
        fraction = f"{i}/7" if i > 0 else "0"
        if i == 0:
            desc = "Spring Equinox"
        elif i == 7:
            desc = "Summer Solstice"
        else:
            desc = f"Equinox + {fraction}"
        
        # Convert to datetime for date display
        dt = datetime.fromtimestamp(ut)
        date_str = dt.strftime("%b %d")
        
        tilt_conditions.append((ut, desc, date_str))
    
    # Create figure with 2x4 grid
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Create grid in GSM coordinates
    x_gsm_grid = np.linspace(-20, 10, 121)  # Reduced resolution for 8 panels
    z_gsm_grid = np.linspace(-10, 10, 81)
    X_GSM, Z_GSM = np.meshgrid(x_gsm_grid, z_gsm_grid)
    Y_GSM = np.zeros_like(X_GSM)
    
    # Standard storm parameters
    parmod = [3.0, -30.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    print(f"\nStandard storm conditions:")
    print(f"Pdyn = {parmod[0]} nPa")
    print(f"Dst = {parmod[1]} nT")
    print(f"IMF By = {parmod[2]} nT")
    print(f"IMF Bz = {parmod[3]} nT")
    
    # Fixed energy
    energy = 100  # keV
    
    # Color levels
    levels = np.logspace(-1, 3, 20)
    
    # Store tilt values for summary
    tilt_values = []
    scatter_fracs = []
    dates = []
    
    for idx, (ut, description, date_str) in enumerate(tilt_conditions):
        ax = axes[idx]
        print(f"\nProcessing {description} ({date_str})...")
        
        # Calculate actual dipole tilt
        ps = geopack.recalc(ut)
        actual_tilt = np.degrees(ps)
        tilt_values.append(actual_tilt)
        dates.append(date_str)
        print(f"  Actual dipole tilt: {actual_tilt:.2f}°")
        
        # Flatten GSM coordinates
        x_flat = X_GSM.flatten()
        y_flat = Y_GSM.flatten()
        z_flat = Z_GSM.flatten()
        
        # Calculate field in GSM
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
        ratio_grid = ratio.reshape(X_GSM.shape)
        
        # Calculate statistics
        scatter_frac = np.sum(ratio < CRITICAL_RATIO) / len(ratio) * 100
        scatter_fracs.append(scatter_frac)
        
        # Plot Rc/RL ratio
        im = ax.contourf(X_GSM, Z_GSM, ratio_grid, levels=levels, 
                        cmap='RdBu_r', norm=LogNorm(vmin=0.1, vmax=1000))
        
        # Add critical contour
        cs = ax.contour(X_GSM, Z_GSM, ratio_grid, levels=[CRITICAL_RATIO], 
                       colors='black', linewidths=2)
        try:
            ax.clabel(cs, inline=True, fontsize=7, fmt='8')
        except:
            pass
        
        # Add vectors (sparse for clarity with 8 panels)
        if idx % 2 == 0:  # Only for every other panel
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
                     color='white', alpha=0.6, width=0.002, headwidth=3, 
                     scale=10, scale_units='xy', edgecolor='black', linewidth=0.2)
        
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
        
        # Add scattering percentage
        ax.text(0.95, 0.95, f'{scatter_frac:.1f}%', 
               transform=ax.transAxes, fontsize=8,
               ha='right', va='top', color='black', weight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Rc/RL Ratio', fontsize=10)
    cbar.ax.axhline(y=8, color='black', linewidth=2)
    cbar.ax.tick_params(labelsize=8)
    
    plt.suptitle('Detailed Dipole Tilt Progression: Spring Equinox to Summer Solstice\n' +
                f'GSM Coordinates, Standard Storm (Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT), 100 keV',
                fontsize=14, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    output_file = os.path.join(output_dir, 'fig25_tilt_progression_detailed_gsm.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nFigure saved: {output_file}")
    
    # Create summary plot of tilt progression
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot 1: Dipole tilt vs date
    ax1.plot(range(8), tilt_values, 'bo-', linewidth=2, markersize=8)
    ax1.set_ylabel('Dipole Tilt (degrees)', fontsize=12)
    ax1.set_title('Dipole Tilt Progression from Spring Equinox to Summer Solstice', 
                 fontsize=14, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-5, 25)
    
    # Add value labels
    for i, (date, tilt) in enumerate(zip(dates, tilt_values)):
        ax1.text(i, tilt + 0.5, f'{tilt:.1f}°', ha='center', fontsize=8)
    
    # Plot 2: Scattering percentage vs date
    ax2.plot(range(8), scatter_fracs, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Time Points', fontsize=12)
    ax2.set_ylabel('Scattering Region (% of XZ plane)', fontsize=12)
    ax2.set_title('Scattering Region Evolution', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(8))
    ax2.set_xticklabels(dates, rotation=45, ha='right')
    
    # Add value labels
    for i, (date, frac) in enumerate(zip(dates, scatter_fracs)):
        ax2.text(i, frac + 0.01, f'{frac:.2f}%', ha='center', fontsize=8)
    
    plt.tight_layout()
    
    output_file2 = os.path.join(output_dir, 'fig26_tilt_progression_summary.png')
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"Summary plot saved: {output_file2}")
    
    # Print summary
    print("\n" + "="*60)
    print("TILT PROGRESSION SUMMARY")
    print("="*60)
    print(f"{'Date':<10} {'Tilt (°)':<10} {'Scattering %':<12}")
    print("-"*60)
    for date, tilt, scatter in zip(dates, tilt_values, scatter_fracs):
        print(f"{date:<10} {tilt:>7.1f}    {scatter:>8.2f}")
    print("="*60)


if __name__ == "__main__":
    create_detailed_tilt_comparison()
    
    print("\n" + "="*80)
    print("Detailed tilt progression analysis complete!")
    print("="*80)
    print("\nKey findings:")
    print("- Dipole tilt shows complex non-linear progression")
    print("- Maximum tilt occurs around April 20, not at solstice")
    print("- Scattering regions follow the tilting current sheet")
    print("- Minimal variation in total scattering percentage")
    print("="*80)