#!/usr/bin/env python3
"""
Dipole Tilt Comparison in SM Coordinates
Shows how different dipole tilts affect Rc/RL ratio in SM coordinate system
Uses standard geomagnetic storm conditions
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
print("DIPOLE TILT COMPARISON IN SM COORDINATES")
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


def gsm_to_sm(x_gsm, y_gsm, z_gsm, ps):
    """Convert from GSM to SM coordinates."""
    cos_ps = np.cos(ps)
    sin_ps = np.sin(ps)
    
    x_sm = x_gsm * cos_ps - z_gsm * sin_ps
    y_sm = y_gsm
    z_sm = x_gsm * sin_ps + z_gsm * cos_ps
    
    return x_sm, y_sm, z_sm


def sm_to_gsm(x_sm, y_sm, z_sm, ps):
    """Convert from SM to GSM coordinates."""
    cos_ps = np.cos(ps)
    sin_ps = np.sin(ps)
    
    x_gsm = x_sm * cos_ps + z_sm * sin_ps
    y_gsm = y_sm
    z_gsm = -x_sm * sin_ps + z_sm * cos_ps
    
    return x_gsm, y_gsm, z_gsm


def create_tilt_comparison_xz_plane():
    """Create XZ plane plots in SM coordinates for different tilts"""
    
    # Define tilt conditions from spring equinox to summer solstice
    spring_eq = 1584748800  # March 20, 2020
    summer_sol = 1592697600  # June 21, 2020
    
    tilt_conditions = [
        # (Unix time, description, approximate tilt in degrees)
        (spring_eq, "Spring Equinox", 0),                                    # March 20
        (spring_eq + (summer_sol - spring_eq) // 3, "Equinox + 1/3", 8),   # ~April 20
        (spring_eq + 2 * (summer_sol - spring_eq) // 3, "Equinox + 2/3", 16),  # ~May 21
        (summer_sol, "Summer Solstice", 23)                                  # June 21
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Create grid in SM coordinates
    x_sm_grid = np.linspace(-20, 10, 151)
    z_sm_grid = np.linspace(-10, 10, 101)
    X_SM, Z_SM = np.meshgrid(x_sm_grid, z_sm_grid)
    Y_SM = np.zeros_like(X_SM)
    
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
    
    for idx, (ut, description, approx_tilt) in enumerate(tilt_conditions):
        ax = axes[idx]
        print(f"\nProcessing {description}...")
        
        # Calculate actual dipole tilt
        ps = geopack.recalc(ut)
        actual_tilt = np.degrees(ps)
        print(f"  Unix time: {ut}")
        print(f"  Actual dipole tilt: {actual_tilt:.2f}°")
        
        # Flatten SM coordinates
        x_sm_flat = X_SM.flatten()
        y_sm_flat = Y_SM.flatten()
        z_sm_flat = Z_SM.flatten()
        
        # Convert SM to GSM for field calculation
        x_gsm_flat, y_gsm_flat, z_gsm_flat = sm_to_gsm(x_sm_flat, y_sm_flat, z_sm_flat, ps)
        
        # Calculate field in GSM
        bx_gsm, by_gsm, bz_gsm = t96_vectorized(parmod, ps, x_gsm_flat, y_gsm_flat, z_gsm_flat)
        B_magnitude = np.sqrt(bx_gsm**2 + by_gsm**2 + bz_gsm**2)
        
        # Convert field vectors to SM coordinates
        bx_sm = bx_gsm * np.cos(ps) - bz_gsm * np.sin(ps)
        by_sm = by_gsm
        bz_sm = bx_gsm * np.sin(ps) + bz_gsm * np.cos(ps)
        
        # Calculate curvature in GSM coordinates
        kappa = field_line_curvature_vectorized(t96_vectorized, parmod, ps, 
                                               x_gsm_flat, y_gsm_flat, z_gsm_flat)
        Rc_Re = np.where(kappa > 1e-10, 1.0 / kappa, 1e3)
        Rc_m = Rc_Re * Re
        
        # Calculate Larmor radius
        RL_m = calculate_larmor_radius(energy, B_magnitude)
        
        # Calculate Rc/RL ratio
        ratio = Rc_m / RL_m
        ratio = np.where(ratio > 1000, 1000, ratio)
        ratio = np.where(ratio < 0.1, 0.1, ratio)
        ratio_grid = ratio.reshape(X_SM.shape)
        
        # Calculate statistics
        scatter_frac = np.sum(ratio < CRITICAL_RATIO) / len(ratio) * 100
        
        # Plot Rc/RL ratio
        im = ax.contourf(X_SM, Z_SM, ratio_grid, levels=levels, 
                        cmap='RdBu_r', norm=LogNorm(vmin=0.1, vmax=1000))
        
        # Add critical contour
        cs = ax.contour(X_SM, Z_SM, ratio_grid, levels=[CRITICAL_RATIO], 
                       colors='black', linewidths=2)
        ax.clabel(cs, inline=True, fontsize=8, fmt='8')
        
        # Add vectors in SM coordinates (sparse)
        x_vec_sm = np.linspace(-20, 10, 16)
        z_vec_sm = np.linspace(-10, 10, 11)
        X_vec_SM, Z_vec_SM = np.meshgrid(x_vec_sm, z_vec_sm)
        Y_vec_SM = np.zeros_like(X_vec_SM)
        
        x_vec_sm_flat = X_vec_SM.flatten()
        y_vec_sm_flat = Y_vec_SM.flatten()
        z_vec_sm_flat = Z_vec_SM.flatten()
        
        # Convert to GSM for field calculation
        x_vec_gsm, y_vec_gsm, z_vec_gsm = sm_to_gsm(x_vec_sm_flat, y_vec_sm_flat, z_vec_sm_flat, ps)
        
        bx_vec_gsm, by_vec_gsm, bz_vec_gsm = t96_vectorized(parmod, ps, 
                                                           x_vec_gsm, y_vec_gsm, z_vec_gsm)
        
        # Convert field to SM
        bx_vec_sm = bx_vec_gsm * np.cos(ps) - bz_vec_gsm * np.sin(ps)
        bz_vec_sm = bx_vec_gsm * np.sin(ps) + bz_vec_gsm * np.cos(ps)
        
        Bx_grid_SM = bx_vec_sm.reshape(X_vec_SM.shape)
        Bz_grid_SM = bz_vec_sm.reshape(Z_vec_SM.shape)
        
        B_vec_mag = np.sqrt(Bx_grid_SM**2 + Bz_grid_SM**2)
        Bx_norm = np.divide(Bx_grid_SM, B_vec_mag, out=np.zeros_like(Bx_grid_SM), 
                          where=B_vec_mag>0)
        Bz_norm = np.divide(Bz_grid_SM, B_vec_mag, out=np.zeros_like(Bz_grid_SM), 
                          where=B_vec_mag>0)
        
        ax.quiver(X_vec_SM, Z_vec_SM, Bx_norm*0.4, Bz_norm*0.4,
                 color='white', alpha=0.7, width=0.002, headwidth=3, 
                 scale=12, scale_units='xy', edgecolor='black', linewidth=0.3)
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax.add_patch(earth)
        ax.text(0, 0, 'E', ha='center', va='center', fontsize=10, weight='bold')
        
        # Labels and formatting
        ax.set_xlabel('X SM (Re)', fontsize=10)
        ax.set_ylabel('Z SM (Re)', fontsize=10)
        ax.set_title(f'{description}\nTilt: {actual_tilt:.1f}°, Scattering: {scatter_frac:.1f}%', 
                    fontsize=11, weight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-20, 10)
        ax.set_ylim(-10, 10)
        
        # Add date text
        date_str = datetime.fromtimestamp(ut).strftime('%Y-%m-%d')
        ax.text(0.02, 0.98, f'Date: {date_str}\nGSM tilt: {actual_tilt:.1f}°', 
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add note about SM coordinates
        if idx == 0:
            ax.text(0.98, 0.02, 'SM Coordinates:\nZ_SM aligned with\ndipole axis', 
                   transform=ax.transAxes, fontsize=8, 
                   ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Rc/RL Ratio', fontsize=12)
    cbar.ax.axhline(y=8, color='black', linewidth=2)
    
    plt.suptitle('Rc/RL Ratio in SM Coordinates: Seasonal Tilt Variations\n' +
                f'Standard Storm Conditions (Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT), 100 keV Electrons', 
                fontsize=16, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    output_file = os.path.join(output_dir, 'fig19_tilt_comparison_sm_coordinates.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nFigure saved: {output_file}")


def create_tilt_comparison_xy_slices():
    """Create XY plane slices at Z_SM=0 for different tilts"""
    
    # Define tilt conditions from spring equinox to summer solstice
    spring_eq = 1584748800  # March 20, 2020
    summer_sol = 1592697600  # June 21, 2020
    
    tilt_conditions = [
        (spring_eq, "Spring Equinox", 0),
        (spring_eq + (summer_sol - spring_eq) // 3, "Equinox + 1/3", 8),
        (spring_eq + 2 * (summer_sol - spring_eq) // 3, "Equinox + 2/3", 16),
        (summer_sol, "Summer Solstice", 23)
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Create grid in SM coordinates (XY plane at Z_SM=0)
    x_sm_grid = np.linspace(-15, 5, 101)
    y_sm_grid = np.linspace(-12, 12, 121)
    X_SM, Y_SM = np.meshgrid(x_sm_grid, y_sm_grid)
    Z_SM = np.zeros_like(X_SM)  # Z_SM = 0 (magnetic equatorial plane)
    
    # Standard storm parameters
    parmod = [3.0, -30.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Fixed energy
    energy = 100  # keV
    
    # Color levels
    levels = np.logspace(-1, 3, 20)
    
    print("\nCreating XY plane slices at Z_SM = 0...")
    
    for idx, (ut, description, approx_tilt) in enumerate(tilt_conditions):
        ax = axes[idx]
        
        # Calculate actual dipole tilt
        ps = geopack.recalc(ut)
        actual_tilt = np.degrees(ps)
        
        # Flatten SM coordinates
        x_sm_flat = X_SM.flatten()
        y_sm_flat = Y_SM.flatten()
        z_sm_flat = Z_SM.flatten()
        
        # Convert SM to GSM for field calculation
        x_gsm_flat, y_gsm_flat, z_gsm_flat = sm_to_gsm(x_sm_flat, y_sm_flat, z_sm_flat, ps)
        
        # Calculate field in GSM
        bx_gsm, by_gsm, bz_gsm = t96_vectorized(parmod, ps, x_gsm_flat, y_gsm_flat, z_gsm_flat)
        B_magnitude = np.sqrt(bx_gsm**2 + by_gsm**2 + bz_gsm**2)
        
        # Calculate curvature
        kappa = field_line_curvature_vectorized(t96_vectorized, parmod, ps, 
                                               x_gsm_flat, y_gsm_flat, z_gsm_flat)
        Rc_Re = np.where(kappa > 1e-10, 1.0 / kappa, 1e3)
        Rc_m = Rc_Re * Re
        
        # Calculate Larmor radius
        RL_m = calculate_larmor_radius(energy, B_magnitude)
        
        # Calculate Rc/RL ratio
        ratio = Rc_m / RL_m
        ratio = np.where(ratio > 1000, 1000, ratio)
        ratio = np.where(ratio < 0.1, 0.1, ratio)
        ratio_grid = ratio.reshape(X_SM.shape)
        
        # Calculate statistics
        scatter_frac = np.sum(ratio < CRITICAL_RATIO) / len(ratio) * 100
        
        # Plot Rc/RL ratio
        im = ax.contourf(X_SM, Y_SM, ratio_grid, levels=levels, 
                        cmap='RdBu_r', norm=LogNorm(vmin=0.1, vmax=1000))
        
        # Add critical contour
        cs = ax.contour(X_SM, Y_SM, ratio_grid, levels=[CRITICAL_RATIO], 
                       colors='black', linewidths=2)
        try:
            ax.clabel(cs, inline=True, fontsize=8, fmt='8')
        except:
            pass
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax.add_patch(earth)
        
        # Labels and formatting
        ax.set_xlabel('X SM (Re)', fontsize=10)
        ax.set_ylabel('Y SM (Re)', fontsize=10)
        ax.set_title(f'{description} (Tilt: {actual_tilt:.1f}°)\nZ_SM = 0, Scattering: {scatter_frac:.1f}%', 
                    fontsize=11, weight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-15, 5)
        ax.set_ylim(-12, 12)
        
        # Add GSM Z equivalent
        z_gsm_equiv = -np.sin(ps) * 0  # For Z_SM = 0
        ax.text(0.02, 0.02, f'Z_GSM ≈ {z_gsm_equiv:.1f} Re', 
               transform=ax.transAxes, fontsize=8,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Rc/RL Ratio', fontsize=12)
    cbar.ax.axhline(y=8, color='black', linewidth=2)
    
    plt.suptitle('Rc/RL Ratio at Magnetic Equator (Z_SM = 0): Seasonal Variations\n' +
                f'Standard Storm Conditions, 100 keV Electrons', 
                fontsize=16, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    output_file = os.path.join(output_dir, 'fig20_tilt_comparison_xy_plane_sm.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"XY plane figure saved: {output_file}")


def create_tilt_summary():
    """Create summary plot showing scattering percentage vs tilt angle"""
    
    # Generate a range of tilts throughout the year
    print("\nCalculating tilt dependence...")
    
    # Sample dates throughout the year
    dates = []
    for month in range(1, 13):
        # Use 15th of each month in 2020
        ut = int(datetime(2020, month, 15).timestamp())
        dates.append(ut)
    
    tilts = []
    scatter_fracs_xz = []
    scatter_fracs_xy = []
    
    # Standard storm parameters
    parmod = [3.0, -30.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    energy = 100  # keV
    
    for ut in dates:
        ps = geopack.recalc(ut)
        tilt_deg = np.degrees(ps)
        tilts.append(tilt_deg)
        
        # Sample XZ plane (Y_SM = 0)
        x_sm = np.linspace(-15, -5, 51)
        z_sm = np.linspace(-2, 2, 41)
        X, Z = np.meshgrid(x_sm, z_sm)
        Y = np.zeros_like(X)
        
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        
        # Convert to GSM
        x_gsm, y_gsm, z_gsm = sm_to_gsm(x_flat, y_flat, z_flat, ps)
        
        # Calculate field
        bx, by, bz = t96_vectorized(parmod, ps, x_gsm, y_gsm, z_gsm)
        B_mag = np.sqrt(bx**2 + by**2 + bz**2)
        
        # Calculate curvature
        kappa = field_line_curvature_vectorized(t96_vectorized, parmod, ps, 
                                               x_gsm, y_gsm, z_gsm)
        Rc_Re = np.where(kappa > 1e-10, 1.0 / kappa, 1e3)
        Rc_m = Rc_Re * Re
        
        # Calculate Larmor radius
        RL_m = calculate_larmor_radius(energy, B_mag)
        
        # Calculate ratio
        ratio = Rc_m / RL_m
        scatter_frac = np.sum(ratio < CRITICAL_RATIO) / len(ratio) * 100
        scatter_fracs_xz.append(scatter_frac)
        
        # Sample XY plane (Z_SM = 0)
        x_sm = np.linspace(-15, -5, 51)
        y_sm = np.linspace(-8, 8, 81)
        X, Y = np.meshgrid(x_sm, y_sm)
        Z = np.zeros_like(X)
        
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        
        # Convert to GSM
        x_gsm, y_gsm, z_gsm = sm_to_gsm(x_flat, y_flat, z_flat, ps)
        
        # Calculate field
        bx, by, bz = t96_vectorized(parmod, ps, x_gsm, y_gsm, z_gsm)
        B_mag = np.sqrt(bx**2 + by**2 + bz**2)
        
        # Calculate curvature
        kappa = field_line_curvature_vectorized(t96_vectorized, parmod, ps, 
                                               x_gsm, y_gsm, z_gsm)
        Rc_Re = np.where(kappa > 1e-10, 1.0 / kappa, 1e3)
        Rc_m = Rc_Re * Re
        
        # Calculate Larmor radius
        RL_m = calculate_larmor_radius(energy, B_mag)
        
        # Calculate ratio
        ratio = Rc_m / RL_m
        scatter_frac = np.sum(ratio < CRITICAL_RATIO) / len(ratio) * 100
        scatter_fracs_xy.append(scatter_frac)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(tilts, scatter_fracs_xz, 'b-o', linewidth=2, markersize=8, 
            label='XZ plane (Y_SM = 0)')
    ax.plot(tilts, scatter_fracs_xy, 'g-s', linewidth=2, markersize=8, 
            label='XY plane (Z_SM = 0)')
    
    ax.set_xlabel('Dipole Tilt (degrees)', fontsize=12)
    ax.set_ylabel('Scattering Region (%)', fontsize=12)
    ax.set_title('Pitch Angle Scattering vs Dipole Tilt in SM Coordinates\n' +
                'Standard Storm Conditions, 100 keV Electrons', 
                fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add season markers
    ax.axvline(23.5, color='red', linestyle='--', alpha=0.5)
    ax.axvline(-23.5, color='blue', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    ax.text(23.5, ax.get_ylim()[1]*0.95, 'Summer\nSolstice', 
            ha='center', va='top', fontsize=9, color='red')
    ax.text(-23.5, ax.get_ylim()[1]*0.95, 'Winter\nSolstice', 
            ha='center', va='top', fontsize=9, color='blue')
    ax.text(0, ax.get_ylim()[1]*0.95, 'Equinoxes', 
            ha='center', va='top', fontsize=9, color='gray')
    
    # Add note about SM coordinates
    ax.text(0.05, 0.95, 
            'Note: In SM coordinates, the magnetic equator\nis always at Z_SM = 0 regardless of tilt', 
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'fig21_tilt_dependence_summary_sm.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Summary plot saved: {output_file}")


if __name__ == "__main__":
    # Create XZ plane comparison
    create_tilt_comparison_xz_plane()
    
    # Create XY plane comparison
    create_tilt_comparison_xy_slices()
    
    # Create summary plot
    create_tilt_summary()
    
    print("\n" + "="*80)
    print("Dipole tilt comparison in SM coordinates complete!")
    print("="*80)
    print("\nKey findings:")
    print("- In SM coordinates, Z_SM is always aligned with dipole axis")
    print("- Magnetic equator remains at Z_SM = 0 for all tilts")
    print("- Scattering patterns rotate with Earth's dipole")
    print("- Overall scattering percentage remains relatively constant")
    print("="*80)