#!/usr/bin/env python3
"""
Magnetic Field Vector Plots with Curvature and Larmor Radius
Shows vector field (Bx, Bz) with either curvature radius or Larmor radius in background
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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

print("="*80)
print("MAGNETIC FIELD VECTOR PLOTS WITH CURVATURE AND LARMOR RADIUS")
print("XZ Plane (Y=0) - T96 Model")
print("="*80)

# Initialize geopack for current conditions
ut = 1584662400  # March 20, 2020 (equinox)
ps = geopack.recalc(ut)

print(f"\nModel conditions:")
print(f"Dipole tilt: {np.degrees(ps):.2f}°")

# Model parameters - standard parameters from previous analysis
parmod = [3.0, -30.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
print(f"\nT96 Parameters:")
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


def create_curvature_radius_plot():
    """Create magnetic field vector plot with curvature radius background"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create grid for background color (curvature radius)
    x_bg = np.linspace(-20, 10, 151)
    z_bg = np.linspace(-10, 10, 101)
    X_bg, Z_bg = np.meshgrid(x_bg, z_bg)
    Y_bg = np.zeros_like(X_bg)
    
    # Calculate field and curvature for background
    print("\nCalculating curvature radius...")
    x_flat = X_bg.flatten()
    y_flat = Y_bg.flatten()
    z_flat = Z_bg.flatten()
    
    # Calculate curvature
    kappa = field_line_curvature_vectorized(t96_vectorized, parmod, ps, x_flat, y_flat, z_flat)
    Rc_Re = np.where(kappa > 1e-10, 1.0 / kappa, 1e3)  # Cap at 1000 Re
    Rc_grid = Rc_Re.reshape(X_bg.shape)
    
    # Also get field for vectors
    bx_bg, by_bg, bz_bg = t96_vectorized(parmod, ps, x_flat, y_flat, z_flat)
    
    # Plot curvature radius as background
    im = ax.contourf(X_bg, Z_bg, Rc_grid, levels=50, cmap='plasma', 
                     norm=LogNorm(vmin=0.1, vmax=100))
    
    # Create coarser grid for vectors
    x_vec = np.linspace(-20, 10, 31)
    z_vec = np.linspace(-10, 10, 21)
    X_vec, Z_vec = np.meshgrid(x_vec, z_vec)
    Y_vec = np.zeros_like(X_vec)
    
    # Calculate field vectors
    print("Calculating field vectors...")
    x_vec_flat = X_vec.flatten()
    y_vec_flat = Y_vec.flatten()
    z_vec_flat = Z_vec.flatten()
    
    bx_vec, by_vec, bz_vec = t96_vectorized(parmod, ps, x_vec_flat, y_vec_flat, z_vec_flat)
    
    # Reshape vectors
    Bx_grid = bx_vec.reshape(X_vec.shape)
    Bz_grid = bz_vec.reshape(Z_vec.shape)
    
    # Normalize vectors for display
    B_vec_mag = np.sqrt(Bx_grid**2 + Bz_grid**2)
    Bx_norm = np.divide(Bx_grid, B_vec_mag, out=np.zeros_like(Bx_grid), where=B_vec_mag>0)
    Bz_norm = np.divide(Bz_grid, B_vec_mag, out=np.zeros_like(Bz_grid), where=B_vec_mag>0)
    
    # Scale vectors by log of field strength for better visualization
    scale_factor = np.log10(B_vec_mag + 1) * 0.3
    
    # Plot vectors
    ax.quiver(X_vec, Z_vec, Bx_norm * scale_factor, Bz_norm * scale_factor,
              color='white', alpha=0.8, width=0.003, headwidth=3, headlength=4,
              scale=15, scale_units='xy')
    
    # Add field lines (streamlines)
    strm = ax.streamplot(x_bg, z_bg, 
                        bx_bg.reshape(len(z_bg), len(x_bg)), 
                        bz_bg.reshape(len(z_bg), len(x_bg)),
                        color='cyan', linewidth=1, density=0.5)
    strm.lines.set_alpha(0.6)
    
    # Add critical contour for Rc = 8 * RL(30keV)
    B_magnitude = np.sqrt(bx_bg**2 + by_bg**2 + bz_bg**2)
    RL_30keV = calculate_larmor_radius(30, B_magnitude) / Re  # Convert to Re
    ratio = Rc_Re.flatten() / RL_30keV
    ratio_grid = ratio.reshape(X_bg.shape)
    
    cs = ax.contour(X_bg, Z_bg, ratio_grid, levels=[8], colors='red', linewidths=2)
    ax.clabel(cs, inline=True, fontsize=10, fmt='Rc/RL=8')
    
    # Add Earth
    earth = plt.Circle((0, 0), 1, color='white', zorder=10)
    ax.add_patch(earth)
    ax.text(0, 0, 'E', ha='center', va='center', fontsize=12, weight='bold')
    
    # Add magnetopause boundary (approximate)
    theta = np.linspace(0, 2*np.pi, 100)
    r_mp = 10 * (2 / (1 + np.cos(theta)))**0.5  # Simple magnetopause model
    x_mp = r_mp * np.cos(theta)
    z_mp = r_mp * np.sin(theta)
    # Only plot dayside part
    mask = x_mp > -15
    ax.plot(x_mp[mask], z_mp[mask], 'w--', linewidth=2, alpha=0.5, label='Magnetopause')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Radius of Curvature Rc (Re)', fontsize=12)
    
    # Labels and formatting
    ax.set_xlabel('X GSM (Re)', fontsize=12)
    ax.set_ylabel('Z GSM (Re)', fontsize=12)
    ax.set_title(f'T96 Magnetic Field: Vectors with Curvature Radius Background\n' +
                f'Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT, IMF By={parmod[2]} nT, Bz={parmod[3]} nT',
                fontsize=14, weight='bold')
    
    ax.set_xlim(-20, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, color='gray')
    
    # Add text annotations
    ax.text(-18, 8, 'Small Rc = High Curvature', fontsize=10, color='white', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
    ax.text(-18, -8, 'Large Rc = Low Curvature', fontsize=10, color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
    ax.text(5, -8, 'Red contour: Rc/RL=8 (30 keV)', fontsize=10, color='red',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, 'fig03a_magnetic_field_vectors_curvature.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nCurvature radius plot saved: {output_file}")
    
    # Print some statistics
    print(f"\nCurvature radius statistics:")
    print(f"  Min Rc: {np.min(Rc_Re):.2f} Re")
    print(f"  Max Rc: {np.min([np.max(Rc_Re), 1000]):.2f} Re")
    print(f"  Median Rc: {np.median(Rc_Re):.2f} Re")


def create_larmor_radius_plots():
    """Create magnetic field vector plots with Larmor radius backgrounds for different energies"""
    
    energies = [10, 100, 1000]  # keV
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Create grid for background
    x_bg = np.linspace(-20, 10, 151)
    z_bg = np.linspace(-10, 10, 101)
    X_bg, Z_bg = np.meshgrid(x_bg, z_bg)
    Y_bg = np.zeros_like(X_bg)
    
    # Calculate field
    print("\nCalculating magnetic field for Larmor radius...")
    x_flat = X_bg.flatten()
    y_flat = Y_bg.flatten()
    z_flat = Z_bg.flatten()
    
    bx_bg, by_bg, bz_bg = t96_vectorized(parmod, ps, x_flat, y_flat, z_flat)
    B_magnitude = np.sqrt(bx_bg**2 + by_bg**2 + bz_bg**2)
    
    # Create coarser grid for vectors
    x_vec = np.linspace(-20, 10, 21)
    z_vec = np.linspace(-10, 10, 15)
    X_vec, Z_vec = np.meshgrid(x_vec, z_vec)
    Y_vec = np.zeros_like(X_vec)
    
    # Calculate field vectors
    x_vec_flat = X_vec.flatten()
    y_vec_flat = Y_vec.flatten()
    z_vec_flat = Z_vec.flatten()
    
    bx_vec, by_vec, bz_vec = t96_vectorized(parmod, ps, x_vec_flat, y_vec_flat, z_vec_flat)
    
    # Reshape vectors
    Bx_grid = bx_vec.reshape(X_vec.shape)
    Bz_grid = bz_vec.reshape(Z_vec.shape)
    
    # Normalize vectors
    B_vec_mag = np.sqrt(Bx_grid**2 + Bz_grid**2)
    Bx_norm = np.divide(Bx_grid, B_vec_mag, out=np.zeros_like(Bx_grid), where=B_vec_mag>0)
    Bz_norm = np.divide(Bz_grid, B_vec_mag, out=np.zeros_like(Bz_grid), where=B_vec_mag>0)
    
    for idx, (ax, energy) in enumerate(zip(axes, energies)):
        print(f"Creating Larmor radius plot for {energy} keV...")
        
        # Calculate Larmor radius
        RL_m = calculate_larmor_radius(energy, B_magnitude)
        RL_Re = RL_m / Re  # Convert to Earth radii
        RL_grid = RL_Re.reshape(X_bg.shape)
        
        # Plot Larmor radius as background
        # Adjust color scale based on energy
        if energy <= 100:
            vmax = 1
        else:  # 1000 keV
            vmax = 10
        
        im = ax.contourf(X_bg, Z_bg, RL_grid, levels=50, cmap='viridis', 
                         norm=LogNorm(vmin=0.001, vmax=vmax))
        
        # Scale vectors
        scale_factor = 0.4
        
        # Plot vectors
        ax.quiver(X_vec, Z_vec, Bx_norm * scale_factor, Bz_norm * scale_factor,
                  color='white', alpha=0.8, width=0.003, headwidth=3, headlength=4,
                  scale=10, scale_units='xy')
        
        # Add field lines
        strm = ax.streamplot(x_bg, z_bg, 
                            bx_bg.reshape(len(z_bg), len(x_bg)), 
                            bz_bg.reshape(len(z_bg), len(x_bg)),
                            color='red', linewidth=0.8, density=0.3)
        strm.lines.set_alpha(0.5)
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax.add_patch(earth)
        ax.text(0, 0, 'E', ha='center', va='center', fontsize=10, weight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(f'Larmor Radius ({energy} keV) [Re]', fontsize=10)
        
        # Labels and formatting
        ax.set_xlabel('X GSM (Re)', fontsize=10)
        ax.set_ylabel('Z GSM (Re)', fontsize=10)
        ax.set_title(f'{energy} keV Electrons', fontsize=12, weight='bold')
        
        ax.set_xlim(-20, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, color='gray')
        
        # Add statistics text
        stats_text = f'Min: {np.min(RL_Re)*1000:.1f} km\nMax: {np.min([np.max(RL_Re), 10])*6371:.0f} km'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.suptitle(f'T96 Magnetic Field Vectors with Larmor Radius Background\n' +
                f'Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT, IMF By={parmod[2]} nT, Bz={parmod[3]} nT',
                fontsize=14, weight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, 'fig03b_magnetic_field_vectors_larmor.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Larmor radius plots saved: {output_file}")


def create_combined_plot():
    """Create a single figure with both curvature and Larmor radius plots"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Create grid
    x_bg = np.linspace(-20, 10, 151)
    z_bg = np.linspace(-10, 10, 101)
    X_bg, Z_bg = np.meshgrid(x_bg, z_bg)
    Y_bg = np.zeros_like(X_bg)
    
    # Calculate field and curvature
    x_flat = X_bg.flatten()
    y_flat = Y_bg.flatten()
    z_flat = Z_bg.flatten()
    
    bx_bg, by_bg, bz_bg = t96_vectorized(parmod, ps, x_flat, y_flat, z_flat)
    B_magnitude = np.sqrt(bx_bg**2 + by_bg**2 + bz_bg**2)
    
    # Calculate curvature
    kappa = field_line_curvature_vectorized(t96_vectorized, parmod, ps, x_flat, y_flat, z_flat)
    Rc_Re = np.where(kappa > 1e-10, 1.0 / kappa, 1e3)
    Rc_grid = Rc_Re.reshape(X_bg.shape)
    
    # Create vectors
    x_vec = np.linspace(-20, 10, 25)
    z_vec = np.linspace(-10, 10, 17)
    X_vec, Z_vec = np.meshgrid(x_vec, z_vec)
    Y_vec = np.zeros_like(X_vec)
    
    x_vec_flat = X_vec.flatten()
    y_vec_flat = Y_vec.flatten()
    z_vec_flat = Z_vec.flatten()
    
    bx_vec, by_vec, bz_vec = t96_vectorized(parmod, ps, x_vec_flat, y_vec_flat, z_vec_flat)
    
    Bx_grid = bx_vec.reshape(X_vec.shape)
    Bz_grid = bz_vec.reshape(Z_vec.shape)
    
    B_vec_mag = np.sqrt(Bx_grid**2 + Bz_grid**2)
    Bx_norm = np.divide(Bx_grid, B_vec_mag, out=np.zeros_like(Bx_grid), where=B_vec_mag>0)
    Bz_norm = np.divide(Bz_grid, B_vec_mag, out=np.zeros_like(Bz_grid), where=B_vec_mag>0)
    
    scale_factor = 0.5
    
    # Plot 1: Curvature Radius
    im1 = ax1.contourf(X_bg, Z_bg, Rc_grid, levels=50, cmap='plasma', 
                       norm=LogNorm(vmin=0.1, vmax=100))
    
    ax1.quiver(X_vec, Z_vec, Bx_norm * scale_factor, Bz_norm * scale_factor,
               color='white', alpha=0.8, width=0.003, headwidth=3, headlength=4,
               scale=12, scale_units='xy')
    
    earth1 = plt.Circle((0, 0), 1, color='white', zorder=10)
    ax1.add_patch(earth1)
    
    cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02)
    cbar1.set_label('Radius of Curvature (Re)', fontsize=11)
    
    ax1.set_xlabel('X GSM (Re)', fontsize=11)
    ax1.set_ylabel('Z GSM (Re)', fontsize=11)
    ax1.set_title('Field Line Curvature Radius', fontsize=12, weight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-20, 10)
    ax1.set_ylim(-10, 10)
    
    # Plot 2: Larmor Radius (30 keV)
    energy = 30  # keV
    RL_m = calculate_larmor_radius(energy, B_magnitude)
    RL_Re = RL_m / Re
    RL_grid = RL_Re.reshape(X_bg.shape)
    
    im2 = ax2.contourf(X_bg, Z_bg, RL_grid, levels=50, cmap='viridis', 
                       norm=LogNorm(vmin=0.001, vmax=1))
    
    ax2.quiver(X_vec, Z_vec, Bx_norm * scale_factor, Bz_norm * scale_factor,
               color='white', alpha=0.8, width=0.003, headwidth=3, headlength=4,
               scale=12, scale_units='xy')
    
    # Add Rc/RL = 8 contour
    ratio_grid = Rc_grid / RL_grid
    cs = ax2.contour(X_bg, Z_bg, ratio_grid, levels=[8], colors='red', linewidths=2)
    ax2.clabel(cs, inline=True, fontsize=10, fmt='Rc/RL=8')
    
    earth2 = plt.Circle((0, 0), 1, color='white', zorder=10)
    ax2.add_patch(earth2)
    
    cbar2 = plt.colorbar(im2, ax=ax2, pad=0.02)
    cbar2.set_label(f'Larmor Radius ({energy} keV) [Re]', fontsize=11)
    
    ax2.set_xlabel('X GSM (Re)', fontsize=11)
    ax2.set_ylabel('Z GSM (Re)', fontsize=11)
    ax2.set_title(f'Electron Larmor Radius ({energy} keV)', fontsize=12, weight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-20, 10)
    ax2.set_ylim(-10, 10)
    
    plt.suptitle(f'T96 Magnetic Field Analysis in XZ Plane (Y=0)\n' +
                f'Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT, IMF By={parmod[2]} nT, Bz={parmod[3]} nT',
                fontsize=14, weight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, 'fig03_magnetic_field_vectors_curvature_larmor.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nCombined plot saved: {output_file}")


if __name__ == "__main__":
    # Create curvature radius plot
    create_curvature_radius_plot()
    
    # Create Larmor radius plots for different energies
    create_larmor_radius_plots()
    
    # Create combined plot
    print("\nCreating combined plot...")
    create_combined_plot()
    
    print("\n" + "="*80)
    print("Magnetic field vector plots with curvature and Larmor radius complete!")
    print("="*80)