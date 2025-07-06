#!/usr/bin/env python3
"""
Curvature Radius and Larmor Radius Parameter Variations
Shows how different model parameters affect curvature radius and Larmor radius
Similar to Figure 11 but showing Rc and RL instead of Rc/RL ratio
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
print("CURVATURE AND LARMOR RADIUS PARAMETER VARIATIONS")
print("="*80)

# Initialize geopack
ut = 1584662400  # March 20, 2020 (equinox)
ps = geopack.recalc(ut)

print(f"\nModel conditions:")
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


def create_curvature_radius_variations():
    """Create XY plane slices showing curvature radius for different parameters"""
    
    # Define parameter sets
    param_sets = [
        # [Pdyn, Dst, By_IMF, Bz_IMF, description]
        ([2.0, 0.0, 0.0, 0.0], "Quiet"),
        ([3.0, -30.0, 1.0, -3.0], "Moderate"),
        ([5.0, -100.0, 5.0, -10.0], "Storm"),
        ([1.0, -200.0, 10.0, -20.0], "Extreme")
    ]
    
    # Z heights to analyze
    z_heights = [-0.4, -0.2, 0.0, 0.2]
    
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    
    # Create XY grid
    x_grid = np.linspace(-15, 5, 101)
    y_grid = np.linspace(-12, 12, 121)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    print("\nCreating curvature radius variations...")
    
    for row_idx, (params, condition) in enumerate(param_sets):
        parmod = params + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        for col_idx, z_height in enumerate(z_heights):
            ax = axes[row_idx, col_idx]
            
            # Create Z array
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
            median_rc = np.median(Rc_Re)
            min_rc = np.min(Rc_Re)
            
            # Plot curvature radius
            im = ax.contourf(X, Y, Rc_grid, levels=50, 
                            cmap='plasma', norm=LogNorm(vmin=0.1, vmax=100))
            
            # Add contours for specific values
            try:
                cs = ax.contour(X, Y, Rc_grid, levels=[0.5, 1, 5, 10], 
                               colors='white', linewidths=1, alpha=0.7)
                ax.clabel(cs, inline=True, fontsize=6, fmt='%.1f Re')
            except:
                pass
            
            # Add Earth
            earth = plt.Circle((0, 0), 1, color='white', zorder=10)
            ax.add_patch(earth)
            
            # Add field vectors (sparse) for middle panels
            if col_idx == 1 or col_idx == 2:
                x_vec = np.linspace(-15, 5, 9)
                y_vec = np.linspace(-12, 12, 11)
                X_vec, Y_vec = np.meshgrid(x_vec, y_vec)
                Z_vec = np.full_like(X_vec, z_height)
                
                x_vec_flat = X_vec.flatten()
                y_vec_flat = Y_vec.flatten()
                z_vec_flat = Z_vec.flatten()
                
                bx_vec, by_vec, bz_vec = t96_vectorized(parmod, ps, 
                                                       x_vec_flat, y_vec_flat, z_vec_flat)
                
                Bx_grid = bx_vec.reshape(X_vec.shape)
                By_grid = by_vec.reshape(Y_vec.shape)
                
                B_vec_mag = np.sqrt(Bx_grid**2 + By_grid**2)
                Bx_norm = np.divide(Bx_grid, B_vec_mag, out=np.zeros_like(Bx_grid), 
                                  where=B_vec_mag>0)
                By_norm = np.divide(By_grid, B_vec_mag, out=np.zeros_like(By_grid), 
                                  where=B_vec_mag>0)
                
                ax.quiver(X_vec, Y_vec, Bx_norm*0.5, By_norm*0.5,
                         color='white', alpha=0.5, width=0.002, headwidth=3, 
                         scale=20, scale_units='xy')
            
            # Labels
            if col_idx == 0:
                ax.set_ylabel(f'{condition}\nY GSM (Re)', fontsize=9)
            else:
                ax.set_ylabel('Y GSM (Re)', fontsize=8)
            
            if row_idx == 3:
                ax.set_xlabel('X GSM (Re)', fontsize=9)
            else:
                ax.set_xlabel('X GSM (Re)', fontsize=8)
            
            if row_idx == 0:
                ax.set_title(f'Z = {z_height} Re', fontsize=10, weight='bold')
            
            # Add statistics
            ax.text(0.95, 0.95, f'Med: {median_rc:.1f} Re', 
                   transform=ax.transAxes, fontsize=7,
                   ha='right', va='top', color='white', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
            
            ax.set_aspect('equal')
            ax.set_xlim(-15, 5)
            ax.set_ylim(-12, 12)
            ax.grid(True, alpha=0.3)
            
            # Reduce tick labels
            ax.set_xticks([-15, -10, -5, 0, 5])
            ax.set_yticks([-10, -5, 0, 5, 10])
            ax.tick_params(labelsize=7)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Radius of Curvature (Re)', fontsize=12)
    cbar.ax.tick_params(labelsize=9)
    
    plt.suptitle('Field Line Curvature Radius: Parameter and Height Variations\n' +
                'Plasma colormap: Purple = High Curvature, Yellow = Low Curvature', 
                fontsize=16, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    output_file = os.path.join(output_dir, 'fig16_curvature_radius_parameter_variations.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Curvature radius variations saved: {output_file}")


def create_larmor_radius_variations():
    """Create XY plane slices showing Larmor radius for different parameters"""
    
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
    
    # Create XY grid
    x_grid = np.linspace(-15, 5, 101)
    y_grid = np.linspace(-12, 12, 121)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Fixed energy
    energy = 100  # keV
    
    print(f"\nCreating Larmor radius variations for {energy} keV electrons...")
    
    for row_idx, (params, condition) in enumerate(param_sets):
        parmod = params + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        for col_idx, z_height in enumerate(z_heights):
            ax = axes[row_idx, col_idx]
            
            # Create Z array
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
            median_rl = np.median(RL_Re)
            max_rl = np.max(RL_Re)
            
            # Plot Larmor radius
            im = ax.contourf(X, Y, RL_grid, levels=50, 
                            cmap='viridis', norm=LogNorm(vmin=0.001, vmax=1))
            
            # Add contours for specific values
            try:
                cs = ax.contour(X, Y, RL_grid, levels=[0.01, 0.05, 0.1, 0.5], 
                               colors='white', linewidths=1, alpha=0.7)
                ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f Re')
            except:
                pass
            
            # Add Earth
            earth = plt.Circle((0, 0), 1, color='white', zorder=10)
            ax.add_patch(earth)
            
            # Add field vectors (sparse) for middle panels
            if col_idx == 1 or col_idx == 2:
                x_vec = np.linspace(-15, 5, 9)
                y_vec = np.linspace(-12, 12, 11)
                X_vec, Y_vec = np.meshgrid(x_vec, y_vec)
                Z_vec = np.full_like(X_vec, z_height)
                
                x_vec_flat = X_vec.flatten()
                y_vec_flat = Y_vec.flatten()
                z_vec_flat = Z_vec.flatten()
                
                bx_vec, by_vec, bz_vec = t96_vectorized(parmod, ps, 
                                                       x_vec_flat, y_vec_flat, z_vec_flat)
                
                Bx_grid = bx_vec.reshape(X_vec.shape)
                By_grid = by_vec.reshape(Y_vec.shape)
                
                B_vec_mag = np.sqrt(Bx_grid**2 + By_grid**2)
                Bx_norm = np.divide(Bx_grid, B_vec_mag, out=np.zeros_like(Bx_grid), 
                                  where=B_vec_mag>0)
                By_norm = np.divide(By_grid, B_vec_mag, out=np.zeros_like(By_grid), 
                                  where=B_vec_mag>0)
                
                ax.quiver(X_vec, Y_vec, Bx_norm*0.5, By_norm*0.5,
                         color='white', alpha=0.5, width=0.002, headwidth=3, 
                         scale=20, scale_units='xy')
            
            # Labels
            if col_idx == 0:
                ax.set_ylabel(f'{condition}\nY GSM (Re)', fontsize=9)
            else:
                ax.set_ylabel('Y GSM (Re)', fontsize=8)
            
            if row_idx == 3:
                ax.set_xlabel('X GSM (Re)', fontsize=9)
            else:
                ax.set_xlabel('X GSM (Re)', fontsize=8)
            
            if row_idx == 0:
                ax.set_title(f'Z = {z_height} Re', fontsize=10, weight='bold')
            
            # Add statistics
            ax.text(0.95, 0.95, f'Med: {median_rl*6371:.0f} km', 
                   transform=ax.transAxes, fontsize=7,
                   ha='right', va='top', color='white', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
            
            ax.set_aspect('equal')
            ax.set_xlim(-15, 5)
            ax.set_ylim(-12, 12)
            ax.grid(True, alpha=0.3)
            
            # Reduce tick labels
            ax.set_xticks([-15, -10, -5, 0, 5])
            ax.set_yticks([-10, -5, 0, 5, 10])
            ax.tick_params(labelsize=7)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label(f'Larmor Radius ({energy} keV) [Re]', fontsize=12)
    cbar.ax.tick_params(labelsize=9)
    
    plt.suptitle(f'Electron Larmor Radius: Parameter and Height Variations ({energy} keV)\n' +
                'Viridis colormap: Dark = Small RL, Bright = Large RL', 
                fontsize=16, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    output_file = os.path.join(output_dir, 'fig17_larmor_radius_parameter_variations.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Larmor radius variations saved: {output_file}")


def create_combined_summary():
    """Create summary plots showing how Rc and RL vary with parameters"""
    
    # Parameter ranges
    dst_values = np.array([0, -30, -50, -100, -150, -200])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Fixed energy
    energy = 100  # keV
    
    # Sample the current sheet region
    x_sample = -10.0
    y_sample = 0.0
    z_sample = 0.0
    
    print("\nCalculating parameter dependencies for Rc and RL...")
    
    # Arrays to store results
    rc_min_values = []
    rc_median_values = []
    rl_max_values = []
    rl_median_values = []
    
    for dst in dst_values:
        parmod = [3.0, dst, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Create small grid around sample point
        x_grid = np.linspace(x_sample-2, x_sample+2, 21)
        z_grid = np.linspace(z_sample-1, z_sample+1, 11)
        X, Z = np.meshgrid(x_grid, z_grid)
        Y = np.full_like(X, y_sample)
        
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        
        # Calculate curvature
        kappa = field_line_curvature_vectorized(t96_vectorized, parmod, ps, 
                                               x_flat, y_flat, z_flat)
        Rc_Re = np.where(kappa > 1e-10, 1.0 / kappa, 1e3)
        
        rc_min_values.append(np.min(Rc_Re))
        rc_median_values.append(np.median(Rc_Re))
        
        # Calculate field and Larmor radius
        bx, by, bz = t96_vectorized(parmod, ps, x_flat, y_flat, z_flat)
        B_magnitude = np.sqrt(bx**2 + by**2 + bz**2)
        RL_m = calculate_larmor_radius(energy, B_magnitude)
        RL_Re = RL_m / Re
        
        rl_max_values.append(np.max(RL_Re))
        rl_median_values.append(np.median(RL_Re))
    
    # Plot 1: Curvature radius vs Dst
    ax1.plot(dst_values, rc_min_values, 'b-o', linewidth=2, markersize=8, label='Min Rc')
    ax1.plot(dst_values, rc_median_values, 'b--s', linewidth=2, markersize=6, label='Median Rc')
    ax1.set_xlabel('Dst (nT)', fontsize=12)
    ax1.set_ylabel('Radius of Curvature (Re)', fontsize=12)
    ax1.set_title('Curvature Radius vs Storm Index', fontsize=14, weight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0.01, 100)
    
    # Plot 2: Larmor radius vs Dst
    ax2.plot(dst_values, np.array(rl_max_values)*6371, 'g-o', linewidth=2, markersize=8, label='Max RL')
    ax2.plot(dst_values, np.array(rl_median_values)*6371, 'g--s', linewidth=2, markersize=6, label='Median RL')
    ax2.set_xlabel('Dst (nT)', fontsize=12)
    ax2.set_ylabel('Larmor Radius (km)', fontsize=12)
    ax2.set_title(f'{energy} keV Electron Larmor Radius vs Storm Index', fontsize=14, weight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(10, 10000)
    
    # Add annotations
    ax1.text(0.05, 0.95, f'Sample region:\nX = {x_sample} Re\nY = {y_sample} Re\nZ = {z_sample} Re',
            transform=ax1.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.suptitle('Curvature and Larmor Radius Dependence on Storm Conditions', 
                fontsize=16, weight='bold')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'fig18_rc_rl_parameter_summary.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Summary plot saved: {output_file}")


if __name__ == "__main__":
    # Create curvature radius variations
    create_curvature_radius_variations()
    
    # Create Larmor radius variations
    create_larmor_radius_variations()
    
    # Create summary plots
    create_combined_summary()
    
    print("\n" + "="*80)
    print("Curvature and Larmor radius parameter variations complete!")
    print("="*80)
    print("\nKey findings:")
    print("- Minimum Rc decreases dramatically during storms")
    print("- Maximum RL increases during storms (weaker B field)")
    print("- Current sheet region shows strongest parameter dependence")
    print("- Extreme conditions create optimal conditions for scattering")
    print("="*80)