#!/usr/bin/env python3
"""
Magnetic Field Parameter Variations using Ts01 Model - Figure 11 Extended to X=-15 Re
Shows how different model parameters affect Rc/RL ratio in XY plane
Extended X range to show full validity range of Ts01 model
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import geopack
from geopack import t01_vectorized, field_line_curvature_vectorized

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
print("CREATING FIGURE 11 - TS01 MODEL (EXTENDED X RANGE)")
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


def create_fig11_extended():
    """Create XY plane slices with different model parameters - extended X range"""
    
    # Define parameter sets
    param_sets = [
        # [Pdyn, Dst, By_IMF, Bz_IMF, G1, G2, description]
        ([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], "Quiet"),
        ([3.0, -30.0, 1.0, -3.0, 1.5, 1.0], "Moderate"),
        ([5.0, -100.0, 5.0, -10.0, 3.0, 2.0], "Storm"),
        ([1.0, -200.0, 10.0, -20.0, 5.0, 3.0], "Extreme")
    ]
    
    # Z heights to show
    z_heights = [-0.2, 0.0, 0.2, 0.4]
    
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    
    # Create XY grid extending to X = -15 Re (actually use -14.95 to avoid edge issues)
    x_grid = np.linspace(-14.95, 5, 101)  # Extended X range
    y_grid = np.linspace(-10, 10, 101)   # Y range
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Fixed energy
    energy = 100  # keV
    
    # Color levels
    levels = np.logspace(-1, 3, 20)
    
    # Store statistics
    all_stats = []
    
    # Process each parameter set
    for row_idx, (params, condition) in enumerate(param_sets):
        print(f"\nProcessing {condition} conditions...")
        
        # Create parmod array
        parmod = params + [0.0, 0.0, 0.0, 0.0]
        
        for col_idx, z_height in enumerate(z_heights):
            ax = axes[row_idx, col_idx]
            print(f"  Z = {z_height} Re...")
            
            # Create Z array for this height
            Z = np.full_like(X, z_height)
            
            # Flatten for calculation
            x_flat = X.flatten()
            y_flat = Y.flatten()
            z_flat = Z.flatten()
            
            # Calculate magnetic field
            bx, by, bz = t01_vectorized(parmod, ps, x_flat, y_flat, z_flat)
            B_magnitude = np.sqrt(bx**2 + by**2 + bz**2)
            
            # Calculate curvature
            kappa = field_line_curvature_vectorized(t01_vectorized, parmod, ps, 
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
            all_stats.append((condition, z_height, scatter_frac))
            
            # Plot Rc/RL ratio
            im = ax.contourf(X, Y, ratio_grid, levels=levels, 
                            cmap='RdBu_r', norm=LogNorm(vmin=0.1, vmax=1000))
            
            # Add critical contour
            try:
                cs = ax.contour(X, Y, ratio_grid, levels=[CRITICAL_RATIO], 
                               colors='black', linewidths=2)
                ax.clabel(cs, inline=True, fontsize=6, fmt='8')
            except:
                pass  # Skip if no contour found
            
            # Add Earth
            earth = plt.Circle((0, 0), 1, color='white', zorder=10)
            ax.add_patch(earth)
            
            # Add field vectors (sparse) for better visibility in extended range
            if row_idx % 2 == 0:  # Only for some panels
                x_vec = np.linspace(-14.95, 5, 13)  # More vectors across extended range
                y_vec = np.linspace(-10, 10, 11)
                X_vec, Y_vec = np.meshgrid(x_vec, y_vec)
                Z_vec = np.full_like(X_vec, z_height)
                
                x_vec_flat = X_vec.flatten()
                y_vec_flat = Y_vec.flatten()
                z_vec_flat = Z_vec.flatten()
                
                bx_vec, by_vec, bz_vec = t01_vectorized(parmod, ps, 
                                                       x_vec_flat, y_vec_flat, z_vec_flat)
                
                Bx_grid = bx_vec.reshape(X_vec.shape)
                By_grid = by_vec.reshape(Y_vec.shape)
                
                B_vec_mag = np.sqrt(Bx_grid**2 + By_grid**2)
                Bx_norm = np.divide(Bx_grid, B_vec_mag, out=np.zeros_like(Bx_grid), 
                                  where=B_vec_mag>0)
                By_norm = np.divide(By_grid, B_vec_mag, out=np.zeros_like(By_grid), 
                                  where=B_vec_mag>0)
                
                ax.quiver(X_vec, Y_vec, Bx_norm, By_norm,
                         color='white', alpha=0.6, width=0.002, headwidth=3, 
                         scale=25, scale_units='xy')
            
            # Labels and formatting
            if col_idx == 0:
                ax.set_ylabel(f'{condition}\nY GSM (Re)', fontsize=9)
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])
                
            if row_idx == 3:
                ax.set_xlabel('X GSM (Re)', fontsize=9)
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])
            
            if row_idx == 0:
                ax.set_title(f'Z = {z_height} Re\n({scatter_frac:.1f}% < 8)', 
                           fontsize=9, weight='bold')
            else:
                ax.set_title(f'{scatter_frac:.1f}% < 8', fontsize=8)
            
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_xlim(-15, 5)  # Set to exactly -15 Re
            ax.set_ylim(-10, 10)
            ax.tick_params(labelsize=8)
            
            # Add X=-15 Re boundary line
            ax.axvline(x=-15, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            if row_idx == 0 and col_idx == 0:
                ax.text(-14.5, 9, 'Ts01 limit', fontsize=7, ha='center', 
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Add a single colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax, 
                       ticks=[0.1, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000])
    cbar.set_label('Rc/RL Ratio', fontsize=12)
    cbar.ax.axhline(y=8, color='black', linewidth=3)
    cbar.ax.text(1.3, 8, 'Critical', fontsize=9, va='center')
    
    plt.suptitle('Rc/RL Ratio in XY Planes: Parameter and Height Variations (Extended to X=-15 Re)\n' +
                'Ts01 Model, 100 keV Electrons', 
                fontsize=16, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    output_file = os.path.join(output_dir, 'fig11_rcrl_xy_parameter_variations_ts01_extended.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nFigure 11 (Ts01 Extended) saved: {output_file}")
    
    # Print statistics summary
    print("\nScattering statistics summary (% with Rc/RL < 8):")
    print("-" * 60)
    print(f"{'Condition':<15} {'Z=-0.2':<8} {'Z=0.0':<8} {'Z=0.2':<8} {'Z=0.4':<8}")
    print("-" * 60)
    
    for condition in ["Quiet", "Moderate", "Storm", "Extreme"]:
        row_stats = [s[2] for s in all_stats if s[0] == condition]
        print(f"{condition:<15} {row_stats[0]:>7.1f} {row_stats[1]:>7.1f} "
              f"{row_stats[2]:>7.1f} {row_stats[3]:>7.1f}")


if __name__ == "__main__":
    create_fig11_extended()
    
    print("\n" + "="*80)
    print("FIGURE 11 EXTENDED COMPLETE (Ts01)")
    print("="*80)
    print("Note: X range extended to -15 Re (Ts01 model validity limit)")
    print("="*80)