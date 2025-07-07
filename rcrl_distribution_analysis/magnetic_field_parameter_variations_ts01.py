#!/usr/bin/env python3
"""
Magnetic Field Parameter Variations using Ts01 Model
Shows how different model parameters affect Rc/RL ratio
Creates variations of Figure 3 (XZ plane) and Figure 5 (XY slices)
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
print("MAGNETIC FIELD PARAMETER VARIATIONS - TS01 MODEL")
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


def create_fig3_variations():
    """Create XZ plane plots with different model parameters (like Figure 3)"""
    
    # Define parameter sets for Ts01
    # [Pdyn, Dst, ByIMF, BzIMF, G1, G2, description]
    param_sets = [
        ([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], "Quiet: Pdyn=2, Dst=0"),
        ([3.0, -30.0, 1.0, -3.0, 1.5, 1.0], "Moderate: Pdyn=3, Dst=-30"),
        ([5.0, -100.0, 5.0, -10.0, 3.0, 2.0], "Storm: Pdyn=5, Dst=-100"),
        ([1.0, -200.0, 10.0, -20.0, 5.0, 3.0], "Extreme: Pdyn=1, Dst=-200")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Create grid for XZ plane
    x_bg = np.linspace(-20, 10, 151)
    z_bg = np.linspace(-10, 10, 101)
    X_bg, Z_bg = np.meshgrid(x_bg, z_bg)
    Y_bg = np.zeros_like(X_bg)
    
    # Fixed energy
    energy = 100  # keV
    
    # Color levels
    levels = np.logspace(-1, 3, 20)
    
    for idx, (params, description) in enumerate(param_sets):
        ax = axes[idx]
        print(f"\nProcessing {description}...")
        
        # Create parmod array for Ts01
        parmod = params + [0.0, 0.0, 0.0, 0.0]
        
        # Calculate field
        x_flat = X_bg.flatten()
        y_flat = Y_bg.flatten()
        z_flat = Z_bg.flatten()
        
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
        ratio_grid = ratio.reshape(X_bg.shape)
        
        # Calculate statistics
        scatter_frac = np.sum(ratio < CRITICAL_RATIO) / len(ratio) * 100
        
        # Plot Rc/RL ratio
        im = ax.contourf(X_bg, Z_bg, ratio_grid, levels=levels, 
                        cmap='RdBu_r', norm=LogNorm(vmin=0.1, vmax=1000))
        
        # Add critical contour
        cs = ax.contour(X_bg, Z_bg, ratio_grid, levels=[CRITICAL_RATIO], 
                       colors='black', linewidths=2)
        ax.clabel(cs, inline=True, fontsize=8, fmt='8')
        
        # Add vectors (sparse)
        x_vec = np.linspace(-20, 10, 16)
        z_vec = np.linspace(-10, 10, 11)
        X_vec, Z_vec = np.meshgrid(x_vec, z_vec)
        Y_vec = np.zeros_like(X_vec)
        
        x_vec_flat = X_vec.flatten()
        y_vec_flat = Y_vec.flatten()
        z_vec_flat = Z_vec.flatten()
        
        bx_vec, by_vec, bz_vec = t01_vectorized(parmod, ps, 
                                               x_vec_flat, y_vec_flat, z_vec_flat)
        
        Bx_grid = bx_vec.reshape(X_vec.shape)
        Bz_grid = bz_vec.reshape(Z_vec.shape)
        
        B_vec_mag = np.sqrt(Bx_grid**2 + Bz_grid**2)
        Bx_norm = np.divide(Bx_grid, B_vec_mag, out=np.zeros_like(Bx_grid), 
                          where=B_vec_mag>0)
        Bz_norm = np.divide(Bz_grid, B_vec_mag, out=np.zeros_like(Bz_grid), 
                          where=B_vec_mag>0)
        
        ax.quiver(X_vec, Z_vec, Bx_norm*0.4, Bz_norm*0.4,
                 color='white', alpha=0.7, width=0.002, headwidth=3, 
                 scale=12, scale_units='xy', edgecolor='black', linewidth=0.3)
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax.add_patch(earth)
        ax.text(0, 0, 'E', ha='center', va='center', fontsize=10, weight='bold')
        
        # Labels and formatting
        ax.set_xlabel('X GSM (Re)', fontsize=10)
        ax.set_ylabel('Z GSM (Re)', fontsize=10)
        ax.set_title(f'{description}\nScattering: {scatter_frac:.1f}%', 
                    fontsize=11, weight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-20, 10)
        ax.set_ylim(-10, 10)
        
        # Add parameter text
        param_text = f'Pdyn={params[0]} nPa\nDst={params[1]} nT\nIMF By={params[2]} nT\nIMF Bz={params[3]} nT'
        if params[4] > 0 or params[5] > 0:
            param_text += f'\nG1={params[4]:.1f}, G2={params[5]:.1f}'
        ax.text(0.02, 0.98, param_text, transform=ax.transAxes, 
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Rc/RL Ratio', fontsize=12)
    cbar.ax.axhline(y=8, color='black', linewidth=2)
    
    plt.suptitle('Rc/RL Ratio in XZ Plane (Y=0): Parameter Variations\nTs01 Model, 100 keV Electrons', 
                fontsize=16, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    output_file = os.path.join(output_dir, 'fig10_rcrl_xz_parameter_variations_ts01.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nFigure 10 (Ts01) saved: {output_file}")


def create_fig5_variations():
    """Create XY plane slices with different model parameters (like Figure 5)"""
    
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
    
    # Create XY grid
    x_grid = np.linspace(-20, 5, 126)
    y_grid = np.linspace(-12, 12, 121)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Fixed energy
    energy = 100  # keV
    
    # Color levels
    levels = np.logspace(-1, 3, 20)
    
    # Store statistics
    all_stats = []
    
    for row_idx, (params, condition) in enumerate(param_sets):
        print(f"\nProcessing {condition} conditions...")
        
        # Create parmod array
        parmod = params + [0.0, 0.0, 0.0, 0.0]
        
        for col_idx, z_height in enumerate(z_heights):
            ax = axes[row_idx, col_idx]
            
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
            cs = ax.contour(X, Y, ratio_grid, levels=[CRITICAL_RATIO], 
                           colors='black', linewidths=2)
            try:
                ax.clabel(cs, inline=True, fontsize=6, fmt='8')
            except:
                pass  # Skip labeling if no contours found
            
            # Add Earth
            earth = plt.Circle((0, 0), 1, color='white', zorder=10)
            ax.add_patch(earth)
            
            # Labels and formatting
            if col_idx == 0:
                ax.set_ylabel(f'{condition}\nY GSM (Re)', fontsize=9)
            if row_idx == 3:
                ax.set_xlabel('X GSM (Re)', fontsize=9)
            
            if row_idx == 0:
                ax.set_title(f'Z = {z_height} Re\n({scatter_frac:.1f}% < 8)', 
                           fontsize=9, weight='bold')
            else:
                ax.set_title(f'{scatter_frac:.1f}% < 8', fontsize=8)
            
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_xlim(-20, 5)
            ax.set_ylim(-12, 12)
            ax.tick_params(labelsize=8)
    
    # Add a single colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax, 
                       ticks=[0.1, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000])
    cbar.set_label('Rc/RL Ratio', fontsize=12)
    cbar.ax.axhline(y=8, color='black', linewidth=3)
    cbar.ax.text(1.3, 8, 'Critical', fontsize=9, va='center')
    
    plt.suptitle('Rc/RL Ratio in XY Planes: Parameter and Height Variations\nTs01 Model, 100 keV Electrons', 
                fontsize=16, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    output_file = os.path.join(output_dir, 'fig11_rcrl_xy_parameter_variations_ts01.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nFigure 11 (Ts01) saved: {output_file}")
    
    # Print statistics summary
    print("\nScattering statistics summary (% with Rc/RL < 8):")
    print("-" * 60)
    print(f"{'Condition':<15} {'Z=-0.2':<8} {'Z=0.0':<8} {'Z=0.2':<8} {'Z=0.4':<8}")
    print("-" * 60)
    
    for condition in ["Quiet", "Moderate", "Storm", "Extreme"]:
        row_stats = [s[2] for s in all_stats if s[0] == condition]
        print(f"{condition:<15} {row_stats[0]:>7.1f} {row_stats[1]:>7.1f} "
              f"{row_stats[2]:>7.1f} {row_stats[3]:>7.1f}")


def create_parameter_summary():
    """Create summary plot showing how scattering varies with parameters"""
    
    # Create more parameter variations for Dst and Pdyn
    dst_values = np.linspace(0, -200, 9)
    pdyn_values = np.linspace(1, 8, 8)
    
    # Fixed IMF conditions
    by_imf = 1.0
    bz_imf = -3.0
    
    # Calculate scattering fraction for each combination
    scatter_dst = []
    scatter_pdyn = []
    
    # Test Dst variation (fixed Pdyn = 3)
    print("\nCalculating Dst variations...")
    for dst in dst_values:
        # G1 and G2 parameters scale with Dst
        g1 = max(0, -dst * 0.025)  # Rough scaling
        g2 = max(0, -dst * 0.015)
        parmod = [3.0, dst, by_imf, bz_imf, g1, g2, 0.0, 0.0, 0.0, 0.0]
        
        # Sample at Z=0 plane
        x_test = np.linspace(-15, 5, 100)
        y_test = np.linspace(-10, 10, 100)
        X_test, Y_test = np.meshgrid(x_test, y_test)
        Z_test = np.zeros_like(X_test)
        
        x_flat = X_test.flatten()
        y_flat = Y_test.flatten()
        z_flat = Z_test.flatten()
        
        bx, by, bz = t01_vectorized(parmod, ps, x_flat, y_flat, z_flat)
        B_magnitude = np.sqrt(bx**2 + by**2 + bz**2)
        
        kappa = field_line_curvature_vectorized(t01_vectorized, parmod, ps, 
                                               x_flat, y_flat, z_flat)
        Rc_Re = np.where(kappa > 1e-10, 1.0 / kappa, 1e3)
        Rc_m = Rc_Re * Re
        
        RL_m = calculate_larmor_radius(100, B_magnitude)
        ratio = Rc_m / RL_m
        
        scatter_frac = np.sum(ratio < CRITICAL_RATIO) / len(ratio) * 100
        scatter_dst.append(scatter_frac)
    
    # Test Pdyn variation (fixed Dst = -30)
    print("Calculating Pdyn variations...")
    for pdyn in pdyn_values:
        parmod = [pdyn, -30.0, by_imf, bz_imf, 1.5, 1.0, 0.0, 0.0, 0.0, 0.0]
        
        x_flat = X_test.flatten()
        y_flat = Y_test.flatten()
        z_flat = Z_test.flatten()
        
        bx, by, bz = t01_vectorized(parmod, ps, x_flat, y_flat, z_flat)
        B_magnitude = np.sqrt(bx**2 + by**2 + bz**2)
        
        kappa = field_line_curvature_vectorized(t01_vectorized, parmod, ps, 
                                               x_flat, y_flat, z_flat)
        Rc_Re = np.where(kappa > 1e-10, 1.0 / kappa, 1e3)
        Rc_m = Rc_Re * Re
        
        RL_m = calculate_larmor_radius(100, B_magnitude)
        ratio = Rc_m / RL_m
        
        scatter_frac = np.sum(ratio < CRITICAL_RATIO) / len(ratio) * 100
        scatter_pdyn.append(scatter_frac)
    
    # Create summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Dst variation
    ax1.plot(dst_values, scatter_dst, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Dst (nT)', fontsize=12)
    ax1.set_ylabel('Scattering Fraction (%)', fontsize=12)
    ax1.set_title('Scattering vs Dst\n(Pdyn=3 nPa, IMF Bz=-3 nT)', fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, -200)
    
    # Pdyn variation
    ax2.plot(pdyn_values, scatter_pdyn, 'r-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Pdyn (nPa)', fontsize=12)
    ax2.set_ylabel('Scattering Fraction (%)', fontsize=12)
    ax2.set_title('Scattering vs Pdyn\n(Dst=-30 nT, IMF Bz=-3 nT)', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 9)
    
    plt.suptitle('Parameter Dependence of Curvature Scattering\nTs01 Model, 100 keV Electrons, Z=0 Re', 
                fontsize=14, weight='bold')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'fig12_parameter_dependence_summary_ts01.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nParameter summary saved: {output_file}")


if __name__ == "__main__":
    # Create all plots
    create_fig3_variations()  # Figure 10
    create_fig5_variations()  # Figure 11
    create_parameter_summary()  # Figure 12
    
    print("\n" + "="*80)
    print("PARAMETER VARIATION ANALYSIS COMPLETE (Ts01)")
    print("="*80)
    print("\nKey findings:")
    print("- Scattering increases during storms (more negative Dst)")
    print("- Lower Pdyn can increase scattering (tail stretching)")
    print("- Southward IMF (negative Bz) enhances scattering")
    print("- Extreme conditions can create large scattering regions")
    print("- Ts01 includes storm-time corrections through G1 and G2 parameters")
    print("="*80)