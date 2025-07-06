#!/usr/bin/env python3
"""
Magnetic Field Parameter Variations
Shows how different model parameters affect Rc/RL ratio
Creates variations of Figure 3 (XZ plane) and Figure 5 (XY slices)
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

# Critical threshold
CRITICAL_RATIO = 8.0

print("="*80)
print("MAGNETIC FIELD PARAMETER VARIATIONS")
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
    
    # Define parameter sets
    param_sets = [
        # [Pdyn, Dst, By_IMF, Bz_IMF, description]
        ([2.0, 0.0, 0.0, 0.0], "Quiet: Pdyn=2, Dst=0"),
        ([3.0, -30.0, 1.0, -3.0], "Moderate: Pdyn=3, Dst=-30"),
        ([5.0, -100.0, 5.0, -10.0], "Storm: Pdyn=5, Dst=-100"),
        ([1.0, -200.0, 10.0, -20.0], "Extreme: Pdyn=1, Dst=-200")
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
        
        # Create parmod array
        parmod = params + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Calculate field
        x_flat = X_bg.flatten()
        y_flat = Y_bg.flatten()
        z_flat = Z_bg.flatten()
        
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
        
        bx_vec, by_vec, bz_vec = t96_vectorized(parmod, ps, 
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
        ax.text(0.02, 0.98, param_text, transform=ax.transAxes, 
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Rc/RL Ratio', fontsize=12)
    cbar.ax.axhline(y=8, color='black', linewidth=2)
    
    plt.suptitle('Rc/RL Ratio in XZ Plane (Y=0): Parameter Variations\n100 keV Electrons', 
                fontsize=16, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    output_file = os.path.join(output_dir, 'fig10_rcrl_xz_parameter_variations.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nFigure 3 variations saved: {output_file}")


def create_fig5_variations():
    """Create XY plane slices with different model parameters (like Figure 5)"""
    
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
    
    # Fixed energy
    energy = 100  # keV
    
    # Color levels
    levels = np.logspace(-1, 3, 20)
    
    print("\nCreating XY plane variations...")
    
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
            
            # Calculate field
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
            ratio_grid = ratio.reshape(X.shape)
            
            # Calculate statistics
            scatter_frac = np.sum(ratio < CRITICAL_RATIO) / len(ratio) * 100
            
            # Plot
            im = ax.contourf(X, Y, ratio_grid, levels=levels, 
                            cmap='RdBu_r', norm=LogNorm(vmin=0.1, vmax=1000))
            
            # Add critical contour
            try:
                cs = ax.contour(X, Y, ratio_grid, levels=[CRITICAL_RATIO], 
                               colors='black', linewidths=2)
                ax.clabel(cs, inline=True, fontsize=7, fmt='8')
            except:
                # Skip labeling if no contours found
                pass
            
            # Add Earth
            earth = plt.Circle((0, 0), 1, color='white', zorder=10)
            ax.add_patch(earth)
            
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
            
            # Add scattering percentage
            ax.text(0.95, 0.95, f'{scatter_frac:.1f}%', 
                   transform=ax.transAxes, fontsize=8,
                   ha='right', va='top', color='black', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            
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
    cbar.set_label('Rc/RL Ratio', fontsize=12)
    cbar.ax.axhline(y=8, color='black', linewidth=2)
    cbar.ax.tick_params(labelsize=9)
    
    plt.suptitle('Rc/RL Ratio in XY Planes: Parameter and Height Variations\n100 keV Electrons', 
                fontsize=16, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    output_file = os.path.join(output_dir, 'fig11_rcrl_xy_parameter_variations.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Figure 5 variations saved: {output_file}")


def create_summary_plot():
    """Create summary plot showing scattering percentage vs parameters"""
    
    # Parameter ranges
    pdyn_values = np.array([1, 2, 3, 5, 10])
    dst_values = np.array([0, -30, -50, -100, -150, -200])
    bz_values = np.array([5, 2, 0, -2, -5, -10, -20])
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Fixed energy
    energy = 100  # keV
    
    # Create small grid for current sheet region
    x_grid = np.linspace(-12, -6, 31)
    z_grid = np.linspace(-0.5, 0.5, 21)
    X, Z = np.meshgrid(x_grid, z_grid)
    Y = np.zeros_like(X)
    
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    
    print("\nCalculating parameter dependencies...")
    
    # 1. Pdyn dependence
    scatter_pdyn = []
    for pdyn in pdyn_values:
        parmod = [pdyn, -30.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        bx, by, bz = t96_vectorized(parmod, ps, x_flat, y_flat, z_flat)
        B_magnitude = np.sqrt(bx**2 + by**2 + bz**2)
        
        kappa = field_line_curvature_vectorized(t96_vectorized, parmod, ps, 
                                               x_flat, y_flat, z_flat)
        Rc_Re = np.where(kappa > 1e-10, 1.0 / kappa, 1e3)
        Rc_m = Rc_Re * Re
        
        RL_m = calculate_larmor_radius(energy, B_magnitude)
        ratio = Rc_m / RL_m
        
        scatter_frac = np.sum(ratio < CRITICAL_RATIO) / len(ratio) * 100
        scatter_pdyn.append(scatter_frac)
    
    ax1.plot(pdyn_values, scatter_pdyn, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Pdyn (nPa)', fontsize=12)
    ax1.set_ylabel('Scattering Region (%)', fontsize=12)
    ax1.set_title('Dynamic Pressure Effect', fontsize=14, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # 2. Dst dependence
    scatter_dst = []
    for dst in dst_values:
        parmod = [3.0, dst, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        bx, by, bz = t96_vectorized(parmod, ps, x_flat, y_flat, z_flat)
        B_magnitude = np.sqrt(bx**2 + by**2 + bz**2)
        
        kappa = field_line_curvature_vectorized(t96_vectorized, parmod, ps, 
                                               x_flat, y_flat, z_flat)
        Rc_Re = np.where(kappa > 1e-10, 1.0 / kappa, 1e3)
        Rc_m = Rc_Re * Re
        
        RL_m = calculate_larmor_radius(energy, B_magnitude)
        ratio = Rc_m / RL_m
        
        scatter_frac = np.sum(ratio < CRITICAL_RATIO) / len(ratio) * 100
        scatter_dst.append(scatter_frac)
    
    ax2.plot(dst_values, scatter_dst, 'r-s', linewidth=2, markersize=8)
    ax2.set_xlabel('Dst (nT)', fontsize=12)
    ax2.set_ylabel('Scattering Region (%)', fontsize=12)
    ax2.set_title('Storm Index Effect', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. IMF Bz dependence
    scatter_bz = []
    for bz in bz_values:
        parmod = [3.0, -30.0, 1.0, bz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        bx, by, bz_field = t96_vectorized(parmod, ps, x_flat, y_flat, z_flat)
        B_magnitude = np.sqrt(bx**2 + by**2 + bz_field**2)
        
        kappa = field_line_curvature_vectorized(t96_vectorized, parmod, ps, 
                                               x_flat, y_flat, z_flat)
        Rc_Re = np.where(kappa > 1e-10, 1.0 / kappa, 1e3)
        Rc_m = Rc_Re * Re
        
        RL_m = calculate_larmor_radius(energy, B_magnitude)
        ratio = Rc_m / RL_m
        
        scatter_frac = np.sum(ratio < CRITICAL_RATIO) / len(ratio) * 100
        scatter_bz.append(scatter_frac)
    
    ax3.plot(bz_values, scatter_bz, 'g-^', linewidth=2, markersize=8)
    ax3.set_xlabel('IMF Bz (nT)', fontsize=12)
    ax3.set_ylabel('Scattering Region (%)', fontsize=12)
    ax3.set_title('IMF Bz Effect', fontsize=14, weight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.suptitle('Pitch Angle Scattering Dependence on Model Parameters\n' +
                'Current Sheet Region (-12 < X < -6 Re, |Z| < 0.5 Re), 100 keV',
                fontsize=16, weight='bold')
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'fig12_parameter_dependence_summary.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Summary plot saved: {output_file}")


if __name__ == "__main__":
    # Create Figure 3 variations (XZ plane)
    create_fig3_variations()
    
    # Create Figure 5 variations (XY slices)
    create_fig5_variations()
    
    # Create summary plot
    create_summary_plot()
    
    print("\n" + "="*80)
    print("Parameter variation analysis complete!")
    print("="*80)
    print("\nKey findings:")
    print("- Scattering increases during storms (more negative Dst)")
    print("- Lower Pdyn can increase scattering (tail stretching)")
    print("- Southward IMF (negative Bz) enhances scattering")
    print("- Extreme conditions can create large scattering regions")
    print("="*80)