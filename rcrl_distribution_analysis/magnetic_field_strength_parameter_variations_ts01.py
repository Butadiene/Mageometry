#!/usr/bin/env python3
"""
Magnetic Field Strength with Parameter Variations using Ts01 Model
Shows how different model parameters affect magnetic field strength
Similar to original Figure 3 but with parameter variations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import geopack
from geopack import t01_vectorized

# Create output directory
output_dir = "figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*80)
print("MAGNETIC FIELD STRENGTH PARAMETER VARIATIONS - TS01 MODEL")
print("="*80)

# Initialize geopack
ut = 1584662400  # March 20, 2020 (equinox)
ps = geopack.recalc(ut)

print(f"\nModel conditions:")
print(f"Dipole tilt: {np.degrees(ps):.2f}°")


def create_field_strength_variations():
    """Create XZ plane plots showing magnetic field strength with different parameters"""
    
    # Define parameter sets for Ts01
    param_sets = [
        # [Pdyn, Dst, By_IMF, Bz_IMF, G1, G2, description]
        ([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], "Quiet: Pdyn=2, Dst=0"),
        ([3.0, -30.0, 1.0, -3.0, 1.5, 1.0], "Moderate: Pdyn=3, Dst=-30"),
        ([5.0, -100.0, 5.0, -10.0, 3.0, 2.0], "Storm: Pdyn=5, Dst=-100"),
        ([1.0, -200.0, 10.0, -20.0, 5.0, 3.0], "Extreme: Pdyn=1, Dst=-200")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Create grid for background color (field strength)
    x_bg = np.linspace(-20, 10, 151)
    z_bg = np.linspace(-10, 10, 101)
    X_bg, Z_bg = np.meshgrid(x_bg, z_bg)
    Y_bg = np.zeros_like(X_bg)
    
    # Create coarser grid for vectors
    x_vec = np.linspace(-20, 10, 31)
    z_vec = np.linspace(-10, 10, 21)
    X_vec, Z_vec = np.meshgrid(x_vec, z_vec)
    Y_vec = np.zeros_like(X_vec)
    
    # Store field strength ranges for comparison
    field_stats = []
    
    for idx, (params, description) in enumerate(param_sets):
        ax = axes[idx]
        print(f"\nProcessing {description}...")
        
        # Create parmod array
        parmod = params + [0.0, 0.0, 0.0, 0.0]
        
        # Calculate field for background
        x_flat = X_bg.flatten()
        y_flat = Y_bg.flatten()
        z_flat = Z_bg.flatten()
        
        bx_bg, by_bg, bz_bg = t01_vectorized(parmod, ps, x_flat, y_flat, z_flat)
        B_magnitude = np.sqrt(bx_bg**2 + by_bg**2 + bz_bg**2)
        B_grid = B_magnitude.reshape(X_bg.shape)
        
        # Calculate statistics
        # Exclude regions very close to Earth (within 2 Re) to avoid singularities
        r_dist = np.sqrt(x_flat**2 + z_flat**2)
        mask_far = r_dist > 2.0
        
        b_min = np.min(B_magnitude[mask_far]) if np.any(mask_far) else np.min(B_magnitude)
        b_max = np.max(B_magnitude[mask_far]) if np.any(mask_far) else np.max(B_magnitude)
        b_median = np.median(B_magnitude[mask_far]) if np.any(mask_far) else np.median(B_magnitude)
        field_stats.append((description, b_min, b_max, b_median))
        
        # Plot field strength as background
        # Cap extreme values to avoid numerical artifacts
        B_grid_capped = np.where(B_grid > 50000, 50000, B_grid)
        
        im = ax.contourf(X_bg, Z_bg, B_grid_capped, levels=50, cmap='viridis', 
                         norm=LogNorm(vmin=1, vmax=10000))
        
        # Add contours for specific field strengths
        contour_levels = [10, 20, 50, 100, 200]
        cs = ax.contour(X_bg, Z_bg, B_grid, levels=contour_levels, 
                       colors='white', linewidths=1, alpha=0.5)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%d nT')
        
        # Calculate field vectors on coarser grid
        x_vec_flat = X_vec.flatten()
        y_vec_flat = Y_vec.flatten()
        z_vec_flat = Z_vec.flatten()
        
        bx_vec, by_vec, bz_vec = t01_vectorized(parmod, ps, 
                                               x_vec_flat, y_vec_flat, z_vec_flat)
        
        Bx_grid = bx_vec.reshape(X_vec.shape)
        Bz_grid = bz_vec.reshape(Z_vec.shape)
        
        # Normalize vectors
        B_vec_mag = np.sqrt(Bx_grid**2 + Bz_grid**2)
        Bx_norm = np.divide(Bx_grid, B_vec_mag, out=np.zeros_like(Bx_grid), 
                          where=B_vec_mag>0)
        Bz_norm = np.divide(Bz_grid, B_vec_mag, out=np.zeros_like(Bz_grid), 
                          where=B_vec_mag>0)
        
        # Plot vectors
        ax.quiver(X_vec, Z_vec, Bx_norm, Bz_norm,
                 color='white', alpha=0.8, width=0.002, headwidth=3,
                 scale=20, scale_units='xy')
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax.add_patch(earth)
        ax.text(0, 0, 'E', ha='center', va='center', fontsize=10, weight='bold')
        
        # Labels and formatting
        ax.set_xlabel('X GSM (Re)', fontsize=10)
        ax.set_ylabel('Z GSM (Re)', fontsize=10)
        ax.set_title(f'{description}\nB range: {b_min:.1f}-{b_max:.0f} nT', 
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
    cbar.set_label('|B| (nT)', fontsize=12)
    
    plt.suptitle('Magnetic Field Strength in XZ Plane (Y=0): Parameter Variations\nTs01 Model', 
                fontsize=16, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    output_file = os.path.join(output_dir, 'fig13_field_strength_parameter_variations_ts01.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nFigure saved: {output_file}")
    
    # Print field statistics summary
    print("\nField strength statistics summary:")
    print("-" * 60)
    print(f"{'Condition':<30} {'Min (nT)':<10} {'Max (nT)':<10} {'Median (nT)':<10}")
    print("-" * 60)
    for desc, b_min, b_max, b_med in field_stats:
        print(f"{desc:<30} {b_min:<10.1f} {b_max:<10.0f} {b_med:<10.1f}")
    
    return field_stats


def create_current_sheet_profiles():
    """Create profiles showing current sheet structure in different conditions"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Define Z profiles at different X locations
    x_locations = [-5, -10, -15, -20]
    z_profile = np.linspace(-5, 5, 101)
    
    # Quiet conditions
    parmod_quiet = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # Storm conditions
    parmod_storm = [5.0, -100.0, 5.0, -10.0, 3.0, 2.0, 0.0, 0.0, 0.0, 0.0]
    
    # Plot Bx component (current sheet indicator)
    for x_loc in x_locations:
        # Quiet conditions
        x_arr = np.full_like(z_profile, x_loc)
        y_arr = np.zeros_like(z_profile)
        
        bx_quiet, _, _ = t01_vectorized(parmod_quiet, ps, x_arr, y_arr, z_profile)
        ax1.plot(z_profile, bx_quiet, label=f'X={x_loc} Re', linewidth=2)
        
        # Storm conditions
        bx_storm, _, _ = t01_vectorized(parmod_storm, ps, x_arr, y_arr, z_profile)
        ax2.plot(z_profile, bx_storm, label=f'X={x_loc} Re', linewidth=2)
    
    # Formatting for quiet conditions
    ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax1.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Z GSM (Re)', fontsize=12)
    ax1.set_ylabel('Bx (nT)', fontsize=12)
    ax1.set_title('Quiet Conditions\n(Pdyn=2 nPa, Dst=0 nT)', fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(-5, 5)
    
    # Formatting for storm conditions
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Z GSM (Re)', fontsize=12)
    ax2.set_ylabel('Bx (nT)', fontsize=12)
    ax2.set_title('Storm Conditions\n(Pdyn=5 nPa, Dst=-100 nT)', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(-5, 5)
    
    plt.suptitle('Current Sheet Profiles (Bx vs Z) at Y=0\nTs01 Model', 
                fontsize=14, weight='bold')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'fig14_field_strength_profiles_ts01.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nCurrent sheet profiles saved: {output_file}")


def create_field_components_comparison():
    """Compare field components in different parameter regimes"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Create grid
    x_grid = np.linspace(-20, 10, 151)
    z_grid = np.linspace(-10, 10, 101)
    X, Z = np.meshgrid(x_grid, z_grid)
    Y = np.zeros_like(X)
    
    # Parameters for comparison
    parmod_quiet = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    parmod_storm = [5.0, -100.0, 5.0, -10.0, 3.0, 2.0, 0.0, 0.0, 0.0, 0.0]
    
    param_sets = [
        (parmod_quiet, "Quiet Conditions"),
        (parmod_storm, "Storm Conditions")
    ]
    
    components = ['Bx', 'By', 'Bz']
    
    for row_idx, (parmod, condition) in enumerate(param_sets):
        print(f"\nProcessing {condition}...")
        
        # Calculate field
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        
        bx, by, bz = t01_vectorized(parmod, ps, x_flat, y_flat, z_flat)
        
        field_components = [
            bx.reshape(X.shape),
            by.reshape(Y.shape),
            bz.reshape(Z.shape)
        ]
        
        for col_idx, (component, comp_name) in enumerate(zip(field_components, components)):
            ax = axes[row_idx, col_idx]
            
            # Use symmetric color scale for components
            vmax = np.percentile(np.abs(component), 95)
            
            im = ax.contourf(X, Z, component, levels=50, cmap='RdBu_r',
                            vmin=-vmax, vmax=vmax)
            
            # Add contours
            cs = ax.contour(X, Z, component, levels=[-50, -20, -10, 0, 10, 20, 50],
                          colors='black', linewidths=0.5, alpha=0.5)
            ax.clabel(cs, inline=True, fontsize=6)
            
            # Add Earth
            earth = plt.Circle((0, 0), 1, color='gray', zorder=10)
            ax.add_patch(earth)
            
            # Labels
            ax.set_xlabel('X GSM (Re)', fontsize=10)
            ax.set_ylabel('Z GSM (Re)', fontsize=10)
            ax.set_title(f'{comp_name} - {condition}', fontsize=11, weight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-20, 10)
            ax.set_ylim(-10, 10)
            
            # Add colorbar for each subplot
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('nT', fontsize=9)
    
    plt.suptitle('Magnetic Field Components in XZ Plane (Y=0)\nTs01 Model', 
                fontsize=16, weight='bold')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'fig15_field_components_comparison_ts01.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nField components comparison saved: {output_file}")


if __name__ == "__main__":
    # Create all plots
    field_stats = create_field_strength_variations()
    create_current_sheet_profiles()
    create_field_components_comparison()
    
    print("\n" + "="*80)
    print("FIELD STRENGTH ANALYSIS COMPLETE (Ts01)")
    print("="*80)
    print("\nKey findings:")
    print("- Field strength varies dramatically with geomagnetic conditions")
    print("- Current sheet becomes thinner during storms")
    print("- Tail field stretching is evident in extreme conditions")
    print("- Field topology changes significantly with Dst")
    print("- Ts01 captures storm-time dynamics through G1 and G2 parameters")
    print("="*80)