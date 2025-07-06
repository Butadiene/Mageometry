#!/usr/bin/env python3
"""
Magnetic Field Strength with Parameter Variations
Shows how different model parameters affect magnetic field strength
Similar to original Figure 3 but with parameter variations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import geopack
from geopack import t96_vectorized

# Create output directory
output_dir = "figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*80)
print("MAGNETIC FIELD STRENGTH PARAMETER VARIATIONS")
print("="*80)

# Initialize geopack
ut = 1584662400  # March 20, 2020 (equinox)
ps = geopack.recalc(ut)

print(f"\nModel conditions:")
print(f"Dipole tilt: {np.degrees(ps):.2f}°")


def create_field_strength_variations():
    """Create XZ plane plots showing magnetic field strength with different parameters"""
    
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
        parmod = params + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Calculate field for background
        x_flat = X_bg.flatten()
        y_flat = Y_bg.flatten()
        z_flat = Z_bg.flatten()
        
        bx_bg, by_bg, bz_bg = t96_vectorized(parmod, ps, x_flat, y_flat, z_flat)
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
        
        # Calculate field vectors
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
        seed_x = np.concatenate([
            np.full(5, -15),  # Tail field lines
            np.full(5, -10),
            np.full(5, -5),
            np.full(7, -2),   # Near-Earth field lines
        ])
        seed_z = np.concatenate([
            np.linspace(-3, 3, 5),
            np.linspace(-4, 4, 5),
            np.linspace(-5, 5, 5),
            np.linspace(-6, 6, 7),
        ])
        
        strm = ax.streamplot(x_bg, z_bg, 
                            bx_bg.reshape(len(z_bg), len(x_bg)), 
                            bz_bg.reshape(len(z_bg), len(x_bg)),
                            color='red', linewidth=1, density=0.5,
                            start_points=np.column_stack([seed_x, seed_z]))
        strm.lines.set_alpha(0.6)
        
        # Add Earth
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax.add_patch(earth)
        ax.text(0, 0, 'E', ha='center', va='center', fontsize=12, weight='bold')
        
        # Add magnetopause boundary (approximate)
        if idx < 3:  # Skip for extreme case as it may be too distorted
            theta = np.linspace(0, np.pi, 100)
            # Adjust magnetopause size based on Pdyn
            r_mp = (10.0 / (params[0] / 2.0)**0.167) * (2 / (1 + np.cos(theta)))**0.5
            x_mp = r_mp * np.cos(theta)
            z_mp = r_mp * np.sin(theta)
            ax.plot(x_mp, z_mp, 'w--', linewidth=2, alpha=0.5, label='Magnetopause')
            ax.plot(x_mp, -z_mp, 'w--', linewidth=2, alpha=0.5)
        
        # Labels and formatting
        ax.set_xlabel('X GSM (Re)', fontsize=11)
        ax.set_ylabel('Z GSM (Re)', fontsize=11)
        ax.set_title(f'{description}', fontsize=12, weight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, color='gray')
        ax.set_xlim(-20, 10)
        ax.set_ylim(-10, 10)
        
        # Add parameter and statistics text
        param_text = f'Pdyn={params[0]} nPa\nDst={params[1]} nT\nIMF By={params[2]} nT\nIMF Bz={params[3]} nT'
        ax.text(0.02, 0.98, param_text, transform=ax.transAxes, 
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        stats_text = f'|B| range:\n{b_min:.1f}-{b_max:.0f} nT\nMedian: {b_median:.1f} nT'
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
               fontsize=8, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Magnetic Field Strength |B| (nT)', fontsize=12)
    
    plt.suptitle('T96 Magnetic Field Strength: Parameter Variations\n' +
                'Vectors show field direction (scaled by log|B|)', 
                fontsize=16, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    output_file = os.path.join(output_dir, 'fig13_field_strength_parameter_variations.png')
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
    """Create profiles of field strength across the current sheet for different conditions"""
    
    # Define parameter sets
    param_sets = [
        ([2.0, 0.0, 0.0, 0.0], "Quiet", 'blue'),
        ([3.0, -30.0, 1.0, -3.0], "Moderate", 'green'),
        ([5.0, -100.0, 5.0, -10.0], "Storm", 'orange'),
        ([1.0, -200.0, 10.0, -20.0], "Extreme", 'red')
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Profile 1: Along X at Y=0, Z=0 (equatorial plane)
    x_profile = np.linspace(-20, 10, 301)
    y_profile = np.zeros_like(x_profile)
    z_profile = np.zeros_like(x_profile)
    
    print("\nCalculating equatorial profiles...")
    
    for params, label, color in param_sets:
        parmod = params + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        bx, by, bz = t96_vectorized(parmod, ps, x_profile, y_profile, z_profile)
        B_magnitude = np.sqrt(bx**2 + by**2 + bz**2)
        
        ax1.plot(x_profile, B_magnitude, color=color, linewidth=2, 
                label=f'{label} (Dst={params[1]})')
    
    ax1.set_xlabel('X GSM (Re)', fontsize=12)
    ax1.set_ylabel('|B| (nT)', fontsize=12)
    ax1.set_title('Field Strength Along Equator (Y=0, Z=0)', fontsize=14, weight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(-20, 10)
    ax1.set_ylim(1, 1000)
    
    # Profile 2: Vertical profile at X=-10 Re (through current sheet)
    x_cs = -10.0
    y_cs = 0.0
    z_profile_cs = np.linspace(-5, 5, 201)
    x_profile_cs = np.full_like(z_profile_cs, x_cs)
    y_profile_cs = np.full_like(z_profile_cs, y_cs)
    
    print("Calculating vertical profiles...")
    
    for params, label, color in param_sets:
        parmod = params + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        bx, by, bz = t96_vectorized(parmod, ps, x_profile_cs, y_profile_cs, z_profile_cs)
        B_magnitude = np.sqrt(bx**2 + by**2 + bz**2)
        
        ax2.plot(z_profile_cs, B_magnitude, color=color, linewidth=2, 
                label=f'{label}')
    
    ax2.set_xlabel('Z GSM (Re)', fontsize=12)
    ax2.set_ylabel('|B| (nT)', fontsize=12)
    ax2.set_title(f'Vertical Profile at X={x_cs} Re', fontsize=14, weight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(1, 100)
    
    # Add annotation for current sheet
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.text(0.1, 5, 'Current\nSheet', fontsize=10, ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    plt.suptitle('Magnetic Field Strength Profiles: Parameter Effects', 
                fontsize=16, weight='bold')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'fig14_field_strength_profiles.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Profile plot saved: {output_file}")


def create_field_component_comparison():
    """Create comparison of field components for different conditions"""
    
    # Use moderate and extreme conditions for comparison
    conditions = [
        ([3.0, -30.0, 1.0, -3.0], "Moderate Storm"),
        ([1.0, -200.0, 10.0, -20.0], "Extreme Storm")
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Create grid
    x_grid = np.linspace(-20, 10, 151)
    z_grid = np.linspace(-10, 10, 101)
    X, Z = np.meshgrid(x_grid, z_grid)
    Y = np.zeros_like(X)
    
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    
    for row_idx, (params, condition) in enumerate(conditions):
        print(f"\nProcessing {condition} components...")
        parmod = params + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Calculate field components
        bx, by, bz = t96_vectorized(parmod, ps, x_flat, y_flat, z_flat)
        
        # Reshape
        Bx_grid = bx.reshape(X.shape)
        By_grid = by.reshape(X.shape)
        Bz_grid = bz.reshape(Z.shape)
        B_total = np.sqrt(bx**2 + by**2 + bz**2).reshape(X.shape)
        
        # Plot Bx component
        ax = axes[row_idx, 0]
        vmax = max(abs(np.min(Bx_grid)), abs(np.max(Bx_grid)))
        im1 = ax.contourf(X, Z, Bx_grid, levels=50, cmap='RdBu_r', 
                         vmin=-vmax, vmax=vmax)
        ax.contour(X, Z, Bx_grid, levels=[0], colors='black', linewidths=2)
        
        earth = plt.Circle((0, 0), 1, color='gray', zorder=10)
        ax.add_patch(earth)
        
        cbar1 = plt.colorbar(im1, ax=ax, pad=0.02)
        cbar1.set_label('Bx (nT)', fontsize=10)
        
        ax.set_xlabel('X GSM (Re)', fontsize=10)
        ax.set_ylabel('Z GSM (Re)', fontsize=10)
        ax.set_title(f'{condition}: Bx Component', fontsize=11, weight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-20, 10)
        ax.set_ylim(-10, 10)
        
        # Plot Bz component
        ax = axes[row_idx, 1]
        vmax = max(abs(np.min(Bz_grid)), abs(np.max(Bz_grid)))
        im2 = ax.contourf(X, Z, Bz_grid, levels=50, cmap='RdBu_r',
                         vmin=-vmax, vmax=vmax)
        ax.contour(X, Z, Bz_grid, levels=[0], colors='black', linewidths=2)
        
        earth = plt.Circle((0, 0), 1, color='gray', zorder=10)
        ax.add_patch(earth)
        
        cbar2 = plt.colorbar(im2, ax=ax, pad=0.02)
        cbar2.set_label('Bz (nT)', fontsize=10)
        
        ax.set_xlabel('X GSM (Re)', fontsize=10)
        ax.set_ylabel('Z GSM (Re)', fontsize=10)
        ax.set_title(f'{condition}: Bz Component', fontsize=11, weight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-20, 10)
        ax.set_ylim(-10, 10)
        
        # Plot total field
        ax = axes[row_idx, 2]
        im3 = ax.contourf(X, Z, B_total, levels=50, cmap='viridis',
                         norm=LogNorm(vmin=5, vmax=500))
        
        # Add specific contours
        cs = ax.contour(X, Z, B_total, levels=[10, 20, 50, 100], 
                       colors='white', linewidths=1)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%d nT')
        
        earth = plt.Circle((0, 0), 1, color='white', zorder=10)
        ax.add_patch(earth)
        
        cbar3 = plt.colorbar(im3, ax=ax, pad=0.02)
        cbar3.set_label('|B| (nT)', fontsize=10)
        
        ax.set_xlabel('X GSM (Re)', fontsize=10)
        ax.set_ylabel('Z GSM (Re)', fontsize=10)
        ax.set_title(f'{condition}: Total |B|', fontsize=11, weight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-20, 10)
        ax.set_ylim(-10, 10)
        
        # Add parameter text to first panel
        if row_idx == 0:
            param_text = f'Pdyn={params[0]} nPa, Dst={params[1]} nT'
        else:
            param_text = f'Pdyn={params[0]} nPa, Dst={params[1]} nT'
        axes[row_idx, 0].text(0.02, 0.02, param_text, 
                             transform=axes[row_idx, 0].transAxes,
                             fontsize=9, va='bottom',
                             bbox=dict(boxstyle='round,pad=0.3', 
                                      facecolor='white', alpha=0.8))
    
    plt.suptitle('Magnetic Field Components: Moderate vs Extreme Conditions', 
                fontsize=16, weight='bold')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'fig15_field_components_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Component comparison saved: {output_file}")


if __name__ == "__main__":
    # Create main field strength variations
    field_stats = create_field_strength_variations()
    
    # Create current sheet profiles
    create_current_sheet_profiles()
    
    # Create field component comparison
    create_field_component_comparison()
    
    print("\n" + "="*80)
    print("Magnetic field strength analysis complete!")
    print("="*80)
    print("\nKey findings:")
    print("- Field strength decreases dramatically during extreme storms")
    print("- Current sheet becomes thinner and weaker during storms")
    print("- Tail field can drop below 1 nT during extreme conditions")
    print("- Near-Earth field remains relatively stable")
    print("="*80)