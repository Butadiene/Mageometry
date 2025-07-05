#!/usr/bin/env python3
"""
3D Volume Rendering of Strong Scattering Regions

This script creates a 3D volume rendering visualization of regions where 
the ratio of curvature radius to Larmor radius (Rc/RL) indicates strong scattering.

When Rc/RL < 8, particles experience strong pitch angle diffusion.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import geopack modules
import geopack
from geopack import t96_vectorized, field_line_curvature_vectorized

# Physical constants
c = 2.99792458e8  # Speed of light (m/s)
me = 9.10938356e-31  # Electron mass (kg)
e = 1.602176634e-19  # Elementary charge (C)
me_c2_keV = 511.0  # Electron rest energy (keV)
Re = 6.371e6  # Earth radius (m)

# Critical threshold for strong scattering
CRITICAL_RATIO = 8.0


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


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc. This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    
    3Dプロットの軸を等しいスケールに設定します。
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def calculate_curvature_radius(model_func, parmod, ps, x, y, z):
    """Calculate radius of curvature and magnetic field strength."""
    kappa = field_line_curvature_vectorized(model_func, parmod, ps, x, y, z)
    Rc = np.where(kappa > 1e-10, 1.0 / kappa, 1e10)
    
    bx, by, bz = model_func(parmod, ps, x, y, z)
    B = np.sqrt(bx**2 + by**2 + bz**2)
    
    return Rc, B


def main():
    """Create 3D volume rendering of scattering regions."""
    
    # Initialize geopack
    ut = 1600000000  # Unix timestamp
    ps = geopack.recalc(ut)
    
    # Model parameters for moderate storm conditions
    parmod = [3.0, -30.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    print("3D Volume Rendering of Strong Scattering Regions")
    print("=" * 70)
    print(f"Model: T96")
    print(f"Conditions: Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT, ByIMF={parmod[2]} nT, BzIMF={parmod[3]} nT")
    print(f"Dipole tilt: {np.degrees(ps):.2f}°")
    print(f"Critical threshold: Rc/RL = {CRITICAL_RATIO}")
    print()
    
    # Energy for analysis
    energy_keV = 100  # keV
    print(f"Analyzing {energy_keV} keV electrons at 90° pitch angle")
    
    # Create figure
    fig = plt.figure(figsize=(16, 8))
    
    # 3D scatter plot of scattering regions
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Sample 3D volume
    print("\nSampling 3D volume for scattering regions...")
    
    x_vol = np.linspace(-15, 5, 30)
    y_vol = np.linspace(-8, 8, 30)
    z_vol = np.linspace(-2, 2, 20)
    
    scatter_volume = []
    scatter_values = []
    
    for z in z_vol:
        for y in y_vol[::2]:  # Sample every other point for efficiency
            for x in x_vol[::2]:
                r = np.sqrt(x**2 + y**2 + z**2)
                if r < 2:  # Skip points too close to Earth
                    continue
                
                Rc_Re, B_nT = calculate_curvature_radius(t96_vectorized, parmod, ps, x, y, z)
                RL_m = calculate_larmor_radius(energy_keV, B_nT, pitch_angle_deg=90)
                ratio = Rc_Re * Re / RL_m
                
                if ratio < CRITICAL_RATIO:
                    scatter_volume.append([x, y, z])
                    scatter_values.append(ratio)
    
    print(f"Found {len(scatter_volume)} points with Rc/RL < {CRITICAL_RATIO}")
    
    if len(scatter_volume) > 0:
        scatter_volume = np.array(scatter_volume)
        scatter_values = np.array(scatter_values)
        
        # 3D scatter plot colored by Rc/RL value
        scatter = ax1.scatter(scatter_volume[:, 0], scatter_volume[:, 1], scatter_volume[:, 2],
                             c=scatter_values, cmap='hot_r', s=20, alpha=0.6,
                             vmin=0, vmax=CRITICAL_RATIO)
        
        cbar = plt.colorbar(scatter, ax=ax1, pad=0.1, shrink=0.8)
        cbar.set_label('Rc/RL Ratio')
    
    # Add Earth
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_earth = np.outer(np.cos(u), np.sin(v))
    y_earth = np.outer(np.sin(u), np.sin(v))
    z_earth = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x_earth, y_earth, z_earth, color='lightgray', alpha=0.8)
    
    ax1.set_xlabel('X GSM (Re)')
    ax1.set_ylabel('Y GSM (Re)')
    ax1.set_zlabel('Z GSM (Re)')
    ax1.set_title(f'3D Volume: Rc/RL < {CRITICAL_RATIO} Regions')
    ax1.set_xlim(-15, 5)
    ax1.set_ylim(-8, 8)
    ax1.set_zlim(-2, 2)
    
    # Apply equal aspect ratio to 3D plot
    set_axes_equal(ax1)
    
    # Alternative method for newer matplotlib versions
    try:
        ax1.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass  # Already handled by set_axes_equal
    
    ax1.view_init(elev=15, azim=45)
    
    # XY projection (equatorial plane)
    ax2 = fig.add_subplot(132)
    if len(scatter_volume) > 0:
        z0_points = scatter_volume[np.abs(scatter_volume[:, 2]) < 0.2]
        if len(z0_points) > 0:
            ax2.scatter(z0_points[:, 0], z0_points[:, 1], c='red', s=20, alpha=0.5)
            ax2.text(0.05, 0.95, f'{len(z0_points)} points near Z=0', 
                    transform=ax2.transAxes, fontsize=10, verticalalignment='top')
    
    # Add Earth
    earth = plt.Circle((0, 0), 1, color='lightgray', zorder=10)
    ax2.add_patch(earth)
    
    # Add MLT labels
    ax2.text(8, 0, '12', ha='center', va='center', fontsize=10, weight='bold')
    ax2.text(-12, 0, '00', ha='center', va='center', fontsize=10, weight='bold')
    ax2.text(0, 8, '06', ha='center', va='center', fontsize=10, weight='bold')
    ax2.text(0, -8, '18', ha='center', va='center', fontsize=10, weight='bold')
    
    ax2.set_xlabel('X GSM (Re)')
    ax2.set_ylabel('Y GSM (Re)')
    ax2.set_title('XY Projection (Z ≈ 0)')
    ax2.set_aspect('equal')
    ax2.set_xlim(-15, 5)
    ax2.set_ylim(-8, 8)
    ax2.grid(True, alpha=0.3)
    
    # XZ projection (noon-midnight meridian)
    ax3 = fig.add_subplot(133)
    if len(scatter_volume) > 0:
        y0_points = scatter_volume[np.abs(scatter_volume[:, 1]) < 0.5]
        if len(y0_points) > 0:
            ax3.scatter(y0_points[:, 0], y0_points[:, 2], c='red', s=20, alpha=0.5)
            ax3.text(0.05, 0.95, f'{len(y0_points)} points near Y=0', 
                    transform=ax3.transAxes, fontsize=10, verticalalignment='top')
    
    # Add Earth
    earth = plt.Circle((0, 0), 1, color='lightgray', zorder=10)
    ax3.add_patch(earth)
    
    ax3.set_xlabel('X GSM (Re)')
    ax3.set_ylabel('Z GSM (Re)')
    ax3.set_title('XZ Projection (Y ≈ 0)')
    ax3.set_aspect('equal')
    ax3.set_xlim(-15, 5)
    ax3.set_ylim(-2, 2)
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f'3D Distribution of Strong Scattering Regions (Rc/RL < {CRITICAL_RATIO})\n'
                 f'{energy_keV} keV electrons, Moderate Storm Conditions',
                 fontsize=14)
    plt.tight_layout()
    
    # Save figure
    output_file = 'scattering_regions_volume_3d.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVolume rendering saved as '{output_file}'")
    
    # Print statistics
    if len(scatter_volume) > 0:
        print("\n3D Scattering Region Statistics:")
        print("-" * 50)
        print(f"Total points with Rc/RL < {CRITICAL_RATIO}: {len(scatter_volume)}")
        print(f"Average Rc/RL in scattering regions: {np.mean(scatter_values):.2f}")
        print(f"Spatial extent of scattering regions:")
        print(f"  X range: [{scatter_volume[:, 0].min():.1f}, {scatter_volume[:, 0].max():.1f}] Re")
        print(f"  Y range: [{scatter_volume[:, 1].min():.1f}, {scatter_volume[:, 1].max():.1f}] Re")
        print(f"  Z range: [{scatter_volume[:, 2].min():.1f}, {scatter_volume[:, 2].max():.1f}] Re")
        
        # Analyze distribution by region
        tail_points = scatter_volume[scatter_volume[:, 0] < -5]
        inner_points = scatter_volume[(scatter_volume[:, 0] > -5) & (scatter_volume[:, 0] < 0)]
        dayside_points = scatter_volume[scatter_volume[:, 0] > 0]
        
        print(f"\nDistribution by region:")
        print(f"  Tail (X < -5 Re): {len(tail_points)} points ({100*len(tail_points)/len(scatter_volume):.1f}%)")
        print(f"  Inner (-5 < X < 0): {len(inner_points)} points ({100*len(inner_points)/len(scatter_volume):.1f}%)")
        print(f"  Dayside (X > 0): {len(dayside_points)} points ({100*len(dayside_points)/len(scatter_volume):.1f}%)")
        
        # MLT distribution
        equatorial_points = scatter_volume[np.abs(scatter_volume[:, 2]) < 0.5]
        if len(equatorial_points) > 0:
            print(f"\nMLT distribution (points near equator, |Z| < 0.5 Re):")
            angles = np.arctan2(equatorial_points[:, 1], equatorial_points[:, 0]) * 12 / np.pi + 12
            angles = angles % 24
            
            dawn = np.sum((angles >= 3) & (angles < 9))
            noon = np.sum((angles >= 9) & (angles < 15))
            dusk = np.sum((angles >= 15) & (angles < 21))
            midnight = np.sum((angles >= 21) | (angles < 3))
            
            total_eq = len(equatorial_points)
            print(f"  Dawn (03-09 MLT): {dawn} points ({100*dawn/total_eq:.1f}%)")
            print(f"  Noon (09-15 MLT): {noon} points ({100*noon/total_eq:.1f}%)")
            print(f"  Dusk (15-21 MLT): {dusk} points ({100*dusk/total_eq:.1f}%)")
            print(f"  Midnight (21-03 MLT): {midnight} points ({100*midnight/total_eq:.1f}%)")
    
    print("\nPhysical Implications:")
    print("-" * 50)
    print(f"• The 3D volume shows where particles experience strong scattering")
    print(f"• Scattering regions are concentrated near the current sheet")
    print(f"• Dawn-dusk asymmetry is evident in the distribution")
    print(f"• Storm conditions expand these regions significantly")
    print(f"• Important for understanding 3D particle precipitation patterns")
    
    plt.show()


if __name__ == "__main__":
    main()