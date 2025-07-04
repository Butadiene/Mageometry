#!/usr/bin/env python3
"""
3D Visualization of Magnetic Field Lines from Strong Scattering Regions

This script creates a 3D visualization of magnetic field lines traced from regions 
where the ratio of curvature radius to Larmor radius (Rc/RL) indicates strong scattering.

When Rc/RL < 8, particles experience strong pitch angle diffusion.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

# Import geopack modules
import geopack
from geopack import t96_vectorized, field_line_curvature_vectorized
from geopack.trace_field_lines_vectorized import trace_vectorized

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


def calculate_curvature_radius(model_func, parmod, ps, x, y, z):
    """Calculate radius of curvature and magnetic field strength."""
    kappa = field_line_curvature_vectorized(model_func, parmod, ps, x, y, z)
    Rc = np.where(kappa > 1e-10, 1.0 / kappa, 1e10)
    
    bx, by, bz = model_func(parmod, ps, x, y, z)
    B = np.sqrt(bx**2 + by**2 + bz**2)
    
    return Rc, B


def find_scattering_regions(energy_keV, parmod, ps, x_grid, y_grid, z_grid):
    """Find regions where Rc/RL < CRITICAL_RATIO."""
    # Flatten grids
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    z_flat = z_grid.flatten()
    
    # Calculate Rc/RL ratio
    Rc_Re, B_nT = calculate_curvature_radius(t96_vectorized, parmod, ps, 
                                             x_flat, y_flat, z_flat)
    RL_m = calculate_larmor_radius(energy_keV, B_nT, pitch_angle_deg=90)
    ratio = Rc_Re * Re / RL_m
    
    # Find scattering regions
    scatter_mask = (ratio < CRITICAL_RATIO) & (np.sqrt(x_flat**2 + y_flat**2 + z_flat**2) > 1.5)
    
    scatter_points = np.column_stack((x_flat[scatter_mask], 
                                      y_flat[scatter_mask], 
                                      z_flat[scatter_mask]))
    scatter_ratios = ratio[scatter_mask]
    
    return scatter_points, scatter_ratios, ratio.reshape(x_grid.shape)


def trace_field_lines_from_points(seed_points, parmod, ps):
    """Trace field lines from seed points in both directions."""
    field_lines = []
    
    if len(seed_points) == 0:
        return field_lines
    
    # Trace in both directions
    for trace_dir in [-1, 1]:
        try:
            # Vectorized tracing
            results = trace_vectorized(
                seed_points[:, 0],  # All x coordinates
                seed_points[:, 1],  # All y coordinates
                seed_points[:, 2],  # All z coordinates
                dir=trace_dir,
                rlim=15.0,
                r0=1.0,
                parmod=parmod,
                exname='t96',
                inname='igrf',
                maxloop=5000,
                return_full_path=True
            )
            
            if results is not None and isinstance(results, tuple) and len(results) >= 7:
                x_paths, y_paths, z_paths = results[3], results[4], results[5]
                
                # Handle masked arrays
                if hasattr(x_paths, 'filled'):
                    x_paths = x_paths.filled(fill_value=np.nan)
                    y_paths = y_paths.filled(fill_value=np.nan)
                    z_paths = z_paths.filled(fill_value=np.nan)
                
                # Process multiple paths
                if hasattr(x_paths, '__len__') and len(x_paths) > 0:
                    if hasattr(x_paths[0], '__len__'):
                        # Multiple paths
                        for i in range(len(x_paths)):
                            x_data = x_paths[i][~np.isnan(x_paths[i])]
                            y_data = y_paths[i][~np.isnan(y_paths[i])]
                            z_data = z_paths[i][~np.isnan(z_paths[i])]
                            
                            if len(x_data) > 10:
                                field_line = np.column_stack((x_data, y_data, z_data))
                                field_lines.append(field_line)
                    else:
                        # Single path
                        if len(x_paths) > 10:
                            field_line = np.column_stack((x_paths, y_paths, z_paths))
                            if np.all(np.isfinite(field_line)):
                                field_lines.append(field_line)
                
        except Exception as e:
            print(f"Error tracing field lines: {str(e)}")
            continue
    
    return field_lines


def main():
    """Create 3D visualization of field lines from scattering regions."""
    
    # Initialize geopack
    ut = 1600000000  # Unix timestamp
    ps = geopack.recalc(ut)
    
    # Model parameters for moderate storm conditions
    parmod = [3.0, -30.0, 1.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    print("3D Structure of Magnetic Field Lines from Strong Scattering Regions")
    print("=" * 70)
    print(f"Model: T96")
    print(f"Conditions: Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT, ByIMF={parmod[2]} nT, BzIMF={parmod[3]} nT")
    print(f"Dipole tilt: {np.degrees(ps):.2f}°")
    print(f"Critical threshold: Rc/RL = {CRITICAL_RATIO}")
    print()
    
    # Energy for analysis
    energy_keV = 100  # keV
    print(f"Analyzing {energy_keV} keV electrons at 90° pitch angle")
    
    # Step 1: Find seed points in scattering regions
    print("\nStep 1: Finding seed points in scattering regions...")
    
    # Create 3D grid to search for scattering regions
    x_search = np.linspace(-15, 5, 41)
    y_search = np.linspace(-8, 8, 33)
    z_search = np.linspace(0, 1.5, 16)
    
    seed_points = []
    
    for z in z_search:
        X, Y = np.meshgrid(x_search, y_search)
        Z = np.full_like(X, z)
        
        scatter_pts, _, _ = find_scattering_regions(energy_keV, parmod, ps, X, Y, Z)
        
        if len(scatter_pts) > 0:
            # Sample points from this Z level
            n_samples = min(5, len(scatter_pts))
            indices = np.random.choice(len(scatter_pts), n_samples, replace=False)
            seed_points.extend(scatter_pts[indices])
    
    seed_points = np.array(seed_points)
    print(f"Found {len(seed_points)} seed points in scattering regions")
    
    # Step 2: Trace field lines
    print("\nStep 2: Tracing field lines from seed points...")
    field_lines = trace_field_lines_from_points(seed_points, parmod, ps)
    print(f"Successfully traced {len(field_lines)} field lines")
    
    # Step 3: Create 3D visualization
    print("\nStep 3: Creating 3D visualization...")
    
    fig = plt.figure(figsize=(18, 10))
    
    # Main 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot field lines
    for fl in field_lines[:50]:  # Limit to 50 for clarity
        ax1.plot(fl[:, 0], fl[:, 1], fl[:, 2], 'b-', alpha=0.3, linewidth=0.8)
    
    # Add Earth
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_earth = np.outer(np.cos(u), np.sin(v))
    y_earth = np.outer(np.sin(u), np.sin(v))
    z_earth = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x_earth, y_earth, z_earth, color='white', alpha=0.8)
    
    # Mark seed points colored by Z level
    if len(seed_points) > 0:
        colors = plt.cm.plasma(seed_points[:, 2] / seed_points[:, 2].max())
        ax1.scatter(seed_points[:, 0], seed_points[:, 1], seed_points[:, 2],
                   c=colors, s=30, alpha=0.8)
    
    ax1.set_xlabel('X GSM (Re)')
    ax1.set_ylabel('Y GSM (Re)')
    ax1.set_zlabel('Z GSM (Re)')
    ax1.set_title(f'3D Field Lines from Rc/RL < {CRITICAL_RATIO} Regions\n{energy_keV} keV electrons')
    ax1.set_xlim(-15, 5)
    ax1.set_ylim(-10, 10)
    ax1.set_zlim(-5, 5)
    ax1.view_init(elev=20, azim=45)
    
    # Meridian projection
    ax2 = fig.add_subplot(122)
    
    # Create grid for background
    x_mer = np.linspace(-15, 5, 81)
    z_mer = np.linspace(-3, 3, 49)
    X_mer, Z_mer = np.meshgrid(x_mer, z_mer)
    Y_mer = np.zeros_like(X_mer)
    
    # Calculate Rc/RL ratio
    _, _, ratio_grid = find_scattering_regions(energy_keV, parmod, ps, X_mer, Y_mer, Z_mer)
    
    # Plot background
    im = ax2.contourf(X_mer, Z_mer, ratio_grid,
                     levels=np.logspace(-1, 3, 30),
                     cmap='RdBu_r', extend='both',
                     norm=LogNorm(vmin=0.1, vmax=1000))
    
    # Add critical contour
    cs = ax2.contour(X_mer, Z_mer, ratio_grid, 
                     levels=[CRITICAL_RATIO],
                     colors='black', linewidths=3)
    ax2.clabel(cs, inline=True, fontsize=10, fmt='Rc/RL=8')
    
    # Project field lines
    for fl in field_lines[:50]:
        ax2.plot(fl[:, 0], fl[:, 2], 'g-', alpha=0.3, linewidth=0.8)
    
    # Mark seed points
    if len(seed_points) > 0:
        ax2.scatter(seed_points[:, 0], seed_points[:, 2], c='red', s=20, alpha=0.6,
                   label='Field line start points')
    
    # Add Earth
    earth = plt.Circle((0, 0), 1, color='white', zorder=10)
    ax2.add_patch(earth)
    
    ax2.set_xlabel('X GSM (Re)')
    ax2.set_ylabel('Z GSM (Re)')
    ax2.set_title('Noon-Midnight Meridian Projection')
    ax2.set_aspect('equal')
    ax2.set_xlim(-15, 5)
    ax2.set_ylim(-3, 3)
    ax2.legend(loc='upper right')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Rc/RL Ratio')
    
    plt.suptitle(f'3D Structure of Magnetic Field Lines from Strong Scattering Regions\n'
                 f'Field lines traced from regions where Rc/RL < {CRITICAL_RATIO}',
                 fontsize=14)
    plt.tight_layout()
    
    # Save figure
    output_file = 'field_lines_scattering_regions_3d.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as '{output_file}'")
    
    # Additional analysis
    print("\nField Line Connectivity Analysis:")
    print("-" * 50)
    
    if len(field_lines) > 0:
        closed_lines = 0
        open_lines = 0
        total_lengths = []
        
        for fl in field_lines:
            start_r = np.sqrt(np.sum(fl[0]**2))
            end_r = np.sqrt(np.sum(fl[-1]**2))
            
            if start_r < 2 and end_r < 2:
                closed_lines += 1
            else:
                open_lines += 1
            
            # Calculate total length
            length = np.sum(np.sqrt(np.sum(np.diff(fl, axis=0)**2, axis=1)))
            total_lengths.append(length)
        
        print(f"Total field lines traced: {len(field_lines)}")
        print(f"Closed field lines (both ends at Earth): {closed_lines} ({100*closed_lines/len(field_lines):.1f}%)")
        print(f"Open field lines: {open_lines} ({100*open_lines/len(field_lines):.1f}%)")
        print(f"Average field line length: {np.mean(total_lengths):.1f} Re")
        print(f"Length range: {np.min(total_lengths):.1f} - {np.max(total_lengths):.1f} Re")
    
    print("\nPhysical Implications:")
    print("-" * 50)
    print(f"• Regions with Rc/RL < {CRITICAL_RATIO} cause strong pitch angle diffusion")
    print("• Field lines passing through these regions can guide particles to precipitation")
    print("• The 3D structure shows how particles at different latitudes can be affected")
    print("• Storm conditions significantly expand the scattering regions")
    print("• Important for understanding radiation belt losses and auroral precipitation")
    
    plt.show()


if __name__ == "__main__":
    main()