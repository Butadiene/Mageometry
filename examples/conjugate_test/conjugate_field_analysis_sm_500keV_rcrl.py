#!/usr/bin/env python
"""
Create a combined plot showing the second row (Rc/RL analysis) 
from each 500 keV time period in a single figure.
Shows: Min Rc/RL, Distance at Min Rc/RL, SM Latitude at Min Rc/RL, SM Longitude at Min Rc/RL
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import geopack
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geopack
from geopack.trace_field_lines_vectorized import trace_vectorized
from geopack.vectorized import t96_vectorized
from geopack.vectorized.field_line_geometry_vectorized import field_line_curvature_vectorized
from geopack.vectorized.field_line_directional_derivatives_new import field_line_directional_derivatives_vectorized
from geopack.coordinates_vectorized import smgsm_vectorized

# Import the original functions from the original script
from conjugate_field_analysis_sm import (
    create_sm_grid, 
    calculate_electron_larmor_radius,
    analyze_field_lines_sm
)


def create_rcrl_analysis_comparison(all_results, times, electron_energy_keV, figsize=(40, 20)):
    """Create 4x4 subplot showing Rc/RL analysis for all times."""
    fig, axes = plt.subplots(4, 4, figsize=figsize)
    
    # Common plot settings
    def setup_axis(ax, title):
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, pad=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.7, 0.7)
        ax.set_xlabel('X_SM (Re)', fontsize=10)
        ax.set_ylabel('Y_SM (Re)', fontsize=10)
        
        # Add Earth circle
        earth = plt.Circle((0, 0), 1.0, fill=False, edgecolor='black', linewidth=1.5, linestyle='--')
        ax.add_patch(earth)
        
        # Add latitude circles
        for lat in [55, 65, 75]:
            r_lat = np.cos(np.radians(lat))
            lat_circle = plt.Circle((0, 0), r_lat, fill=False, edgecolor='gray', 
                                   linewidth=0.5, linestyle=':', alpha=0.5)
            ax.add_patch(lat_circle)
            # Add latitude label at 45 degree angle to avoid overlap
            angle = np.pi/4  # 45 degrees
            x_label = r_lat * np.cos(angle)
            y_label = r_lat * np.sin(angle)
            ax.text(x_label, y_label, f'{lat}°', fontsize=7, ha='center', va='center',
                    color='gray', alpha=0.8, bbox=dict(boxstyle='round,pad=0.2', 
                                                       facecolor='white', edgecolor='none', alpha=0.7))
    
    # Create time labels
    time_labels = ['00:00', '06:00', '12:00', '18:00']
    
    # Plot each time in a row
    for row, (time_label, results, time) in enumerate(zip(time_labels, all_results, times)):
        # Extract data
        sm_lat = results['sm_lat']
        sm_lon = results['sm_lon']
        min_rc_rl = results['min_rc_rl']
        min_rc_rl_dist = results['min_rc_rl_dist']
        min_rc_rl_lat = results['min_rc_rl_lat']
        min_rc_rl_lon = results['min_rc_rl_lon']
        conjugate_mask = results['conjugate_mask']
        
        # Convert spherical to Cartesian for plotting
        sm_lon_rad = sm_lon * np.pi / 180
        sm_lat_rad = sm_lat * np.pi / 180
        x_plot = np.cos(sm_lat_rad) * np.cos(sm_lon_rad)
        y_plot = np.cos(sm_lat_rad) * np.sin(sm_lon_rad)
        
        # Get dipole tilt for this time
        ut = time.timestamp()
        ps = geopack.recalc(ut)
        dipole_tilt = np.degrees(ps)
        
        # Column 1: Minimum Rc/RL
        ax = axes[row, 0]
        setup_axis(ax, f'Minimum Rc/RL Ratio ({electron_energy_keV} keV)\n{time_label} UTC, Dipole Tilt: {dipole_tilt:.1f}°')
        
        # Non-conjugate points
        non_conj = ~conjugate_mask
        if np.any(non_conj):
            ax.scatter(x_plot[non_conj], y_plot[non_conj], 
                      c='gray', s=20, alpha=0.3)
        
        # Conjugate points
        conj = conjugate_mask & ~np.isnan(min_rc_rl) & (min_rc_rl > 0)
        if np.any(conj):
            vmin, vmax = 1.0, 64.0
            min_rc_rl_clipped = np.clip(min_rc_rl[conj], vmin, vmax)
            
            sc = ax.scatter(x_plot[conj], y_plot[conj], 
                           c=min_rc_rl_clipped, s=30,
                           cmap='RdBu_r',
                           norm=colors.LogNorm(vmin=vmin, vmax=vmax))
            cbar = plt.colorbar(sc, ax=ax, pad=0.1)
            cbar.set_label(r'Min $R_c/R_L$', fontsize=10)
            cbar.ax.axhline(y=8, color='black', linestyle='--', linewidth=1)
        
        # Column 2: Distance at minimum Rc/RL
        ax = axes[row, 1]
        setup_axis(ax, f'Distance at Minimum Rc/RL (Re)\n{time_label} UTC')
        
        if np.any(non_conj):
            ax.scatter(x_plot[non_conj], y_plot[non_conj], 
                      c='gray', s=20, alpha=0.3)
        
        conj = conjugate_mask & ~np.isnan(min_rc_rl_dist)
        if np.any(conj):
            sc = ax.scatter(x_plot[conj], y_plot[conj], 
                           c=min_rc_rl_dist[conj], s=30,
                           cmap='plasma',
                           vmin=1.0, vmax=min(20.0, np.nanmax(min_rc_rl_dist[conj])))
            cbar = plt.colorbar(sc, ax=ax, pad=0.1)
            cbar.set_label('Distance (Re)', fontsize=10)
        
        # Column 3: SM Latitude at minimum Rc/RL
        ax = axes[row, 2]
        setup_axis(ax, f'SM Latitude at Minimum Rc/RL (°)\n{time_label} UTC')
        
        if np.any(non_conj):
            ax.scatter(x_plot[non_conj], y_plot[non_conj], 
                      c='gray', s=20, alpha=0.3)
        
        conj = conjugate_mask & ~np.isnan(min_rc_rl_lat)
        if np.any(conj):
            sc = ax.scatter(x_plot[conj], y_plot[conj], 
                           c=min_rc_rl_lat[conj], s=30,
                           cmap='coolwarm',
                           vmin=-90, vmax=90)
            cbar = plt.colorbar(sc, ax=ax, pad=0.1)
            cbar.set_label('SM Latitude (°)', fontsize=10)
        
        # Column 4: SM longitude at minimum Rc/RL
        ax = axes[row, 3]
        setup_axis(ax, f'SM Longitude at Minimum Rc/RL\n{time_label} UTC')
        
        if np.any(non_conj):
            ax.scatter(x_plot[non_conj], y_plot[non_conj], 
                      c='gray', s=20, alpha=0.3)
        
        conj = conjugate_mask & ~np.isnan(min_rc_rl_lon)
        if np.any(conj):
            sc = ax.scatter(x_plot[conj], y_plot[conj], 
                           c=min_rc_rl_lon[conj], s=30,
                           cmap='hsv',
                           vmin=0, vmax=360)
            cbar = plt.colorbar(sc, ax=ax, pad=0.1)
            cbar.set_label('SM Longitude (°)', fontsize=10)
            cbar.set_ticks([0, 90, 180, 270, 360])
            cbar.set_ticklabels(['0°', '90°', '180°', '270°', '360°'])
    
    # Overall title
    fig.suptitle(f'Rc/RL Analysis for {electron_energy_keV} keV Electrons at Different Times\n' + 
                 'T96 Model, Spring Equinox 2024, Moderate Storm Conditions\n' +
                 'SM Coordinates, Starting from Northern Hemisphere at 1 Re', 
                 fontsize=16, y=0.995)
    
    plt.tight_layout()
    return fig, axes


def main():
    """Main function."""
    # Base date: Spring Equinox (March 20, 2024)
    base_date = datetime(2024, 3, 20, 0, 0, 0)
    
    # Create times for 0:00, 6:00, 12:00, 18:00
    times = [
        base_date,  # 00:00
        base_date + timedelta(hours=6),   # 06:00
        base_date + timedelta(hours=12),  # 12:00  
        base_date + timedelta(hours=18)   # 18:00
    ]
    
    time_labels = ['0000', '0600', '1200', '1800']
    
    # Set T96 model parameters (moderate storm conditions)
    parmod = np.array([3.0, -30.0, 0.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Set electron energy to 500 keV
    electron_energy_keV = 500.0
    
    # Create starting grid
    print("Creating grid directly in SM coordinates...")
    print("  Latitude range: 55° to 75°")
    print("  Longitude range: 0° to 360° (full coverage)")
    x_start_sm, y_start_sm, z_start_sm, sm_lat_start, sm_lon_start = create_sm_grid(
        radius=1.0,
        nlat=16,   # Doubled latitude density
        nlon=72    # Doubled longitude density
    )
    
    print(f"\nGrid points: {len(x_start_sm)}")
    print(f"Coordinate system: SM (Solar Magnetic)")
    print(f"Electron energy: {electron_energy_keV} keV")
    print(f"T96 Parameters:")
    print(f"  Pdyn = {parmod[0]} nPa")
    print(f"  Dst = {parmod[1]} nT")
    print(f"  ByIMF = {parmod[2]} nT")
    print(f"  BzIMF = {parmod[3]} nT\n")
    
    # Analyze field lines for each time
    all_results = []
    for i, (time, label) in enumerate(zip(times, time_labels)):
        print(f"\nProcessing time {i+1}/4: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        ut = time.timestamp()
        ps = geopack.recalc(ut)
        print(f"  Dipole tilt angle: {np.degrees(ps):.1f}°")
        
        # Analyze field lines
        results = analyze_field_lines_sm(
            ut, parmod, x_start_sm, y_start_sm, z_start_sm, 
            sm_lat_start, sm_lon_start, electron_energy_keV
        )
        
        conjugate_mask = results['conjugate_mask']
        print(f"  Conjugate field lines: {np.sum(conjugate_mask)}/{len(x_start_sm)} ({100*np.sum(conjugate_mask)/len(x_start_sm):.1f}%)")
        
        all_results.append(results)
    
    # Create combined plot showing Rc/RL analysis (second row)
    print("\nCreating combined Rc/RL analysis plot...")
    fig, axes = create_rcrl_analysis_comparison(all_results, times, electron_energy_keV)
    
    # Save figure
    filename = os.path.join(os.path.dirname(__file__), 'conjugate_field_analysis_sm_500keV_rcrl_analysis.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.show()


if __name__ == '__main__':
    main()