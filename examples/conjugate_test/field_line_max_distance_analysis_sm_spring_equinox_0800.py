#!/usr/bin/env python
"""
Analyze and plot the maximum distance points along magnetic field lines.
Spring Equinox at 8:00 AM UTC
Creates Cartesian plots showing:
1. Maximum distance from Earth center (Re)
2. SM latitude at maximum distance point
3. SM longitude at maximum distance point
Uses T89 magnetospheric model with SM (Solar Magnetic) coordinates.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from datetime import datetime
import sys
import os

# Add parent directory to path to import geopack
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geopack
from geopack.trace_field_lines_vectorized import trace_vectorized
from geopack.vectorized import t89_vectorized
from geopack.coordinates_vectorized import smgsm_vectorized


def create_sm_grid(radius=1.0, nlat=20, nlon=48):
    """
    Create a grid of starting points directly in SM coordinates.
    
    In SM coordinates:
    - Z_SM axis: aligned with magnetic dipole axis
    - Y_SM axis: perpendicular to both dipole axis and Sun-Earth line
      +Y_SM points toward 90° SM longitude, -Y_SM points toward 270° SM longitude
    - X_SM axis: completes right-handed system
    """
    # Create latitude grid in SM coordinates (0° = SM equator, 90° = north magnetic pole)
    sm_lat = np.linspace(55, 75, nlat)
    
    # Create longitude grid (0-360°)
    # In SM coordinates: longitude measured from X_SM axis
    sm_lon = np.linspace(0, 360, nlon, endpoint=False)
    
    # Create meshgrid
    SM_LON_GRID, SM_LAT_GRID = np.meshgrid(sm_lon, sm_lat)
    
    # Flatten for processing
    sm_lon_flat = SM_LON_GRID.flatten()
    sm_lat_flat = SM_LAT_GRID.flatten()
    
    # Convert to radians
    sm_lon_rad = sm_lon_flat * np.pi / 180
    sm_lat_rad = sm_lat_flat * np.pi / 180
    
    # Convert to Cartesian SM coordinates
    x_sm = radius * np.cos(sm_lat_rad) * np.cos(sm_lon_rad)
    y_sm = radius * np.cos(sm_lat_rad) * np.sin(sm_lon_rad)
    z_sm = radius * np.sin(sm_lat_rad)
    
    return x_sm, y_sm, z_sm, sm_lat_flat, sm_lon_flat


def analyze_field_line_max_distance(ut, parmod, x_start_sm, y_start_sm, z_start_sm, 
                                   sm_lat_start, sm_lon_start):
    """
    Trace field lines and find the maximum distance point along each line.
    Also identifies open vs closed field lines.
    """
    # Update geopack parameters
    ps = geopack.recalc(ut)
    
    print(f"Tracing {len(x_start_sm)} field lines with T89 model...")
    
    # Convert SM to GSM for field line tracing
    x_start_gsm, y_start_gsm, z_start_gsm = smgsm_vectorized(x_start_sm, y_start_sm, z_start_sm, 1)
    
    # Trace in both directions and combine
    all_max_dist = np.full(len(x_start_sm), np.nan)
    all_max_dist_lat = np.full(len(x_start_sm), np.nan)
    all_max_dist_lon = np.full(len(x_start_sm), np.nan)  # SM longitude
    all_status = np.full(len(x_start_sm), -1)
    all_is_open = np.full(len(x_start_sm), False)  # Track open field lines
    
    # Trace antiparallel (dir=1)
    xf1_gsm, yf1_gsm, zf1_gsm, fl_x1_gsm, fl_y1_gsm, fl_z1_gsm, status1 = trace_vectorized(
        x_start_gsm, y_start_gsm, z_start_gsm,
        dir=1,
        rlim=30.0,  # Increased limit to capture distant field lines
        r0=1.0,
        parmod=parmod,
        exname='t89',
        inname='igrf',
        maxloop=2000,
        return_full_path=True
    )
    
    # Convert to SM
    fl_x1, fl_y1, fl_z1 = smgsm_vectorized(fl_x1_gsm, fl_y1_gsm, fl_z1_gsm, -1)
    
    # Trace parallel (dir=-1)
    xf2_gsm, yf2_gsm, zf2_gsm, fl_x2_gsm, fl_y2_gsm, fl_z2_gsm, status2 = trace_vectorized(
        x_start_gsm, y_start_gsm, z_start_gsm,
        dir=-1,
        rlim=30.0,
        r0=1.0,
        parmod=parmod,
        exname='t89',
        inname='igrf',
        maxloop=2000,
        return_full_path=True
    )
    
    # Convert to SM
    fl_x2, fl_y2, fl_z2 = smgsm_vectorized(fl_x2_gsm, fl_y2_gsm, fl_z2_gsm, -1)
    
    print("Analyzing maximum distance points...")
    
    # Process each field line
    for i in range(len(x_start_sm)):
        if i % 100 == 0 and i > 0:
            print(f"  Progress: {i}/{len(x_start_sm)} ({100*i/len(x_start_sm):.1f}%)")
        
        # Combine both directions
        if hasattr(fl_x1, 'mask'):
            # Antiparallel direction
            valid1 = ~fl_x1.mask[i, :]
            x_line1 = fl_x1.data[i, valid1] if np.sum(valid1) > 0 else np.array([])
            y_line1 = fl_y1.data[i, valid1] if np.sum(valid1) > 0 else np.array([])
            z_line1 = fl_z1.data[i, valid1] if np.sum(valid1) > 0 else np.array([])
            
            # Parallel direction (reverse to maintain continuity)
            valid2 = ~fl_x2.mask[i, :]
            x_line2 = fl_x2.data[i, valid2][::-1] if np.sum(valid2) > 0 else np.array([])
            y_line2 = fl_y2.data[i, valid2][::-1] if np.sum(valid2) > 0 else np.array([])
            z_line2 = fl_z2.data[i, valid2][::-1] if np.sum(valid2) > 0 else np.array([])
        else:
            # Antiparallel direction
            valid1 = ~np.isnan(fl_x1[i, :])
            x_line1 = fl_x1[i, valid1] if np.sum(valid1) > 0 else np.array([])
            y_line1 = fl_y1[i, valid1] if np.sum(valid1) > 0 else np.array([])
            z_line1 = fl_z1[i, valid1] if np.sum(valid1) > 0 else np.array([])
            
            # Parallel direction (reverse to maintain continuity)
            valid2 = ~np.isnan(fl_x2[i, :])
            x_line2 = fl_x2[i, valid2][::-1] if np.sum(valid2) > 0 else np.array([])
            y_line2 = fl_y2[i, valid2][::-1] if np.sum(valid2) > 0 else np.array([])
            z_line2 = fl_z2[i, valid2][::-1] if np.sum(valid2) > 0 else np.array([])
        
        # Skip if no valid points
        if len(x_line1) == 0 and len(x_line2) == 0:
            continue
            
        # Combine both directions (excluding duplicate starting point)
        if len(x_line2) > 1 and len(x_line1) > 0:
            x_line = np.concatenate([x_line2[:-1], x_line1])
            y_line = np.concatenate([y_line2[:-1], y_line1])
            z_line = np.concatenate([z_line2[:-1], z_line1])
        elif len(x_line1) > 0:
            x_line = x_line1
            y_line = y_line1
            z_line = z_line1
        else:
            x_line = x_line2
            y_line = y_line2
            z_line = z_line2
        
        if len(x_line) < 3:
            continue
        
        # Calculate distances
        distances = np.sqrt(x_line**2 + y_line**2 + z_line**2)
        
        # Find maximum distance point
        idx_max = np.argmax(distances)
        all_max_dist[i] = distances[idx_max]
        
        # Get coordinates at maximum distance
        x_max = x_line[idx_max]
        y_max = y_line[idx_max]
        z_max = z_line[idx_max]
        
        # Convert to spherical SM coordinates
        r_max = distances[idx_max]
        lat_max_rad = np.arcsin(z_max / r_max) if r_max > 0 else 0
        lon_max_rad = np.arctan2(y_max, x_max)
        
        all_max_dist_lat[i] = np.degrees(lat_max_rad)
        
        # Store SM longitude directly
        sm_lon_deg = np.degrees(lon_max_rad) % 360
        all_max_dist_lon[i] = sm_lon_deg
        
        # Set status based on successful tracing
        if status1[i] == 0 or status2[i] == 0:
            all_status[i] = 0
        
        # Determine if field line is open
        # A field line is considered closed if both ends reach r0 (1 Re)
        # Check endpoints
        r1_end = np.sqrt(xf1_gsm[i]**2 + yf1_gsm[i]**2 + zf1_gsm[i]**2)
        r2_end = np.sqrt(xf2_gsm[i]**2 + yf2_gsm[i]**2 + zf2_gsm[i]**2)
        
        # Field line is open if either end doesn't reach r0 or reaches boundary
        is_closed = (status1[i] == 0 and abs(r1_end - 1.0) < 0.1) and \
                   (status2[i] == 0 and abs(r2_end - 1.0) < 0.1)
        all_is_open[i] = not is_closed and (status1[i] >= 0 or status2[i] >= 0)
    
    print(f"Found {np.sum(~np.isnan(all_max_dist))} field lines with valid max distance points")
    print(f"  Open field lines: {np.sum(all_is_open)}")
    print(f"  Closed field lines: {np.sum(~all_is_open & ~np.isnan(all_max_dist))}")
    
    return {
        'max_dist': all_max_dist,
        'max_dist_lat': all_max_dist_lat,
        'max_dist_lon': all_max_dist_lon,  # SM longitude
        'status': all_status,
        'is_open': all_is_open,
        'sm_lat': sm_lat_start,
        'sm_lon': sm_lon_start
    }


def create_max_distance_plots(results, ut, figsize=(18, 6)):
    """
    Create 3 Cartesian plots showing maximum distance properties.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Extract data
    sm_lat = results['sm_lat']
    sm_lon = results['sm_lon']
    max_dist = results['max_dist']
    max_dist_lat = results['max_dist_lat']
    max_dist_lon = results['max_dist_lon']  # SM longitude
    status = results['status']
    is_open = results['is_open']
    
    # Convert spherical to Cartesian for plotting
    # Starting positions at r=1 Re
    sm_lon_rad = sm_lon * np.pi / 180
    sm_lat_rad = sm_lat * np.pi / 180
    
    # Calculate X,Y positions at Earth's surface (r=1)
    x_start = np.cos(sm_lat_rad) * np.cos(sm_lon_rad)
    y_start = np.cos(sm_lat_rad) * np.sin(sm_lon_rad)
    
    # Create masks for different field line types
    valid_mask = ~np.isnan(max_dist) & (status >= 0)
    closed_mask = valid_mask & ~is_open
    open_mask = valid_mask & is_open
    invalid_mask = ~valid_mask
    
    # Plot 1: Maximum distance from Earth
    ax1 = axes[0]
    ax1.set_aspect('equal')
    
    # Plot invalid points
    if np.any(invalid_mask):
        ax1.scatter(x_start[invalid_mask], y_start[invalid_mask], 
                   c='gray', s=20, alpha=0.3, label='Invalid')
    
    # Plot closed field lines
    if np.any(closed_mask):
        # Clip distances for colorbar range
        max_dist_clipped = np.clip(max_dist[closed_mask], 0, 8)
        
        sc1_closed = ax1.scatter(x_start[closed_mask], y_start[closed_mask], 
                                c=max_dist_clipped, s=30,
                                cmap='viridis',
                                vmin=0, vmax=8,
                                marker='o',
                                label='Closed')
        
    # Plot open field lines with circle and cross
    if np.any(open_mask):
        max_dist_clipped_open = np.clip(max_dist[open_mask], 0, 8)
        
        # Plot circles for open field lines
        sc1_open = ax1.scatter(x_start[open_mask], y_start[open_mask], 
                              c=max_dist_clipped_open, s=30,
                              cmap='viridis',
                              vmin=0, vmax=8,
                              marker='o',
                              label='Open (×)')
        
        # Add crosses on top (don't include in legend)
        ax1.scatter(x_start[open_mask], y_start[open_mask], 
                   c='black', s=30,
                   marker='x',
                   linewidths=1,
                   label='_nolegend_')
    
    # Add Earth circle
    earth1 = plt.Circle((0, 0), 1.0, fill=False, edgecolor='black', linewidth=2, linestyle='--')
    ax1.add_patch(earth1)
    
    # Add latitude circles
    for i, lat in enumerate([55, 65, 75]):
        r_lat = np.cos(np.radians(lat))
        lat_circle = plt.Circle((0, 0), r_lat, fill=False, edgecolor='gray', 
                               linewidth=0.5, linestyle=':', alpha=0.5)
        ax1.add_patch(lat_circle)
        # Add latitude label at 45 degree angle to avoid overlap
        angle = np.pi/4  # 45 degrees
        x_label = r_lat * np.cos(angle)
        y_label = r_lat * np.sin(angle)
        ax1.text(x_label, y_label, f'{lat}°', fontsize=8, ha='center', va='center',
                color='gray', alpha=0.7, bbox=dict(boxstyle='round,pad=0.2', 
                                                   facecolor='white', edgecolor='none', alpha=0.7))
    
    
    # Set limits and labels
    ax1.set_xlim(-0.7, 0.7)
    ax1.set_ylim(-0.7, 0.7)
    ax1.set_xlabel('X_SM (Re)', fontsize=12)
    ax1.set_ylabel('Y_SM (Re)', fontsize=12)
    ax1.set_title('Maximum Distance from Earth (Re)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    if np.any(valid_mask):
        # Use the closed field line scatter for colorbar if it exists, otherwise use open
        sc_for_cbar = sc1_closed if np.any(closed_mask) else sc1_open
        cbar1 = plt.colorbar(sc_for_cbar, ax=ax1, pad=0.1)
        cbar1.set_label('Max Distance (Re)', fontsize=10)
    
    # Plot 2: SM Latitude at maximum distance
    ax2 = axes[1]
    ax2.set_aspect('equal')
    
    if np.any(invalid_mask):
        ax2.scatter(x_start[invalid_mask], y_start[invalid_mask], 
                   c='gray', s=20, alpha=0.3)
    
    # Plot closed field lines
    if np.any(closed_mask):
        sc2_closed = ax2.scatter(x_start[closed_mask], y_start[closed_mask], 
                                c=max_dist_lat[closed_mask], s=30,
                                cmap='coolwarm',
                                vmin=-90, vmax=90,
                                marker='o')
    
    # Plot open field lines
    if np.any(open_mask):
        sc2_open = ax2.scatter(x_start[open_mask], y_start[open_mask], 
                              c=max_dist_lat[open_mask], s=30,
                              cmap='coolwarm',
                              vmin=-90, vmax=90,
                              marker='o')
        
        # Add crosses on top
        ax2.scatter(x_start[open_mask], y_start[open_mask], 
                   c='black', s=30,
                   marker='x',
                   linewidths=1)
    
    # Add Earth circle
    earth2 = plt.Circle((0, 0), 1.0, fill=False, edgecolor='black', linewidth=2, linestyle='--')
    ax2.add_patch(earth2)
    
    # Add latitude circles
    for i, lat in enumerate([55, 65, 75]):
        r_lat = np.cos(np.radians(lat))
        lat_circle = plt.Circle((0, 0), r_lat, fill=False, edgecolor='gray', 
                               linewidth=0.5, linestyle=':', alpha=0.5)
        ax2.add_patch(lat_circle)
        # Add latitude label at 45 degree angle to avoid overlap
        angle = np.pi/4  # 45 degrees
        x_label = r_lat * np.cos(angle)
        y_label = r_lat * np.sin(angle)
        ax2.text(x_label, y_label, f'{lat}°', fontsize=8, ha='center', va='center',
                color='gray', alpha=0.7, bbox=dict(boxstyle='round,pad=0.2', 
                                                   facecolor='white', edgecolor='none', alpha=0.7))
    
    
    ax2.set_xlim(-0.7, 0.7)
    ax2.set_ylim(-0.7, 0.7)
    ax2.set_xlabel('X_SM (Re)', fontsize=12)
    ax2.set_ylabel('Y_SM (Re)', fontsize=12)
    ax2.set_title('SM Latitude at Maximum Distance (°)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    if np.any(valid_mask):
        sc_for_cbar = sc2_closed if np.any(closed_mask) else sc2_open
        cbar2 = plt.colorbar(sc_for_cbar, ax=ax2, pad=0.1)
        cbar2.set_label('SM Latitude (°)', fontsize=10)
    
    # Plot 3: SM longitude at maximum distance
    ax3 = axes[2]
    ax3.set_aspect('equal')
    
    if np.any(invalid_mask):
        ax3.scatter(x_start[invalid_mask], y_start[invalid_mask], 
                   c='gray', s=20, alpha=0.3)
    
    # Plot closed field lines
    if np.any(closed_mask):
        sc3_closed = ax3.scatter(x_start[closed_mask], y_start[closed_mask], 
                                c=max_dist_lon[closed_mask], s=30,
                                cmap='hsv',
                                vmin=0, vmax=360,
                                marker='o')
    
    # Plot open field lines
    if np.any(open_mask):
        sc3_open = ax3.scatter(x_start[open_mask], y_start[open_mask], 
                              c=max_dist_lon[open_mask], s=30,
                              cmap='hsv',
                              vmin=0, vmax=360,
                              marker='o')
        
        # Add crosses on top
        ax3.scatter(x_start[open_mask], y_start[open_mask], 
                   c='black', s=30,
                   marker='x',
                   linewidths=1)
    
    # Add Earth circle
    earth3 = plt.Circle((0, 0), 1.0, fill=False, edgecolor='black', linewidth=2, linestyle='--')
    ax3.add_patch(earth3)
    
    # Add latitude circles
    for i, lat in enumerate([55, 65, 75]):
        r_lat = np.cos(np.radians(lat))
        lat_circle = plt.Circle((0, 0), r_lat, fill=False, edgecolor='gray', 
                               linewidth=0.5, linestyle=':', alpha=0.5)
        ax3.add_patch(lat_circle)
        # Add latitude label at 45 degree angle to avoid overlap
        angle = np.pi/4  # 45 degrees
        x_label = r_lat * np.cos(angle)
        y_label = r_lat * np.sin(angle)
        ax3.text(x_label, y_label, f'{lat}°', fontsize=8, ha='center', va='center',
                color='gray', alpha=0.7, bbox=dict(boxstyle='round,pad=0.2', 
                                                   facecolor='white', edgecolor='none', alpha=0.7))
    
    
    ax3.set_xlim(-0.7, 0.7)
    ax3.set_ylim(-0.7, 0.7)
    ax3.set_xlabel('X_SM (Re)', fontsize=12)
    ax3.set_ylabel('Y_SM (Re)', fontsize=12)
    ax3.set_title('SM Longitude at Maximum Distance', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    if np.any(valid_mask):
        sc_for_cbar = sc3_closed if np.any(closed_mask) else sc3_open
        cbar3 = plt.colorbar(sc_for_cbar, ax=ax3, pad=0.1)
        cbar3.set_label('SM Longitude (°)', fontsize=10)
        cbar3.set_ticks([0, 90, 180, 270, 360])
        cbar3.set_ticklabels(['0°', '90°', '180°', '270°', '360°'])
    
    # Add legend to first plot
    ax1.legend(loc='upper left', fontsize=10)
    
    # Get dipole tilt
    ps = geopack.recalc(ut)
    dipole_tilt = np.degrees(ps)
    
    # Overall title
    fig.suptitle(f'Field Line Maximum Distance Analysis - T89 Model (Kp=0)\n' + 
                 f'Spring Equinox (March 20, 2024) at 08:00 UTC, Dipole Tilt: {dipole_tilt:.1f}°\n' +
                 'Starting from Northern Hemisphere at 1 Re, Lat: 55-75°', 
                 fontsize=16, y=1.02)
    
    plt.tight_layout()
    return fig, axes


def main():
    """Main function."""
    # Set time to Spring Equinox (March 20, 2024) at 8:00 AM UTC
    spring_equinox = datetime(2024, 3, 20, 8, 0, 0)
    ut = spring_equinox.timestamp()
    ps = geopack.recalc(ut)
    
    print(f"Using date: {spring_equinox}")
    
    # Set T89 model parameters (quiet conditions)
    # For T89: parmod[0] = iopt (1-7), where iopt=1 corresponds to Kp=0
    parmod = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Create starting grid directly in SM coordinates
    print("Creating grid directly in SM coordinates...")
    print("  Latitude range: 55° to 75°")
    print("  Longitude range: 0° to 360° (full coverage)")
    x_start_sm, y_start_sm, z_start_sm, sm_lat_start, sm_lon_start = create_sm_grid(
        radius=1.0,
        nlat=20,
        nlon=48
    )
    
    # Print summary
    print(f"\nConfiguration:")
    print(f"  Grid points: {len(x_start_sm)}")
    print(f"  Coordinate system: SM (Solar Magnetic)")
    print(f"  Dipole tilt angle: {np.degrees(ps):.1f}°")
    print(f"  T89 Parameters:")
    print(f"    Kp = 0 (iopt={parmod[0]:.0f}, quietest conditions)")
    
    # Analyze field lines
    results = analyze_field_line_max_distance(
        ut, parmod, x_start_sm, y_start_sm, z_start_sm, 
        sm_lat_start, sm_lon_start
    )
    
    # Create plots
    print("\nCreating maximum distance plots...")
    fig, axes = create_max_distance_plots(results, ut)
    
    # Save figure
    output_file = os.path.join(os.path.dirname(__file__), 'field_line_max_distance_sm_spring_equinox_0800.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved {output_file}")
    plt.show()
    
    # Print summary statistics
    valid_mask = ~np.isnan(results['max_dist'])
    open_mask = results['is_open']
    if np.any(valid_mask):
        print(f"\nSummary statistics:")
        print(f"  Valid field lines: {np.sum(valid_mask)} / {len(x_start_sm)}")
        print(f"  Open field lines: {np.sum(open_mask)} ({100*np.sum(open_mask)/np.sum(valid_mask):.1f}% of valid)")
        print(f"  Closed field lines: {np.sum(valid_mask & ~open_mask)}")
        print(f"  Max distance range: {np.nanmin(results['max_dist']):.1f} - {np.nanmax(results['max_dist']):.1f} Re")
        print(f"  Mean max distance: {np.nanmean(results['max_dist']):.1f} Re")
        print(f"  Median max distance: {np.nanmedian(results['max_dist']):.1f} Re")
        print(f"  Mean max distance (open): {np.nanmean(results['max_dist'][open_mask]):.1f} Re")
        print(f"  Mean max distance (closed): {np.nanmean(results['max_dist'][valid_mask & ~open_mask]):.1f} Re")


if __name__ == '__main__':
    main()