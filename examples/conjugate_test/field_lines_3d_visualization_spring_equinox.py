#!/usr/bin/env python
"""
3D Visualization of magnetic field lines for Spring Equinox at 8:00 AM UTC.
Creates an interactive 3D plot showing field lines traced from various starting points.
Uses T89 magnetospheric model with SM (Solar Magnetic) coordinates.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as mcolors
from datetime import datetime
import sys
import os

# Add parent directory to path to import geopack
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geopack
from geopack.trace_field_lines_vectorized import trace_vectorized
from geopack.vectorized import t89_vectorized
from geopack.coordinates_vectorized import smgsm_vectorized


def create_sm_grid_3d(nlat=10, nlon=12, lat_range=(55, 75)):
    """
    Create a sparse grid of starting points for 3D visualization.
    """
    # Create sparse grid for better visualization
    sm_lat = np.linspace(lat_range[0], lat_range[1], nlat)
    sm_lon = np.linspace(0, 360, nlon, endpoint=False)
    
    # Create meshgrid
    SM_LON_GRID, SM_LAT_GRID = np.meshgrid(sm_lon, sm_lat)
    
    # Flatten for processing
    sm_lon_flat = SM_LON_GRID.flatten()
    sm_lat_flat = SM_LAT_GRID.flatten()
    
    # Convert to radians
    sm_lon_rad = sm_lon_flat * np.pi / 180
    sm_lat_rad = sm_lat_flat * np.pi / 180
    
    # Convert to Cartesian SM coordinates at Earth's surface (r=1 Re)
    x_sm = np.cos(sm_lat_rad) * np.cos(sm_lon_rad)
    y_sm = np.cos(sm_lat_rad) * np.sin(sm_lon_rad)
    z_sm = np.sin(sm_lat_rad)
    
    return x_sm, y_sm, z_sm, sm_lat_flat, sm_lon_flat


def trace_field_lines_3d(ut, parmod, x_start_sm, y_start_sm, z_start_sm):
    """
    Trace field lines in both directions and return full paths.
    """
    # Update geopack parameters
    ps = geopack.recalc(ut)
    
    print(f"Tracing {len(x_start_sm)} field lines with T89 model...")
    
    # Convert SM to GSM for field line tracing
    x_start_gsm, y_start_gsm, z_start_gsm = smgsm_vectorized(x_start_sm, y_start_sm, z_start_sm, 1)
    
    # Trace antiparallel (dir=1)
    xf1_gsm, yf1_gsm, zf1_gsm, fl_x1_gsm, fl_y1_gsm, fl_z1_gsm, status1 = trace_vectorized(
        x_start_gsm, y_start_gsm, z_start_gsm,
        dir=1,
        rlim=30.0,
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
    
    # Combine field line segments
    field_lines = []
    statuses = []
    
    for i in range(len(x_start_sm)):
        # Extract valid points for both directions
        if hasattr(fl_x1, 'mask'):
            valid1 = ~fl_x1.mask[i, :]
            x_line1 = fl_x1.data[i, valid1] if np.sum(valid1) > 0 else np.array([])
            y_line1 = fl_y1.data[i, valid1] if np.sum(valid1) > 0 else np.array([])
            z_line1 = fl_z1.data[i, valid1] if np.sum(valid1) > 0 else np.array([])
            
            valid2 = ~fl_x2.mask[i, :]
            x_line2 = fl_x2.data[i, valid2][::-1] if np.sum(valid2) > 0 else np.array([])
            y_line2 = fl_y2.data[i, valid2][::-1] if np.sum(valid2) > 0 else np.array([])
            z_line2 = fl_z2.data[i, valid2][::-1] if np.sum(valid2) > 0 else np.array([])
        else:
            valid1 = ~np.isnan(fl_x1[i, :])
            x_line1 = fl_x1[i, valid1] if np.sum(valid1) > 0 else np.array([])
            y_line1 = fl_y1[i, valid1] if np.sum(valid1) > 0 else np.array([])
            z_line1 = fl_z1[i, valid1] if np.sum(valid1) > 0 else np.array([])
            
            valid2 = ~np.isnan(fl_x2[i, :])
            x_line2 = fl_x2[i, valid2][::-1] if np.sum(valid2) > 0 else np.array([])
            y_line2 = fl_y2[i, valid2][::-1] if np.sum(valid2) > 0 else np.array([])
            z_line2 = fl_z2[i, valid2][::-1] if np.sum(valid2) > 0 else np.array([])
        
        # Combine both directions
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
        
        if len(x_line) > 2:
            field_lines.append((x_line, y_line, z_line))
            
            # Determine if field line is closed
            r1_end = np.sqrt(xf1_gsm[i]**2 + yf1_gsm[i]**2 + zf1_gsm[i]**2)
            r2_end = np.sqrt(xf2_gsm[i]**2 + yf2_gsm[i]**2 + zf2_gsm[i]**2)
            is_closed = (status1[i] == 0 and abs(r1_end - 1.0) < 0.1) and \
                       (status2[i] == 0 and abs(r2_end - 1.0) < 0.1)
            statuses.append(is_closed)
    
    return field_lines, statuses


def create_3d_visualization(field_lines, statuses, ut):
    """
    Create interactive 3D visualization of magnetic field lines.
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create color maps for closed and open field lines
    cmap_closed = plt.colormaps['viridis']
    cmap_open = plt.colormaps['plasma']
    
    # Plot field lines
    for i, ((x, y, z), is_closed) in enumerate(zip(field_lines, statuses)):
        # Calculate maximum distance for color coding
        distances = np.sqrt(x**2 + y**2 + z**2)
        max_dist = np.max(distances)
        
        # Normalize color based on max distance
        if is_closed:
            color = cmap_closed(np.clip(max_dist / 8.0, 0, 1))
            alpha = 0.8
            linewidth = 1.5
        else:
            color = cmap_open(np.clip(max_dist / 30.0, 0, 1))
            alpha = 0.6
            linewidth = 1.2
        
        # Plot the field line
        ax.plot(x, y, z, color=color, alpha=alpha, linewidth=linewidth)
    
    # Add Earth sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_earth = np.outer(np.cos(u), np.sin(v))
    y_earth = np.outer(np.sin(u), np.sin(v))
    z_earth = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x_earth, y_earth, z_earth, color='blue', alpha=0.3, shade=True)
    
    # Add coordinate axes
    axis_length = 5
    ax.plot([0, axis_length], [0, 0], [0, 0], 'r-', linewidth=2, label='X_SM')
    ax.plot([0, 0], [0, axis_length], [0, 0], 'g-', linewidth=2, label='Y_SM')
    ax.plot([0, 0], [0, 0], [0, axis_length], 'b-', linewidth=2, label='Z_SM')
    
    # Add magnetopause boundary (approximate)
    theta = np.linspace(0, 2*np.pi, 100)
    phi = np.linspace(0, np.pi, 50)
    THETA, PHI = np.meshgrid(theta, phi)
    
    # Simple magnetopause shape
    with np.errstate(divide='ignore', invalid='ignore'):
        r_mp = 10.0 * (2.0 / (1 + np.cos(PHI)))**0.5
        r_mp = np.where(PHI > np.pi/2, np.clip(r_mp, 0, 15), r_mp)
        r_mp = np.nan_to_num(r_mp, nan=15.0, posinf=15.0)
    
    X_mp = r_mp * np.sin(PHI) * np.cos(THETA)
    Y_mp = r_mp * np.sin(PHI) * np.sin(THETA)
    Z_mp = r_mp * np.cos(PHI)
    
    ax.plot_surface(X_mp, Y_mp, Z_mp, alpha=0.1, color='gray', shade=False)
    
    # Set labels and title
    ax.set_xlabel('X_SM (Re)', fontsize=12)
    ax.set_ylabel('Y_SM (Re)', fontsize=12)
    ax.set_zlabel('Z_SM (Re)', fontsize=12)
    
    # Get dipole tilt
    ps = geopack.recalc(ut)
    dipole_tilt = np.degrees(ps)
    
    ax.set_title(f'3D Magnetic Field Lines - T89 Model (Kp=0)\n' + 
                 f'Spring Equinox (March 20, 2024) at 08:00 UTC\n' +
                 f'Dipole Tilt: {dipole_tilt:.1f}°', 
                 fontsize=14, pad=20)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Set axis limits
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_zlim(-15, 15)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    closed_line = plt.Line2D([0], [0], color=cmap_closed(0.5), linewidth=2, label='Closed field lines')
    open_line = plt.Line2D([0], [0], color=cmap_open(0.5), linewidth=2, label='Open field lines')
    ax.legend(handles=[closed_line, open_line], loc='upper left')
    
    # Add color note
    fig.text(0.95, 0.5, 'Color indicates\nmax distance\nfrom Earth', 
             transform=fig.transFigure, fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    return fig, ax


def create_field_line_cross_sections(field_lines, statuses, ut):
    """
    Create cross-sectional views of field lines in different planes.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # XY plane (view from North)
    ax1 = axes[0, 0]
    for (x, y, z), is_closed in zip(field_lines, statuses):
        color = 'blue' if is_closed else 'red'
        alpha = 0.7 if is_closed else 0.5
        ax1.plot(x, y, color=color, alpha=alpha, linewidth=1)
    
    # Add Earth
    earth_circle = plt.Circle((0, 0), 1.0, fill=True, color='lightblue', alpha=0.8)
    ax1.add_patch(earth_circle)
    ax1.set_xlim(-20, 20)
    ax1.set_ylim(-20, 20)
    ax1.set_xlabel('X_SM (Re)')
    ax1.set_ylabel('Y_SM (Re)')
    ax1.set_title('XY Plane (View from North)')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # XZ plane (view from Dawn)
    ax2 = axes[0, 1]
    for (x, y, z), is_closed in zip(field_lines, statuses):
        color = 'blue' if is_closed else 'red'
        alpha = 0.7 if is_closed else 0.5
        ax2.plot(x, z, color=color, alpha=alpha, linewidth=1)
    
    # Add Earth
    earth_circle = plt.Circle((0, 0), 1.0, fill=True, color='lightblue', alpha=0.8)
    ax2.add_patch(earth_circle)
    ax2.set_xlim(-20, 20)
    ax2.set_ylim(-15, 15)
    ax2.set_xlabel('X_SM (Re)')
    ax2.set_ylabel('Z_SM (Re)')
    ax2.set_title('XZ Plane (View from Dawn)')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # YZ plane (view from Sun)
    ax3 = axes[1, 0]
    for (x, y, z), is_closed in zip(field_lines, statuses):
        color = 'blue' if is_closed else 'red'
        alpha = 0.7 if is_closed else 0.5
        ax3.plot(y, z, color=color, alpha=alpha, linewidth=1)
    
    # Add Earth
    earth_circle = plt.Circle((0, 0), 1.0, fill=True, color='lightblue', alpha=0.8)
    ax3.add_patch(earth_circle)
    ax3.set_xlim(-20, 20)
    ax3.set_ylim(-15, 15)
    ax3.set_xlabel('Y_SM (Re)')
    ax3.set_ylabel('Z_SM (Re)')
    ax3.set_title('YZ Plane (View from Sun)')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # 3D perspective in 2D
    ax4 = axes[1, 1]
    ax4.text(0.5, 0.5, 'Field Line Statistics', ha='center', va='center', 
             fontsize=16, transform=ax4.transAxes)
    
    n_closed = sum(statuses)
    n_open = len(statuses) - n_closed
    
    stats_text = f"Total field lines: {len(statuses)}\n"
    stats_text += f"Closed field lines: {n_closed} ({100*n_closed/len(statuses):.1f}%)\n"
    stats_text += f"Open field lines: {n_open} ({100*n_open/len(statuses):.1f}%)\n\n"
    
    # Get dipole tilt
    ps = geopack.recalc(ut)
    dipole_tilt = np.degrees(ps)
    stats_text += f"Dipole tilt: {dipole_tilt:.1f}°\n"
    stats_text += f"Model: T89 (Kp=0)"
    
    ax4.text(0.5, 0.3, stats_text, ha='center', va='center', 
             fontsize=12, transform=ax4.transAxes,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    ax4.axis('off')
    
    # Overall title
    fig.suptitle('Magnetic Field Line Cross Sections - Spring Equinox at 08:00 UTC', 
                 fontsize=16)
    
    plt.tight_layout()
    return fig


def main():
    """Main function."""
    # Set time to Spring Equinox (March 20, 2024) at 8:00 AM UTC
    spring_equinox = datetime(2024, 3, 20, 8, 0, 0)
    ut = spring_equinox.timestamp()
    ps = geopack.recalc(ut)
    
    print(f"Using date: {spring_equinox}")
    print(f"Dipole tilt angle: {np.degrees(ps):.1f}°")
    
    # Set T89 model parameters (quiet conditions, Kp=0)
    parmod = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Create starting grid for 3D visualization
    print("\nCreating starting grid for 3D visualization...")
    x_start_sm, y_start_sm, z_start_sm, sm_lat, sm_lon = create_sm_grid_3d(
        nlat=8,  # Fewer points for clearer visualization
        nlon=16,
        lat_range=(55, 75)
    )
    
    print(f"Grid points: {len(x_start_sm)}")
    
    # Trace field lines
    field_lines, statuses = trace_field_lines_3d(ut, parmod, x_start_sm, y_start_sm, z_start_sm)
    
    print(f"\nSuccessfully traced {len(field_lines)} field lines")
    print(f"Closed field lines: {sum(statuses)}")
    print(f"Open field lines: {len(statuses) - sum(statuses)}")
    
    # Create 3D visualization
    print("\nCreating 3D visualization...")
    fig_3d, ax_3d = create_3d_visualization(field_lines, statuses, ut)
    
    # Save 3D plot
    output_file_3d = os.path.join(os.path.dirname(__file__), 'field_lines_3d_spring_equinox.png')
    plt.savefig(output_file_3d, dpi=300, bbox_inches='tight')
    print(f"Saved {output_file_3d}")
    
    # Create cross-sectional views
    print("\nCreating cross-sectional views...")
    fig_cross = create_field_line_cross_sections(field_lines, statuses, ut)
    
    # Save cross-sectional plot
    output_file_cross = os.path.join(os.path.dirname(__file__), 'field_lines_cross_sections_spring_equinox.png')
    fig_cross.savefig(output_file_cross, dpi=300, bbox_inches='tight')
    print(f"Saved {output_file_cross}")
    
    # plt.show()  # Comment out to avoid blocking


if __name__ == '__main__':
    main()