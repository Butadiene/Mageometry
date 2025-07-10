#!/usr/bin/env python
"""
Create two rows of heatmaps showing field line properties using SM (Solar Magnetic) coordinates.
Row 1: B-field analysis (minimum B, distance, SM latitude, SM longitude)
Row 2: Rc/RL analysis (minimum Rc/RL, distance, SM latitude, SM longitude)
Uses T96 magnetospheric model with Cartesian XY plots.
Grid is generated directly in SM coordinates without geographic conversion.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from datetime import datetime, timezone
import sys
import os
# Add parent directory to path to import geopack
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geopack
from geopack.trace_field_lines_vectorized import trace_vectorized
from geopack.vectorized import t96_vectorized
from geopack.vectorized.field_line_geometry_vectorized import field_line_curvature_vectorized
from geopack.coordinates_vectorized import smgsm_vectorized


def create_sm_grid(radius=1.0, nlat=8, nlon=8):
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


def calculate_electron_larmor_radius(B_magnitude, electron_energy_keV=100.0, momentum_factor=None):
    """Calculate the electron Larmor radius."""
    if momentum_factor is None:
        # Constants
        m_e = 9.10938356e-31  # electron mass in kg
        e = 1.602176634e-19   # elementary charge in C
        c = 299792458.0       # speed of light in m/s
        
        # Convert energy to Joules
        E_joules = electron_energy_keV * 1000 * e
        
        # Calculate relativistic momentum
        E_rest = m_e * c**2
        E_total = E_joules + E_rest
        p = np.sqrt(E_total**2 - E_rest**2) / c
        
        # Pre-compute momentum factor for reuse
        momentum_factor = p / e
    
    # Convert B to Tesla and calculate Larmor radius
    B_tesla = B_magnitude * 1e-9
    RL_m = momentum_factor / B_tesla
    
    # Convert to km
    RL_km = RL_m / 1000.0
    
    return RL_km, momentum_factor


def analyze_field_lines_sm(ut, parmod, x_start_sm, y_start_sm, z_start_sm, 
                          sm_lat_start, sm_lon_start, electron_energy_keV=100.0):
    """
    Trace field lines using T96 magnetospheric model in SM coordinates.
    """
    # Update geopack parameters
    ps = geopack.recalc(ut)
    
    print(f"Tracing {len(x_start_sm)} field lines with T96 model...")
    
    # Convert SM to GSM for field line tracing (since trace_vectorized expects GSM)
    x_start_gsm, y_start_gsm, z_start_gsm = smgsm_vectorized(x_start_sm, y_start_sm, z_start_sm, 1)
    
    # First trace antiparallel (dir=1)
    xf1_gsm, yf1_gsm, zf1_gsm, fl_x1_gsm, fl_y1_gsm, fl_z1_gsm, status1 = trace_vectorized(
        x_start_gsm, y_start_gsm, z_start_gsm,
        dir=1,  # Trace antiparallel to B
        rlim=20.0,
        r0=1.0,
        parmod=parmod,
        exname='t96',
        inname='igrf',
        maxloop=2000,
        return_full_path=True
    )
    
    # Convert results back to SM
    xf1, yf1, zf1 = smgsm_vectorized(xf1_gsm, yf1_gsm, zf1_gsm, -1)
    fl_x1, fl_y1, fl_z1 = smgsm_vectorized(fl_x1_gsm, fl_y1_gsm, fl_z1_gsm, -1)
    
    # Also trace parallel (dir=-1) 
    xf2_gsm, yf2_gsm, zf2_gsm, fl_x2_gsm, fl_y2_gsm, fl_z2_gsm, status2 = trace_vectorized(
        x_start_gsm, y_start_gsm, z_start_gsm,
        dir=-1,  # Trace parallel to B
        rlim=20.0,
        r0=1.0,
        parmod=parmod,
        exname='t96',
        inname='igrf',
        maxloop=2000,
        return_full_path=True
    )
    
    # Convert results back to SM
    xf2, yf2, zf2 = smgsm_vectorized(xf2_gsm, yf2_gsm, zf2_gsm, -1)
    fl_x2, fl_y2, fl_z2 = smgsm_vectorized(fl_x2_gsm, fl_y2_gsm, fl_z2_gsm, -1)
    
    # Choose the trace that goes to southern hemisphere (if any)
    xf, yf, zf = xf1.copy(), yf1.copy(), zf1.copy()
    fl_x, fl_y, fl_z = fl_x1, fl_y1, fl_z1
    status = status1.copy()
    
    # Check which direction reaches southern hemisphere
    for i in range(len(x_start_sm)):
        r1 = np.sqrt(xf1[i]**2 + yf1[i]**2 + zf1[i]**2)
        r2 = np.sqrt(xf2[i]**2 + yf2[i]**2 + zf2[i]**2)
        
        # If parallel trace reaches southern hemisphere at r0, use it
        if status2[i] == 0 and zf2[i] < 0 and abs(r2 - 1.0) < 0.1:
            # But only if antiparallel didn't also reach southern hemisphere
            if not (status1[i] == 0 and zf1[i] < 0 and abs(r1 - 1.0) < 0.1):
                xf[i], yf[i], zf[i] = xf2[i], yf2[i], zf2[i]
                fl_x[i,:] = fl_x2[i,:]
                fl_y[i,:] = fl_y2[i,:]
                fl_z[i,:] = fl_z2[i,:]
                status[i] = status2[i]
    
    # Initialize result arrays
    nlines = len(x_start_sm)
    min_b = np.full(nlines, np.nan)
    min_b_dist = np.full(nlines, np.nan)
    min_b_lat = np.full(nlines, np.nan)  # SM latitude at min B
    min_b_lon = np.full(nlines, np.nan)   # SM longitude at min B
    min_rc_rl = np.full(nlines, np.nan)
    min_rc_rl_dist = np.full(nlines, np.nan)
    min_rc_rl_lat = np.full(nlines, np.nan)  # SM latitude at min Rc/RL
    min_rc_rl_lon = np.full(nlines, np.nan)  # SM longitude at min Rc/RL
    conjugate_mask = np.zeros(nlines, dtype=bool)
    
    print("Analyzing field lines...")
    
    # Create T96 SM wrapper function once (not inside loop)
    def t96_sm_wrapper(parmod, ps, x_sm, y_sm, z_sm):
        x_gsm, y_gsm, z_gsm = smgsm_vectorized(x_sm, y_sm, z_sm, 1)
        bx_gsm, by_gsm, bz_gsm = t96_vectorized(parmod, ps, x_gsm, y_gsm, z_gsm)
        bx_sm, by_sm, bz_sm = smgsm_vectorized(bx_gsm, by_gsm, bz_gsm, -1)
        return bx_sm, by_sm, bz_sm
    
    # Vectorized conjugate check
    r_final = np.sqrt(xf**2 + yf**2 + zf**2)
    conjugate_mask = (status == 0) & (zf < 0) & (np.abs(r_final - 1.0) < 0.1)
    
    # Process conjugate field lines
    n_conjugate = np.sum(conjugate_mask)
    print(f"  Processing {n_conjugate} conjugate field lines...")
    
    # Pre-compute electron momentum factor for Larmor radius calculation
    _, momentum_factor = calculate_electron_larmor_radius(1.0, electron_energy_keV)
    
    for i in range(nlines):
        if i % 100 == 0 and i > 0:
            print(f"    Progress: {i}/{nlines} ({100*i/nlines:.1f}%)")
            
        if conjugate_mask[i]:
            # Get valid points
            if hasattr(fl_x, 'mask'):
                valid = ~fl_x.mask[i, :]
                x_line = fl_x.data[i, valid]
                y_line = fl_y.data[i, valid]
                z_line = fl_z.data[i, valid]
            else:
                valid = ~np.isnan(fl_x[i, :])
                x_line = fl_x[i, valid]
                y_line = fl_y[i, valid]
                z_line = fl_z[i, valid]
            
            if np.sum(valid) < 10:
                continue
            
            # Convert SM coordinates to GSM for T96 field calculation
            x_line_gsm, y_line_gsm, z_line_gsm = smgsm_vectorized(x_line, y_line, z_line, 1)
            
            # Calculate B field using T96 model (in GSM)
            bx_gsm, by_gsm, bz_gsm = t96_vectorized(parmod, ps, x_line_gsm, y_line_gsm, z_line_gsm)
            
            # Convert B field back to SM coordinates
            bx, by, bz = smgsm_vectorized(bx_gsm, by_gsm, bz_gsm, -1)
            
            b_mag = np.sqrt(bx**2 + by**2 + bz**2)
            distances = np.sqrt(x_line**2 + y_line**2 + z_line**2)
            
            # Find minimum B
            idx_min_b = np.argmin(b_mag)
            min_b[i] = b_mag[idx_min_b]
            min_b_dist[i] = distances[idx_min_b]
            
            # Calculate SM coordinates of minimum B location
            x_min_b = x_line[idx_min_b]
            y_min_b = y_line[idx_min_b]
            z_min_b = z_line[idx_min_b]
            
            # Convert to spherical SM coordinates
            r_min_b = np.sqrt(x_min_b**2 + y_min_b**2 + z_min_b**2)
            lat_min_b_rad = np.arcsin(z_min_b / r_min_b)
            lon_min_b_rad = np.arctan2(y_min_b, x_min_b)
            
            min_b_lat[i] = np.degrees(lat_min_b_rad)
            # Store SM longitude directly
            sm_lon_deg = np.degrees(lon_min_b_rad) % 360
            min_b_lon[i] = sm_lon_deg
            
            # Calculate curvature and Rc/RL
            kappa = field_line_curvature_vectorized(
                t96_sm_wrapper, parmod, ps, x_line, y_line, z_line
            )
            
            # Calculate Rc and RL
            Rc_km = np.zeros_like(kappa)
            valid_kappa = kappa > 0
            Rc_km[valid_kappa] = (1.0 / kappa[valid_kappa]) * 6371.2
            
            RL_km, _ = calculate_electron_larmor_radius(b_mag, electron_energy_keV, momentum_factor)
            
            # Calculate Rc/RL ratio
            rc_rl_ratio = np.zeros_like(Rc_km)
            valid_ratio = (RL_km > 0) & valid_kappa
            rc_rl_ratio[valid_ratio] = Rc_km[valid_ratio] / RL_km[valid_ratio]
            
            # Find minimum Rc/RL
            if np.any(valid_ratio):
                valid_ratios = rc_rl_ratio[valid_ratio]
                if len(valid_ratios) > 0:
                    idx_min = np.argmin(rc_rl_ratio[valid_ratio])
                    valid_indices = np.where(valid_ratio)[0]
                    idx_original = valid_indices[idx_min]
                    
                    min_rc_rl[i] = rc_rl_ratio[idx_original]
                    min_rc_rl_dist[i] = distances[idx_original]
                    
                    # Calculate SM coordinates of minimum Rc/RL location
                    x_min_rc = x_line[idx_original]
                    y_min_rc = y_line[idx_original]
                    z_min_rc = z_line[idx_original]
                    
                    # Convert to spherical SM coordinates
                    r_min_rc = np.sqrt(x_min_rc**2 + y_min_rc**2 + z_min_rc**2)
                    lat_min_rc_rad = np.arcsin(z_min_rc / r_min_rc)
                    lon_min_rc_rad = np.arctan2(y_min_rc, x_min_rc)
                    
                    min_rc_rl_lat[i] = np.degrees(lat_min_rc_rad)
                    # Store SM longitude directly
                    sm_lon_deg = np.degrees(lon_min_rc_rad) % 360
                    min_rc_rl_lon[i] = sm_lon_deg
    
    return (min_b, min_b_dist, min_b_lat, min_b_lon,
            min_rc_rl, min_rc_rl_dist, min_rc_rl_lat, min_rc_rl_lon,
            conjugate_mask, sm_lat_start, sm_lon_start)


def plot_results(min_b, min_b_dist, min_b_lat, min_b_lon,
                min_rc_rl, min_rc_rl_dist, min_rc_rl_lat, min_rc_rl_lon,
                conjugate_mask, sm_lat, sm_lon, ut, electron_energy_keV=100.0,
                output_filename='conjugate_field_analysis_sm_basic.png',
                tilt_angle=None, parmod=None):
    """Plot the field line analysis results in SM coordinates."""
    
    # Create figure with 2x5 subplots (B-field, Rc/RL analyses, and differences)
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    
    # Build title with storm conditions and tilt angle
    title_lines = ['Conjugate Field Line Analysis - SM Coordinates (T96 Model)']
    
    if tilt_angle is not None:
        title_lines.append(f'Dipole Tilt: {tilt_angle:.1f}°')
    
    if parmod is not None:
        storm_info = (f'Storm Conditions: Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT, '
                     f'ByIMF={parmod[2]} nT, BzIMF={parmod[3]} nT')
        title_lines.append(storm_info)
    
    title_lines.append(f'Time: {datetime.fromtimestamp(ut, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")} UTC, '
                      f'Electron Energy: {electron_energy_keV} keV')
    
    fig.suptitle('\n'.join(title_lines), fontsize=14)
    
    def setup_axis(ax, title):
        """Setup common axis properties for SM coordinate plots"""
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('X_SM (Re)', fontsize=9)
        ax.set_ylabel('Y_SM (Re)', fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add SM latitude circles
        angles = np.linspace(0, 2*np.pi, 100)
        for lat in [55, 60, 65, 70, 75]:
            r_lat = np.cos(np.radians(lat))
            ax.plot(r_lat * np.cos(angles), r_lat * np.sin(angles), 
                   'k--', alpha=0.2, linewidth=0.5)
            # Add latitude labels at 45 degree angle
            angle = np.pi/4  # 45 degrees
            x_label = r_lat * np.cos(angle)
            y_label = r_lat * np.sin(angle)
            ax.text(x_label, y_label, f'{lat}°', fontsize=7, ha='center', va='center',
                    color='gray', alpha=0.8, bbox=dict(boxstyle='round,pad=0.2', 
                                                       facecolor='white', edgecolor='none', alpha=0.7))
        
    
    # Convert spherical to Cartesian for plotting
    # Starting positions at r=1 Re
    sm_lon_rad = sm_lon * np.pi / 180
    sm_lat_rad = sm_lat * np.pi / 180
    
    # Calculate X,Y positions at Earth's surface (r=1)
    x_plot = np.cos(sm_lat_rad) * np.cos(sm_lon_rad)
    y_plot = np.cos(sm_lat_rad) * np.sin(sm_lon_rad)
    
    # Row 1: B-field analysis
    # Plot 1: Minimum B-field
    ax1 = axes[0, 0]
    setup_axis(ax1, 'Minimum B-field Strength (nT) - T96 Model')
    
    # Non-conjugate points
    non_conj = ~conjugate_mask
    if np.any(non_conj):
        ax1.scatter(x_plot[non_conj], y_plot[non_conj], 
                   c='gray', s=20, alpha=0.3, label='Open')
    
    # Conjugate points
    conj = conjugate_mask & ~np.isnan(min_b)
    if np.any(conj):
        sc1 = ax1.scatter(x_plot[conj], y_plot[conj], 
                         c=min_b[conj], s=30,
                         cmap='viridis',
                         norm=colors.LogNorm(vmin=np.nanmin(min_b[conj]), 
                                           vmax=np.nanmax(min_b[conj])))
        cbar1 = plt.colorbar(sc1, ax=ax1, pad=0.1)
        cbar1.set_label('Min B (nT)', fontsize=10)
    
    # Plot 2: Distance at minimum B
    ax2 = axes[0, 1]
    setup_axis(ax2, 'Distance at Minimum B-field (Re) - T96 Model')
    
    if np.any(non_conj):
        ax2.scatter(x_plot[non_conj], y_plot[non_conj], 
                   c='gray', s=20, alpha=0.3)
    
    conj = conjugate_mask & ~np.isnan(min_b_dist)
    if np.any(conj):
        sc2 = ax2.scatter(x_plot[conj], y_plot[conj], 
                         c=min_b_dist[conj], s=30,
                         cmap='plasma',
                         vmin=1.0, vmax=min(20.0, np.nanmax(min_b_dist[conj])))
        cbar2 = plt.colorbar(sc2, ax=ax2, pad=0.1)
        cbar2.set_label('Distance (Re)', fontsize=10)
    
    # Plot 3: SM Latitude at minimum B
    ax3 = axes[0, 2]
    setup_axis(ax3, 'SM Latitude at Minimum B-field (°) - T96 Model')
    
    if np.any(non_conj):
        ax3.scatter(x_plot[non_conj], y_plot[non_conj], 
                   c='gray', s=20, alpha=0.3)
    
    conj = conjugate_mask & ~np.isnan(min_b_lat)
    if np.any(conj):
        sc3 = ax3.scatter(x_plot[conj], y_plot[conj], 
                         c=min_b_lat[conj], s=30,
                         cmap='coolwarm',
                         vmin=-90, vmax=90)
        cbar3 = plt.colorbar(sc3, ax=ax3, pad=0.1)
        cbar3.set_label('SM Latitude (°)', fontsize=10)
    
    # Plot 4: SM longitude at minimum B
    ax4 = axes[0, 3]
    setup_axis(ax4, 'SM Longitude at Minimum B-field - T96 Model')
    
    if np.any(non_conj):
        ax4.scatter(x_plot[non_conj], y_plot[non_conj], 
                   c='gray', s=20, alpha=0.3)
    
    conj = conjugate_mask & ~np.isnan(min_b_lon)
    if np.any(conj):
        sc4 = ax4.scatter(x_plot[conj], y_plot[conj], 
                         c=min_b_lon[conj], s=30,
                         cmap='hsv',
                         vmin=0, vmax=360)
        cbar4 = plt.colorbar(sc4, ax=ax4, pad=0.1)
        cbar4.set_label('SM Longitude (°)', fontsize=10)
        cbar4.set_ticks([0, 90, 180, 270, 360])
        cbar4.set_ticklabels(['0°', '90°', '180°', '270°', '360°'])
    
    # Row 2: Rc/RL analysis
    # Plot 5: Minimum Rc/RL
    ax5 = axes[1, 0]
    setup_axis(ax5, f'Minimum Rc/RL Ratio ({electron_energy_keV} keV) - T96 Model')
    
    if np.any(non_conj):
        ax5.scatter(x_plot[non_conj], y_plot[non_conj], 
                   c='gray', s=20, alpha=0.3)
    
    conj = conjugate_mask & ~np.isnan(min_rc_rl) & (min_rc_rl > 0)
    if np.any(conj):
        vmin, vmax = 1.0, 64.0
        min_rc_rl_clipped = np.clip(min_rc_rl[conj], vmin, vmax)
        
        sc5 = ax5.scatter(x_plot[conj], y_plot[conj], 
                         c=min_rc_rl_clipped, s=30,
                         cmap='RdBu_r',
                         norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        cbar5 = plt.colorbar(sc5, ax=ax5, pad=0.1)
        cbar5.set_label(r'Min $R_c/R_L$', fontsize=10)
        cbar5.ax.axhline(y=8, color='black', linestyle='--', linewidth=1)
    
    # Plot 6: Distance at minimum Rc/RL
    ax6 = axes[1, 1]
    setup_axis(ax6, 'Distance at Minimum Rc/RL (Re) - T96 Model')
    
    if np.any(non_conj):
        ax6.scatter(x_plot[non_conj], y_plot[non_conj], 
                   c='gray', s=20, alpha=0.3)
    
    conj = conjugate_mask & ~np.isnan(min_rc_rl_dist)
    if np.any(conj):
        sc6 = ax6.scatter(x_plot[conj], y_plot[conj], 
                         c=min_rc_rl_dist[conj], s=30,
                         cmap='plasma',
                         vmin=1.0, vmax=min(20.0, np.nanmax(min_rc_rl_dist[conj])))
        cbar6 = plt.colorbar(sc6, ax=ax6, pad=0.1)
        cbar6.set_label('Distance (Re)', fontsize=10)
    
    # Plot 7: SM Latitude at minimum Rc/RL
    ax7 = axes[1, 2]
    setup_axis(ax7, f'SM Latitude at Minimum Rc/RL (°) - T96 Model')
    
    if np.any(non_conj):
        ax7.scatter(x_plot[non_conj], y_plot[non_conj], 
                   c='gray', s=20, alpha=0.3)
    
    conj = conjugate_mask & ~np.isnan(min_rc_rl_lat)
    if np.any(conj):
        sc7 = ax7.scatter(x_plot[conj], y_plot[conj], 
                         c=min_rc_rl_lat[conj], s=30,
                         cmap='coolwarm',
                         vmin=-90, vmax=90)
        cbar7 = plt.colorbar(sc7, ax=ax7, pad=0.1)
        cbar7.set_label('SM Latitude (°)', fontsize=10)
    
    # Plot 8: SM longitude at minimum Rc/RL
    ax8 = axes[1, 3]
    setup_axis(ax8, f'SM Longitude at Minimum Rc/RL - T96 Model')
    
    if np.any(non_conj):
        ax8.scatter(x_plot[non_conj], y_plot[non_conj], 
                   c='gray', s=20, alpha=0.3)
    
    conj = conjugate_mask & ~np.isnan(min_rc_rl_lon)
    if np.any(conj):
        sc8 = ax8.scatter(x_plot[conj], y_plot[conj], 
                         c=min_rc_rl_lon[conj], s=30,
                         cmap='hsv',
                         vmin=0, vmax=360)
        cbar8 = plt.colorbar(sc8, ax=ax8, pad=0.1)
        cbar8.set_label('SM Longitude (°)', fontsize=10)
        cbar8.set_ticks([0, 90, 180, 270, 360])
        cbar8.set_ticklabels(['0°', '90°', '180°', '270°', '360°'])
    
    # Column 5: Difference plots
    # Plot 9: Distance difference (min_b_dist - min_rc_rl_dist)
    ax9 = axes[0, 4]
    setup_axis(ax9, 'Distance Difference: Min B - Min Rc/RL (Re)')
    
    if np.any(non_conj):
        ax9.scatter(x_plot[non_conj], y_plot[non_conj], 
                   c='gray', s=20, alpha=0.3)
    
    # Calculate distance difference
    dist_diff = min_b_dist - min_rc_rl_dist
    conj = conjugate_mask & ~np.isnan(dist_diff)
    
    # Identify points where Rc/RL < 8
    rc_rl_below_8 = (min_rc_rl < 8) & ~np.isnan(min_rc_rl)
    
    if np.any(conj):
        # Center colormap around zero
        max_abs_diff = np.nanpercentile(np.abs(dist_diff[conj]), 95)
        
        # Plot regular points (Rc/RL >= 8) as circles
        regular_mask = conj & ~rc_rl_below_8
        if np.any(regular_mask):
            sc9a = ax9.scatter(x_plot[regular_mask], y_plot[regular_mask], 
                             c=dist_diff[regular_mask], s=30,
                             cmap='RdBu_r', marker='o',
                             vmin=-max_abs_diff, vmax=max_abs_diff)
        
        # Plot points with Rc/RL < 8 as squares
        highlight_mask = conj & rc_rl_below_8
        if np.any(highlight_mask):
            sc9b = ax9.scatter(x_plot[highlight_mask], y_plot[highlight_mask], 
                             c=dist_diff[highlight_mask], s=60,
                             cmap='RdBu_r', marker='s',
                             vmin=-max_abs_diff, vmax=max_abs_diff,
                             edgecolors='black', linewidth=1.5)
        
        # Use the first scatter for colorbar
        if np.any(regular_mask):
            cbar9 = plt.colorbar(sc9a, ax=ax9, pad=0.1)
        elif np.any(highlight_mask):
            cbar9 = plt.colorbar(sc9b, ax=ax9, pad=0.1)
        cbar9.set_label('Distance Difference (Re)', fontsize=10)
        
        # Add legend for markers
        legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor='gray', markersize=8, label='Rc/RL ≥ 8'),
                         Line2D([0], [0], marker='s', color='w', 
                                markerfacecolor='gray', markersize=10, 
                                markeredgecolor='black', markeredgewidth=1.5, label='Rc/RL < 8')]
        ax9.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    # Plot 10: Latitude difference (min_b_lat - min_rc_rl_lat)
    ax10 = axes[1, 4]
    setup_axis(ax10, 'SM Latitude Difference: Min B - Min Rc/RL (°)')
    
    if np.any(non_conj):
        ax10.scatter(x_plot[non_conj], y_plot[non_conj], 
                    c='gray', s=20, alpha=0.3)
    
    # Calculate latitude difference
    lat_diff = min_b_lat - min_rc_rl_lat
    conj = conjugate_mask & ~np.isnan(lat_diff)
    
    if np.any(conj):
        # Center colormap around zero
        max_abs_diff = np.nanpercentile(np.abs(lat_diff[conj]), 95)
        
        # Plot regular points (Rc/RL >= 8) as circles
        regular_mask = conj & ~rc_rl_below_8
        if np.any(regular_mask):
            sc10a = ax10.scatter(x_plot[regular_mask], y_plot[regular_mask], 
                               c=lat_diff[regular_mask], s=30,
                               cmap='RdBu_r', marker='o',
                               vmin=-max_abs_diff, vmax=max_abs_diff)
        
        # Plot points with Rc/RL < 8 as squares
        highlight_mask = conj & rc_rl_below_8
        if np.any(highlight_mask):
            sc10b = ax10.scatter(x_plot[highlight_mask], y_plot[highlight_mask], 
                               c=lat_diff[highlight_mask], s=60,
                               cmap='RdBu_r', marker='s',
                               vmin=-max_abs_diff, vmax=max_abs_diff,
                               edgecolors='black', linewidth=1.5)
        
        # Use the first scatter for colorbar
        if np.any(regular_mask):
            cbar10 = plt.colorbar(sc10a, ax=ax10, pad=0.1)
        elif np.any(highlight_mask):
            cbar10 = plt.colorbar(sc10b, ax=ax10, pad=0.1)
        cbar10.set_label('Latitude Difference (°)', fontsize=10)
        
        # Add legend for markers
        legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor='gray', markersize=8, label='Rc/RL ≥ 8'),
                         Line2D([0], [0], marker='s', color='w', 
                                markerfacecolor='gray', markersize=10, 
                                markeredgecolor='black', markeredgewidth=1.5, label='Rc/RL < 8')]
        ax10.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {output_filename}")
    plt.close()


def run_single_analysis(ut, parmod, scenario_name, electron_energy_keV=100.0):
    """Run analysis for a single scenario."""
    # Update geopack parameters
    ps = geopack.recalc(ut)
    
    print(f"\n{'='*60}")
    print(f"Running scenario: {scenario_name}")
    print(f"  Dipole tilt angle: {np.degrees(ps):.1f}°")
    print(f"  T96 Parameters: Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT, "
          f"ByIMF={parmod[2]} nT, BzIMF={parmod[3]} nT")
    
    # Create SM coordinate grid
    x_sm, y_sm, z_sm, sm_lat, sm_lon = create_sm_grid(radius=1.0, nlat=32, nlon=48)
    
    # Analyze field lines
    results = analyze_field_lines_sm(ut, parmod, x_sm, y_sm, z_sm, 
                                    sm_lat, sm_lon, electron_energy_keV)
    
    # Unpack results
    (min_b, min_b_dist, min_b_lat, min_b_lon,
     min_rc_rl, min_rc_rl_dist, min_rc_rl_lat, min_rc_rl_lon,
     conjugate_mask, sm_lat_out, sm_lon_out) = results
    
    # Plot results - save to conjugate_field_analysis_sm_diff folder
    output_dir = os.path.join(os.path.dirname(__file__), 'conjugate_field_analysis_sm_diff')
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f'{scenario_name}.png')
    
    plot_results(min_b, min_b_dist, min_b_lat, min_b_lon,
                min_rc_rl, min_rc_rl_dist, min_rc_rl_lat, min_rc_rl_lon,
                conjugate_mask, sm_lat_out, sm_lon_out, ut, electron_energy_keV,
                output_filename=output_filename,
                tilt_angle=np.degrees(ps), parmod=parmod)
    
    # Print summary statistics
    print(f"\nSummary for {scenario_name}:")
    if np.any(conjugate_mask):
        conj_idx = conjugate_mask
        print(f"  Conjugate field lines: {np.sum(conj_idx)}/{len(conjugate_mask)} ({100*np.sum(conj_idx)/len(conjugate_mask):.1f}%)")
        
        if np.any(~np.isnan(min_rc_rl[conj_idx])):
            threshold = 8.0
            below_threshold = np.sum(min_rc_rl[conj_idx] < threshold)
            print(f"  Field lines with Rc/RL < {threshold}: {below_threshold} "
                  f"({100*below_threshold/np.sum(conj_idx):.1f}% of conjugate)")


def main():
    """Main function to run multiple conjugate field analyses."""
    
    # Electron energy in keV
    electron_energy_keV = 300.0
    
    # Define different tilt scenarios (dates throughout the year)
    tilt_scenarios = [
        # (datetime, scenario_name)
        (datetime(2024, 12, 21, 12, 0, 0), "winter_solstice"),  # Maximum negative tilt
        (datetime(2024, 11, 15, 12, 0, 0), "november"),
        (datetime(2024, 10, 15, 12, 0, 0), "october"),
        (datetime(2024, 9, 23, 0, 0, 0), "autumn_equinox"),     # Near zero tilt
        (datetime(2024, 8, 15, 12, 0, 0), "august"),
        (datetime(2024, 7, 15, 12, 0, 0), "july"),
        (datetime(2024, 6, 21, 12, 0, 0), "summer_solstice"),   # Maximum positive tilt
        (datetime(2024, 5, 15, 12, 0, 0), "may"),
        (datetime(2024, 4, 15, 12, 0, 0), "april"),
        (datetime(2024, 3, 20, 12, 0, 0), "spring_equinox"),    # Near zero tilt
        (datetime(2024, 2, 15, 12, 0, 0), "february"),
        (datetime(2024, 1, 15, 12, 0, 0), "january"),
    ]
    
    # Define different storm conditions
    # Parameters for T96 model: [Pdyn, Dst, ByIMF, BzIMF, unused...]
    storm_conditions = [
        # (parmod, condition_name)
        ([2.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0], "quiet"),           # Quiet conditions
        ([3.0, -20.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0], "moderate"),     # Moderate activity
        ([5.0, -50.0, 5.0, -10.0, 0, 0, 0, 0, 0, 0], "storm"),       # Storm conditions
        ([8.0, -100.0, 10.0, -15.0, 0, 0, 0, 0, 0, 0], "intense"),   # Intense storm
        ([2.0, -30.0, -5.0, 5.0, 0, 0, 0, 0, 0, 0], "northward"),    # Northward IMF
    ]
    
    print("="*80)
    print("Running conjugate field analysis for multiple scenarios")
    print(f"Electron energy: {electron_energy_keV} keV")
    print(f"Total scenarios: {len(tilt_scenarios)} tilts × {len(storm_conditions)} conditions = "
          f"{len(tilt_scenarios) * len(storm_conditions)} images")
    print("="*80)
    
    # Run analysis for each combination
    for storm_parmod, storm_name in storm_conditions:
        for date_time, tilt_name in tilt_scenarios:
            ut = date_time.timestamp()
            scenario_name = f"{tilt_name}_{storm_name}"
            
            try:
                run_single_analysis(ut, storm_parmod, scenario_name, electron_energy_keV)
            except Exception as e:
                print(f"\nError in scenario {scenario_name}: {str(e)}")
                continue
    
    print("\n" + "="*80)
    print("All scenarios completed!")
    print(f"Images saved in: examples/conjugate_test/conjugate_field_analysis_sm_diff/")
    print("="*80)


if __name__ == "__main__":
    main()