#!/usr/bin/env python
"""
Create a combined plot showing the second row (Rc/RL analysis) 
from each 500 keV time period in a single figure using T89 model with Kp=0.
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
from geopack.vectorized import t89_vectorized
from geopack.vectorized.field_line_geometry_vectorized import field_line_curvature_vectorized
from geopack.vectorized.field_line_directional_derivatives_new import field_line_directional_derivatives_vectorized
from geopack.coordinates_vectorized import smgsm_vectorized


def create_sm_grid(radius=1.0, nlat=8, nlon=8):
    """Create a grid of starting points directly in SM coordinates."""
    # Create latitude grid in SM coordinates (0° = SM equator, 90° = north magnetic pole)
    sm_lat = np.linspace(55, 75, nlat)
    
    # Create longitude grid (0-360°)
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


def analyze_field_lines_sm_t89(ut, iopt, x_start_sm, y_start_sm, z_start_sm, 
                               sm_lat_start, sm_lon_start, electron_energy_keV=100.0):
    """Trace field lines using T89 magnetospheric model in SM coordinates."""
    # Update geopack parameters
    ps = geopack.recalc(ut)
    
    # Convert SM to GSM for field line tracing
    x_start_gsm, y_start_gsm, z_start_gsm = smgsm_vectorized(x_start_sm, y_start_sm, z_start_sm, 1)
    
    # First trace antiparallel (dir=1)
    xf1_gsm, yf1_gsm, zf1_gsm, fl_x1_gsm, fl_y1_gsm, fl_z1_gsm, status1 = trace_vectorized(
        x_start_gsm, y_start_gsm, z_start_gsm,
        dir=1,  # Trace antiparallel to B
        rlim=20.0,
        r0=1.0,
        parmod=iopt,  # For T89, parmod is just the Kp index
        exname='t89',
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
        parmod=iopt,
        exname='t89',
        inname='igrf',
        maxloop=2000,
        return_full_path=True
    )
    
    # Convert results back to SM
    xf2, yf2, zf2 = smgsm_vectorized(xf2_gsm, yf2_gsm, zf2_gsm, -1)
    fl_x2, fl_y2, fl_z2 = smgsm_vectorized(fl_x2_gsm, fl_y2_gsm, fl_z2_gsm, -1)
    
    # Choose the trace that goes to southern hemisphere
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
    min_rc_rl = np.full(nlines, np.nan)
    min_rc_rl_dist = np.full(nlines, np.nan)
    min_rc_rl_lat = np.full(nlines, np.nan)
    min_rc_rl_lon = np.full(nlines, np.nan)
    conjugate_mask = np.zeros(nlines, dtype=bool)
    
    # Create T89 SM wrapper function
    def t89_sm_wrapper(parmod, ps, x_sm, y_sm, z_sm):
        x_gsm, y_gsm, z_gsm = smgsm_vectorized(x_sm, y_sm, z_sm, 1)
        bx_gsm, by_gsm, bz_gsm = t89_vectorized(parmod, ps, x_gsm, y_gsm, z_gsm)
        bx_sm, by_sm, bz_sm = smgsm_vectorized(bx_gsm, by_gsm, bz_gsm, -1)
        return bx_sm, by_sm, bz_sm
    
    # Vectorized conjugate check
    r_final = np.sqrt(xf**2 + yf**2 + zf**2)
    conjugate_mask = (status == 0) & (zf < 0) & (np.abs(r_final - 1.0) < 0.1)
    
    # Pre-compute electron momentum factor
    _, momentum_factor = calculate_electron_larmor_radius(1.0, electron_energy_keV)
    
    for i in range(nlines):
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
            
            # Convert SM coordinates to GSM for T89 field calculation
            x_line_gsm, y_line_gsm, z_line_gsm = smgsm_vectorized(x_line, y_line, z_line, 1)
            
            # Calculate B field using T89 model
            bx_gsm, by_gsm, bz_gsm = t89_vectorized(iopt, ps, x_line_gsm, y_line_gsm, z_line_gsm)
            
            # Convert B field back to SM coordinates
            bx, by, bz = smgsm_vectorized(bx_gsm, by_gsm, bz_gsm, -1)
            
            b_mag = np.sqrt(bx**2 + by**2 + bz**2)
            distances = np.sqrt(x_line**2 + y_line**2 + z_line**2)
            
            # Find minimum B
            idx_min_b = np.argmin(b_mag)
            min_b[i] = b_mag[idx_min_b]
            min_b_dist[i] = distances[idx_min_b]
            
            # Calculate curvature and Rc/RL
            kappa = field_line_curvature_vectorized(
                t89_sm_wrapper, iopt, ps, x_line, y_line, z_line
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
                    
                    # Calculate SM lat/lon at minimum Rc/RL
                    r_at_min = distances[idx_original]
                    x_at_min = x_line[idx_original]
                    y_at_min = y_line[idx_original]
                    z_at_min = z_line[idx_original]
                    
                    # Convert to spherical SM coordinates
                    min_rc_rl_lat[i] = np.degrees(np.arcsin(z_at_min / r_at_min))
                    min_rc_rl_lon[i] = np.degrees(np.arctan2(y_at_min, x_at_min))
                    if min_rc_rl_lon[i] < 0:
                        min_rc_rl_lon[i] += 360
    
    return {
        'min_b': min_b,
        'min_b_dist': min_b_dist,
        'min_rc_rl': min_rc_rl,
        'min_rc_rl_dist': min_rc_rl_dist,
        'min_rc_rl_lat': min_rc_rl_lat,
        'min_rc_rl_lon': min_rc_rl_lon,
        'conjugate_mask': conjugate_mask,
        'sm_lat': sm_lat_start,
        'sm_lon': sm_lon_start
    }


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
    time_labels = ['07:00', '08:00', '09:00', '10:00']
    
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
                 'T89 Model (Kp=0), Spring Equinox 2024, Quiet Conditions\n' +
                 'SM Coordinates, Starting from Northern Hemisphere at 1 Re', 
                 fontsize=16, y=0.995)
    
    plt.tight_layout()
    return fig, axes


def main():
    """Main function."""
    # Base date: Spring Equinox (March 20, 2024)
    base_date = datetime(2024, 3, 20, 0, 0, 0)
    
    # Create times for 7:00, 8:00, 9:00, 10:00
    times = [
        base_date + timedelta(hours=7),   # 07:00
        base_date + timedelta(hours=8),   # 08:00
        base_date + timedelta(hours=9),   # 09:00  
        base_date + timedelta(hours=10)   # 10:00
    ]
    
    time_labels = ['0700', '0800', '0900', '1000']
    
    # Set T89 model parameter (Kp index)
    # Kp=0 corresponds to iopt=1 in T89 model
    iopt = 1  # Kp=0 (quietest conditions)
    
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
    print(f"T89 Model Parameters:")
    print(f"  Kp index = 0 (iopt = {iopt})")
    print(f"  Conditions: Quiet magnetosphere\n")
    
    # Analyze field lines for each time
    all_results = []
    for i, (time, label) in enumerate(zip(times, time_labels)):
        print(f"\nProcessing time {i+1}/4: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        ut = time.timestamp()
        ps = geopack.recalc(ut)
        print(f"  Dipole tilt angle: {np.degrees(ps):.1f}°")
        
        # Analyze field lines
        results = analyze_field_lines_sm_t89(
            ut, iopt, x_start_sm, y_start_sm, z_start_sm, 
            sm_lat_start, sm_lon_start, electron_energy_keV
        )
        
        conjugate_mask = results['conjugate_mask']
        print(f"  Conjugate field lines: {np.sum(conjugate_mask)}/{len(x_start_sm)} ({100*np.sum(conjugate_mask)/len(x_start_sm):.1f}%)")
        
        all_results.append(results)
    
    # Create combined plot showing Rc/RL analysis (second row)
    print("\nCreating combined Rc/RL analysis plot...")
    fig, axes = create_rcrl_analysis_comparison(all_results, times, electron_energy_keV)
    
    # Save figure
    filename = os.path.join(os.path.dirname(__file__), 'conjugate_field_analysis_sm_500keV_rcrl_t89_kp0.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.show()


if __name__ == '__main__':
    main()