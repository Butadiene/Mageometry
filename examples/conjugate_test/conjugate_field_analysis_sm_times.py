#!/usr/bin/env python
"""
Create field line property plots at four different times of day (0:00, 6:00, 12:00, 18:00).
Uses T96 magnetospheric model with SM coordinates.
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


def analyze_field_lines_sm(ut, parmod, x_start_sm, y_start_sm, z_start_sm, 
                          sm_lat_start, sm_lon_start, electron_energy_keV=100.0):
    """Trace field lines using T96 magnetospheric model in SM coordinates."""
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
    conjugate_mask = np.zeros(nlines, dtype=bool)
    
    # Create T96 SM wrapper function
    def t96_sm_wrapper(parmod, ps, x_sm, y_sm, z_sm):
        x_gsm, y_gsm, z_gsm = smgsm_vectorized(x_sm, y_sm, z_sm, 1)
        bx_gsm, by_gsm, bz_gsm = t96_vectorized(parmod, ps, x_gsm, y_gsm, z_gsm)
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
            
            # Convert SM coordinates to GSM for T96 field calculation
            x_line_gsm, y_line_gsm, z_line_gsm = smgsm_vectorized(x_line, y_line, z_line, 1)
            
            # Calculate B field using T96 model
            bx_gsm, by_gsm, bz_gsm = t96_vectorized(parmod, ps, x_line_gsm, y_line_gsm, z_line_gsm)
            
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
    
    return {
        'min_b': min_b,
        'min_b_dist': min_b_dist,
        'min_rc_rl': min_rc_rl,
        'min_rc_rl_dist': min_rc_rl_dist,
        'conjugate_mask': conjugate_mask,
        'sm_lat': sm_lat_start,
        'sm_lon': sm_lon_start
    }


def create_time_comparison_plots(all_results, times, electron_energy_keV, figsize=(20, 20)):
    """Create 4x4 subplot comparing field line properties at different times."""
    fig, axes = plt.subplots(4, 4, figsize=figsize)
    
    # Common plot settings
    def setup_axis(ax, title):
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12, pad=10)
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
    
    # Create time labels
    time_labels = ['00:00', '06:00', '12:00', '18:00']
    
    # Plot each time in a column
    for col, (time_label, results) in enumerate(zip(time_labels, all_results)):
        # Extract data
        sm_lat = results['sm_lat']
        sm_lon = results['sm_lon']
        min_b = results['min_b']
        min_b_dist = results['min_b_dist']
        min_rc_rl = results['min_rc_rl']
        min_rc_rl_dist = results['min_rc_rl_dist']
        conjugate_mask = results['conjugate_mask']
        
        # Convert spherical to Cartesian for plotting
        sm_lon_rad = sm_lon * np.pi / 180
        sm_lat_rad = sm_lat * np.pi / 180
        x_plot = np.cos(sm_lat_rad) * np.cos(sm_lon_rad)
        y_plot = np.cos(sm_lat_rad) * np.sin(sm_lon_rad)
        
        # Row 1: Minimum B-field
        ax = axes[0, col]
        setup_axis(ax, f'Min B-field (nT)\n{time_label}')
        
        # Non-conjugate points
        non_conj = ~conjugate_mask
        if np.any(non_conj):
            ax.scatter(x_plot[non_conj], y_plot[non_conj], 
                      c='gray', s=20, alpha=0.3)
        
        # Conjugate points
        conj = conjugate_mask & ~np.isnan(min_b)
        if np.any(conj):
            sc = ax.scatter(x_plot[conj], y_plot[conj], 
                           c=min_b[conj], s=30,
                           cmap='viridis',
                           norm=colors.LogNorm(vmin=np.nanmin(min_b[conj]), 
                                             vmax=np.nanmax(min_b[conj])))
            if col == 3:  # Add colorbar to last column
                cbar = plt.colorbar(sc, ax=ax, pad=0.1)
                cbar.set_label('Min B (nT)', fontsize=10)
        
        # Row 2: Distance at minimum B
        ax = axes[1, col]
        setup_axis(ax, f'Distance at Min B (Re)\n{time_label}')
        
        if np.any(non_conj):
            ax.scatter(x_plot[non_conj], y_plot[non_conj], 
                      c='gray', s=20, alpha=0.3)
        
        conj = conjugate_mask & ~np.isnan(min_b_dist)
        if np.any(conj):
            sc = ax.scatter(x_plot[conj], y_plot[conj], 
                           c=min_b_dist[conj], s=30,
                           cmap='plasma',
                           vmin=1.0, vmax=min(20.0, np.nanmax(min_b_dist[conj])))
            if col == 3:
                cbar = plt.colorbar(sc, ax=ax, pad=0.1)
                cbar.set_label('Distance (Re)', fontsize=10)
        
        # Row 3: Minimum Rc/RL
        ax = axes[2, col]
        setup_axis(ax, f'Min Rc/RL ({electron_energy_keV} keV)\n{time_label}')
        
        if np.any(non_conj):
            ax.scatter(x_plot[non_conj], y_plot[non_conj], 
                      c='gray', s=20, alpha=0.3)
        
        conj = conjugate_mask & ~np.isnan(min_rc_rl) & (min_rc_rl > 0)
        if np.any(conj):
            vmin, vmax = 1.0, 64.0
            min_rc_rl_clipped = np.clip(min_rc_rl[conj], vmin, vmax)
            
            sc = ax.scatter(x_plot[conj], y_plot[conj], 
                           c=min_rc_rl_clipped, s=30,
                           cmap='RdBu_r',
                           norm=colors.LogNorm(vmin=vmin, vmax=vmax))
            if col == 3:
                cbar = plt.colorbar(sc, ax=ax, pad=0.1)
                cbar.set_label(r'Min $R_c/R_L$', fontsize=10)
                cbar.ax.axhline(y=8, color='black', linestyle='--', linewidth=1)
        
        # Row 4: Distance at minimum Rc/RL
        ax = axes[3, col]
        setup_axis(ax, f'Distance at Min Rc/RL (Re)\n{time_label}')
        
        if np.any(non_conj):
            ax.scatter(x_plot[non_conj], y_plot[non_conj], 
                      c='gray', s=20, alpha=0.3)
        
        conj = conjugate_mask & ~np.isnan(min_rc_rl_dist)
        if np.any(conj):
            sc = ax.scatter(x_plot[conj], y_plot[conj], 
                           c=min_rc_rl_dist[conj], s=30,
                           cmap='plasma',
                           vmin=1.0, vmax=min(20.0, np.nanmax(min_rc_rl_dist[conj])))
            if col == 3:
                cbar = plt.colorbar(sc, ax=ax, pad=0.1)
                cbar.set_label('Distance (Re)', fontsize=10)
    
    # Overall title
    fig.suptitle('T96 Magnetospheric Field Analysis at Different Times\n' + 
                 'SM Coordinates, Spring Equinox 2024, Moderate Storm Conditions', 
                 fontsize=16, y=0.99)
    
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
    
    # Set T96 model parameters (moderate storm conditions)
    parmod = np.array([3.0, -30.0, 0.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Set electron energy
    electron_energy_keV = 100.0
    
    # Create starting grid
    print("Creating grid directly in SM coordinates...")
    x_start_sm, y_start_sm, z_start_sm, sm_lat_start, sm_lon_start = create_sm_grid(
        radius=1.0,
        nlat=16,   # Original density
        nlon=72    # Original density
    )
    
    print(f"Grid points: {len(x_start_sm)}")
    print(f"T96 Parameters: Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT, BzIMF={parmod[3]} nT\n")
    
    # Analyze field lines for each time
    all_results = []
    for i, time in enumerate(times):
        print(f"Processing time {i+1}/4: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        ut = time.timestamp()
        ps = geopack.recalc(ut)
        print(f"  Dipole tilt angle: {np.degrees(ps):.1f}°")
        
        results = analyze_field_lines_sm(
            ut, parmod, x_start_sm, y_start_sm, z_start_sm, 
            sm_lat_start, sm_lon_start, electron_energy_keV
        )
        
        conjugate_mask = results['conjugate_mask']
        print(f"  Conjugate field lines: {np.sum(conjugate_mask)}/{len(x_start_sm)} ({100*np.sum(conjugate_mask)/len(x_start_sm):.1f}%)\n")
        
        all_results.append(results)
    
    # Create comparison plots
    print("Creating time comparison plots...")
    fig, axes = create_time_comparison_plots(all_results, times, electron_energy_keV)
    
    # Save figure
    output_file = os.path.join(os.path.dirname(__file__), 'conjugate_field_analysis_sm_times.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved {output_file}")
    plt.show()


if __name__ == '__main__':
    main()