#!/usr/bin/env python
"""
Create four heatmaps showing field line properties using SM (Solar Magnetic) coordinates.
Uses T96 magnetospheric model with inverted polar plots (90° at center).
Grid is generated directly in SM coordinates without geographic conversion.
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
from geopack.igrf_vectorized import igrf_gsm_vectorized
from geopack.vectorized import t96_vectorized
from geopack.vectorized.field_line_geometry_vectorized import field_line_curvature_vectorized
from geopack.coordinates_vectorized import smgsm_vectorized, geomag_vectorized, magsm_vectorized


def create_sm_grid(radius=1.0, nlat=12, nlon=16):
    """
    Create a grid of starting points directly in SM coordinates.
    
    In SM coordinates:
    - Z_SM axis: aligned with magnetic dipole axis
    - Y_SM axis: perpendicular to both dipole axis and Sun-Earth line
    - X_SM axis: completes right-handed system
    """
    # Create latitude grid in SM coordinates (0° = SM equator, 90° = north magnetic pole)
    sm_lat = np.linspace(40, 80, nlat)
    
    # Create longitude grid in SM coordinates (0° = noon meridian)
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


def calculate_electron_larmor_radius(B_magnitude, electron_energy_keV=100.0):
    """Calculate the electron Larmor radius."""
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
    
    # Convert B to Tesla
    B_tesla = B_magnitude * 1e-9
    
    # Calculate Larmor radius: RL = p / (eB)
    RL_m = p / (e * B_tesla)
    
    # Convert to km
    RL_km = RL_m / 1000.0
    
    return RL_km


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
    min_rc_rl = np.full(nlines, np.nan)
    min_rc_rl_dist = np.full(nlines, np.nan)
    conjugate_mask = np.zeros(nlines, dtype=bool)
    
    print("Analyzing field lines...")
    
    # Check if conjugate
    for i in range(nlines):
        r_final = np.sqrt(xf[i]**2 + yf[i]**2 + zf[i]**2)
        if status[i] == 0 and zf[i] < 0 and abs(r_final - 1.0) < 0.1:
            conjugate_mask[i] = True
    
    # Process conjugate field lines
    for i in range(nlines):
        if i % 50 == 0:
            print(f"  Processing field line {i}/{nlines}...")
            
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
            
            # Calculate curvature and Rc/RL
            # Need to create a wrapper function that handles SM coordinates
            def t96_sm_wrapper(parmod, ps, x_sm, y_sm, z_sm):
                x_gsm, y_gsm, z_gsm = smgsm_vectorized(x_sm, y_sm, z_sm, 1)
                bx_gsm, by_gsm, bz_gsm = t96_vectorized(parmod, ps, x_gsm, y_gsm, z_gsm)
                bx_sm, by_sm, bz_sm = smgsm_vectorized(bx_gsm, by_gsm, bz_gsm, -1)
                return bx_sm, by_sm, bz_sm
            
            kappa = field_line_curvature_vectorized(
                t96_sm_wrapper, parmod, ps, x_line, y_line, z_line
            )
            
            # Calculate Rc and RL
            Rc_km = np.zeros_like(kappa)
            valid_kappa = kappa > 0
            Rc_km[valid_kappa] = (1.0 / kappa[valid_kappa]) * 6371.2
            
            RL_km = calculate_electron_larmor_radius(b_mag, electron_energy_keV)
            
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
    
    print(f"Found {np.sum(conjugate_mask)} conjugate field lines out of {nlines} total")
    
    return {
        'min_b': min_b,
        'min_b_dist': min_b_dist,
        'min_rc_rl': min_rc_rl,
        'min_rc_rl_dist': min_rc_rl_dist,
        'conjugate_mask': conjugate_mask,
        'sm_lat': sm_lat_start,
        'sm_lon': sm_lon_start
    }


def create_sm_coord_plots(results, electron_energy_keV, figsize=(20, 18)):
    """
    Create 2x2 subplot with inverted polar plots showing SM coordinate grid.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, subplot_kw=dict(projection='polar'))
    
    # Extract data
    sm_lat = results['sm_lat']
    sm_lon = results['sm_lon']
    min_b = results['min_b']
    min_b_dist = results['min_b_dist']
    min_rc_rl = results['min_rc_rl']
    min_rc_rl_dist = results['min_rc_rl_dist']
    conjugate_mask = results['conjugate_mask']
    
    # Common plot settings for polar plots
    def setup_axis(ax, title):
        ax.set_theta_zero_location('N')  # 0° longitude at top
        ax.set_theta_direction(-1)  # Clockwise
        ax.set_ylim(0, 60)  # 90° at center (r=0) to 30° at edge (r=60)
        ax.set_title(title, fontsize=14, pad=20)
        ax.grid(True, alpha=0.3)
        
        # Set theta (longitude) labels for SM coordinates
        lon_angles = np.array([0, 90, 180, 270])  # SM longitude
        ax.set_thetagrids(lon_angles, ['0°', '90°', '180°', '270°'])
        
        # Add radial labels (inverted: high lat at center)
        ax.set_rticks([10, 20, 30, 40, 50], ['80°', '70°', '60°', '50°', '40°'])
        ax.set_rlabel_position(45)
    
    # Convert SM longitude to angle in radians for polar plot
    theta_plot = sm_lon * np.pi / 180  # Convert degrees to radians
    
    # Invert radius: 90° at center (r=0), lower latitudes outward
    r_plot = 90 - sm_lat
    
    # Plot 1: Minimum B-field
    ax1 = axes[0, 0]
    setup_axis(ax1, 'Minimum B-field Strength (nT) - T96 Model')
    
    # Non-conjugate points
    non_conj = ~conjugate_mask
    if np.any(non_conj):
        ax1.scatter(theta_plot[non_conj], r_plot[non_conj], 
                   c='gray', s=20, alpha=0.3, label='Open')
    
    # Conjugate points
    conj = conjugate_mask & ~np.isnan(min_b)
    if np.any(conj):
        sc1 = ax1.scatter(theta_plot[conj], r_plot[conj], 
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
        ax2.scatter(theta_plot[non_conj], r_plot[non_conj], 
                   c='gray', s=20, alpha=0.3)
    
    conj = conjugate_mask & ~np.isnan(min_b_dist)
    if np.any(conj):
        sc2 = ax2.scatter(theta_plot[conj], r_plot[conj], 
                         c=min_b_dist[conj], s=30,
                         cmap='plasma',
                         vmin=1.0, vmax=min(20.0, np.nanmax(min_b_dist[conj])))
        cbar2 = plt.colorbar(sc2, ax=ax2, pad=0.1)
        cbar2.set_label('Distance (Re)', fontsize=10)
    
    # Plot 3: Minimum Rc/RL
    ax3 = axes[1, 0]
    setup_axis(ax3, f'Minimum Rc/RL Ratio ({electron_energy_keV} keV) - T96 Model')
    
    if np.any(non_conj):
        ax3.scatter(theta_plot[non_conj], r_plot[non_conj], 
                   c='gray', s=20, alpha=0.3)
    
    conj = conjugate_mask & ~np.isnan(min_rc_rl) & (min_rc_rl > 0)
    if np.any(conj):
        vmin, vmax = 1.0, 64.0
        min_rc_rl_clipped = np.clip(min_rc_rl[conj], vmin, vmax)
        
        sc3 = ax3.scatter(theta_plot[conj], r_plot[conj], 
                         c=min_rc_rl_clipped, s=30,
                         cmap='RdBu_r',
                         norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        cbar3 = plt.colorbar(sc3, ax=ax3, pad=0.1)
        cbar3.set_label(r'Min $R_c/R_L$', fontsize=10)
        cbar3.ax.axhline(y=8, color='black', linestyle='--', linewidth=1)
    
    # Plot 4: Distance at minimum Rc/RL
    ax4 = axes[1, 1]
    setup_axis(ax4, 'Distance at Minimum Rc/RL (Re) - T96 Model')
    
    if np.any(non_conj):
        ax4.scatter(theta_plot[non_conj], r_plot[non_conj], 
                   c='gray', s=20, alpha=0.3)
    
    conj = conjugate_mask & ~np.isnan(min_rc_rl_dist)
    if np.any(conj):
        sc4 = ax4.scatter(theta_plot[conj], r_plot[conj], 
                         c=min_rc_rl_dist[conj], s=30,
                         cmap='plasma',
                         vmin=1.0, vmax=min(20.0, np.nanmax(min_rc_rl_dist[conj])))
        cbar4 = plt.colorbar(sc4, ax=ax4, pad=0.1)
        cbar4.set_label('Distance (Re)', fontsize=10)
    
    # Add legend to first plot
    ax1.legend(loc='upper left', fontsize=10)
    
    # Overall title
    fig.suptitle('T96 Magnetospheric Field Analysis in SM Coordinates\n' + 
                 'Starting from Northern Hemisphere at 1 Re (90° at center)', fontsize=16, y=0.95)
    
    plt.tight_layout()
    return fig, axes


def main():
    """Main function."""
    # Set time
    ut = datetime.now().timestamp()
    ps = geopack.recalc(ut)
    
    # Set T96 model parameters (moderate activity)
    # For T96: [Pdyn, Dst, ByIMF, BzIMF, unused...]
    parmod = np.array([2.0, -20.0, 0.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Set electron energy
    electron_energy_keV = 100.0
    
    # Create starting grid directly in SM coordinates
    print("Creating grid directly in SM coordinates...")
    x_start_sm, y_start_sm, z_start_sm, sm_lat_start, sm_lon_start = create_sm_grid(
        radius=1.0,
        nlat=12,  # Reasonable grid size
        nlon=16
    )
    
    # Print summary
    print(f"\nConfiguration:")
    print(f"  Grid points: {len(x_start_sm)}")
    print(f"  Coordinate system: SM (Solar Magnetic)")
    print(f"  Dipole tilt angle: {np.degrees(ps):.1f}°")
    print(f"  T96 Parameters:")
    print(f"    Pdyn = {parmod[0]} nPa")
    print(f"    Dst = {parmod[1]} nT")
    print(f"    ByIMF = {parmod[2]} nT")
    print(f"    BzIMF = {parmod[3]} nT")
    
    # Analyze field lines
    results = analyze_field_lines_sm(
        ut, parmod, x_start_sm, y_start_sm, z_start_sm, 
        sm_lat_start, sm_lon_start, electron_energy_keV
    )
    
    # Create plots
    print("\nCreating SM coordinate plots...")
    fig, axes = create_sm_coord_plots(results, electron_energy_keV)
    
    # Save figure
    plt.savefig('conjugate_field_analysis_sm.png', dpi=300, bbox_inches='tight')
    print("Saved conjugate_field_analysis_sm.png")
    plt.show()
    
    # Print summary
    conjugate_mask = results['conjugate_mask']
    print(f"\nSummary:")
    print(f"Total field lines traced: {len(x_start_sm)}")
    print(f"Conjugate field lines: {np.sum(conjugate_mask)}")
    print(f"Percentage conjugate: {100*np.sum(conjugate_mask)/len(x_start_sm):.1f}%")


if __name__ == '__main__':
    main()