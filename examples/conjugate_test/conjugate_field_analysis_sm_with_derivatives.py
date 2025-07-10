#!/usr/bin/env python
"""
Create conjugate field analysis plots with eight directional derivative graphs below.
This script creates a combined visualization showing both the basic analysis (top)
and directional derivatives at minimum Rc/RL location (bottom).
Points where Rc/RL < 8 and |SM Latitude Difference| < 5 are highlighted as squares.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from datetime import datetime, timezone
import sys
import os

# Add parent directory to path to import geopack
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the analyze function that returns derivatives from conjugate_field_analysis_sm.py
from conjugate_field_analysis_sm import (
    create_sm_grid, analyze_field_lines_sm
)
import geopack


def plot_combined_with_derivatives(results, ut, electron_energy_keV,
                                  output_filename='conjugate_field_analysis_combined.png',
                                  tilt_angle=None, parmod=None):
    """
    Create a combined plot with the basic analysis (2x5) on top and 
    directional derivatives (2x4) below.
    """
    # Extract data from results dictionary
    min_b = results['min_b']
    min_b_dist = results['min_b_dist']
    min_b_lat = results['min_b_lat']
    min_b_lon = results['min_b_lon']
    min_rc_rl = results['min_rc_rl']
    min_rc_rl_dist = results['min_rc_rl_dist']
    min_rc_rl_lat = results['min_rc_rl_lat']
    min_rc_rl_lon = results['min_rc_rl_lon']
    conjugate_mask = results['conjugate_mask']
    sm_lat = results['sm_lat']
    sm_lon = results['sm_lon']
    derivatives_at_min_rc_rl = results['derivatives_at_min_rc_rl']
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(25, 20))
    
    # Build title
    title_lines = ['Conjugate Field Line Analysis with Directional Derivatives - SM Coordinates (T96 Model)']
    
    if tilt_angle is not None:
        title_lines.append(f'Dipole Tilt: {tilt_angle:.1f}°')
    
    if parmod is not None:
        storm_info = (f'Storm Conditions: Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT, '
                     f'ByIMF={parmod[2]} nT, BzIMF={parmod[3]} nT')
        title_lines.append(storm_info)
    
    title_lines.append(f'Time: {datetime.fromtimestamp(ut, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")} UTC, '
                      f'Electron Energy: {electron_energy_keV} keV')
    
    fig.suptitle('\n'.join(title_lines), fontsize=16, y=0.995)
    
    # Create GridSpec for custom layout
    gs = gridspec.GridSpec(4, 5, figure=fig, hspace=0.3, wspace=0.3)
    
    # Top 2 rows: Basic analysis (2x5)
    basic_axes = []
    for row in range(2):
        for col in range(5):
            ax = fig.add_subplot(gs[row, col])
            basic_axes.append(ax)
    basic_axes = np.array(basic_axes).reshape(2, 5)
    
    # Bottom 2 rows: Directional derivatives (2x4, centered)
    deriv_axes = []
    for row in range(2):
        for col in range(4):
            # Center the 4 columns by using columns 0.5-3.5
            ax = fig.add_subplot(gs[row+2, col:col+1])
            deriv_axes.append(ax)
    deriv_axes = np.array(deriv_axes).reshape(2, 4)
    
    # Common setup function
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
            # Add latitude labels
            angle = np.pi/4  # 45 degrees
            x_label = r_lat * np.cos(angle)
            y_label = r_lat * np.sin(angle)
            ax.text(x_label, y_label, f'{lat}°', fontsize=7, ha='center', va='center',
                    color='gray', alpha=0.8, bbox=dict(boxstyle='round,pad=0.2', 
                                                       facecolor='white', edgecolor='none', alpha=0.7))
    
    # Convert spherical to Cartesian for plotting
    sm_lon_rad = sm_lon * np.pi / 180
    sm_lat_rad = sm_lat * np.pi / 180
    x_plot = np.cos(sm_lat_rad) * np.cos(sm_lon_rad)
    y_plot = np.cos(sm_lat_rad) * np.sin(sm_lon_rad)
    
    # === PART 1: BASIC ANALYSIS (Top 2 rows) ===
    
    # Row 1: B-field analysis
    # Plot 1: Minimum B-field
    ax1 = basic_axes[0, 0]
    setup_axis(ax1, 'Minimum B-field Strength (nT) - T96 Model')
    
    non_conj = ~conjugate_mask
    if np.any(non_conj):
        ax1.scatter(x_plot[non_conj], y_plot[non_conj], 
                   c='gray', s=6, alpha=0.3, label='Open')
    
    conj = conjugate_mask & ~np.isnan(min_b)
    if np.any(conj):
        sc1 = ax1.scatter(x_plot[conj], y_plot[conj], 
                         c=min_b[conj], s=8,
                         cmap='viridis',
                         norm=matplotlib.colors.LogNorm(vmin=np.nanmin(min_b[conj]), 
                                           vmax=np.nanmax(min_b[conj])))
        cbar1 = plt.colorbar(sc1, ax=ax1, pad=0.1)
        cbar1.set_label('Min B (nT)', fontsize=10)
    
    # Plot 2: Distance at minimum B
    ax2 = basic_axes[0, 1]
    setup_axis(ax2, 'Distance at Minimum B-field (Re) - T96 Model')
    
    if np.any(non_conj):
        ax2.scatter(x_plot[non_conj], y_plot[non_conj], 
                   c='gray', s=6, alpha=0.3)
    
    conj = conjugate_mask & ~np.isnan(min_b_dist)
    if np.any(conj):
        sc2 = ax2.scatter(x_plot[conj], y_plot[conj], 
                         c=min_b_dist[conj], s=8,
                         cmap='plasma',
                         vmin=1.0, vmax=min(20.0, np.nanmax(min_b_dist[conj])))
        cbar2 = plt.colorbar(sc2, ax=ax2, pad=0.1)
        cbar2.set_label('Distance (Re)', fontsize=10)
    
    # Plot 3: SM Latitude at minimum B
    ax3 = basic_axes[0, 2]
    setup_axis(ax3, 'SM Latitude at Minimum B-field (°) - T96 Model')
    
    if np.any(non_conj):
        ax3.scatter(x_plot[non_conj], y_plot[non_conj], 
                   c='gray', s=6, alpha=0.3)
    
    conj = conjugate_mask & ~np.isnan(min_b_lat)
    if np.any(conj):
        sc3 = ax3.scatter(x_plot[conj], y_plot[conj], 
                         c=min_b_lat[conj], s=8,
                         cmap='coolwarm',
                         vmin=-90, vmax=90)
        cbar3 = plt.colorbar(sc3, ax=ax3, pad=0.1)
        cbar3.set_label('SM Latitude (°)', fontsize=10)
    
    # Plot 4: SM longitude at minimum B
    ax4 = basic_axes[0, 3]
    setup_axis(ax4, 'SM Longitude at Minimum B-field - T96 Model')
    
    if np.any(non_conj):
        ax4.scatter(x_plot[non_conj], y_plot[non_conj], 
                   c='gray', s=6, alpha=0.3)
    
    conj = conjugate_mask & ~np.isnan(min_b_lon)
    if np.any(conj):
        sc4 = ax4.scatter(x_plot[conj], y_plot[conj], 
                         c=min_b_lon[conj], s=8,
                         cmap='hsv',
                         vmin=0, vmax=360)
        cbar4 = plt.colorbar(sc4, ax=ax4, pad=0.1)
        cbar4.set_label('SM Longitude (°)', fontsize=10)
        cbar4.set_ticks([0, 90, 180, 270, 360])
        cbar4.set_ticklabels(['0°', '90°', '180°', '270°', '360°'])
    
    # Row 2: Rc/RL analysis
    # Plot 5: Minimum Rc/RL
    ax5 = basic_axes[1, 0]
    setup_axis(ax5, f'Minimum Rc/RL Ratio ({electron_energy_keV} keV) - T96 Model')
    
    if np.any(non_conj):
        ax5.scatter(x_plot[non_conj], y_plot[non_conj], 
                   c='gray', s=6, alpha=0.3)
    
    conj = conjugate_mask & ~np.isnan(min_rc_rl) & (min_rc_rl > 0)
    if np.any(conj):
        vmin, vmax = 1.0, 64.0
        min_rc_rl_clipped = np.clip(min_rc_rl[conj], vmin, vmax)
        
        sc5 = ax5.scatter(x_plot[conj], y_plot[conj], 
                         c=min_rc_rl_clipped, s=8,
                         cmap='RdBu_r',
                         norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
        cbar5 = plt.colorbar(sc5, ax=ax5, pad=0.1)
        cbar5.set_label(r'Min $R_c/R_L$', fontsize=10)
        cbar5.ax.axhline(y=8, color='black', linestyle='--', linewidth=1)
    
    # Plot 6: Distance at minimum Rc/RL
    ax6 = basic_axes[1, 1]
    setup_axis(ax6, 'Distance at Minimum Rc/RL (Re) - T96 Model')
    
    rc_rl_below_8 = (min_rc_rl < 8) & ~np.isnan(min_rc_rl)
    
    if np.any(non_conj):
        ax6.scatter(x_plot[non_conj], y_plot[non_conj], 
                   c='gray', s=6, alpha=0.3)
    
    conj = conjugate_mask & ~np.isnan(min_rc_rl_dist)
    if np.any(conj):
        # Regular points (Rc/RL >= 8)
        regular_mask = conj & ~rc_rl_below_8
        if np.any(regular_mask):
            sc6a = ax6.scatter(x_plot[regular_mask], y_plot[regular_mask], 
                             c=min_rc_rl_dist[regular_mask], s=8,
                             cmap='plasma', marker='o',
                             vmin=1.0, vmax=min(20.0, np.nanmax(min_rc_rl_dist[conj])))
        
        # Highlight points (Rc/RL < 8)
        highlight_mask = conj & rc_rl_below_8
        if np.any(highlight_mask):
            sc6b = ax6.scatter(x_plot[highlight_mask], y_plot[highlight_mask], 
                             c=min_rc_rl_dist[highlight_mask], s=8,
                             cmap='plasma', marker='s',
                             vmin=1.0, vmax=min(20.0, np.nanmax(min_rc_rl_dist[conj])),
                             edgecolors='black', linewidth=0.2)
        
        if np.any(regular_mask):
            cbar6 = plt.colorbar(sc6a, ax=ax6, pad=0.1)
        elif np.any(highlight_mask):
            cbar6 = plt.colorbar(sc6b, ax=ax6, pad=0.1)
        cbar6.set_label('Distance (Re)', fontsize=10)
    
    # Plot 7: SM Latitude at minimum Rc/RL
    ax7 = basic_axes[1, 2]
    setup_axis(ax7, f'SM Latitude at Minimum Rc/RL (°) - T96 Model')
    
    if np.any(non_conj):
        ax7.scatter(x_plot[non_conj], y_plot[non_conj], 
                   c='gray', s=6, alpha=0.3)
    
    conj = conjugate_mask & ~np.isnan(min_rc_rl_lat)
    if np.any(conj):
        # Regular points (Rc/RL >= 8)
        regular_mask = conj & ~rc_rl_below_8
        if np.any(regular_mask):
            sc7a = ax7.scatter(x_plot[regular_mask], y_plot[regular_mask], 
                             c=min_rc_rl_lat[regular_mask], s=8,
                             cmap='coolwarm', marker='o',
                             vmin=-90, vmax=90)
        
        # Highlight points (Rc/RL < 8)
        highlight_mask = conj & rc_rl_below_8
        if np.any(highlight_mask):
            sc7b = ax7.scatter(x_plot[highlight_mask], y_plot[highlight_mask], 
                             c=min_rc_rl_lat[highlight_mask], s=8,
                             cmap='coolwarm', marker='s',
                             vmin=-90, vmax=90,
                             edgecolors='black', linewidth=0.2)
        
        if np.any(regular_mask):
            cbar7 = plt.colorbar(sc7a, ax=ax7, pad=0.1)
        elif np.any(highlight_mask):
            cbar7 = plt.colorbar(sc7b, ax=ax7, pad=0.1)
        cbar7.set_label('SM Latitude (°)', fontsize=10)
    
    # Plot 8: SM longitude at minimum Rc/RL
    ax8 = basic_axes[1, 3]
    setup_axis(ax8, f'SM Longitude at Minimum Rc/RL - T96 Model')
    
    if np.any(non_conj):
        ax8.scatter(x_plot[non_conj], y_plot[non_conj], 
                   c='gray', s=6, alpha=0.3)
    
    conj = conjugate_mask & ~np.isnan(min_rc_rl_lon)
    if np.any(conj):
        # Regular points (Rc/RL >= 8)
        regular_mask = conj & ~rc_rl_below_8
        if np.any(regular_mask):
            sc8a = ax8.scatter(x_plot[regular_mask], y_plot[regular_mask], 
                             c=min_rc_rl_lon[regular_mask], s=8,
                             cmap='hsv', marker='o',
                             vmin=0, vmax=360)
        
        # Highlight points (Rc/RL < 8)
        highlight_mask = conj & rc_rl_below_8
        if np.any(highlight_mask):
            sc8b = ax8.scatter(x_plot[highlight_mask], y_plot[highlight_mask], 
                             c=min_rc_rl_lon[highlight_mask], s=8,
                             cmap='hsv', marker='s',
                             vmin=0, vmax=360,
                             edgecolors='black', linewidth=0.2)
        
        if np.any(regular_mask):
            cbar8 = plt.colorbar(sc8a, ax=ax8, pad=0.1)
        elif np.any(highlight_mask):
            cbar8 = plt.colorbar(sc8b, ax=ax8, pad=0.1)
        cbar8.set_label('SM Longitude (°)', fontsize=10)
        cbar8.set_ticks([0, 90, 180, 270, 360])
        cbar8.set_ticklabels(['0°', '90°', '180°', '270°', '360°'])
    
    # Column 5: Basic difference plots
    # Plot 9: Distance difference
    ax9 = basic_axes[0, 4]
    setup_axis(ax9, 'Distance Difference: Min B - Min Rc/RL (Re)')
    
    dist_diff = min_b_dist - min_rc_rl_dist
    rc_rl_below_8 = (min_rc_rl < 8) & ~np.isnan(min_rc_rl)
    
    if np.any(non_conj):
        ax9.scatter(x_plot[non_conj], y_plot[non_conj], 
                   c='gray', s=6, alpha=0.3)
    
    conj = conjugate_mask & ~np.isnan(dist_diff)
    if np.any(conj):
        max_abs_diff = np.nanpercentile(np.abs(dist_diff[conj]), 95)
        
        regular_mask = conj & ~rc_rl_below_8
        if np.any(regular_mask):
            sc9a = ax9.scatter(x_plot[regular_mask], y_plot[regular_mask], 
                             c=dist_diff[regular_mask], s=6.5,
                             cmap='RdBu_r', marker='o',
                             vmin=-max_abs_diff, vmax=max_abs_diff)
        
        highlight_mask = conj & rc_rl_below_8
        if np.any(highlight_mask):
            sc9b = ax9.scatter(x_plot[highlight_mask], y_plot[highlight_mask], 
                             c=dist_diff[highlight_mask], s=8,
                             cmap='RdBu_r', marker='s',
                             vmin=-max_abs_diff, vmax=max_abs_diff,
                             edgecolors='black', linewidth=0.2)
        
        if np.any(regular_mask):
            cbar9 = plt.colorbar(sc9a, ax=ax9, pad=0.1)
        elif np.any(highlight_mask):
            cbar9 = plt.colorbar(sc9b, ax=ax9, pad=0.1)
        cbar9.set_label('Distance Difference (Re)', fontsize=10)
        
        legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor='gray', markersize=5.6, label='Rc/RL ≥ 8'),
                         Line2D([0], [0], marker='s', color='w', 
                                markerfacecolor='gray', markersize=7, 
                                markeredgecolor='black', markeredgewidth=1.5, label='Rc/RL < 8')]
        ax9.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    # Plot 10: Latitude difference
    ax10 = basic_axes[1, 4]
    setup_axis(ax10, 'SM Latitude Difference: Min B - Min Rc/RL (°)')
    
    lat_diff = min_b_lat - min_rc_rl_lat
    
    if np.any(non_conj):
        ax10.scatter(x_plot[non_conj], y_plot[non_conj], 
                    c='gray', s=6, alpha=0.3)
    
    conj = conjugate_mask & ~np.isnan(lat_diff)
    if np.any(conj):
        vmin_lat, vmax_lat = -15, 15
        
        regular_mask = conj & ~rc_rl_below_8
        if np.any(regular_mask):
            sc10a = ax10.scatter(x_plot[regular_mask], y_plot[regular_mask], 
                               c=lat_diff[regular_mask], s=6.5,
                               cmap='RdBu_r', marker='o',
                               vmin=vmin_lat, vmax=vmax_lat)
        
        highlight_mask = conj & rc_rl_below_8
        if np.any(highlight_mask):
            sc10b = ax10.scatter(x_plot[highlight_mask], y_plot[highlight_mask], 
                               c=lat_diff[highlight_mask], s=8,
                               cmap='RdBu_r', marker='s',
                               vmin=vmin_lat, vmax=vmax_lat,
                               edgecolors='black', linewidth=0.2)
        
        if np.any(regular_mask):
            cbar10 = plt.colorbar(sc10a, ax=ax10, pad=0.1)
        elif np.any(highlight_mask):
            cbar10 = plt.colorbar(sc10b, ax=ax10, pad=0.1)
        cbar10.set_label('Latitude Difference (°)', fontsize=10)
        
        legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor='gray', markersize=5.6, label='Rc/RL ≥ 8'),
                         Line2D([0], [0], marker='s', color='w', 
                                markerfacecolor='gray', markersize=7, 
                                markeredgecolor='black', markeredgewidth=1.5, label='Rc/RL < 8')]
        ax10.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    # === PART 2: DIRECTIONAL DERIVATIVES (Bottom 2 rows) ===
    
    # Add subtitle for differential analysis section
    fig.text(0.5, 0.48, 'Directional Derivatives at Minimum Rc/RL Location', 
             ha='center', fontsize=14, weight='bold')
    
    # Special mask for highlighting
    special_mask = (min_rc_rl < 8) & (np.abs(lat_diff) < 5) & ~np.isnan(min_rc_rl) & ~np.isnan(lat_diff)
    
    # Helper function for derivative plots
    def plot_derivative_with_special_markers(ax, deriv_data, cmap, label, symmetric=True):
        """Plot derivative data with special square markers for Rc/RL < 8 and |lat_diff| < 5."""
        non_conj = ~conjugate_mask
        if np.any(non_conj):
            ax.scatter(x_plot[non_conj], y_plot[non_conj], 
                      c='gray', s=6, alpha=0.3)
        
        conj = conjugate_mask & ~np.isnan(deriv_data)
        if np.any(conj):
            if symmetric:
                max_abs_val = np.nanpercentile(np.abs(deriv_data[conj]), 95)
                vmin, vmax = -max_abs_val, max_abs_val
            else:
                vmin = 0
                vmax = np.nanpercentile(deriv_data[conj], 95)
            
            regular_mask = conj & ~special_mask
            if np.any(regular_mask):
                sc1 = ax.scatter(x_plot[regular_mask], y_plot[regular_mask], 
                                c=deriv_data[regular_mask], s=8, marker='o',
                                cmap=cmap, vmin=vmin, vmax=vmax)
            
            special_conj = conj & special_mask
            if np.any(special_conj):
                sc2 = ax.scatter(x_plot[special_conj], y_plot[special_conj], 
                                c=deriv_data[special_conj], s=16, marker='s',
                                cmap=cmap, vmin=vmin, vmax=vmax,
                                edgecolors='black', linewidth=0.2)
            
            if np.any(regular_mask):
                cbar = plt.colorbar(sc1, ax=ax, pad=0.1)
            elif np.any(special_conj):
                cbar = plt.colorbar(sc2, ax=ax, pad=0.1)
            cbar.set_label(label, fontsize=9)
    
    # Row 3: First set of directional derivatives
    # Plot 1: κ = (∂T/∂T)·n (curvature)
    ax_d1 = deriv_axes[0, 0]
    setup_axis(ax_d1, 'Curvature κ = (∂T/∂T)·n at Min Rc/RL')
    plot_derivative_with_special_markers(ax_d1, derivatives_at_min_rc_rl['dT_dT_n'], 
                                        'viridis', 'κ = (∂T/∂T)·n', symmetric=False)
    
    # Plot 2: τ = (∂n/∂T)·b (torsion)
    ax_d2 = deriv_axes[0, 1]
    setup_axis(ax_d2, 'Torsion τ = (∂n/∂T)·b at Min Rc/RL')
    plot_derivative_with_special_markers(ax_d2, derivatives_at_min_rc_rl['dn_dT_b'], 
                                        'RdBu_r', 'τ = (∂n/∂T)·b', symmetric=True)
    
    # Plot 3: (∂T/∂n)·b
    ax_d3 = deriv_axes[0, 2]
    setup_axis(ax_d3, '(∂T/∂n)·b at Min Rc/RL')
    plot_derivative_with_special_markers(ax_d3, derivatives_at_min_rc_rl['dT_dn_b'], 
                                        'coolwarm', '(∂T/∂n)·b', symmetric=True)
    
    # Plot 4: (∂n/∂n)·b
    ax_d4 = deriv_axes[0, 3]
    setup_axis(ax_d4, '(∂n/∂n)·b at Min Rc/RL')
    plot_derivative_with_special_markers(ax_d4, derivatives_at_min_rc_rl['dn_dn_b'], 
                                        'PuOr_r', '(∂n/∂n)·b', symmetric=True)
    
    # Row 4: Second set of directional derivatives
    # Plot 5: (∂n/∂b)·T
    ax_d5 = deriv_axes[1, 0]
    setup_axis(ax_d5, '(∂n/∂b)·T at Min Rc/RL')
    plot_derivative_with_special_markers(ax_d5, derivatives_at_min_rc_rl['dn_db_T'], 
                                        'seismic', '(∂n/∂b)·T', symmetric=True)
    
    # Plot 6: (∂b/∂b)·T
    ax_d6 = deriv_axes[1, 1]
    setup_axis(ax_d6, '(∂b/∂b)·T at Min Rc/RL')
    plot_derivative_with_special_markers(ax_d6, derivatives_at_min_rc_rl['db_db_T'], 
                                        'BrBG_r', '(∂b/∂b)·T', symmetric=True)
    
    # Plot 7: (∂n/∂b)·b
    ax_d7 = deriv_axes[1, 2]
    setup_axis(ax_d7, '(∂n/∂b)·b at Min Rc/RL')
    plot_derivative_with_special_markers(ax_d7, derivatives_at_min_rc_rl['dn_db_b'], 
                                        'PRGn_r', '(∂n/∂b)·b', symmetric=True)
    
    # Plot 8: (∂T/∂n)·n
    ax_d8 = deriv_axes[1, 3]
    setup_axis(ax_d8, '(∂T/∂n)·n at Min Rc/RL')
    plot_derivative_with_special_markers(ax_d8, derivatives_at_min_rc_rl['dT_dn_n'], 
                                        'twilight_shifted', '(∂T/∂n)·n', symmetric=True)
    
    # Add legend to first derivative plot
    if ax_d1 == deriv_axes[0, 0]:
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                  markersize=5.6, label='Regular'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                  markersize=8, markeredgecolor='black', markeredgewidth=1.5, 
                  label='Rc/RL<8 & |Δlat|<5°')
        ]
        ax_d1.legend(handles=legend_elements, loc='lower right', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Combined analysis figure saved as {output_filename}")
    plt.close()


def run_single_analysis(ut, parmod, scenario_name, electron_energy_keV=100.0):
    """Run analysis for a single scenario and create combined plot."""
    # Update geopack parameters
    ps = geopack.recalc(ut)
    
    print(f"\n{'='*60}")
    print(f"Running scenario: {scenario_name}")
    print(f"  Dipole tilt angle: {np.degrees(ps):.1f}°")
    print(f"  T96 Parameters: Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT, "
          f"ByIMF={parmod[2]} nT, BzIMF={parmod[3]} nT")
    
    # Create SM coordinate grid
    x_sm, y_sm, z_sm, sm_lat, sm_lon = create_sm_grid(radius=1.0, nlat=32, nlon=48)
    
    # Analyze field lines (this now returns derivatives)
    results = analyze_field_lines_sm(ut, parmod, x_sm, y_sm, z_sm, 
                                    sm_lat, sm_lon, electron_energy_keV)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'conjugate_field_analysis_sm_diff')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create combined plot with directional derivatives
    output_filename = os.path.join(output_dir, f'{scenario_name}_with_derivatives.png')
    plot_combined_with_derivatives(results, ut, electron_energy_keV,
                                  output_filename=output_filename,
                                  tilt_angle=np.degrees(ps), parmod=parmod)
    
    # Print summary statistics
    print(f"\nSummary for {scenario_name}:")
    conjugate_mask = results['conjugate_mask']
    if np.any(conjugate_mask):
        conj_idx = conjugate_mask
        print(f"  Conjugate field lines: {np.sum(conj_idx)}/{len(conjugate_mask)} ({100*np.sum(conj_idx)/len(conjugate_mask):.1f}%)")
        
        min_rc_rl = results['min_rc_rl']
        if np.any(~np.isnan(min_rc_rl[conj_idx])):
            threshold = 8.0
            below_threshold = np.sum(min_rc_rl[conj_idx] < threshold)
            print(f"  Field lines with Rc/RL < {threshold}: {below_threshold} "
                  f"({100*below_threshold/np.sum(conj_idx):.1f}% of conjugate)")
            
            # Report special points
            lat_diff = results['min_b_lat'] - results['min_rc_rl_lat']
            special_count = np.sum((min_rc_rl[conj_idx] < 8) & (np.abs(lat_diff[conj_idx]) < 5))
            print(f"  Special points (Rc/RL<8 & |Δlat|<5°): {special_count} "
                  f"({100*special_count/np.sum(conj_idx):.1f}% of conjugate)")
            
            # Report on derivatives
            derivatives_at_min_rc_rl = results['derivatives_at_min_rc_rl']
            n_valid_deriv = np.sum(~np.isnan(derivatives_at_min_rc_rl['dT_dT_n']))
            print(f"  Points with calculated derivatives: {n_valid_deriv}")


def main():
    """Main function to run analyses with directional derivative plots."""
    
    # Electron energy in keV
    electron_energy_keV = 300.0
    
    # Define test scenarios
    tilt_scenarios = [
        (datetime(2024, 3, 20, 12, 0, 0), "spring_equinox"),
        (datetime(2024, 6, 21, 12, 0, 0), "summer_solstice"),
        (datetime(2024, 12, 21, 12, 0, 0), "winter_solstice"),
    ]
    
    storm_conditions = [
        ([2.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0], "quiet"),
        ([3.0, -20.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0], "moderate"),
        ([5.0, -50.0, 5.0, -10.0, 0, 0, 0, 0, 0, 0], "storm"),
    ]
    
    print("="*80)
    print("Running conjugate field analysis with directional derivative plots")
    print(f"Electron energy: {electron_energy_keV} keV")
    print(f"Creating combined plots with basic analysis (top) and")
    print(f"eight directional derivatives at minimum Rc/RL (bottom)")
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
                import traceback
                traceback.print_exc()
                continue
    
    print("\n" + "="*80)
    print("All scenarios completed!")
    print(f"Images saved in: examples/conjugate_test/conjugate_field_analysis_sm_diff/")
    print("Files created: *_with_derivatives.png")
    print("="*80)


if __name__ == "__main__":
    main()