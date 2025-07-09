#!/usr/bin/env python
"""
Verify SM coordinate system by displaying the geomagnetic axis on the XY plane.
In SM coordinates, the geomagnetic axis should always be along the Z-axis.
This script shows the geomagnetic dipole axis for different dates.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os

# Add parent directory to path to import geopack
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geopack
from geopack.coordinates_vectorized import smgsm_vectorized, geogsm_vectorized, magsm_vectorized


def plot_geomag_axis_in_sm(dates_info):
    """
    Plot the geomagnetic axis in SM coordinates for multiple dates.
    In SM coordinates, the geomagnetic axis should always point along +Z.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(aspect='equal'))
    axes = axes.flatten()
    
    for idx, (name, date) in enumerate(dates_info):
        ax = axes[idx]
        
        # Calculate dipole parameters
        ut = date.timestamp()
        ps = geopack.recalc(ut)
        
        # In GEO coordinates, the geomagnetic dipole axis is defined by:
        # The dipole moment direction (from geopack internals)
        # For simplicity, we'll use the fact that in MAG coordinates,
        # the dipole is along the Z-axis, then transform to SM
        
        # Create points along the magnetic dipole axis in MAG coordinates
        # (0, 0, ±1) in MAG represents the magnetic poles
        mag_north = np.array([0.0, 0.0, 1.0])   # North magnetic pole
        mag_south = np.array([0.0, 0.0, -1.0])  # South magnetic pole
        
        # Also create some reference points in MAG coordinates
        # to show the MAG coordinate frame
        mag_x = np.array([1.0, 0.0, 0.0])
        mag_y = np.array([0.0, 1.0, 0.0])
        
        # Transform from MAG to GEO coordinates
        # First, we need to use the geomagnetic pole location
        # The transformation uses the fact that MAG Z-axis points to magnetic north
        
        # For visualization, let's create a circle in the MAG XY plane
        theta = np.linspace(0, 2*np.pi, 100)
        mag_circle_x = np.cos(theta)
        mag_circle_y = np.sin(theta)
        mag_circle_z = np.zeros_like(theta)
        
        # Transform MAG to SM via GEO
        # First MAG to GEO (using the fact that we know the dipole tilt)
        # In GEO, the dipole axis has a tilt angle ps from the Z-axis
        # and is in the XZ plane (for the dates we're considering)
        
        # The dipole axis in GEO coordinates
        geo_dipole_x = np.sin(ps)
        geo_dipole_y = 0.0
        geo_dipole_z = np.cos(ps)
        
        # Now transform this to SM coordinates via GSM
        # First GEO to GSM
        gsm_dipole_x, gsm_dipole_y, gsm_dipole_z = geogsm_vectorized(
            geo_dipole_x, geo_dipole_y, geo_dipole_z, 1
        )
        
        # Then GSM to SM
        sm_dipole_x, sm_dipole_y, sm_dipole_z = smgsm_vectorized(
            gsm_dipole_x, gsm_dipole_y, gsm_dipole_z, -1
        )
        
        # Also transform the negative dipole direction
        gsm_dipole_x_neg, gsm_dipole_y_neg, gsm_dipole_z_neg = geogsm_vectorized(
            -geo_dipole_x, -geo_dipole_y, -geo_dipole_z, 1
        )
        sm_dipole_x_neg, sm_dipole_y_neg, sm_dipole_z_neg = smgsm_vectorized(
            gsm_dipole_x_neg, gsm_dipole_y_neg, gsm_dipole_z_neg, -1
        )
        
        # Create XY plane grid in SM coordinates
        x_range = np.linspace(-2, 2, 20)
        y_range = np.linspace(-2, 2, 20)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)
        
        # Plot the XY plane grid
        ax.plot(X, Y, 'k-', alpha=0.2, linewidth=0.5)
        ax.plot(X.T, Y.T, 'k-', alpha=0.2, linewidth=0.5)
        
        # Plot SM coordinate axes
        ax.arrow(0, 0, 1.5, 0, head_width=0.1, head_length=0.1, 
                fc='red', ec='red', linewidth=2, label='X_SM')
        ax.arrow(0, 0, 0, 1.5, head_width=0.1, head_length=0.1, 
                fc='green', ec='green', linewidth=2, label='Y_SM')
        
        # Plot the projection of the geomagnetic dipole axis onto XY plane
        # This should be near zero if SM is working correctly
        ax.plot([0, sm_dipole_x], [0, sm_dipole_y], 'b-', linewidth=3, 
                label=f'Dipole projection')
        ax.plot([0, sm_dipole_x_neg], [0, sm_dipole_y_neg], 'b--', linewidth=2)
        
        # Add a point at the origin
        ax.plot(0, 0, 'ko', markersize=8)
        
        # Add dipole information
        dipole_xy_mag = np.sqrt(sm_dipole_x**2 + sm_dipole_y**2)
        
        # Add circle to show unit radius
        circle = plt.Circle((0, 0), 1.0, fill=False, edgecolor='gray', 
                           linestyle='--', alpha=0.5)
        ax.add_patch(circle)
        
        # Labels and title
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_xlabel('X_SM', fontsize=12)
        ax.set_ylabel('Y_SM', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{name}\n{date.strftime("%Y-%m-%d")}\n' + 
                    f'Dipole tilt (ps) = {np.degrees(ps):.1f}°\n' +
                    f'Dipole XY magnitude = {dipole_xy_mag:.4f}',
                    fontsize=11)
        
        # Add text showing the SM dipole components
        ax.text(0.05, 0.95, 
                f'SM dipole:\nX = {sm_dipole_x:.4f}\nY = {sm_dipole_y:.4f}\nZ = {sm_dipole_z:.4f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9)
        
        # Add SM longitude labels around the edge
        sm_angles = [0, 90, 180, 270]
        for angle_deg in sm_angles:
            angle_rad = angle_deg * np.pi / 180
            x_pos = 1.8 * np.cos(angle_rad)
            y_pos = 1.8 * np.sin(angle_rad)
            ax.text(x_pos, y_pos, f'{angle_deg}°', ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                   fontsize=10)
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)
    
    # Overall title
    fig.suptitle('Geomagnetic Dipole Axis in SM Coordinate System (XY Plane)\n' +
                 'In SM coordinates, the dipole should always be along Z-axis (minimal XY projection)',
                 fontsize=14)
    
    plt.tight_layout()
    return fig


def main():
    """Main function."""
    # Define dates throughout the year
    dates_info = [
        ("Spring Equinox", datetime(2024, 3, 20, 12, 0, 0)),
        ("Mid-May", datetime(2024, 5, 15, 12, 0, 0)),
        ("Summer Solstice", datetime(2024, 6, 21, 12, 0, 0)),
        ("Mid-July", datetime(2024, 7, 15, 12, 0, 0)),
        ("Autumn Equinox", datetime(2024, 9, 22, 12, 0, 0)),
        ("Mid-December", datetime(2024, 12, 15, 12, 0, 0))
    ]
    
    print("Verifying SM coordinate system...")
    print("In SM coordinates, the geomagnetic dipole axis should always")
    print("point along the Z-axis, with minimal projection in the XY plane.")
    print()
    
    # Create the plot
    fig = plot_geomag_axis_in_sm(dates_info)
    
    # Save the figure
    plt.savefig('sm_geomag_axis_verification.png', dpi=300, bbox_inches='tight')
    print("Saved sm_geomag_axis_verification.png")
    
    # Print summary
    print("\nSummary of dipole axis in SM coordinates:")
    print("-" * 60)
    for name, date in dates_info:
        ut = date.timestamp()
        ps = geopack.recalc(ut)
        
        # Calculate dipole in GEO and transform to SM
        geo_dipole_x = np.sin(ps)
        geo_dipole_y = 0.0
        geo_dipole_z = np.cos(ps)
        
        # GEO to GSM
        gsm_dipole_x, gsm_dipole_y, gsm_dipole_z = geogsm_vectorized(
            geo_dipole_x, geo_dipole_y, geo_dipole_z, 1
        )
        
        # GSM to SM
        sm_dipole_x, sm_dipole_y, sm_dipole_z = smgsm_vectorized(
            gsm_dipole_x, gsm_dipole_y, gsm_dipole_z, -1
        )
        
        dipole_xy_mag = np.sqrt(sm_dipole_x**2 + sm_dipole_y**2)
        
        print(f"{name:20s}: ps = {np.degrees(ps):6.1f}°, "
              f"SM dipole XY mag = {dipole_xy_mag:.6f}")
    
    plt.show()


if __name__ == '__main__':
    main()