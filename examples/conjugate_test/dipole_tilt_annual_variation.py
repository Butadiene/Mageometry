#!/usr/bin/env python
"""
Plot the annual variation of dipole tilt angle with hourly resolution.
Shows both the seasonal variation and daily oscillation.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import geopack
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geopack


def calculate_annual_tilt():
    """Calculate dipole tilt angle for every hour in 2024."""
    # Start date: January 1, 2024, 00:00 UTC
    start_date = datetime(2024, 1, 1, 0, 0, 0)
    
    # Create hourly timestamps for entire year
    hours_in_year = 365 * 24  # 2024 is a leap year, so 366 days
    if start_date.year % 4 == 0:  # Leap year check
        hours_in_year = 366 * 24
    
    dates = []
    tilts = []
    
    print(f"Calculating dipole tilt for {hours_in_year} hours in {start_date.year}...")
    
    for hour in range(hours_in_year):
        if hour % (24 * 30) == 0:  # Progress update every month
            print(f"  Processing month {hour // (24 * 30) + 1}/12...")
        
        # Current time
        current_time = start_date + timedelta(hours=hour)
        dates.append(current_time)
        
        # Calculate tilt
        ut = current_time.timestamp()
        ps = geopack.recalc(ut)
        tilt_deg = np.degrees(ps)
        tilts.append(tilt_deg)
    
    return np.array(dates), np.array(tilts)


def plot_annual_tilt(dates, tilts):
    """Create visualization of annual tilt variation."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    
    # Convert dates to day of year for x-axis
    start_date = dates[0]
    days_of_year = np.array([(d - start_date).total_seconds() / 86400 for d in dates])
    
    # 1. Full year view
    ax1.plot(days_of_year, tilts, 'b-', linewidth=0.5, alpha=0.7)
    ax1.set_xlabel('Day of Year (2024)')
    ax1.set_ylabel('Dipole Tilt (degrees)')
    ax1.set_title('Annual Variation of Dipole Tilt Angle (Hourly Resolution)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 366)
    
    # Add vertical lines for equinoxes and solstices
    ax1.axvline(79, color='green', linestyle='--', alpha=0.5, label='Spring Equinox')  # Mar 20
    ax1.axvline(172, color='red', linestyle='--', alpha=0.5, label='Summer Solstice')  # Jun 21
    ax1.axvline(266, color='orange', linestyle='--', alpha=0.5, label='Fall Equinox')  # Sep 23
    ax1.axvline(355, color='blue', linestyle='--', alpha=0.5, label='Winter Solstice')  # Dec 21
    ax1.legend(loc='upper right')
    
    # Add envelope showing daily extremes
    daily_max = []
    daily_min = []
    for day in range(366):
        day_start = day * 24
        day_end = min(day_start + 24, len(tilts))
        if day_start < len(tilts):
            daily_max.append(np.max(tilts[day_start:day_end]))
            daily_min.append(np.min(tilts[day_start:day_end]))
    
    days = np.arange(len(daily_max))
    ax1.fill_between(days, daily_min, daily_max, alpha=0.2, color='blue', label='Daily range')
    
    # 2. Seasonal trend (smoothed)
    # Calculate daily average
    daily_avg = []
    for day in range(366):
        day_start = day * 24
        day_end = min(day_start + 24, len(tilts))
        if day_start < len(tilts):
            daily_avg.append(np.mean(tilts[day_start:day_end]))
    
    ax2.plot(days[:len(daily_avg)], daily_avg, 'r-', linewidth=2, label='Daily Average')
    ax2.set_xlabel('Day of Year (2024)')
    ax2.set_ylabel('Average Dipole Tilt (degrees)')
    ax2.set_title('Seasonal Variation of Dipole Tilt (Daily Averages)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 366)
    
    # Add reference lines
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax2.axvline(79, color='green', linestyle='--', alpha=0.5)
    ax2.axvline(172, color='red', linestyle='--', alpha=0.5)
    ax2.axvline(266, color='orange', linestyle='--', alpha=0.5)
    ax2.axvline(355, color='blue', linestyle='--', alpha=0.5)
    
    # 3. Example daily variations at different seasons
    example_days = [
        (79, 'Spring Equinox'),
        (172, 'Summer Solstice'),
        (266, 'Fall Equinox'),
        (355, 'Winter Solstice')
    ]
    
    colors = ['green', 'red', 'orange', 'blue']
    
    for (day, label), color in zip(example_days, colors):
        day_start = day * 24
        day_end = day_start + 24
        if day_end <= len(tilts):
            hours = np.arange(24)
            day_tilts = tilts[day_start:day_end]
            ax3.plot(hours, day_tilts, '-', color=color, linewidth=2, label=label)
    
    ax3.set_xlabel('Hour of Day (UTC)')
    ax3.set_ylabel('Dipole Tilt (degrees)')
    ax3.set_title('Daily Variation of Dipole Tilt at Different Seasons', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 23)
    ax3.legend()
    ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # Print some statistics
    print("\nDipole Tilt Statistics for 2024:")
    print(f"  Maximum tilt: {np.max(tilts):.2f}° (around summer solstice)")
    print(f"  Minimum tilt: {np.min(tilts):.2f}° (around winter solstice)")
    print(f"  Annual average: {np.mean(tilts):.2f}°")
    print(f"  Daily variation range: ~{np.mean(daily_max) - np.mean(daily_min):.1f}°")
    
    # Find exact times of extremes
    max_idx = np.argmax(tilts)
    min_idx = np.argmin(tilts)
    print(f"\n  Maximum occurs at: {dates[max_idx].strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Minimum occurs at: {dates[min_idx].strftime('%Y-%m-%d %H:%M UTC')}")
    
    return fig


def main():
    """Main function."""
    print("Calculating annual variation of dipole tilt angle...")
    print("This will take a moment as we calculate 8760+ hourly values...")
    
    # Calculate tilt for entire year
    dates, tilts = calculate_annual_tilt()
    
    # Create visualization
    fig = plot_annual_tilt(dates, tilts)
    
    # Save figure
    filename = os.path.join(os.path.dirname(__file__), 'dipole_tilt_annual_variation.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to: {filename}")
    
    plt.show()


if __name__ == '__main__':
    main()