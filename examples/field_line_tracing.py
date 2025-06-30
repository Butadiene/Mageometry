#!/usr/bin/env python
"""
Field line tracing example using Geopack models.

This example demonstrates how to trace magnetic field lines using
the magnetospheric models.
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
import geopack

def trace_field_line(model_func, params, ps, start_pos, step_size=0.1, max_steps=1000):
    """
    Simple field line tracer using Runge-Kutta 4th order method.
    
    Parameters:
    -----------
    model_func : function
        Field model function (e.g., geopack.t96)
    params : array-like
        Model parameters
    ps : float
        Dipole tilt angle
    start_pos : tuple
        Starting position (x, y, z) in Re
    step_size : float
        Integration step size in Re
    max_steps : int
        Maximum number of steps
    
    Returns:
    --------
    positions : ndarray
        Array of positions along field line
    """
    positions = [start_pos]
    x, y, z = start_pos
    
    for _ in range(max_steps):
        # Get field direction at current position
        bx, by, bz = model_func(params, ps, x, y, z)
        b_mag = np.sqrt(bx**2 + by**2 + bz**2)
        
        if b_mag < 1e-10:  # Stop if field is too weak
            break
            
        # Normalize to unit vector
        dx, dy, dz = bx/b_mag, by/b_mag, bz/b_mag
        
        # RK4 integration
        k1x, k1y, k1z = step_size * dx, step_size * dy, step_size * dz
        
        # Half step
        bx2, by2, bz2 = model_func(params, ps, x + k1x/2, y + k1y/2, z + k1z/2)
        b_mag2 = np.sqrt(bx2**2 + by2**2 + bz2**2)
        if b_mag2 < 1e-10:
            break
        k2x, k2y, k2z = step_size * bx2/b_mag2, step_size * by2/b_mag2, step_size * bz2/b_mag2
        
        # Another half step
        bx3, by3, bz3 = model_func(params, ps, x + k2x/2, y + k2y/2, z + k2z/2)
        b_mag3 = np.sqrt(bx3**2 + by3**2 + bz3**2)
        if b_mag3 < 1e-10:
            break
        k3x, k3y, k3z = step_size * bx3/b_mag3, step_size * by3/b_mag3, step_size * bz3/b_mag3
        
        # Full step
        bx4, by4, bz4 = model_func(params, ps, x + k3x, y + k3y, z + k3z)
        b_mag4 = np.sqrt(bx4**2 + by4**2 + bz4**2)
        if b_mag4 < 1e-10:
            break
        k4x, k4y, k4z = step_size * bx4/b_mag4, step_size * by4/b_mag4, step_size * bz4/b_mag4
        
        # Update position
        x += (k1x + 2*k2x + 2*k3x + k4x) / 6
        y += (k1y + 2*k2y + 2*k3y + k4y) / 6
        z += (k1z + 2*k2z + 2*k3z + k4z) / 6
        
        positions.append((x, y, z))
        
        # Stop if we've reached Earth's surface or gone too far
        r = np.sqrt(x**2 + y**2 + z**2)
        if r < 1.0 or r > 50.0:
            break
    
    return np.array(positions)

# Set up time and model parameters
dt = datetime.datetime(2023, 3, 15, 12, 0, 0)
ut = dt.timestamp()
ps = geopack.recalc(ut)

# T96 model parameters
parmod = np.array([2.0, -20.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0])

# Trace field lines from different starting points
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

# Starting positions along x-axis
start_x_values = [3, 4, 5, 6, 7, 8]
colors = plt.cm.viridis(np.linspace(0, 1, len(start_x_values)))

for x_start, color in zip(start_x_values, colors):
    start_pos = (x_start, 0.0, 0.0)
    
    # Trace field line in both directions
    # Forward (along field)
    field_line_fwd = trace_field_line(geopack.t96, parmod, ps, start_pos, step_size=0.1)
    
    # Backward (against field)
    field_line_bwd = trace_field_line(geopack.t96, parmod, ps, start_pos, step_size=-0.1)
    
    # Combine both directions
    field_line = np.vstack([field_line_bwd[::-1], field_line_fwd[1:]])
    
    # Plot in XZ plane
    ax.plot(field_line[:, 0], field_line[:, 2], color=color, 
            label=f'Start at X={x_start} Re', linewidth=2)

# Add Earth
earth = plt.Circle((0, 0), 1.0, color='blue', alpha=0.5)
ax.add_patch(earth)

# Formatting
ax.set_xlabel('X (Re)', fontsize=12)
ax.set_ylabel('Z (Re)', fontsize=12)
ax.set_title('Magnetic Field Lines - T96 Model\n' + 
             f'Pdyn={parmod[0]} nPa, Dst={parmod[1]} nT, ' +
             f'By_IMF={parmod[2]} nT, Bz_IMF={parmod[3]} nT',
             fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.legend(loc='upper right')
ax.set_xlim(-5, 15)
ax.set_ylim(-10, 10)

plt.tight_layout()
plt.savefig('field_lines_example.png', dpi=150)
print("Field line plot saved as 'field_lines_example.png'")

# Calculate field line footpoints
print("\nField line footpoints (where they intersect Earth's surface):")
for x_start in start_x_values:
    start_pos = (x_start, 0.0, 0.0)
    
    # Trace backward to find footpoint
    field_line = trace_field_line(geopack.t96, parmod, ps, start_pos, step_size=-0.05)
    
    # Find where it intersects Earth's surface
    for pos in field_line:
        r = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        if r <= 1.01:  # Close to Earth's surface
            lat = np.degrees(np.arcsin(pos[2]/r))
            lon = np.degrees(np.arctan2(pos[1], pos[0]))
            print(f"  X={x_start} Re -> Lat={lat:.1f}°, Lon={lon:.1f}°")
            break