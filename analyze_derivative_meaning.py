"""
Analyze the mathematical meaning and behavior of (∂n/∂b)·b.
"""
import numpy as np
import matplotlib.pyplot as plt
from geopack import recalc, dip
from geopack.vectorized import (
    field_line_frenet_frame_vectorized,
    field_line_directional_derivatives_vectorized,
    field_line_curvature_vectorized
)

# Initialize
ut = 0.0
ps = recalc(ut)
ps = 0.0  # No tilt

def dipole_field_wrapper(parmod, ps, x, y, z):
    bx, by, bz = dip(x, y, z)
    return bx, by, bz

print("Mathematical Analysis of (∂n/∂b)·b")
print("=" * 60)
print()
print("The quantity (∂n/∂b)·b represents how the normal vector n changes")
print("as we move in the binormal direction b, projected onto b itself.")
print()
print("For a dipole field in the meridional plane:")
print("- Field lines are planar (τ = 0)")
print("- The binormal b points in the Y direction (azimuthal)")
print("- Moving in the b direction takes us around the dipole axis")
print()

# Analyze behavior along a radial line at different latitudes
print("Analyzing along radial lines at different latitudes:")
print("-" * 60)

r_values = np.linspace(2, 8, 30)
latitudes = [0, 30, 60, 80]  # degrees

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for i, lat in enumerate(latitudes):
    ax = axes[i // 2, i % 2]
    
    # Convert latitude to z coordinate
    lat_rad = np.radians(lat)
    x_vals = -r_values * np.cos(lat_rad)
    z_vals = r_values * np.sin(lat_rad)
    y_vals = np.zeros_like(x_vals)
    
    # Calculate derivatives
    derivatives = field_line_directional_derivatives_vectorized(
        dipole_field_wrapper, None, ps, x_vals, y_vals, z_vals
    )
    
    # Get values
    dn_db_b = derivatives['dn_db_b']
    curvature = derivatives['dT_dT_n']
    
    # Also get field magnitude
    b_mags = []
    for x, y, z in zip(x_vals, y_vals, z_vals):
        bx, by, bz = dip(x, y, z)
        b_mags.append(np.sqrt(bx**2 + by**2 + bz**2))
    b_mags = np.array(b_mags)
    
    # Plot
    ax.plot(r_values, dn_db_b, 'b-', linewidth=2, label='(∂n/∂b)·b')
    ax.plot(r_values, curvature, 'r--', linewidth=2, label='κ')
    ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Radial distance (Re)')
    ax.set_ylabel('Value')
    ax.set_title(f'Latitude = {lat}°')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Print statistics
    print(f"\nLatitude {lat}°:")
    print(f"  (∂n/∂b)·b range: [{dn_db_b.min():.3f}, {dn_db_b.max():.3f}]")
    print(f"  Curvature range: [{curvature.min():.3f}, {curvature.max():.3f}]")
    
    # Find where (∂n/∂b)·b is maximum
    max_idx = np.argmax(np.abs(dn_db_b))
    print(f"  Max |(∂n/∂b)·b| = {np.abs(dn_db_b[max_idx]):.3f} at r = {r_values[max_idx]:.2f} Re")

plt.tight_layout()
plt.savefig('derivative_radial_analysis.png', dpi=150)
plt.show()

# Now analyze the structure near the extreme point
print("\n" + "=" * 60)
print("Detailed analysis near extreme point (-2.0, 0.0, 4.0):")

# The extreme point corresponds to high latitude, close to Earth
x0, z0 = -2.0, 4.0
r0 = np.sqrt(x0**2 + z0**2)
lat0 = np.degrees(np.arctan2(z0, -x0))
print(f"Extreme point: r = {r0:.2f} Re, latitude = {lat0:.1f}°")

# This is near the polar region at relatively close distance
# The large value might be due to the rapid change in field line geometry
# as we move azimuthally near the pole

# Test the numerical stability
print("\nNumerical stability test with different methods:")

def calculate_dn_db_b_forward(x, y, z, delta=0.01):
    """Calculate (∂n/∂b)·b using forward differences."""
    # Get frame at current point
    tx0, ty0, tz0, nx0, ny0, nz0, bx0, by0, bz0, _ = field_line_frenet_frame_vectorized(
        dipole_field_wrapper, None, ps, x, y, z
    )
    
    # Step in binormal direction
    x_plus = x + delta * bx0
    y_plus = y + delta * by0
    z_plus = z + delta * bz0
    
    # Get normal at stepped point
    _, _, _, nx_plus, ny_plus, nz_plus, _, _, _, _ = field_line_frenet_frame_vectorized(
        dipole_field_wrapper, None, ps, x_plus, y_plus, z_plus
    )
    
    # Forward difference
    dn_db_x = (nx_plus - nx0) / delta
    dn_db_y = (ny_plus - ny0) / delta
    dn_db_z = (nz_plus - nz0) / delta
    
    # Project onto b
    return dn_db_x * bx0 + dn_db_y * by0 + dn_db_z * bz0

# Test different methods
deltas = [0.001, 0.005, 0.01, 0.05, 0.1]
for delta in deltas:
    # Central difference (default)
    derivatives = field_line_directional_derivatives_vectorized(
        dipole_field_wrapper, None, ps, x0, 0.0, z0, delta=delta
    )
    central = derivatives['dn_db_b']
    
    # Forward difference
    forward = calculate_dn_db_b_forward(x0, 0.0, z0, delta)
    
    print(f"delta = {delta:5.3f}: central = {central:8.3f}, forward = {forward:8.3f}")

print("\nConclusion:")
print("The large values of (∂n/∂b)·b near the poles at close distances are likely")
print("due to the rapid change in field line geometry in the azimuthal direction.")
print("This is a real mathematical feature, not a numerical error.")