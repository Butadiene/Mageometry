"""
Debug script to investigate the large (∂n/∂b)·b values in dipole field analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from geopack import recalc, dip
from geopack.vectorized import (
    field_line_directional_derivatives_vectorized,
    field_line_frenet_frame_vectorized,
    verify_unit_vectors
)

# Initialize geopack
ut = 0.0
recalc(ut)

# Dipole field wrapper
def dipole_field_wrapper(parmod, ps, x, y, z):
    """Wrapper for dipole field that matches T96 interface."""
    bx, by, bz = dip(x, y, z)
    return bx, by, bz

# Test at specific points
ps = 0.0  # No tilt

# Select test points where the issue is prominent
# From the notebook, we see large values in certain regions
test_points = [
    # Point near Earth where curvature is high
    (-3.0, 0.0, 0.0),
    # Point in problematic region from the plot
    (-4.0, 0.0, 2.0),
    (-4.0, 0.0, -2.0),
    # Point farther out
    (-6.0, 0.0, 0.0),
    # Additional test points
    (-2.5, 0.0, 1.0),
    (-2.5, 0.0, -1.0)
]

print("Analyzing (∂n/∂b)·b values at test points")
print("=" * 70)

for i, (x, y, z) in enumerate(test_points):
    print(f"\nPoint {i+1}: ({x:.1f}, {y:.1f}, {z:.1f}) Re")
    print("-" * 50)
    
    # Get Frenet frame
    tx, ty, tz, nx, ny, nz, bx, by, bz, info = field_line_frenet_frame_vectorized(
        dipole_field_wrapper, None, ps, x, y, z
    )
    
    # Verify unit vectors
    unit_errors = verify_unit_vectors(tx, ty, tz, nx, ny, nz, bx, by, bz)
    print(f"Unit vector checks:")
    print(f"  |T| - 1 = {unit_errors['|T| - 1']:.2e}")
    print(f"  |n| - 1 = {unit_errors['|n| - 1']:.2e}")
    print(f"  |b| - 1 = {unit_errors['|b| - 1']:.2e}")
    print(f"  T·n = {unit_errors['T·n']:.2e}")
    print(f"  T·b = {unit_errors['T·b']:.2e}")
    print(f"  n·b = {unit_errors['n·b']:.2e}")
    
    # Get all derivatives
    derivatives = field_line_directional_derivatives_vectorized(
        dipole_field_wrapper, None, ps, x, y, z
    )
    
    # Print key values
    print(f"\nDerivative values:")
    print(f"  κ = (∂T/∂T)·n = {derivatives['dT_dT_n']:.6f}")
    print(f"  τ = (∂n/∂T)·b = {derivatives['dn_dT_b']:.2e}")
    print(f"  (∂n/∂b)·b = {derivatives['dn_db_b']:.6f}")
    print(f"  (∂n/∂b)·T = {derivatives['dn_db_T']:.6f}")
    print(f"  (∂b/∂b)·T = {derivatives['db_db_T']:.6f}")
    
    # Check antisymmetry
    print(f"\nAntisymmetry checks:")
    print(f"  (∂n/∂b)·b + (∂b/∂b)·n = {derivatives['dn_db_b'] + derivatives['db_db_n']:.2e}")
    print(f"  (∂n/∂b)·T + (∂T/∂b)·n = {derivatives['dn_db_T'] + derivatives['dT_db_n']:.2e}")
    
    # Test different delta values
    print(f"\nEffect of delta on (∂n/∂b)·b:")
    for delta in [0.001, 0.01, 0.1]:
        derivatives_delta = field_line_directional_derivatives_vectorized(
            dipole_field_wrapper, None, ps, x, y, z, delta=delta
        )
        print(f"  delta = {delta:5.3f}: (∂n/∂b)·b = {derivatives_delta['dn_db_b']:8.4f}")

# Now let's look at the spatial pattern
print("\n" + "=" * 70)
print("Analyzing spatial pattern of (∂n/∂b)·b")

# Create a finer grid around problematic region
x_fine = np.linspace(-5, -3, 30)
z_fine = np.linspace(-3, 3, 30)
X_fine, Z_fine = np.meshgrid(x_fine, z_fine)
Y_fine = np.zeros_like(X_fine)

# Flatten for calculations
x_flat = X_fine.flatten()
y_flat = Y_fine.flatten()
z_flat = Z_fine.flatten()

# Calculate derivatives
derivatives_grid = field_line_directional_derivatives_vectorized(
    dipole_field_wrapper, None, ps, x_flat, y_flat, z_flat
)

# Reshape
dn_db_b_grid = derivatives_grid['dn_db_b'].reshape(X_fine.shape)
curvature_grid = derivatives_grid['dT_dT_n'].reshape(X_fine.shape)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# (∂n/∂b)·b
im1 = ax1.contourf(X_fine, Z_fine, dn_db_b_grid, levels=20, cmap='RdBu_r')
ax1.set_title('(∂n/∂b)·b in Meridional Plane')
ax1.set_xlabel('X (Re)')
ax1.set_ylabel('Z (Re)')
ax1.set_aspect('equal')
plt.colorbar(im1, ax=ax1, label='(∂n/∂b)·b')

# Add contour lines for extreme values
extreme_vals = np.percentile(np.abs(dn_db_b_grid), [90, 95, 99])
ax1.contour(X_fine, Z_fine, np.abs(dn_db_b_grid), levels=extreme_vals, 
            colors='black', linewidths=1, alpha=0.5)

# Curvature for comparison
im2 = ax2.contourf(X_fine, Z_fine, curvature_grid, levels=20, cmap='viridis')
ax2.set_title('Curvature κ')
ax2.set_xlabel('X (Re)')
ax2.set_ylabel('Z (Re)')
ax2.set_aspect('equal')
plt.colorbar(im2, ax=ax2, label='κ (1/Re)')

plt.tight_layout()
plt.savefig('debug_dn_db_b_pattern.png', dpi=150)
plt.show()

# Find points with extreme values
max_idx = np.argmax(np.abs(dn_db_b_grid.flatten()))
x_max = x_flat[max_idx]
y_max = y_flat[max_idx]
z_max = z_flat[max_idx]
val_max = dn_db_b_grid.flatten()[max_idx]

print(f"\nMaximum |∂n/∂b)·b| = {np.abs(val_max):.2f} at ({x_max:.2f}, {y_max:.2f}, {z_max:.2f}) Re")

# Check magnetic field strength at extreme points
bx, by, bz = dipole_field_wrapper(None, ps, x_max, y_max, z_max)
b_mag = np.sqrt(bx**2 + by**2 + bz**2)
print(f"Magnetic field strength at max point: {b_mag:.1f} nT")

# Distance from Earth
r_max = np.sqrt(x_max**2 + y_max**2 + z_max**2)
print(f"Distance from Earth: {r_max:.2f} Re")