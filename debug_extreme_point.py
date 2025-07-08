"""
Investigate the extreme (∂n/∂b)·b value at (-2.0, 0.0, 4.0).
"""
import numpy as np
import matplotlib.pyplot as plt
from geopack import recalc, dip
from geopack.vectorized import (
    field_line_directional_derivatives_vectorized,
    field_line_frenet_frame_vectorized,
    field_line_tangent_vectorized,
    field_line_normal_vectorized,
    field_line_binormal_vectorized,
    verify_unit_vectors
)

# Initialize
ut = 0.0
ps = recalc(ut)
ps = 0.0  # No tilt

# Dipole field wrapper
def dipole_field_wrapper(parmod, ps, x, y, z):
    bx, by, bz = dip(x, y, z)
    return bx, by, bz

# The problematic point
x0, y0, z0 = -2.0, 0.0, 4.0
print(f"Analyzing extreme point: ({x0}, {y0}, {z0}) Re")
print("=" * 60)

# Basic field properties
bx, by, bz = dip(x0, y0, z0)
b_mag = np.sqrt(bx**2 + by**2 + bz**2)
r = np.sqrt(x0**2 + y0**2 + z0**2)
print(f"Distance from Earth: {r:.2f} Re")
print(f"Magnetic field: Bx={bx:.1f}, By={by:.1f}, Bz={bz:.1f} nT")
print(f"Field magnitude: {b_mag:.1f} nT")

# Get Frenet frame
tx, ty, tz, nx, ny, nz, bx_frame, by_frame, bz_frame, info = field_line_frenet_frame_vectorized(
    dipole_field_wrapper, None, ps, x0, y0, z0
)

print(f"\nFrenet frame vectors:")
print(f"T = ({tx:.6f}, {ty:.6f}, {tz:.6f})")
print(f"n = ({nx:.6f}, {ny:.6f}, {nz:.6f})")  
print(f"b = ({bx_frame:.6f}, {by_frame:.6f}, {bz_frame:.6f})")

# Verify orthonormality
unit_errors = verify_unit_vectors(tx, ty, tz, nx, ny, nz, bx_frame, by_frame, bz_frame)
print(f"\nUnit vector verification:")
for key, val in unit_errors.items():
    print(f"  {key:10} = {val:.2e}")

# Get derivatives at this point
print(f"\nCalculating derivatives with different delta values:")
for delta in [0.001, 0.01, 0.1]:
    derivatives = field_line_directional_derivatives_vectorized(
        dipole_field_wrapper, None, ps, x0, y0, z0, delta=delta
    )
    print(f"\ndelta = {delta}:")
    print(f"  κ = (∂T/∂T)·n = {derivatives['dT_dT_n']:.6f}")
    print(f"  (∂n/∂b)·b = {derivatives['dn_db_b']:.6f}")
    print(f"  (∂n/∂b)·T = {derivatives['dn_db_T']:.6f}")
    print(f"  (∂b/∂b)·T = {derivatives['db_db_T']:.6f}")
    
    # Check antisymmetry
    antisym_check = derivatives['dn_db_b'] + derivatives['db_db_n']
    print(f"  Antisymmetry: (∂n/∂b)·b + (∂b/∂b)·n = {antisym_check:.2e}")

# Investigate the neighborhood
print("\n" + "=" * 60)
print("Analyzing neighborhood around extreme point")

# Create small grid around the point
eps = 0.5
x_near = np.linspace(x0 - eps, x0 + eps, 21)
z_near = np.linspace(z0 - eps, z0 + eps, 21)
X_near, Z_near = np.meshgrid(x_near, z_near)
Y_near = np.zeros_like(X_near)

# Calculate derivatives on grid
x_flat = X_near.flatten()
y_flat = Y_near.flatten()
z_flat = Z_near.flatten()

derivatives_near = field_line_directional_derivatives_vectorized(
    dipole_field_wrapper, None, ps, x_flat, y_flat, z_flat
)

dn_db_b_near = derivatives_near['dn_db_b'].reshape(X_near.shape)
curvature_near = derivatives_near['dT_dT_n'].reshape(X_near.shape)

# Plot neighborhood
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# (∂n/∂b)·b
im1 = ax1.contourf(X_near, Z_near, dn_db_b_near, levels=20, cmap='RdBu_r')
ax1.plot(x0, z0, 'k*', markersize=10, label='Extreme point')
ax1.set_title(f'(∂n/∂b)·b near ({x0}, {z0})')
ax1.set_xlabel('X (Re)')
ax1.set_ylabel('Z (Re)')
ax1.set_aspect('equal')
plt.colorbar(im1, ax=ax1)

# Curvature
im2 = ax2.contourf(X_near, Z_near, curvature_near, levels=20, cmap='viridis')
ax2.plot(x0, z0, 'k*', markersize=10)
ax2.set_title('Curvature κ')
ax2.set_xlabel('X (Re)')
ax2.set_ylabel('Z (Re)')
ax2.set_aspect('equal')
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.savefig('extreme_point_neighborhood.png', dpi=150)
plt.show()

# Check if this is near a singular region
print(f"\nChecking for singular behavior:")
# Magnetic field gradient
delta_grad = 0.01
bx_plus_x, _, _ = dip(x0 + delta_grad, y0, z0)
bx_minus_x, _, _ = dip(x0 - delta_grad, y0, z0)
dbx_dx = (bx_plus_x - bx_minus_x) / (2 * delta_grad)

_, _, bz_plus_z = dip(x0, y0, z0 + delta_grad)
_, _, bz_minus_z = dip(x0, y0, z0 - delta_grad)
dbz_dz = (bz_plus_z - bz_minus_z) / (2 * delta_grad)

divergence = dbx_dx + dbz_dz  # ∂By/∂y = 0 by symmetry
print(f"∇·B = {divergence:.6f} nT/Re (should be ~0)")

# Check if we're near Earth's surface
earth_radius = 1.0  # Re
min_distance = 1.5  # Minimum safe distance
if r < min_distance:
    print(f"\nWARNING: Point is very close to Earth (r = {r:.2f} Re < {min_distance} Re)")
    print("This may cause numerical issues in derivative calculations.")