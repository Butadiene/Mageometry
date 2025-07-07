"""
Example demonstrating field line directional derivatives.

This script shows how to use the vectorized directional derivative functions
to analyze the geometry of magnetic field lines.
"""

import numpy as np
import matplotlib.pyplot as plt
from geopack import recalc
from geopack.vectorized import t96_vectorized
from geopack.vectorized import (
    field_line_tangent_vectorized,
    field_line_normal_vectorized,
    field_line_binormal_vectorized,
    field_line_curvature_vectorized,
    field_line_torsion_vectorized,
    field_line_tangent_normal_derivative_vectorized,
    field_line_normal_normal_derivative_vectorized,
    field_line_tangent_binormal_derivative_vectorized,
    field_line_normal_binormal_derivative_vectorized
)

# Set up time and model parameters
ut = 0.0  # 1970-01-01 00:00:00
ps = recalc(ut)

# T96 model parameters: [Pdyn, Dst, ByIMF, BzIMF, ...]
parmod = [2.0, -18.0, 2.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Create a grid of points in the equatorial plane
x = np.linspace(-10, -3, 50)
y = np.zeros_like(x)
z = np.zeros_like(x)

# Calculate field line geometry
print("Calculating field line geometry...")
tx, ty, tz = field_line_tangent_vectorized(t96_vectorized, parmod, ps, x, y, z)
nx, ny, nz = field_line_normal_vectorized(t96_vectorized, parmod, ps, x, y, z)
bx, by, bz = field_line_binormal_vectorized(t96_vectorized, parmod, ps, x, y, z)
curvature = field_line_curvature_vectorized(t96_vectorized, parmod, ps, x, y, z)
torsion = field_line_torsion_vectorized(t96_vectorized, parmod, ps, x, y, z)

# Calculate directional derivatives
print("Calculating directional derivatives...")

# Normal derivative of tangent
dT_dn_x, dT_dn_y, dT_dn_z, dT_dn_t, dT_dn_n, dT_dn_b = \
    field_line_tangent_normal_derivative_vectorized(t96_vectorized, parmod, ps, x, y, z)

# Normal derivative of normal
dN_dn_x, dN_dn_y, dN_dn_z, dN_dn_t, dN_dn_n, dN_dn_b = \
    field_line_normal_normal_derivative_vectorized(t96_vectorized, parmod, ps, x, y, z)

# Binormal derivative of tangent
dT_db_x, dT_db_y, dT_db_z, dT_db_t, dT_db_n, dT_db_b = \
    field_line_tangent_binormal_derivative_vectorized(t96_vectorized, parmod, ps, x, y, z)

# Binormal derivative of normal
dN_db_x, dN_db_y, dN_db_z, dN_db_t, dN_db_n, dN_db_b = \
    field_line_normal_binormal_derivative_vectorized(t96_vectorized, parmod, ps, x, y, z)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Curvature and torsion
ax = axes[0, 0]
ax.plot(x, curvature, 'b-', label='Curvature κ')
ax.plot(x, torsion, 'r-', label='Torsion τ')
ax.set_xlabel('X (Re)')
ax.set_ylabel('1/Re')
ax.set_title('Field Line Curvature and Torsion')
ax.legend()
ax.grid(True)

# Plot 2: Components of ∂T/∂n
ax = axes[0, 1]
ax.plot(x, dT_dn_t, 'g-', label='Tangent component (≈0)')
ax.plot(x, dT_dn_n, 'b-', label='Normal component')
ax.plot(x, dT_dn_b, 'r-', label='Binormal component')
ax.set_xlabel('X (Re)')
ax.set_ylabel('1/Re')
ax.set_title('Components of ∂T/∂n')
ax.legend()
ax.grid(True)

# Plot 3: Antisymmetry check
ax = axes[1, 0]
antisym_n = dT_dn_n + dN_dn_t  # Should be ~0
antisym_b = dT_db_n + dN_db_t  # Should be ~0
ax.plot(x, antisym_n, 'b-', label='(∂T/∂n)·N + (∂N/∂n)·T')
ax.plot(x, antisym_b, 'r-', label='(∂T/∂b)·N + (∂N/∂b)·T')
ax.set_xlabel('X (Re)')
ax.set_ylabel('Residual')
ax.set_title('Antisymmetry Relations (should be ~0)')
ax.legend()
ax.grid(True)

# Plot 4: Self-component check
ax = axes[1, 1]
ax.plot(x, dT_dn_t, 'b-', label='(∂T/∂n)·T')
ax.plot(x, dT_db_t, 'r-', label='(∂T/∂b)·T')
ax.plot(x, dN_dn_n, 'g-', label='(∂N/∂n)·N')
ax.set_xlabel('X (Re)')
ax.set_ylabel('Component')
ax.set_title('Self-Components (should be ~0)')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('field_line_directional_derivatives.png', dpi=150)
plt.show()

# Print summary statistics
print("\nSummary Statistics:")
print(f"Curvature range: {curvature.min():.4f} to {curvature.max():.4f} 1/Re")
print(f"Torsion range: {torsion.min():.4f} to {torsion.max():.4f} 1/Re")
print(f"\nSelf-component residuals (should be ~0):")
print(f"  max|(∂T/∂n)·T| = {np.abs(dT_dn_t).max():.2e}")
print(f"  max|(∂T/∂b)·T| = {np.abs(dT_db_t).max():.2e}")
print(f"  max|(∂N/∂n)·N| = {np.abs(dN_dn_n).max():.2e}")
print(f"\nAntisymmetry residuals (should be ~0):")
print(f"  max|(∂T/∂n)·N + (∂N/∂n)·T| = {np.abs(antisym_n).max():.2e}")
print(f"  max|(∂T/∂b)·N + (∂N/∂b)·T| = {np.abs(antisym_b).max():.2e}")

# Note about limitations
print("\nNote: The finite difference implementation has limitations for strongly")
print("curved field lines. The relationships (∂N/∂n)·T = -κ and (∂N/∂b)·T = -τ")
print("may not be exactly satisfied due to numerical approximations.")