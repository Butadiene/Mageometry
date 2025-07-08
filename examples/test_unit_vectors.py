"""
Demonstration that T, n, and b are unit vectors and the implications for directional derivatives.
"""

import numpy as np
from geopack import recalc
from geopack.vectorized import (
    t96_vectorized,
    field_line_frenet_frame_vectorized,
    field_line_directional_derivatives_vectorized,
    verify_antisymmetry_relations,
    verify_unit_vectors
)

# Set up parameters
ut = 0.0
ps = recalc(ut)
parmod = [2.0, -18.0, 2.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Test points
x_arr = np.array([-5.0, -6.0, -7.0, -8.0])
y_arr = np.array([0.0, 1.0, 0.0, -1.0])
z_arr = np.array([0.0, 0.0, 1.0, 0.0])

print("Verification of Unit Vector Properties")
print("=" * 60)

# Get Frenet frame
tx, ty, tz, nx, ny, nz, bx, by, bz, curv = field_line_frenet_frame_vectorized(
    t96_vectorized, parmod, ps, x_arr, y_arr, z_arr
)

# Verify unit vectors
errors = verify_unit_vectors(tx, ty, tz, nx, ny, nz, bx, by, bz)

print("\n1. Unit Length Verification:")
print("-" * 40)
for key in ['|T| - 1', '|n| - 1', '|b| - 1']:
    max_error = np.max(np.abs(errors[key]))
    print(f"{key:8} max error: {max_error:.2e}")

print("\n2. Orthogonality Verification:")
print("-" * 40)
for key in ['T·n', 'T·b', 'n·b']:
    max_error = np.max(np.abs(errors[key]))
    print(f"{key:8} max error: {max_error:.2e}")

print("\n3. b = T × n Verification:")
print("-" * 40)
max_error = np.max(np.abs(errors['b - T×n']))
print(f"b - T×n max error: {max_error:.2e}")

# Calculate directional derivatives
derivatives = field_line_directional_derivatives_vectorized(
    t96_vectorized, parmod, ps, x_arr, y_arr, z_arr
)

print("\n\n4. Implications for Directional Derivatives:")
print("=" * 60)
print("\nSince T, n, and b are unit vectors, their derivatives are perpendicular to themselves.")
print("This means the following self-components are always zero:")
print("- (∂T/∂T)·T = 0")
print("- (∂n/∂n)·n = 0")
print("- (∂b/∂b)·b = 0")

print("\nThe 9 formulas capture the non-zero components:")
print("-" * 40)
print("Frenet-Serret formulas:")
print(f"  (∂T/∂T)·n = κ : max = {np.max(derivatives['dT_dT_n']):.4f}")
print(f"  (∂T/∂T)·b = 0 : max = {np.max(np.abs(derivatives['dT_dT_b'])):.2e}")
print(f"  (∂n/∂T)·b = τ : max = {np.max(np.abs(derivatives['dn_dT_b'])):.4f}")

print("\nNote: (∂T/∂T)·T is not included because it's always zero for unit vector T.")

# Demonstrate why (∂T/∂T)·T = 0
print("\n\n5. Mathematical Proof that (∂T/∂T)·T = 0:")
print("=" * 60)
print("Since T·T = 1 (constant), taking the directional derivative:")
print("  d/ds(T·T) = 0")
print("  (∂T/∂s)·T + T·(∂T/∂s) = 0")
print("  2T·(∂T/∂s) = 0")
print("  T·(∂T/∂s) = 0")
print("\nThis holds for any direction s, including s = T.")
print("Therefore: (∂T/∂T)·T = 0")

# Verify antisymmetry still holds
print("\n\n6. Antisymmetry Relations (still valid with unit vectors):")
print("=" * 60)
errors_antisym = verify_antisymmetry_relations(derivatives)
for name, error in errors_antisym.items():
    max_error = np.max(np.abs(error))
    print(f"{name:20} max error = {max_error:.2e}")

print("\n\nConclusion:")
print("-" * 40)
print("The 9 formulas correctly capture all non-zero directional derivatives")
print("while respecting the unit vector constraints of the Frenet-Serret frame.")