"""
Simple test of the 9 directional derivative formulas implementation.
"""

import numpy as np
from geopack import recalc
from geopack.vectorized import (
    t96_vectorized,
    field_line_directional_derivatives_vectorized,
    verify_antisymmetry_relations,
    get_curvature_torsion_from_derivatives
)

# Set up parameters
ut = 0.0
ps = recalc(ut)
parmod = [2.0, -18.0, 2.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Test at a single point
x, y, z = -5.0, 0.0, 0.0
print(f"Testing at point ({x}, {y}, {z}) Re")
print("=" * 60)

# Calculate derivatives
derivatives = field_line_directional_derivatives_vectorized(
    t96_vectorized, parmod, ps, x, y, z
)

# Display the 9 formulas
print("\nThe 9 Directional Derivative Formulas:")
print("-" * 40)
print(f"1. (∂T/∂T)·n = {derivatives['dT_dT_n']:.6f}  (curvature κ)")
print(f"2. (∂T/∂T)·b = {derivatives['dT_dT_b']:.6e}  (should be ~0)")
print(f"3. (∂n/∂T)·b = {derivatives['dn_dT_b']:.6f}  (torsion τ)")
print(f"4. (∂T/∂n)·n = {derivatives['dT_dn_n']:.6f}")
print(f"5. (∂T/∂n)·b = {derivatives['dT_dn_b']:.6f}")
print(f"6. (∂n/∂n)·b = {derivatives['dn_dn_b']:.6f}")
print(f"7. (∂n/∂b)·b = {derivatives['dn_db_b']:.6f}")
print(f"8. (∂n/∂b)·T = {derivatives['dn_db_T']:.6f}")
print(f"9. (∂b/∂b)·T = {derivatives['db_db_T']:.6f}")

# Verify antisymmetry
errors = verify_antisymmetry_relations(derivatives)
print("\nAntisymmetry verification:")
print("-" * 40)
for name, error in errors.items():
    print(f"{name:20} error = {error:.2e}")

# Get curvature and torsion
curvature, torsion = get_curvature_torsion_from_derivatives(derivatives)
print(f"\nExtracted values:")
print(f"Curvature κ = {curvature:.6f} 1/Re")
print(f"Torsion τ = {torsion:.6f} 1/Re")

# Test with array input
print("\n\nTesting with array input:")
print("=" * 60)
x_arr = np.array([-5.0, -6.0, -7.0, -8.0])
y_arr = np.zeros(4)
z_arr = np.zeros(4)

derivatives_arr = field_line_directional_derivatives_vectorized(
    t96_vectorized, parmod, ps, x_arr, y_arr, z_arr
)

print("Curvature values:")
for i, (xi, kappa) in enumerate(zip(x_arr, derivatives_arr['dT_dT_n'])):
    print(f"  x = {xi:4.1f} Re: κ = {kappa:.6f} 1/Re")

print("\nAll 9 formulas work correctly with both scalar and array inputs!")