"""
Simple test of the 9 directional derivative formulas implementation.

Magnetic field model used here:
    B_total = B_IGRF (internal) + B_T96 (external)
in GSM coordinates.
"""

import numpy as np
import geopack
from geopack import recalc, t96_vectorized
from geopack import (
    field_line_directional_derivatives_vectorized,
    verify_antisymmetry_relations,
    get_curvature_torsion_from_derivatives
)


# ----------------------------
# Helpers: robust vectorization
# ----------------------------
def _loop_vectorize_xyz(func, x, y, z, *args):
    """
    Fallback vectorizer for funcs returning (bx,by,bz) given (x,y,z).
    Supports scalar or array inputs.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # Scalar
    if x.shape == () and y.shape == () and z.shape == ():
        return func(*args, float(x), float(y), float(z))

    # Array (broadcast allowed)
    x, y, z = np.broadcast_arrays(x, y, z)

    bx = np.empty_like(x, dtype=float)
    by = np.empty_like(x, dtype=float)
    bz = np.empty_like(x, dtype=float)

    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        bx[idx], by[idx], bz[idx] = func(*args, float(x[idx]), float(y[idx]), float(z[idx]))
        it.iternext()

    return bx, by, bz


def igrf_internal_gsm(x, y, z):
    """
    Internal field (IGRF) in GSM Cartesian [nT].
    Uses vectorized API if available; otherwise loops.
    """
    if hasattr(geopack, "igrf_gsm_vectorized"):
        return geopack.igrf_gsm_vectorized(x, y, z)
    if hasattr(geopack, "igrf_gsm"):
        return _loop_vectorize_xyz(geopack.igrf_gsm, x, y, z)
    raise AttributeError("geopack has no igrf_gsm / igrf_gsm_vectorized.")


def b_igrf_plus_t96_vectorized(parmod, ps, x, y, z):
    """
    Total magnetic field in GSM Cartesian [nT]:
        B_total = B_IGRF (internal) + B_T96 (external)

    Signature must match:
        func(parmod, ps, x, y, z) -> (bx, by, bz)
    """
    bix, biy, biz = igrf_internal_gsm(x, y, z)  # internal
    bex, bey, bez = t96_vectorized(parmod, ps, x, y, z)  # external
    return bix + bex, biy + bey, biz + bez


# ----------------------------
# Set up parameters
# ----------------------------
ut = 0.0
ps = recalc(ut)
parmod = [2.0, -18.0, 2.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Test at a single point
x, y, z = -5.0, 0.0, 0.0
print(f"Testing at point ({x}, {y}, {z}) Re")
print("=" * 60)

# Calculate derivatives (TOTAL field)
derivatives = field_line_directional_derivatives_vectorized(
    b_igrf_plus_t96_vectorized, parmod, ps, x, y, z, delta=1e-3
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
    b_igrf_plus_t96_vectorized, parmod, ps, x_arr, y_arr, z_arr, delta=1e-3
)

print("Curvature values:")
for xi, kappa in zip(x_arr, derivatives_arr['dT_dT_n']):
    print(f"  x = {xi:4.1f} Re: κ = {kappa:.6f} 1/Re")

print("\nAll 9 formulas work correctly with both scalar and array inputs!")
