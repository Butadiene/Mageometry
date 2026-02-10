"""
Demonstration that T, n, and b are unit vectors and the implications for directional derivatives.

Magnetic field model used here:
    B_total = B_IGRF (internal) + B_T96 (external)
in GSM coordinates.
"""

import numpy as np
import geopack
from geopack import recalc, t96_vectorized
from geopack import (
    field_line_frenet_frame_vectorized,
    field_line_directional_derivatives_vectorized,
    verify_antisymmetry_relations,
    verify_unit_vectors
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
    bix, biy, biz = igrf_internal_gsm(x, y, z)          # internal
    bex, bey, bez = t96_vectorized(parmod, ps, x, y, z) # external
    return bix + bex, biy + bey, biz + bez


# ----------------------------
# Set up parameters
# ----------------------------
ut = 0.0
ps = recalc(ut)
parmod = [2.0, -18.0, 2.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Test points
x_arr = np.array([-5.0, -6.0, -7.0, -8.0])
y_arr = np.array([0.0, 1.0, 0.0, -1.0])
z_arr = np.array([0.0, 0.0, 1.0, 0.0])

print("Verification of Unit Vector Properties")
print("=" * 60)
print("Model: IGRF(internal) + T96(external)")

# Get Frenet frame (TOTAL field)
tx, ty, tz, nx, ny, nz, bx, by, bz, curv = field_line_frenet_frame_vectorized(
    b_igrf_plus_t96_vectorized, parmod, ps, x_arr, y_arr, z_arr
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

# Calculate directional derivatives (TOTAL field)
derivatives = field_line_directional_derivatives_vectorized(
    b_igrf_plus_t96_vectorized, parmod, ps, x_arr, y_arr, z_arr
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
