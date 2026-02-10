"""
Basic usage examples for the Geopack library.

This script demonstrates how to calculate the TOTAL magnetic field as:
    B_total = B_IGRF (internal) + B_T96 (external)
in GSM coordinates.

Notes:
- Call geopack.recalc(ut) whenever the time changes.
- IGRF is computed as internal field; T96 as external field.
"""

import numpy as np
import datetime
import time
import geopack


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


def t96_external_gsm(parmod, ps, x, y, z):
    """
    External field (T96) in GSM Cartesian [nT].
    Uses vectorized API if available; otherwise loops.
    """
    if hasattr(geopack, "t96_vectorized"):
        return geopack.t96_vectorized(parmod, ps, x, y, z)
    if hasattr(geopack, "t96"):
        return _loop_vectorize_xyz(geopack.t96, x, y, z, parmod, ps)
    raise AttributeError("geopack has no t96 / t96_vectorized.")


def total_field_igrf_plus_t96(parmod, ps, x, y, z):
    """
    Total field in GSM Cartesian [nT]:
        B_total = B_IGRF + B_T96
    """
    bix, biy, biz = igrf_internal_gsm(x, y, z)
    bex, bey, bez = t96_external_gsm(parmod, ps, x, y, z)
    return bix + bex, biy + bey, biz + bez, (bix, biy, biz), (bex, bey, bez)


# ----------------------------
# Set up time and calculate dipole tilt
# ----------------------------
dt = datetime.datetime(2023, 3, 15, 12, 0, 0)
ut = dt.timestamp()
ps = geopack.recalc(ut)

print(f"Dipole tilt angle: {np.degrees(ps):.2f} degrees")


# ----------------------------
# T96 parameters
# ----------------------------
# Model parameters: [Pdyn, Dst, ByIMF, BzIMF, unused...]
parmod = np.array([2.0, -20.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0])


# ----------------------------
# Example 1: Scalar total field = IGRF + T96
# ----------------------------
print("\n=== TOTAL Field = IGRF(internal) + T96(external) (Scalar) ===")
x, y, z = 5.0, 0.0, 0.0  # Position in GSM coordinates (Re)

btx, bty, btz, (bix, biy, biz), (bex, bey, bez) = total_field_igrf_plus_t96(parmod, ps, x, y, z)

b_int_mag = np.sqrt(bix**2 + biy**2 + biz**2)
b_ext_mag = np.sqrt(bex**2 + bey**2 + bez**2)
b_tot_mag = np.sqrt(btx**2 + bty**2 + btz**2)

print(f"Position (GSM): ({x}, {y}, {z}) Re")
print(f"IGRF (internal): ({bix:.2f}, {biy:.2f}, {biz:.2f}) nT   |B|={b_int_mag:.2f} nT")
print(f"T96  (external): ({bex:.2f}, {bey:.2f}, {bez:.2f}) nT   |B|={b_ext_mag:.2f} nT")
print(f"TOTAL (sum)    : ({btx:.2f}, {bty:.2f}, {btz:.2f}) nT   |B|={b_tot_mag:.2f} nT")


# ----------------------------
# Example 2: Vectorized total field along X (GSM)
# ----------------------------
print("\n=== TOTAL Field = IGRF + T96 (Vectorized) ===")
n_points = 1000
x = np.linspace(-10, 10, n_points)
y = np.zeros(n_points)
z = np.zeros(n_points)

btx, bty, btz, (bix, biy, biz), (bex, bey, bez) = total_field_igrf_plus_t96(parmod, ps, x, y, z)

indices = [0, n_points // 2, -1]
print("Sample points:")
for i in indices:
    b_int_mag = np.sqrt(bix[i]**2 + biy[i]**2 + biz[i]**2)
    b_ext_mag = np.sqrt(bex[i]**2 + bey[i]**2 + bez[i]**2)
    b_tot_mag = np.sqrt(btx[i]**2 + bty[i]**2 + btz[i]**2)
    print(
        f"  X={x[i]:6.1f} Re | "
        f"IGRF=({bix[i]:7.2f},{biy[i]:7.2f},{biz[i]:7.2f}) |B|={b_int_mag:7.2f}  "
        f"T96=({bex[i]:7.2f},{bey[i]:7.2f},{bez[i]:7.2f}) |B|={b_ext_mag:7.2f}  "
        f"TOTAL=({btx[i]:7.2f},{bty[i]:7.2f},{btz[i]:7.2f}) |B|={b_tot_mag:7.2f}"
    )


# ----------------------------
# Example 3: Performance comparison for TOTAL field
# ----------------------------
print("\n=== Performance Comparison (TOTAL = IGRF + T96) ===")

n_test = 10000
x_test = np.random.uniform(-10, 5, n_test)
y_test = np.random.uniform(-5, 5, n_test)
z_test = np.random.uniform(-3, 3, n_test)

# Scalar version (sample, then extrapolate)
n_sample = 100
t0 = time.time()
for i in range(n_sample):
    _ = total_field_igrf_plus_t96(parmod, ps, float(x_test[i]), float(y_test[i]), float(z_test[i]))[0:3]
t_scalar = (time.time() - t0) * n_test / n_sample

# Vectorized version (if IGRF vectorized not available, this will be slower due to loop fallback)
t0 = time.time()
_ = total_field_igrf_plus_t96(parmod, ps, x_test, y_test, z_test)[0:3]
t_vector = time.time() - t0

print(f"Scalar (estimated): {t_scalar:.3f} seconds for {n_test} points")
print(f"Vectorized:         {t_vector:.3f} seconds for {n_test} points")
print(f"Speedup:            {t_scalar/t_vector:.1f}x")
print(f"Throughput:         {n_test/t_vector:.0f} points/second")

if not hasattr(geopack, "igrf_gsm_vectorized"):
    print("Note: igrf_gsm_vectorized not found -> internal IGRF part used loop fallback.")


# ----------------------------
# Example 4: Coordinate transformations (unchanged)
# ----------------------------
print("\n=== Coordinate Transformations ===")
x_gsm, y_gsm, z_gsm = 5.0, 0.0, 0.0
x_geo, y_geo, z_geo = geopack.geogsm(x_gsm, y_gsm, z_gsm, -1)  # GSM -> GEO
print(f"GSM: ({x_gsm}, {y_gsm}, {z_gsm}) Re")
print(f"GEO: ({x_geo:.3f}, {y_geo:.3f}, {z_geo:.3f}) Re")


# ----------------------------
# Example 5: Internal IGRF only (optional)
# ----------------------------
print("\n=== IGRF (internal only) at GSM point ===")
x_gsm, y_gsm, z_gsm = 2.0, 1.0, -0.5
bix, biy, biz = igrf_internal_gsm(x_gsm, y_gsm, z_gsm)
b_int_mag = np.sqrt(bix**2 + biy**2 + biz**2)
print(f"Position (GSM): ({x_gsm}, {y_gsm}, {z_gsm}) Re")
print(f"IGRF field:     ({bix:.2f}, {biy:.2f}, {biz:.2f}) nT, |B|={b_int_mag:.2f} nT")
