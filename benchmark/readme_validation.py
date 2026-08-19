"""
Scalar vs Vectorized Accuracy Validation

Computes IGRF + T96 total magnetic field over a grid in the X-Z meridian
plane (Y=0) using both scalar (loop) and vectorized implementations, then
generates two figures:

  1. Histogram of the relative error distribution
  2. Colormap of relative error projected onto the magnetosphere

Usage:
    python benchmark/readme_validation.py
"""

import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from mageometry import geopack

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
NX, NZ = 100, 100
X_RANGE = (2.0, -10.0)   # Re  (sunward → tailward)
Z_RANGE = (6.0, -6.0)    # Re
R_EARTH = 1.0             # mask radius

ut = datetime(2020, 1, 1, 12, 0, 0).timestamp()
ps = geopack.recalc(ut)
parmod = np.array([2.0, -20.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0], dtype=np.float64)

# ---------------------------------------------------------------------------
# Grid (X-Z meridian plane, Y = 0)
# ---------------------------------------------------------------------------
x1d = np.linspace(*X_RANGE, NX)
z1d = np.linspace(*Z_RANGE, NZ)
X, Z = np.meshgrid(x1d, z1d)        # shape (NZ, NX)
Y = np.zeros_like(X)

x_flat = X.ravel()
y_flat = Y.ravel()
z_flat = Z.ravel()

r_flat = np.sqrt(x_flat**2 + y_flat**2 + z_flat**2)
outside = r_flat >= R_EARTH

# ---------------------------------------------------------------------------
# Scalar computation (loop over valid points)
# ---------------------------------------------------------------------------
n_pts = x_flat.size
bx_s = np.full(n_pts, np.nan)
by_s = np.full(n_pts, np.nan)
bz_s = np.full(n_pts, np.nan)

print(f"Computing scalar field at {outside.sum()} points ...")
t0 = time.perf_counter()
for i in np.where(outside)[0]:
    xi, yi, zi = float(x_flat[i]), float(y_flat[i]), float(z_flat[i])
    ix, iy, iz = geopack.igrf_gsm(xi, yi, zi)
    ex, ey, ez = geopack.t96(parmod, ps, xi, yi, zi)
    bx_s[i] = ix + ex
    by_s[i] = iy + ey
    bz_s[i] = iz + ez
t_scalar = time.perf_counter() - t0
print(f"  scalar: {t_scalar:.2f} s")

# ---------------------------------------------------------------------------
# Vectorized computation (single call on full arrays)
# ---------------------------------------------------------------------------
bx_v = np.full(n_pts, np.nan)
by_v = np.full(n_pts, np.nan)
bz_v = np.full(n_pts, np.nan)

print(f"Computing vectorized field ...")
t0 = time.perf_counter()
xo, yo, zo = x_flat[outside], y_flat[outside], z_flat[outside]
ix, iy, iz = geopack.igrf_gsm_vectorized(xo, yo, zo)
ex, ey, ez = geopack.t96_vectorized(parmod, ps, xo, yo, zo)
bx_v[outside] = ix + ex
by_v[outside] = iy + ey
bz_v[outside] = iz + ez
t_vec = time.perf_counter() - t0
print(f"  vectorized: {t_vec:.4f} s  ({t_scalar / max(t_vec, 1e-15):.1f}x speedup)")

# ---------------------------------------------------------------------------
# Relative error:  (B_scalar - B_vector) / |B_scalar|
# ---------------------------------------------------------------------------
B_s = np.sqrt(bx_s**2 + by_s**2 + bz_s**2)
B_v = np.sqrt(bx_v**2 + by_v**2 + bz_v**2)

valid = outside & (B_s > 0)
rel_err = np.full(n_pts, np.nan)
rel_err[valid] = (B_s[valid] - B_v[valid]) / np.abs(B_s[valid])

err_valid = rel_err[valid]
abs_err = np.abs(err_valid)

print("\n--- Relative error statistics ---")
print(f"  points evaluated : {valid.sum()}")
print(f"  max |rel err|    : {abs_err.max():.3e}")
print(f"  mean |rel err|   : {abs_err.mean():.3e}")
print(f"  median |rel err| : {np.median(abs_err):.3e}")

# ---------------------------------------------------------------------------
# Figure 1 — Histogram of relative error
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-darkgrid")

fig1, ax1 = plt.subplots(figsize=(10, 6))

# Use absolute relative error on a log10 scale for the histogram
log_abs_err = np.log10(np.clip(abs_err, 1e-18, None))

ax1.hist(log_abs_err, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.4)
ax1.set_xlabel("log$_{10}$  |relative error|")
ax1.set_ylabel("Count")
ax1.set_title(
    "IGRF + T96 : Scalar vs Vectorized Relative Error Distribution\n"
    f"X = [{X_RANGE[0]}, {X_RANGE[1]}] Re,  "
    f"Z = [{Z_RANGE[0]}, {Z_RANGE[1]}] Re,  Y = 0"
)

# Annotate statistics
speedup = t_scalar / max(t_vec, 1e-15)
stats_text = (
    f"max  |rel err| = {abs_err.max():.2e}\n"
    f"mean |rel err| = {abs_err.mean():.2e}\n"
    f"med  |rel err| = {np.median(abs_err):.2e}\n"
    f"N = {valid.sum()}\n"
    f"\n"
    f"scalar     : {t_scalar:7.2f} s\n"
    f"vectorized : {t_vec:7.4f} s\n"
    f"speedup    : {speedup:.0f}x"
)
ax1.text(
    0.97, 0.95, stats_text,
    transform=ax1.transAxes, ha="right", va="top",
    fontsize=10, family="monospace",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
)

fig1.tight_layout()
fig1.savefig("benchmark/readme_validation_histogram.png", dpi=300, bbox_inches="tight")
print("\nSaved  benchmark/readme_validation_histogram.png")

# ---------------------------------------------------------------------------
# Figure 2 — Colormap of relative error on the magnetosphere
# ---------------------------------------------------------------------------
err_grid = rel_err.reshape(NZ, NX)

fig2, ax2 = plt.subplots(figsize=(12, 7))

# Symmetric log-scale colour map centred on zero
abs_max = np.nanmax(np.abs(err_grid))
if abs_max == 0:
    abs_max = 1e-16
from matplotlib.colors import SymLogNorm
norm = SymLogNorm(linthresh=abs_max * 1e-2, vmin=-abs_max, vmax=abs_max)

im = ax2.pcolormesh(
    x1d, z1d, err_grid,
    cmap="RdBu_r", norm=norm, shading="auto",
)
cbar = fig2.colorbar(im, ax=ax2, pad=0.02)
cbar.set_label("Relative error  (scalar $-$ vector) / |scalar|")

# Earth
earth = plt.Circle((0, 0), R_EARTH, color="white", ec="black", lw=0.8, zorder=10)
ax2.add_patch(earth)
ax2.text(0, 0, "Earth", ha="center", va="center", fontsize=9, zorder=11)

ax2.set_xlabel("X  (Re)")
ax2.set_ylabel("Z  (Re)")
ax2.set_title(
    "IGRF + T96 : Relative Error Map  (X-Z meridian, Y = 0)\n"
    f"max |rel err| = {abs_max:.2e}"
)
ax2.set_aspect("equal")

fig2.tight_layout()
fig2.savefig("benchmark/readme_validation_colormap.png", dpi=300, bbox_inches="tight")
print("Saved  benchmark/readme_validation_colormap.png")

plt.show()
