"""
Usage examples from the README.

Demonstrates the six core vectorized capabilities:
  1. Coordinate transformations
  2. IGRF internal field
  3. Tsyganenko external field models
  4. Field line tracing
  5. Field line geometry (Frenet-Serret frame)
  6. Field line directional derivatives
"""

import geopack
from geopack import (
    geogsm_vectorized, igrf_gsm_vectorized, t96_vectorized, trace_vectorized,
    field_line_curvature_vectorized, field_line_frenet_frame_vectorized,
    field_line_directional_derivatives_vectorized,
)
import numpy as np

# --- Setup ---
ut = 100  # Unix timestamp (seconds since 1970-01-01)
ps = geopack.recalc(ut)

# --- 1. Coordinate Transformations ---
# Convert multiple GEO points to GSM (j=1: GEO->GSM, j=-1: GSM->GEO)
x_geo = np.array([1.0, 2.0, 3.0])
y_geo = np.array([0.5, 1.0, 1.5])
z_geo = np.array([0.0, 0.0, 0.0])

x_gsm, y_gsm, z_gsm = geogsm_vectorized(x_geo, y_geo, z_geo, j=1)
print("=== Coordinate Transformations (GEO -> GSM) ===")
for i in range(len(x_geo)):
    print(f"  GEO ({x_geo[i]}, {y_geo[i]}, {z_geo[i]}) -> GSM ({x_gsm[i]:.4f}, {y_gsm[i]:.4f}, {z_gsm[i]:.4f})")

# --- 2. IGRF Internal Field ---
# IGRF magnetic field at multiple GSM positions (Earth radii)
x = np.array([2.0, 3.0, 4.0, 5.0])
y = np.zeros(4)
z = np.zeros(4)

bx, by, bz = igrf_gsm_vectorized(x, y, z)  # returns nT
print("\n=== IGRF Internal Field (GSM) ===")
for i in range(len(x)):
    print(f"  r = {x[i]} Re -> B = ({bx[i]:.2f}, {by[i]:.2f}, {bz[i]:.2f}) nT")

# Dipole field at the same positions (accepts scalars or arrays)
dx, dy, dz = geopack.dip(x, y, z)
print("\n=== Dipole Internal Field (GSM) ===")
for i in range(len(x)):
    print(f"  r = {x[i]} Re -> B = ({dx[i]:.2f}, {dy[i]:.2f}, {dz[i]:.2f}) nT")

# --- 3. Tsyganenko External Field Models ---
# T96 parameters: [Pdyn, Dst, ByIMF, BzIMF, 0, 0, 0, 0, 0, 0]
parmod = np.array([2.0, -20.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0])

x = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
y = np.zeros(5)
z = np.zeros(5)

bx, by, bz = t96_vectorized(parmod, ps, x, y, z)  # returns nT in GSM
print("\n=== T96 External Field (GSM) ===")
for i in range(len(x)):
    print(f"  r = {x[i]} Re -> B = ({bx[i]:.2f}, {by[i]:.2f}, {bz[i]:.2f}) nT")

# --- 4. Field Line Tracing ---
# Trace using dipole (internal) + T96 (external)
x0 = np.array([5.0, 6.0, 7.0, 8.0])
y0 = np.zeros(4)
z0 = np.zeros(4)

xf, yf, zf, status = trace_vectorized(
    x0, y0, z0, dir=-1, rlim=30, parmod=parmod, exname="t96", inname="dip"
)
# status: 0 = hit inner boundary, 1 = hit outer boundary, 2 = max steps
print("\n=== Field Line Tracing (dipole + T96) ===")
status_labels = {0: "inner boundary", 1: "outer boundary", 2: "max steps"}
for i in range(len(x0)):
    print(f"  start ({x0[i]}, {y0[i]}, {z0[i]}) -> end ({xf[i]:.4f}, {yf[i]:.4f}, {zf[i]:.4f})  status: {status_labels.get(int(status[i]), '?')}")

# --- 5. Field Line Geometry (Frenet-Serret Frame) ---
# Combined model: dipole (internal) + T96 (external)
def dip_plus_t96(parmod, ps, x, y, z):
    bx_d, by_d, bz_d = geopack.dip(x, y, z)
    bx_t, by_t, bz_t = t96_vectorized(parmod, ps, x, y, z)
    return bx_d + bx_t, by_d + by_t, bz_d + bz_t

# Curvature at several points along the noon meridian
x = np.array([5.0, 6.0, 7.0, 8.0])
y = np.zeros(4)
z = np.zeros(4)

kappa = field_line_curvature_vectorized(dip_plus_t96, parmod, ps, x, y, z)
print("\n=== Field Line Curvature (dipole + T96) ===")
for i in range(len(x)):
    print(f"  r = {x[i]} Re -> curvature = {kappa[i]:.6f} 1/Re")

# Full Frenet-Serret frame (tangent, normal, binormal) + curvature
tx, ty, tz, nx, ny, nz, bnx, bny, bnz, curvature = \
    field_line_frenet_frame_vectorized(dip_plus_t96, parmod, ps, x, y, z)
print("\n=== Frenet-Serret Frame (dipole + T96) ===")
for i in range(len(x)):
    print(f"  r = {x[i]} Re -> T=({tx[i]:.4f}, {ty[i]:.4f}, {tz[i]:.4f})  "
          f"N=({nx[i]:.4f}, {ny[i]:.4f}, {nz[i]:.4f})  "
          f"B=({bnx[i]:.4f}, {bny[i]:.4f}, {bnz[i]:.4f})")

# --- 6. Field Line Directional Derivatives ---
derivs = field_line_directional_derivatives_vectorized(
    dip_plus_t96, parmod, ps, x, y, z
)
print("\n=== Field Line Directional Derivatives ===")

# All 9 components, grouped by derivative direction
labels = [
    # Tangential derivatives (∂/∂T)
    ("(∂T/∂T)·n = κ", "dT_dT_n"),
    ("(∂T/∂T)·b    ", "dT_dT_b"),
    ("(∂n/∂T)·b = τ", "dn_dT_b"),
    # Normal derivatives (∂/∂n)
    ("(∂T/∂n)·n    ", "dT_dn_n"),
    ("(∂T/∂n)·b    ", "dT_dn_b"),
    ("(∂n/∂n)·b    ", "dn_dn_b"),
    # Binormal derivatives (∂/∂b)
    ("(∂n/∂b)·b    ", "dn_db_b"),
    ("(∂n/∂b)·T    ", "dn_db_T"),
    ("(∂b/∂b)·T    ", "db_db_T"),
]
for i in range(len(x)):
    print(f"  r = {x[i]} Re:")
    for label, key in labels:
        print(f"    {label} = {derivs[key][i]:+.6f}")
    print()
