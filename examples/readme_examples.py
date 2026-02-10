"""
Usage examples from the README.

Demonstrates the four core vectorized capabilities:
  1. Coordinate transformations
  2. IGRF internal field
  3. Tsyganenko external field models
  4. Field line tracing
"""

import geopack
from geopack import geogsm_vectorized, igrf_gsm_vectorized, t96_vectorized, trace_vectorized
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
# Trace multiple field lines simultaneously
x0 = np.array([5.0, 6.0, 7.0, 8.0])
y0 = np.zeros(4)
z0 = np.zeros(4)

xf, yf, zf, status = trace_vectorized(x0, y0, z0, dir=-1, rlim=30)
# status: 0 = hit inner boundary, 1 = hit outer boundary, 2 = max steps
print("\n=== Field Line Tracing ===")
status_labels = {0: "inner boundary", 1: "outer boundary", 2: "max steps"}
for i in range(len(x0)):
    print(f"  start ({x0[i]}, {y0[i]}, {z0[i]}) -> end ({xf[i]:.4f}, {yf[i]:.4f}, {zf[i]:.4f})  status: {status_labels.get(int(status[i]), '?')}")
