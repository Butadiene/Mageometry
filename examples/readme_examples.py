"""
Usage examples from the README.

Runs the numerical README examples without plotting or simulation files:
  1. Coordinate transformations
  2. IGRF internal field
  3. Tsyganenko external field models
  4. Field line tracing
  5. Field line geometry (Frenet-Serret frame)
  6. Field line directional derivatives
  7. Field-magnitude gradients and current density
  8. Transverse rotation and shear
"""

from mageometry import geopack
from mageometry import (
    geopack_field,
    trace_field_lines,
    field_line_curvature, field_line_frenet_frame,
    field_line_directional_derivatives,
    field_magnitude_derivatives, field_line_current_density,
    field_line_transverse_geometry,
)
from mageometry.geopack import (
    geogsm_vectorized, igrf_gsm_vectorized, t96_vectorized,
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

x = np.array([5.0, 6.0, 7.0, 8.0, 9.0])  # GSM coordinates (Re)
y = np.zeros(5)
z = np.zeros(5)

bx, by, bz = t96_vectorized(parmod, ps, x, y, z)  # GSM components (nT)
print("\n=== T96 External Field (GSM) ===")
for i in range(len(x)):
    print(f"  r = {x[i]} Re -> B = ({bx[i]:.2f}, {by[i]:.2f}, {bz[i]:.2f}) nT")

# Tsyganenko models give only the external (magnetospheric) field.
# Add an internal field to get the total magnetic field:
bx_int, by_int, bz_int = geopack.dip(x, y, z)
bx_total = bx + bx_int
by_total = by + by_int
bz_total = bz + bz_int
print("\n=== Total Field: dipole (internal) + T96 (external) ===")
for i in range(len(x)):
    print(f"  r = {x[i]} Re -> B = ({bx_total[i]:.2f}, {by_total[i]:.2f}, {bz_total[i]:.2f}) nT")

# --- 4. Field Line Tracing ---
# Trace using dipole (internal) + T96 (external)
x0 = np.array([5.0, 6.0, 7.0, 8.0])
y0 = np.zeros(4)
z0 = np.zeros(4)

field = geopack_field('t96', 'dip', parmod, ps)
tr = trace_field_lines(
    field, x0, y0, z0, direction='both', ds=0.1, r0=1.0, rlim=30.0
)
# status: +B end; status_backward: -B end.
xf, yf, zf = tr.end
print("\n=== Field Line Tracing (dipole + T96) ===")
status_labels = {0: "inner boundary", 1: "outer boundary", 2: "max steps",
                 3: "undefined field", 4: "custom stop"}
for i in range(len(x0)):
    print(f"  seed ({x0[i]}, {y0[i]}, {z0[i]}) -> +B end "
          f"({xf[i]:.4f}, {yf[i]:.4f}, {zf[i]:.4f}); "
          f"+B: {status_labels[int(tr.status[i])]}, "
          f"-B: {status_labels[int(tr.status_backward[i])]}")
x_line, y_line, z_line = tr.path(0)
s = tr.arc_length(0)
kappa_line = field_line_curvature(field, x_line, y_line, z_line)
print(f"  First line: {len(s)} points; seed arc length = {s[tr.start_index[0]]:.1f} Re")

# --- 5. Field Line Geometry (Frenet-Serret Frame) ---
# Geometry functions take any callable field(x, y, z) -> (bx, by, bz);
# geopack_field wraps the geopack models into that form.
field = geopack_field(external='t96', internal='dip', parmod=parmod, ps=ps)

# Curvature at several points along the noon meridian
x = np.array([5.0, 6.0, 7.0, 8.0])
y = np.zeros(4)
z = np.zeros(4)

kappa = field_line_curvature(field, x, y, z, delta=1e-3)
# kappa: field line curvature [1/Re]
print("\n=== Field Line Curvature (dipole + T96) ===")
for i in range(len(x)):
    print(f"  r = {x[i]} Re -> curvature = {kappa[i]:.6f} 1/Re")

# Full Frenet-Serret frame (tangent, normal, binormal) + curvature
tx, ty, tz, nx, ny, nz, bnx, bny, bnz, curvature = \
    field_line_frenet_frame(field, x, y, z, delta=1e-3)
# curvature [1/Re]; tangent, normal, binormal are unit vectors (dimensionless)
print("\n=== Frenet-Serret Frame (dipole + T96) ===")
for i in range(len(x)):
    print(f"  r = {x[i]} Re -> T=({tx[i]:.4f}, {ty[i]:.4f}, {tz[i]:.4f})  "
          f"N=({nx[i]:.4f}, {ny[i]:.4f}, {nz[i]:.4f})  "
          f"B=({bnx[i]:.4f}, {bny[i]:.4f}, {bnz[i]:.4f})")

# --- 6. Field Line Directional Derivatives ---
derivs = field_line_directional_derivatives(
    field, x, y, z, delta=1e-3
)
# All derivative values are in units of [1/Re]
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

# --- 7. Field-Magnitude Gradients and Current Density ---
mag = field_magnitude_derivatives(field, x, y, z, delta=1e-3)
cur = field_line_current_density(field, x, y, z, delta=1e-3)
print("=== Field-Magnitude Gradients and Current Density ===")
for i in range(len(x)):
    print(f"  r = {x[i]} Re: |B| = {mag['B'][i]:.3f} nT; "
          f"mu0J (T, n, b) = ({cur['mu0J_T'][i]:+.6f}, "
          f"{cur['mu0J_n'][i]:+.6f}, {cur['mu0J_b'][i]:+.6f}) nT/Re")

# --- 8. Transverse Rotation and Shear ---
rates = field_line_transverse_geometry(field, x, y, z, delta=1e-3,
                                      curvature_tol=1e-8)
print("\n=== Transverse Geometry (1/Re) ===")
for i in range(len(x)):
    print(f"  r = {x[i]} Re: " + ', '.join(
        f"{key} = {rates[key][i]:+.6f}"
        for key in ('alpha', 'sigma', 'q', 'gamma', 'omega_c')))
