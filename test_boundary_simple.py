import numpy as np
import geopack
from geopack.trace_vectorized_no_interp import trace_vectorized_no_interp

# Initialize
ut = 100.0
ps = geopack.recalc(ut)

# Test a single trace that will hit the boundary
x0, y0, z0 = -29.5, 0.0, 1.662  # Start very close to boundary

print(f"Starting at: ({x0}, {y0}, {z0})")
print(f"Initial radius: {np.sqrt(x0**2 + y0**2 + z0**2):.3f}")
print(f"Boundary: 30.0 Re")
print()

# Scalar trace
xf_s, yf_s, zf_s, _, _, _ = geopack.geopack.trace(x0, y0, z0, dir=1, rlim=30)
r_s = np.sqrt(xf_s**2 + yf_s**2 + zf_s**2)
print(f"Scalar result: ({xf_s:.3f}, {yf_s:.3f}, {zf_s:.3f})")
print(f"Scalar radius: {r_s:.3f}")
print()

# Vectorized trace
xf_v, yf_v, zf_v, status_v = trace_vectorized_no_interp(x0, y0, z0, dir=1, rlim=30)
r_v = np.sqrt(xf_v**2 + yf_v**2 + zf_v**2)
print(f"Vector result: ({xf_v:.3f}, {yf_v:.3f}, {zf_v:.3f})")
print(f"Vector radius: {r_v:.3f}")
print(f"Vector status: {status_v}")
print()

print(f"Difference in radius: {r_v - r_s:.3f} Re")