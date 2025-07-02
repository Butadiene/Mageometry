import numpy as np
import geopack
from geopack.trace_vectorized_no_interp_fixed import trace_vectorized_no_interp_fixed as trace_vectorized_no_interp

# Initialize
ut = 100.0
ps = geopack.recalc(ut)

# Test the tail region case that had large error
x0, y0, z0 = -10.0, 0.0, 2.0

# Scalar trace
xf_s, yf_s, zf_s, _, _, _ = geopack.geopack.trace(x0, y0, z0, dir=1, rlim=30)
r_s = np.sqrt(xf_s**2 + yf_s**2 + zf_s**2)

# Vectorized trace (fixed version)
xf_v, yf_v, zf_v, status_v = trace_vectorized_no_interp(x0, y0, z0, dir=1, rlim=30)
r_v = np.sqrt(xf_v**2 + yf_v**2 + zf_v**2)

# Calculate error
dr = np.sqrt((xf_v-xf_s)**2 + (yf_v-yf_s)**2 + (zf_v-zf_s)**2)

print(f'Tail region test: ({x0}, {y0}, {z0})')
print(f'Scalar endpoint: ({xf_s:.3f}, {yf_s:.3f}, {zf_s:.3f}), r={r_s:.3f}')
print(f'Vector endpoint: ({xf_v:.3f}, {yf_v:.3f}, {zf_v:.3f}), r={r_v:.3f}')
print(f'Error: {dr:.3f} Re')
print(f'Status: {status_v}')

# Test multiple cases
print("\n" + "="*60 + "\n")

test_cases = [
    (5.0, 0.0, 0.0, "Equatorial, noon"),
    (-10.0, 0.0, 2.0, "Tail region"),
    (-20.0, 0.0, 2.0, "Far tail"),
    (3.0, 0.0, 3.0, "High latitude"),
    (1.5, 0.0, 0.0, "Near Earth")
]

for x, y, z, desc in test_cases:
    # Scalar
    xf_s, yf_s, zf_s, _, _, _ = geopack.geopack.trace(x, y, z, dir=1, rlim=30)
    r_s = np.sqrt(xf_s**2 + yf_s**2 + zf_s**2)
    
    # Vectorized
    xf_v, yf_v, zf_v, status_v = trace_vectorized_no_interp(x, y, z, dir=1, rlim=30)
    r_v = np.sqrt(xf_v**2 + yf_v**2 + zf_v**2)
    
    # Error
    dr = np.sqrt((xf_v-xf_s)**2 + (yf_v-yf_s)**2 + (zf_v-zf_s)**2)
    
    print(f"{desc}: error={dr:.3e} Re, scalar r={r_s:.3f}, vector r={r_v:.3f}")