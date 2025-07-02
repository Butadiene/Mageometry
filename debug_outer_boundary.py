import numpy as np
import geopack
from geopack.trace_vectorized_no_interp import trace_vectorized_no_interp

# Initialize
ut = 100.0
ps = geopack.recalc(ut)

# Test the tail region case
x0, y0, z0 = -10.0, 0.0, 2.0

# Trace with full path
xf_s, yf_s, zf_s, xx_s, yy_s, zz_s = geopack.geopack.trace(x0, y0, z0, dir=1, rlim=30)
xf_v, yf_v, zf_v, xx_v, yy_v, zz_v, status_v = trace_vectorized_no_interp(
    x0, y0, z0, dir=1, rlim=30, return_full_path=True
)

# Calculate radii along paths
r_s = np.sqrt(xx_s**2 + yy_s**2 + zz_s**2)
# Convert masked array to regular array for vectorized
xx_v_data = np.ma.filled(xx_v, np.nan)
yy_v_data = np.ma.filled(yy_v, np.nan)
zz_v_data = np.ma.filled(zz_v, np.nan)
r_v = np.sqrt(xx_v_data**2 + yy_v_data**2 + zz_v_data**2)

# Find where mask starts (where tracing stopped)
valid_mask = ~xx_v.mask[0]
n_steps_v = np.sum(valid_mask)
r_v_valid = r_v[0, :n_steps_v]

print("Scalar trace - last 5 steps:")
for i in range(max(0, len(r_s)-5), len(r_s)):
    print(f"  Step {i}: r={r_s[i]:.3f}, x={xx_s[i]:.3f}")

print(f"\nScalar final: r={r_s[-1]:.3f}")

print("\nVectorized trace - last 5 steps:")
for i in range(max(0, n_steps_v-5), n_steps_v):
    print(f"  Step {i}: r={r_v_valid[i]:.3f}, x={xx_v_data[0,i]:.3f}")

print(f"\nVectorized final: r={r_v_valid[-1]:.3f}")

# Check if vectorized is checking boundary after step
print(f"\nBoundary = 30.0 Re")
print(f"Scalar stops at r={r_s[-1]:.3f} (before crossing)")
print(f"Vectorized stops at r={r_v_valid[-1]:.3f} (after crossing?)")