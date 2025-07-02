import numpy as np
import geopack
from geopack.trace_vectorized_no_interp import trace_vectorized_no_interp

# Initialize
ut = 100.0
ps = geopack.recalc(ut)

# Test the tail region case with arrays to get masked arrays
x0 = np.array([-10.0, -20.0])
y0 = np.array([0.0, 0.0])
z0 = np.array([2.0, 2.0])

# Trace with full path
xf_v, yf_v, zf_v, xx_v, yy_v, zz_v, status_v = trace_vectorized_no_interp(
    x0, y0, z0, dir=1, rlim=30, return_full_path=True
)

# Also trace scalar for comparison
xf_s1, yf_s1, zf_s1, xx_s1, yy_s1, zz_s1 = geopack.geopack.trace(x0[0], y0[0], z0[0], dir=1, rlim=30)

# Calculate radii
r_s1 = np.sqrt(xx_s1**2 + yy_s1**2 + zz_s1**2)

print("Case 1: Start at (-10, 0, 2)")
print("Scalar trace - last 5 steps:")
for i in range(max(0, len(r_s1)-5), len(r_s1)):
    print(f"  Step {i}: r={r_s1[i]:.3f}, x={xx_s1[i]:.3f}")

print(f"\nScalar final: r={r_s1[-1]:.3f}, x={xx_s1[-1]:.3f}")

# For vectorized, find valid steps
valid_mask = ~xx_v.mask[0]
n_steps = np.sum(valid_mask)
print(f"\nVectorized trace - last 5 steps (total {n_steps} steps):")
for i in range(max(0, n_steps-5), n_steps):
    r = np.sqrt(xx_v.data[0,i]**2 + yy_v.data[0,i]**2 + zz_v.data[0,i]**2)
    print(f"  Step {i}: r={r:.3f}, x={xx_v.data[0,i]:.3f}")

r_final = np.sqrt(xf_v[0]**2 + yf_v[0]**2 + zf_v[0]**2)
print(f"\nVectorized final: r={r_final:.3f}, x={xf_v[0]:.3f}")

print(f"\nBoundary = 30.0 Re")
print(f"Scalar stops at r={r_s1[-1]:.3f} (slightly past boundary)")
print(f"Vectorized stops at r={r_final:.3f} (far past boundary)")
print(f"Difference: {r_final - r_s1[-1]:.3f} Re")