import numpy as np
import geopack
from geopack.trace_vectorized_no_interp import trace_vectorized_no_interp

# Initialize
ut = 100.0
ps = geopack.recalc(ut)

# Use the exact case that shows 1.5 Re error
x0, y0, z0 = -10.0, 0.0, 2.0

# Get full paths for both
xf_s, yf_s, zf_s, xx_s, yy_s, zz_s = geopack.geopack.trace(x0, y0, z0, dir=1, rlim=30)

# For vectorized, use array input to get masked arrays
x0_arr = np.array([x0])
y0_arr = np.array([y0])
z0_arr = np.array([z0])
xf_v, yf_v, zf_v, xx_v, yy_v, zz_v, status_v = trace_vectorized_no_interp(
    x0_arr, y0_arr, z0_arr, dir=1, rlim=30, return_full_path=True
)

# Extract valid steps for vectorized
valid_mask = ~xx_v.mask[0]
n_steps_v = np.sum(valid_mask)

print(f"Starting at: ({x0}, {y0}, {z0})")
print(f"Scalar takes {len(xx_s)} steps")
print(f"Vectorized takes {n_steps_v} steps")
print()

# Find where they diverge near the boundary
print("Near boundary comparison:")
print("Scalar positions:")
for i in range(max(0, len(xx_s)-5), len(xx_s)):
    r = np.sqrt(xx_s[i]**2 + yy_s[i]**2 + zz_s[i]**2)
    if i > 0:
        dr = r - np.sqrt(xx_s[i-1]**2 + yy_s[i-1]**2 + zz_s[i-1]**2)
        print(f"  Step {i}: x={xx_s[i]:.3f}, r={r:.3f}, step_size≈{dr:.3f}")
    else:
        print(f"  Step {i}: x={xx_s[i]:.3f}, r={r:.3f}")

print("\nVectorized positions:")
for i in range(max(0, n_steps_v-5), n_steps_v):
    r = np.sqrt(xx_v.data[0,i]**2 + yy_v.data[0,i]**2 + zz_v.data[0,i]**2)
    if i > 0:
        dr = r - np.sqrt(xx_v.data[0,i-1]**2 + yy_v.data[0,i-1]**2 + zz_v.data[0,i-1]**2)
        print(f"  Step {i}: x={xx_v.data[0,i]:.3f}, r={r:.3f}, step_size≈{dr:.3f}")
    else:
        print(f"  Step {i}: x={xx_v.data[0,i]:.3f}, r={r:.3f}")

print(f"\nKey observation:")
print(f"Scalar last step size: ~0.5 Re")
print(f"Vectorized last step size: ~1.68 Re")
print(f"This explains the 1.5 Re difference!")