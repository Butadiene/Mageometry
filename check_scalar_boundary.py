import numpy as np
import geopack

# Initialize
ut = 100.0
ps = geopack.recalc(ut)

# Start very close to boundary
x0, y0, z0 = -29.7, 0.0, 1.66
r0 = np.sqrt(x0**2 + y0**2 + z0**2)

print(f"Starting at: ({x0}, {y0}, {z0}), r={r0:.3f}")
print(f"Boundary: r=30.0")
print()

# Trace with full path
xf_s, yf_s, zf_s, xx_s, yy_s, zz_s = geopack.geopack.trace(x0, y0, z0, dir=1, rlim=30)

print("Scalar trace positions:")
for i in range(len(xx_s)):
    r = np.sqrt(xx_s[i]**2 + yy_s[i]**2 + zz_s[i]**2)
    print(f"  Step {i}: x={xx_s[i]:.3f}, r={r:.3f}")

rf = np.sqrt(xf_s**2 + yf_s**2 + zf_s**2)
print(f"\nFinal returned: x={xf_s:.3f}, r={rf:.3f}")
print(f"This is position from step {len(xx_s)-1}")

# Key insight: Does scalar return the position BEFORE or AFTER crossing boundary?
if rf > 30.0:
    print("\nScalar returns position AFTER crossing boundary")
else:
    print("\nScalar returns position BEFORE crossing boundary")