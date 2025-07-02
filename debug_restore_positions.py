import numpy as np
import geopack

# Patch the trace function to debug
import geopack.trace_vectorized_no_interp as tvni

# Store original
original_trace = tvni.trace_vectorized_no_interp

def debug_trace(*args, **kwargs):
    # Add debug flag
    tvni._debug = True
    result = original_trace(*args, **kwargs)
    tvni._debug = False
    return result

# Replace
tvni.trace_vectorized_no_interp = debug_trace

# Also patch the main loop
original_code = tvni.trace_vectorized_no_interp
exec(open('/home/skipjack/Documents/geopack-vectorize/geopack/trace_vectorized_no_interp.py').read())

# Initialize
ut = 100.0
ps = geopack.recalc(ut)

# Test
x0, y0, z0 = -29.5, 0.0, 1.66

# Compare with scalar
xf_s, yf_s, zf_s, _, _, _ = geopack.geopack.trace(x0, y0, z0, dir=1, rlim=30)
r_s = np.sqrt(xf_s**2 + yf_s**2 + zf_s**2)

print(f"Start: ({x0}, {y0}, {z0}), r={np.sqrt(x0**2+y0**2+z0**2):.3f}")
print(f"Scalar result: x={xf_s:.3f}, r={r_s:.3f}")
print()

# I need to understand what the scalar version does when it hits the boundary
# Let me trace with full path
xf_s, yf_s, zf_s, xx_s, yy_s, zz_s = geopack.geopack.trace(x0, y0, z0, dir=1, rlim=30)

print("Scalar trace last 3 positions:")
for i in range(max(0, len(xx_s)-3), len(xx_s)):
    r = np.sqrt(xx_s[i]**2 + yy_s[i]**2 + zz_s[i]**2)
    print(f"  Step {i}: x={xx_s[i]:.3f}, r={r:.3f}")
    
print(f"\nScalar returns position at step {len(xx_s)-1}")