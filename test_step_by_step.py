import numpy as np
import geopack

# Initialize
ut = 100.0
ps = geopack.recalc(ut)

# Add some debug prints to trace_vectorized_no_interp
import geopack.trace_vectorized_no_interp as tvni

# Monkey patch to add debug
original_trace = tvni.trace_vectorized_no_interp

def debug_trace(xi, yi, zi, **kwargs):
    print(f"Starting trace from ({xi}, {yi}, {zi})")
    
    # Call original with modifications
    result = original_trace(xi, yi, zi, **kwargs)
    return result

# Temporarily replace
tvni.trace_vectorized_no_interp = debug_trace

# Now trace
x0, y0, z0 = -28.0, 0.0, 1.7
print(f"Testing from ({x0}, {y0}, {z0}), r={np.sqrt(x0**2+y0**2+z0**2):.3f}")

# Import the function directly to use original
from geopack.trace_vectorized_no_interp import trace_vectorized_no_interp as original_func

# Test
xf_v, yf_v, zf_v, status_v = original_func(x0, y0, z0, dir=1, rlim=30)
r_v = np.sqrt(xf_v**2 + yf_v**2 + zf_v**2)

print(f"\nResult: ({xf_v:.3f}, {yf_v:.3f}, {zf_v:.3f})")
print(f"Final radius: {r_v:.3f}")
print(f"Status: {status_v}")

# Also test scalar
xf_s, yf_s, zf_s, _, _, _ = geopack.geopack.trace(x0, y0, z0, dir=1, rlim=30)
r_s = np.sqrt(xf_s**2 + yf_s**2 + zf_s**2)
print(f"\nScalar result: ({xf_s:.3f}, {yf_s:.3f}, {zf_s:.3f})")
print(f"Scalar radius: {r_s:.3f}")