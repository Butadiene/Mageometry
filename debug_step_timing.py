import numpy as np
import geopack
from geopack.trace_vectorized_no_interp import trace_vectorized_no_interp

# Initialize
ut = 100.0
ps = geopack.recalc(ut)

# Monkey patch to add debug
import geopack.trace_vectorized_no_interp as tvni

# Store original
original_step = tvni.step_vectorized

def debug_step(x, y, z, ds_array, errin, parmod, exname, inname, active_mask, status, iteration_count):
    print(f"step_vectorized called with {np.sum(active_mask)} active traces")
    print(f"  Positions before: x[0]={x[0]:.3f}, active={active_mask[0]}")
    result = original_step(x, y, z, ds_array, errin, parmod, exname, inname, active_mask, status, iteration_count)
    print(f"  Positions after: x[0]={result[0][0]:.3f}")
    return result

# Replace temporarily
tvni.step_vectorized = debug_step

# Test with array input to trace single field line
x0 = np.array([-29.0])
y0 = np.array([0.0])
z0 = np.array([1.7])

print(f"Starting at r={np.sqrt(x0[0]**2+y0[0]**2+z0[0]**2):.3f}")
print("Boundary at r=30.0")
print()

# Trace
xf_v, yf_v, zf_v, status_v = trace_vectorized_no_interp(x0, y0, z0, dir=1, rlim=30)

print(f"\nFinal position: ({xf_v[0]:.3f}, {yf_v[0]:.3f}, {zf_v[0]:.3f})")
print(f"Final radius: {np.sqrt(xf_v[0]**2+yf_v[0]**2+zf_v[0]**2):.3f}")
print(f"Status: {status_v[0]}")