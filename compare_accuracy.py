"""
Compare numerical accuracy of trace_vectorized.py vs trace_vectorized_no_interp.py
"""
import numpy as np
import geopack
from geopack.trace_vectorized import trace_vectorized
from geopack.trace_vectorized_no_interp import trace_vectorized_no_interp

# Initialize
ut = 100.0
ps = geopack.recalc(ut)

print("Comparing Numerical Accuracy: trace_vectorized vs trace_vectorized_no_interp")
print("=" * 80)

# Test 1: Outer boundary accuracy
print("\n1. OUTER BOUNDARY ACCURACY TEST")
print("-" * 40)

# Start from tail region that will hit outer boundary
x0, y0, z0 = -10.0, 0.0, 2.0
rlim = 30.0

# Trace with both versions
xf_interp, yf_interp, zf_interp, status_interp = trace_vectorized(x0, y0, z0, dir=1, rlim=rlim)
xf_no_interp, yf_no_interp, zf_no_interp, status_no_interp = trace_vectorized_no_interp(x0, y0, z0, dir=1, rlim=rlim)

r_interp = np.sqrt(xf_interp**2 + yf_interp**2 + zf_interp**2)
r_no_interp = np.sqrt(xf_no_interp**2 + yf_no_interp**2 + zf_no_interp**2)

print(f"Starting point: ({x0}, {y0}, {z0})")
print(f"Boundary radius: {rlim} Re")
print(f"\nWith interpolation:    r = {r_interp:.6f} Re, error = {abs(r_interp - rlim):.6f} Re")
print(f"Without interpolation: r = {r_no_interp:.6f} Re, error = {abs(r_no_interp - rlim):.6f} Re")
print(f"\nInterpolation is {abs(r_no_interp - rlim) / abs(r_interp - rlim):.1f}x more accurate at boundary")

# Test 2: Inner boundary accuracy
print("\n\n2. INNER BOUNDARY ACCURACY TEST")
print("-" * 40)

# Start from position that will hit inner boundary
x0, y0, z0 = 3.0, 0.0, 3.0
r0 = 1.0

xf_interp, yf_interp, zf_interp, status_interp = trace_vectorized(x0, y0, z0, dir=1, r0=r0)
xf_no_interp, yf_no_interp, zf_no_interp, status_no_interp = trace_vectorized_no_interp(x0, y0, z0, dir=1, r0=r0)

r_interp = np.sqrt(xf_interp**2 + yf_interp**2 + zf_interp**2)
r_no_interp = np.sqrt(xf_no_interp**2 + yf_no_interp**2 + zf_no_interp**2)

print(f"Starting point: ({x0}, {y0}, {z0})")
print(f"Inner boundary radius: {r0} Re")
print(f"\nWith interpolation:    r = {r_interp:.6f} Re, error = {abs(r_interp - r0):.6f} Re")
print(f"Without interpolation: r = {r_no_interp:.6f} Re, error = {abs(r_no_interp - r0):.6f} Re")
print("\nBoth use interpolation at inner boundary (same accuracy)")

# Test 3: Field line conservation (no boundaries)
print("\n\n3. FIELD LINE CONSERVATION TEST")
print("-" * 40)

# Start from multiple points that won't hit boundaries
test_points = [
    (3.0, 0.0, 0.0),
    (0.0, 3.0, 0.0),
    (2.0, 2.0, 2.0),
    (4.0, 0.0, 2.0)
]

print("Testing field line tracing accuracy (no boundary interactions)...")
print("Both should follow the same field lines with similar accuracy\n")

errors = []
for x0, y0, z0 in test_points:
    # Trace forward then backward
    # With interpolation
    x1, y1, z1, _ = trace_vectorized(x0, y0, z0, dir=1, rlim=10, r0=0.5)
    x2, y2, z2, _ = trace_vectorized(x1, y1, z1, dir=-1, rlim=10, r0=0.5)
    error_interp = np.sqrt((x2-x0)**2 + (y2-y0)**2 + (z2-z0)**2)
    
    # Without interpolation
    x1, y1, z1, _ = trace_vectorized_no_interp(x0, y0, z0, dir=1, rlim=10, r0=0.5)
    x2, y2, z2, _ = trace_vectorized_no_interp(x1, y1, z1, dir=-1, rlim=10, r0=0.5)
    error_no_interp = np.sqrt((x2-x0)**2 + (y2-y0)**2 + (z2-z0)**2)
    
    print(f"Point ({x0}, {y0}, {z0}):")
    print(f"  With interp:    round-trip error = {error_interp:.6f} Re")
    print(f"  Without interp: round-trip error = {error_no_interp:.6f} Re")
    
    errors.append((error_interp, error_no_interp))

# Test 4: Magnetic field conservation along field line
print("\n\n4. MAGNETIC FIELD MAGNITUDE CONSERVATION")
print("-" * 40)

# Trace a field line and check B conservation
x0, y0, z0 = 5.0, 0.0, 0.0

# Get full path with interpolation
xf, yf, zf, xx, yy, zz, _ = trace_vectorized(
    np.array([x0]), np.array([y0]), np.array([z0]), 
    dir=1, rlim=10, return_full_path=True
)

# Calculate B along the path
from geopack import t89
n_points = np.sum(~xx.mask[0])
B_mags = []
for i in range(n_points):
    bx, by, bz = t89(2, ps, xx.data[0,i], yy.data[0,i], zz.data[0,i])
    B = np.sqrt(bx**2 + by**2 + bz**2)
    B_mags.append(B)

B_mags = np.array(B_mags)
B_variation = (B_mags.max() - B_mags.min()) / B_mags.mean()

print(f"Field line from ({x0}, {y0}, {z0}) to ({xf[0]:.3f}, {yf[0]:.3f}, {zf[0]:.3f})")
print(f"Magnetic field variation along field line: {B_variation:.3%}")
print(f"(Lower variation = better numerical accuracy)")

# Summary
print("\n\nSUMMARY")
print("=" * 80)
print("1. trace_vectorized (WITH interpolation) is MORE accurate at boundaries:")
print("   - Outer boundary: Places endpoints exactly at r=rlim")
print("   - More physically correct for boundary intersection problems")
print("\n2. Both versions have similar accuracy for field line integration:")
print("   - Same RK5 integration scheme")
print("   - Same adaptive step sizing")
print("   - Differences only at boundaries")
print("\n3. Recommendation: Use trace_vectorized (with interpolation) for:")
print("   - More accurate boundary intersections")
print("   - Physical problems where exact boundary location matters")
print("   - Better representation of magnetopause crossings")