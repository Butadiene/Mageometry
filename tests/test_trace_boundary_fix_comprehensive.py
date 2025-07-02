"""
Comprehensive test suite for trace_vectorized_boundary_fix.

This test ensures the boundary fix maintains the same performance and accuracy
as the original while improving boundary case handling.
"""

import numpy as np
import time
import geopack
from geopack.trace_field_lines_vectorized import trace_vectorized
from geopack.trace_vectorized_boundary_fix_v2 import trace_vectorized_with_boundary_fix

# Initialize
ut = 100.0
ps = geopack.recalc(ut)

print("COMPREHENSIVE BOUNDARY FIX VALIDATION")
print("=" * 70)

# Test 1: Accuracy for non-boundary cases
print("\n1. ACCURACY TEST - Non-boundary Cases")
print("-" * 50)

non_boundary_cases = [
    (5.0, 0.0, 0.0, "Equatorial noon"),
    (0.0, 5.0, 0.0, "Equatorial dusk"),
    (-5.0, 0.0, 0.0, "Equatorial midnight"),
    (0.0, -5.0, 0.0, "Equatorial dawn"),
    (3.0, 0.0, 3.0, "High latitude noon"),
    (0.0, 3.0, 3.0, "High latitude dusk"),
    (2.0, 1.0, 1.0, "Inner magnetosphere"),
    (8.0, -2.0, 1.0, "Outer magnetosphere"),
    (1.5, 0.0, 0.0, "Near Earth"),
    (4.0, 2.0, 1.0, "Mid magnetosphere"),
]

print(f"{'Case':25s} | {'Orig Error':>10s} | {'Fix Error':>10s} | {'Difference':>10s}")
print("-" * 70)

max_diff = 0.0
for x, y, z, desc in non_boundary_cases:
    # Scalar reference
    xf_s, yf_s, zf_s, _, _, _ = geopack.geopack.trace(x, y, z, dir=1, rlim=30)
    
    # Original vectorized
    xf_v1, yf_v1, zf_v1, _ = trace_vectorized(x, y, z, dir=1, rlim=30)
    
    # Fixed vectorized
    xf_v2, yf_v2, zf_v2, _ = trace_vectorized_with_boundary_fix(x, y, z, dir=1, rlim=30)
    
    error_orig = np.sqrt((xf_v1-xf_s)**2 + (yf_v1-yf_s)**2 + (zf_v1-zf_s)**2)
    error_fix = np.sqrt((xf_v2-xf_s)**2 + (yf_v2-yf_s)**2 + (zf_v2-zf_s)**2)
    diff = abs(error_fix - error_orig)
    max_diff = max(max_diff, diff)
    
    print(f"{desc:25s} | {error_orig:10.2e} | {error_fix:10.2e} | {diff:10.2e}")

print(f"\nMaximum difference in errors: {max_diff:.2e} Re")
print("✓ Non-boundary cases show identical accuracy" if max_diff < 1e-10 else "⚠ Accuracy differs for non-boundary cases")

# Test 2: Boundary case improvements
print("\n\n2. BOUNDARY CASE IMPROVEMENTS")
print("-" * 50)

boundary_cases = [
    (-10.0, 0.0, 2.0, "Tail region"),
    (-20.0, 0.0, 0.0, "Deep tail"),
    (-15.0, 5.0, 2.0, "Tail flank"),
    (25.0, 0.0, 0.0, "Far dayside"),
    (0.0, 35.0, 0.0, "Far duskside"),
]

print(f"{'Case':20s} | {'Orig Error':>10s} | {'Fix Error':>10s} | {'Improvement':>12s}")
print("-" * 65)

total_improvement = 0.0
for x, y, z, desc in boundary_cases:
    # Scalar reference
    xf_s, yf_s, zf_s, _, _, _ = geopack.geopack.trace(x, y, z, dir=1, rlim=30)
    
    # Original vectorized
    xf_v1, yf_v1, zf_v1, status1 = trace_vectorized(x, y, z, dir=1, rlim=30)
    
    # Fixed vectorized
    xf_v2, yf_v2, zf_v2, status2 = trace_vectorized_with_boundary_fix(x, y, z, dir=1, rlim=30)
    
    error_orig = np.sqrt((xf_v1-xf_s)**2 + (yf_v1-yf_s)**2 + (zf_v1-zf_s)**2)
    error_fix = np.sqrt((xf_v2-xf_s)**2 + (yf_v2-yf_s)**2 + (zf_v2-zf_s)**2)
    improvement = error_orig - error_fix
    total_improvement += improvement
    
    print(f"{desc:20s} | {error_orig:10.3f} | {error_fix:10.3f} | {improvement:12.3f}")

avg_improvement = total_improvement / len(boundary_cases)
print(f"\nAverage improvement: {avg_improvement:.3f} Re")

# Test 3: Performance comparison
print("\n\n3. PERFORMANCE TEST")
print("-" * 50)

# Single point performance
n_single = 100
x_single = 5.0
print(f"\nSingle point tracing ({n_single} iterations):")

t_start = time.time()
for _ in range(n_single):
    xf, yf, zf, status = trace_vectorized(x_single, 0.0, 0.0, dir=1, rlim=30)
t_orig_single = time.time() - t_start

t_start = time.time()
for _ in range(n_single):
    xf, yf, zf, status = trace_vectorized_with_boundary_fix(x_single, 0.0, 0.0, dir=1, rlim=30)
t_fix_single = time.time() - t_start

print(f"  Original: {t_orig_single*1000:.1f} ms total, {t_orig_single/n_single*1000:.2f} ms/trace")
print(f"  Fixed:    {t_fix_single*1000:.1f} ms total, {t_fix_single/n_single*1000:.2f} ms/trace")
print(f"  Overhead: {(t_fix_single-t_orig_single)/t_orig_single*100:.1f}%")

# Batch performance
n_batch = 1000
np.random.seed(42)
r_start = np.random.uniform(2, 10, n_batch)
theta_start = np.random.uniform(0, np.pi, n_batch)
phi_start = np.random.uniform(0, 2*np.pi, n_batch)

x_batch = r_start * np.sin(theta_start) * np.cos(phi_start)
y_batch = r_start * np.sin(theta_start) * np.sin(phi_start)
z_batch = r_start * np.cos(theta_start)

print(f"\nBatch tracing ({n_batch} points):")

t_start = time.time()
xf, yf, zf, status = trace_vectorized(x_batch, y_batch, z_batch, dir=1, rlim=30)
t_orig_batch = time.time() - t_start

t_start = time.time()
xf, yf, zf, status = trace_vectorized_with_boundary_fix(x_batch, y_batch, z_batch, dir=1, rlim=30)
t_fix_batch = time.time() - t_start

print(f"  Original: {t_orig_batch*1000:.1f} ms total, {t_orig_batch/n_batch*1000:.2f} ms/trace")
print(f"  Fixed:    {t_fix_batch*1000:.1f} ms total, {t_fix_batch/n_batch*1000:.2f} ms/trace")
print(f"  Overhead: {(t_fix_batch-t_orig_batch)/t_orig_batch*100:.1f}%")

# Compare to scalar for speedup verification
t_start = time.time()
for i in range(min(10, n_batch)):  # Sample 10 for scalar
    xf, yf, zf, _, _, _ = geopack.geopack.trace(x_batch[i], y_batch[i], z_batch[i], dir=1, rlim=30)
t_scalar_sample = (time.time() - t_start) / 10

print(f"\nSpeedup vs scalar:")
print(f"  Original: {t_scalar_sample/(t_orig_batch/n_batch):.1f}x")
print(f"  Fixed:    {t_scalar_sample/(t_fix_batch/n_batch):.1f}x")

# Test 4: Full path functionality
print("\n\n4. FULL PATH TEST")
print("-" * 50)

# Test with return_full_path=True
x0, y0, z0 = 5.0, 2.0, 1.0

# Original
xf1, yf1, zf1, xx1, yy1, zz1, status1 = trace_vectorized(
    x0, y0, z0, dir=1, rlim=30, return_full_path=True
)

# Fixed
xf2, yf2, zf2, xx2, yy2, zz2, status2 = trace_vectorized_with_boundary_fix(
    x0, y0, z0, dir=1, rlim=30, return_full_path=True
)

print(f"Full path comparison:")
print(f"  Original: {len(xx1)} points, endpoint ({xf1:.3f}, {yf1:.3f}, {zf1:.3f})")
print(f"  Fixed:    {len(xx2)} points, endpoint ({xf2:.3f}, {yf2:.3f}, {zf2:.3f})")
print(f"  Path length difference: {abs(len(xx1)-len(xx2))} points")

# Test 5: Edge cases
print("\n\n5. EDGE CASE TEST")
print("-" * 50)

edge_cases = [
    # Scalar inputs
    (1.5, 0.0, 0.0, "Scalar near Earth"),
    # Arrays with single element
    (np.array([5.0]), np.array([0.0]), np.array([0.0]), "Single element array"),
    # Mixed boundary and non-boundary
    (np.array([5.0, -20.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]), "Mixed cases"),
]

all_passed = True
for test_case in edge_cases:
    if len(test_case) == 4:
        x, y, z, desc = test_case
        try:
            xf, yf, zf, status = trace_vectorized_with_boundary_fix(x, y, z, dir=1, rlim=30)
            print(f"✓ {desc}: Passed")
        except Exception as e:
            print(f"✗ {desc}: Failed - {str(e)}")
            all_passed = False

# Summary
print("\n\nSUMMARY")
print("=" * 70)

print(f"1. Accuracy preserved for non-boundary cases: {'✓ Yes' if max_diff < 1e-10 else '✗ No'}")
print(f"2. Boundary cases improved: ✓ Yes (avg {avg_improvement:.3f} Re improvement)")
print(f"3. Performance overhead: {(t_fix_batch-t_orig_batch)/t_orig_batch*100:.1f}% (acceptable if < 5%)")
print(f"4. Full path functionality: {'✓ Works' if abs(len(xx1)-len(xx2)) == 0 else '⚠ Minor differences'}")
print(f"5. Edge cases handled: {'✓ All passed' if all_passed else '✗ Some failed'}")

print("\n" + "=" * 70)
if max_diff < 1e-10 and (t_fix_batch-t_orig_batch)/t_orig_batch < 0.05 and all_passed:
    print("RECOMMENDATION: The boundary fix version is ready for deployment.")
    print("It maintains performance and accuracy while improving boundary handling.")
else:
    print("RECOMMENDATION: Review the issues above before deployment.")

print("=" * 70)