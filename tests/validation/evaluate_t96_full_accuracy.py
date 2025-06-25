#!/usr/bin/env python3
"""
Comprehensive accuracy evaluation for the vectorized T96 implementation.

This script performs extensive testing of the vectorized T96 model against the
scalar version to verify numerical accuracy across a wide parameter space.
"""

import numpy as np
import time
import sys
from geopack import t96
from geopack.t96_vectorized import t96_vectorized


def generate_test_cases(n_cases=10000):
    """Generate diverse test cases covering the full parameter space."""
    np.random.seed(42)  # For reproducibility
    
    # Spatial coordinates - cover near/far field, all quadrants
    r = np.random.uniform(1.5, 30.0, n_cases)  # Distance from 1.5 to 30 Re
    theta = np.random.uniform(0, np.pi, n_cases)  # Polar angle
    phi = np.random.uniform(0, 2*np.pi, n_cases)  # Azimuthal angle
    
    # Convert to Cartesian
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    # Add some special cases
    # Near-Earth points
    x[:100] = np.random.uniform(-3, 3, 100)
    y[:100] = np.random.uniform(-3, 3, 100)
    z[:100] = np.random.uniform(-3, 3, 100)
    
    # Tail region
    x[100:200] = np.random.uniform(-50, -10, 100)
    y[100:200] = np.random.uniform(-10, 10, 100)
    z[100:200] = np.random.uniform(-5, 5, 100)
    
    # High-latitude points
    x[200:300] = np.random.uniform(-5, 5, 100)
    y[200:300] = np.random.uniform(-5, 5, 100)
    z[200:300] = np.random.uniform(10, 20, 100) * np.random.choice([-1, 1], 100)
    
    # Model parameters - diverse conditions
    pdyn_values = np.random.uniform(0.5, 10.0, n_cases)  # Solar wind pressure
    dst_values = np.random.uniform(-200, 50, n_cases)   # Dst index
    byimf_values = np.random.uniform(-10, 10, n_cases)  # IMF By
    bzimf_values = np.random.uniform(-10, 10, n_cases)  # IMF Bz
    
    # Tilt angles
    ps_values = np.random.uniform(-0.5, 0.5, n_cases)   # Dipole tilt
    
    return x, y, z, pdyn_values, dst_values, byimf_values, bzimf_values, ps_values


def evaluate_accuracy(x, y, z, parmod, ps):
    """Compare scalar and vectorized results for given inputs."""
    # Scalar computation
    n = len(x)
    bx_scalar = np.zeros(n)
    by_scalar = np.zeros(n)
    bz_scalar = np.zeros(n)
    
    for i in range(n):
        bx_scalar[i], by_scalar[i], bz_scalar[i] = t96.t96(
            parmod[i], ps[i], x[i], y[i], z[i]
        )
    
    # Vectorized computation - test both single and batch
    # Test single point interface
    bx_vec_single, by_vec_single, bz_vec_single = t96_vectorized(
        parmod[0], ps[0], x[0], y[0], z[0]
    )
    
    # Test batch interface
    bx_vec = np.zeros(n)
    by_vec = np.zeros(n)
    bz_vec = np.zeros(n)
    
    for i in range(n):
        bx_vec[i], by_vec[i], bz_vec[i] = t96_vectorized(
            parmod[i], ps[i], x[i], y[i], z[i]
        )
    
    # Calculate differences
    diff_x = bx_vec - bx_scalar
    diff_y = by_vec - by_scalar
    diff_z = bz_vec - bz_scalar
    
    # Calculate relative errors (avoid division by zero)
    b_magnitude = np.sqrt(bx_scalar**2 + by_scalar**2 + bz_scalar**2)
    safe_magnitude = np.where(b_magnitude > 1e-10, b_magnitude, 1e-10)
    
    rel_diff = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2) / safe_magnitude
    
    # Component-wise relative errors
    rel_diff_x = np.abs(diff_x) / (np.abs(bx_scalar) + 1e-10)
    rel_diff_y = np.abs(diff_y) / (np.abs(by_scalar) + 1e-10)
    rel_diff_z = np.abs(diff_z) / (np.abs(bz_scalar) + 1e-10)
    
    return {
        'diff_x': diff_x,
        'diff_y': diff_y,
        'diff_z': diff_z,
        'rel_diff': rel_diff,
        'rel_diff_x': rel_diff_x,
        'rel_diff_y': rel_diff_y,
        'rel_diff_z': rel_diff_z,
        'bx_scalar': bx_scalar,
        'by_scalar': by_scalar,
        'bz_scalar': bz_scalar,
        'bx_vec': bx_vec,
        'by_vec': by_vec,
        'bz_vec': bz_vec,
        'single_test': (bx_vec_single, by_vec_single, bz_vec_single),
        'single_scalar': (bx_scalar[0], by_scalar[0], bz_scalar[0])
    }


def analyze_results(results, x, y, z, parmod, ps):
    """Analyze accuracy results and identify problem areas."""
    rel_diff = results['rel_diff']
    
    # Basic statistics
    print("\n=== ACCURACY STATISTICS ===")
    print(f"Number of test points: {len(rel_diff)}")
    print(f"\nRelative error statistics:")
    print(f"  Mean: {np.mean(rel_diff):.2e}")
    print(f"  Median: {np.median(rel_diff):.2e}")
    print(f"  Max: {np.max(rel_diff):.2e}")
    print(f"  99th percentile: {np.percentile(rel_diff, 99):.2e}")
    print(f"  95th percentile: {np.percentile(rel_diff, 95):.2e}")
    
    # Component-wise statistics
    print(f"\nComponent-wise max relative errors:")
    print(f"  Bx: {np.max(results['rel_diff_x']):.2e}")
    print(f"  By: {np.max(results['rel_diff_y']):.2e}")
    print(f"  Bz: {np.max(results['rel_diff_z']):.2e}")
    
    # Test single point interface
    single_vec = results['single_test']
    single_scalar = results['single_scalar']
    print(f"\nSingle point interface test:")
    print(f"  Scalar: Bx={single_scalar[0]:.6f}, By={single_scalar[1]:.6f}, Bz={single_scalar[2]:.6f}")
    print(f"  Vector: Bx={single_vec[0]:.6f}, By={single_vec[1]:.6f}, Bz={single_vec[2]:.6f}")
    print(f"  Difference: {np.max(np.abs(np.array(single_vec) - np.array(single_scalar))):.2e}")
    
    # Identify worst cases
    worst_idx = np.argmax(rel_diff)
    print(f"\nWorst case (highest relative error):")
    print(f"  Position: x={x[worst_idx]:.3f}, y={y[worst_idx]:.3f}, z={z[worst_idx]:.3f}")
    print(f"  Distance from origin: {np.sqrt(x[worst_idx]**2 + y[worst_idx]**2 + z[worst_idx]**2):.3f} Re")
    print(f"  Parameters: Pdyn={parmod[worst_idx,0]:.2f}, Dst={parmod[worst_idx,1]:.1f}")
    print(f"             ByIMF={parmod[worst_idx,2]:.2f}, BzIMF={parmod[worst_idx,3]:.2f}")
    print(f"  Tilt angle: {np.degrees(ps[worst_idx]):.1f} degrees")
    print(f"  Scalar field: Bx={results['bx_scalar'][worst_idx]:.3f}, By={results['by_scalar'][worst_idx]:.3f}, Bz={results['bz_scalar'][worst_idx]:.3f}")
    print(f"  Vector field: Bx={results['bx_vec'][worst_idx]:.3f}, By={results['by_vec'][worst_idx]:.3f}, Bz={results['bz_vec'][worst_idx]:.3f}")
    print(f"  Relative error: {rel_diff[worst_idx]:.2e}")
    
    # Error distribution by region
    r = np.sqrt(x**2 + y**2 + z**2)
    
    print(f"\nError distribution by distance:")
    for r_min, r_max in [(0, 3), (3, 10), (10, 20), (20, 50)]:
        mask = (r >= r_min) & (r < r_max)
        if np.any(mask):
            print(f"  {r_min}-{r_max} Re: mean={np.mean(rel_diff[mask]):.2e}, max={np.max(rel_diff[mask]):.2e}")
    
    # Error distribution by IMF conditions
    print(f"\nError distribution by IMF conditions:")
    # Northward IMF
    mask_north = parmod[:, 3] > 0
    if np.any(mask_north):
        print(f"  Northward IMF (Bz>0): mean={np.mean(rel_diff[mask_north]):.2e}, max={np.max(rel_diff[mask_north]):.2e}")
    
    # Southward IMF
    mask_south = parmod[:, 3] < 0
    if np.any(mask_south):
        print(f"  Southward IMF (Bz<0): mean={np.mean(rel_diff[mask_south]):.2e}, max={np.max(rel_diff[mask_south]):.2e}")
    
    # Strong By
    mask_by = np.abs(parmod[:, 2]) > 5
    if np.any(mask_by):
        print(f"  Strong By (|By|>5): mean={np.mean(rel_diff[mask_by]):.2e}, max={np.max(rel_diff[mask_by]):.2e}")
    
    # Count cases exceeding thresholds
    print(f"\nError threshold analysis:")
    for threshold in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
        count = np.sum(rel_diff > threshold)
        percentage = 100.0 * count / len(rel_diff)
        print(f"  Points with error > {threshold:.0e}: {count} ({percentage:.2f}%)")
    
    return worst_idx


def benchmark_performance(n_points=10000):
    """Benchmark performance of scalar vs vectorized implementations."""
    print("\n=== PERFORMANCE BENCHMARK ===")
    
    # Generate test data
    x = np.random.uniform(-20, 10, n_points)
    y = np.random.uniform(-15, 15, n_points)
    z = np.random.uniform(-10, 10, n_points)
    
    parmod = np.array([2.0, -20.0, 3.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ps = 0.1
    
    # Time scalar implementation
    start = time.time()
    for i in range(n_points):
        t96.t96(parmod, ps, x[i], y[i], z[i])
    scalar_time = time.time() - start
    
    # Time vectorized implementation (single points)
    start = time.time()
    for i in range(n_points):
        t96_vectorized(parmod, ps, x[i], y[i], z[i])
    vec_single_time = time.time() - start
    
    # Time vectorized implementation (batch - if supported)
    start = time.time()
    try:
        t96_vectorized(parmod, ps, x, y, z)
        vec_batch_time = time.time() - start
        batch_supported = True
    except:
        vec_batch_time = None
        batch_supported = False
    
    print(f"Processing {n_points} points:")
    print(f"  Scalar implementation: {scalar_time:.3f} seconds ({n_points/scalar_time:.0f} points/sec)")
    print(f"  Vectorized (single calls): {vec_single_time:.3f} seconds ({n_points/vec_single_time:.0f} points/sec)")
    if batch_supported:
        print(f"  Vectorized (batch): {vec_batch_time:.3f} seconds ({n_points/vec_batch_time:.0f} points/sec)")
        print(f"  Batch speedup: {scalar_time/vec_batch_time:.1f}x")
    print(f"  Single call speedup: {scalar_time/vec_single_time:.1f}x")


def main():
    """Main evaluation routine."""
    print("T96 Vectorization Comprehensive Accuracy Evaluation")
    print("=" * 60)
    
    # Generate test cases
    print("\nGenerating test cases...")
    x, y, z, pdyn, dst, byimf, bzimf, ps = generate_test_cases(10000)
    
    # Prepare parameter arrays
    n = len(x)
    parmod = np.zeros((n, 10))
    parmod[:, 0] = pdyn
    parmod[:, 1] = dst
    parmod[:, 2] = byimf
    parmod[:, 3] = bzimf
    
    # Evaluate accuracy
    print("Evaluating accuracy...")
    results = evaluate_accuracy(x, y, z, parmod, ps)
    
    # Analyze results
    worst_idx = analyze_results(results, x, y, z, parmod, ps)
    
    # Performance benchmark
    benchmark_performance(1000)
    
    # Save detailed results for worst cases
    print("\n=== DETAILED WORST CASES ===")
    rel_diff = results['rel_diff']
    worst_indices = np.argsort(rel_diff)[-10:][::-1]
    
    with open('t96_worst_cases.txt', 'w') as f:
        f.write("Top 10 worst cases by relative error:\n\n")
        for i, idx in enumerate(worst_indices):
            f.write(f"Case {i+1}:\n")
            f.write(f"  Position: ({x[idx]:.6f}, {y[idx]:.6f}, {z[idx]:.6f})\n")
            f.write(f"  Parameters: Pdyn={parmod[idx,0]:.3f}, Dst={parmod[idx,1]:.1f}, ")
            f.write(f"ByIMF={parmod[idx,2]:.3f}, BzIMF={parmod[idx,3]:.3f}\n")
            f.write(f"  Tilt: {np.degrees(ps[idx]):.2f} degrees\n")
            f.write(f"  Scalar: ({results['bx_scalar'][idx]:.6f}, {results['by_scalar'][idx]:.6f}, {results['bz_scalar'][idx]:.6f})\n")
            f.write(f"  Vector: ({results['bx_vec'][idx]:.6f}, {results['by_vec'][idx]:.6f}, {results['bz_vec'][idx]:.6f})\n")
            f.write(f"  Difference: ({results['diff_x'][idx]:.2e}, {results['diff_y'][idx]:.2e}, {results['diff_z'][idx]:.2e})\n")
            f.write(f"  Relative error: {rel_diff[idx]:.2e}\n\n")
    
    print("\nWorst cases saved to t96_worst_cases.txt")
    
    # Final summary
    print("\n=== SUMMARY ===")
    if np.max(rel_diff) < 1e-6:
        print("✓ Vectorized implementation achieves excellent accuracy (< 1e-6 relative error)")
    elif np.max(rel_diff) < 1e-4:
        print("✓ Vectorized implementation achieves good accuracy (< 1e-4 relative error)")
    elif np.max(rel_diff) < 1e-2:
        print("⚠ Vectorized implementation has moderate accuracy (< 1e-2 relative error)")
    else:
        print("✗ Vectorized implementation has significant accuracy issues (> 1e-2 relative error)")
    
    return results


if __name__ == "__main__":
    main()