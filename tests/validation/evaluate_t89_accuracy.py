#!/usr/bin/env python3
"""
Comprehensive accuracy evaluation for the vectorized T89 implementation.

This script follows the testing methodology established for T96 to verify
numerical accuracy across a wide parameter space.
"""

import numpy as np
import time
import sys
sys.path.append('../..')
from geopack import t89
from geopack.t89_vectorized import t89_vectorized


def generate_test_cases(n_cases=10000):
    """Generate diverse test cases covering the full parameter space."""
    np.random.seed(42)  # For reproducibility
    
    # Spatial coordinates - cover near/far field, all quadrants
    r = np.random.uniform(1.5, 70.0, n_cases)  # T89 valid up to 70 Re
    theta = np.random.uniform(0, np.pi, n_cases)  # Polar angle
    phi = np.random.uniform(0, 2*np.pi, n_cases)  # Azimuthal angle
    
    # Convert to Cartesian
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    # Add special test cases following T96 methodology
    # Near-Earth points
    x[:100] = np.random.uniform(-3, 3, 100)
    y[:100] = np.random.uniform(-3, 3, 100)
    z[:100] = np.random.uniform(-3, 3, 100)
    
    # Tail region (T89 extends further than T96)
    x[100:200] = np.random.uniform(-70, -10, 100)
    y[100:200] = np.random.uniform(-20, 20, 100)
    z[100:200] = np.random.uniform(-10, 10, 100)
    
    # High-latitude points
    x[200:300] = np.random.uniform(-10, 10, 100)
    y[200:300] = np.random.uniform(-10, 10, 100)
    z[200:300] = np.random.uniform(15, 30, 100) * np.random.choice([-1, 1], 100)
    
    # Model parameters
    # T89 uses Kp indices (1-7)
    iopt_values = np.random.randint(1, 8, n_cases)
    
    # Tilt angles (similar range to T96)
    ps_values = np.random.uniform(-0.5, 0.5, n_cases)  # Dipole tilt in radians
    
    return x, y, z, iopt_values, ps_values


def evaluate_accuracy(x, y, z, iopt_values, ps_values):
    """Compare scalar and vectorized results for given inputs."""
    n = len(x)
    
    # Scalar computation
    bx_scalar = np.zeros(n)
    by_scalar = np.zeros(n)
    bz_scalar = np.zeros(n)
    
    print("Computing scalar results...")
    t0 = time.time()
    for i in range(n):
        bx_scalar[i], by_scalar[i], bz_scalar[i] = t89.t89(
            iopt_values[i], ps_values[i], x[i], y[i], z[i]
        )
    t_scalar = time.time() - t0
    print(f"Scalar computation time: {t_scalar:.3f}s")
    
    # Vectorized computation - test both single and batch
    print("Testing single point interface...")
    bx_vec_single, by_vec_single, bz_vec_single = t89_vectorized(
        iopt_values[0], ps_values[0], x[0], y[0], z[0]
    )
    
    # Test batch interface (one at a time for different iopt/ps values)
    print("Computing vectorized results...")
    bx_vec = np.zeros(n)
    by_vec = np.zeros(n)
    bz_vec = np.zeros(n)
    
    t0 = time.time()
    for i in range(n):
        bx_vec[i], by_vec[i], bz_vec[i] = t89_vectorized(
            iopt_values[i], ps_values[i], x[i], y[i], z[i]
        )
    t_vec = time.time() - t0
    print(f"Vectorized computation time: {t_vec:.3f}s")
    
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
        'single_scalar': (bx_scalar[0], by_scalar[0], bz_scalar[0]),
        't_scalar': t_scalar,
        't_vec': t_vec
    }


def analyze_results(results, x, y, z, iopt_values, ps_values):
    """Analyze accuracy results and identify problem areas."""
    rel_diff = results['rel_diff']
    
    # Basic statistics
    print("\n=== T89 VECTORIZATION ACCURACY STATISTICS ===")
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
    
    # Error distribution
    print(f"\nError distribution:")
    thresholds = [1e-10, 1e-08, 1e-06, 1e-04, 1e-02]
    for threshold in thresholds:
        count = np.sum(rel_diff > threshold)
        percentage = 100.0 * count / len(rel_diff)
        print(f"  > {threshold:.0e}: {count:6d} points ({percentage:5.2f}%)")
    
    # Find worst cases
    worst_idx = np.argmax(rel_diff)
    print(f"\nWorst case:")
    print(f"  Position: ({x[worst_idx]:.3f}, {y[worst_idx]:.3f}, {z[worst_idx]:.3f}) Re")
    print(f"  Distance: {np.sqrt(x[worst_idx]**2 + y[worst_idx]**2 + z[worst_idx]**2):.3f} Re")
    print(f"  Kp index (iopt): {iopt_values[worst_idx]}")
    print(f"  Tilt: {np.degrees(ps_values[worst_idx]):.1f}°")
    print(f"  Scalar: Bx={results['bx_scalar'][worst_idx]:.3f}, By={results['by_scalar'][worst_idx]:.3f}, Bz={results['bz_scalar'][worst_idx]:.3f} nT")
    print(f"  Vector: Bx={results['bx_vec'][worst_idx]:.3f}, By={results['by_vec'][worst_idx]:.3f}, Bz={results['bz_vec'][worst_idx]:.3f} nT")
    print(f"  Relative error: {rel_diff[worst_idx]:.2e}")
    
    # Regional analysis
    print(f"\nRegional accuracy analysis:")
    r = np.sqrt(x**2 + y**2 + z**2)
    regions = [(0, 3), (3, 10), (10, 20), (20, 50), (50, 70)]
    print(f"{'Region (Re)':15s} {'Mean Error':12s} {'Max Error':12s}")
    print("-" * 40)
    for r_min, r_max in regions:
        mask = (r >= r_min) & (r < r_max)
        if np.any(mask):
            mean_err = np.mean(rel_diff[mask])
            max_err = np.max(rel_diff[mask])
            print(f"{r_min:2d}-{r_max:2d}            {mean_err:12.2e} {max_err:12.2e}")
    
    # Performance summary
    print(f"\nPerformance summary:")
    print(f"  Scalar time: {results['t_scalar']:.3f}s ({len(x)/results['t_scalar']:.0f} points/sec)")
    print(f"  Vector time: {results['t_vec']:.3f}s ({len(x)/results['t_vec']:.0f} points/sec)")
    print(f"  Speedup: {results['t_scalar']/results['t_vec']:.1f}x")


def test_batch_processing():
    """Test batch processing performance."""
    print("\n=== BATCH PROCESSING TEST ===")
    
    # Generate batch data with constant parameters
    n_batch = 1000
    x_batch = np.random.uniform(-20, 20, n_batch)
    y_batch = np.random.uniform(-20, 20, n_batch)
    z_batch = np.random.uniform(-20, 20, n_batch)
    iopt = 3  # Middle Kp value
    ps = 0.1  # Small tilt
    
    # Time scalar loop
    t0 = time.time()
    bx_scalar = np.zeros(n_batch)
    by_scalar = np.zeros(n_batch)
    bz_scalar = np.zeros(n_batch)
    for i in range(n_batch):
        bx_scalar[i], by_scalar[i], bz_scalar[i] = t89.t89(
            iopt, ps, x_batch[i], y_batch[i], z_batch[i]
        )
    t_scalar = time.time() - t0
    
    # Time vectorized batch
    t0 = time.time()
    bx_vec, by_vec, bz_vec = t89_vectorized(iopt, ps, x_batch, y_batch, z_batch)
    t_vec = time.time() - t0
    
    # Verify accuracy
    diff = np.sqrt((bx_vec - bx_scalar)**2 + (by_vec - by_scalar)**2 + (bz_vec - bz_scalar)**2)
    b_mag = np.sqrt(bx_scalar**2 + by_scalar**2 + bz_scalar**2)
    rel_error = diff / (b_mag + 1e-10)
    
    print(f"Batch size: {n_batch} points")
    print(f"Scalar time: {t_scalar:.3f}s")
    print(f"Vector time: {t_vec:.3f}s")
    print(f"Speedup: {t_scalar/t_vec:.1f}x")
    print(f"Throughput: {n_batch/t_vec:.0f} points/sec")
    print(f"Max relative error: {np.max(rel_error):.2e}")


def main():
    """Run comprehensive T89 vectorization tests."""
    print("T89 Vectorized Implementation Accuracy Evaluation")
    print("=" * 50)
    
    # Generate test cases
    print("\nGenerating test cases...")
    x, y, z, iopt_values, ps_values = generate_test_cases(10000)
    
    # Evaluate accuracy
    print("\nEvaluating accuracy...")
    results = evaluate_accuracy(x, y, z, iopt_values, ps_values)
    
    # Analyze results
    analyze_results(results, x, y, z, iopt_values, ps_values)
    
    # Test batch processing
    test_batch_processing()
    
    # Summary
    print("\n" + "=" * 50)
    if np.max(results['rel_diff']) < 1e-6:
        print("✓ T89 vectorization PASSED accuracy requirements")
        print("  Maximum relative error < 1e-6")
    else:
        print("✗ T89 vectorization FAILED accuracy requirements")
        print(f"  Maximum relative error: {np.max(results['rel_diff']):.2e}")


if __name__ == "__main__":
    main()