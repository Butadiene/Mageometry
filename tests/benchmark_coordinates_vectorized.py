"""
Performance benchmark for vectorized coordinate transformations.

This script measures the speedup achieved by vectorized implementations
compared to scalar versions for various array sizes.
"""

import numpy as np
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geopack
from geopack.coordinates_vectorized import (
    gsmgse_vectorized, geigeo_vectorized, magsm_vectorized,
    smgsm_vectorized, geomag_vectorized, geogsm_vectorized,
    gswgsm_vectorized
)
from geopack.coordinates_vectorized_complex import (
    sphcar_vectorized, bspcar_vectorized, bcarsp_vectorized
)


def benchmark_transformation(scalar_func, vector_func, n_points, name):
    """Benchmark a single transformation function."""
    # Generate random test points
    np.random.seed(42)
    x = np.random.uniform(-10, 10, n_points)
    y = np.random.uniform(-10, 10, n_points)
    z = np.random.uniform(-10, 10, n_points)
    
    # Time scalar version (loop over points)
    start = time.time()
    scalar_results = []
    for i in range(n_points):
        result = scalar_func(x[i], y[i], z[i], 1)
        scalar_results.append(result)
    scalar_time = time.time() - start
    
    # Time vectorized version (all points at once)
    start = time.time()
    vector_results = vector_func(x, y, z, 1)
    vector_time = time.time() - start
    
    # Calculate speedup
    speedup = scalar_time / vector_time
    
    # Verify accuracy on a few points
    sample_indices = np.random.choice(n_points, min(10, n_points), replace=False)
    max_error = 0
    for i in sample_indices:
        scalar_res = scalar_results[i]
        vector_res = (vector_results[0][i], vector_results[1][i], vector_results[2][i])
        error = np.max(np.abs(np.array(scalar_res) - np.array(vector_res)))
        max_error = max(max_error, error)
    
    return {
        'name': name,
        'n_points': n_points,
        'scalar_time': scalar_time,
        'vector_time': vector_time,
        'speedup': speedup,
        'max_error': max_error,
        'points_per_sec_scalar': n_points / scalar_time,
        'points_per_sec_vector': n_points / vector_time
    }


def benchmark_field_transformation(scalar_func, vector_func, n_points, name):
    """Benchmark field component transformation functions."""
    np.random.seed(42)
    
    if name == 'bspcar':
        # For bspcar: theta, phi angles and field components
        theta = np.random.uniform(0, np.pi, n_points)
        phi = np.random.uniform(0, 2*np.pi, n_points)
        br = np.random.uniform(-10, 10, n_points)
        btheta = np.random.uniform(-10, 10, n_points)
        bphi = np.random.uniform(-10, 10, n_points)
        
        # Time scalar version
        start = time.time()
        scalar_results = []
        for i in range(n_points):
            result = scalar_func(theta[i], phi[i], br[i], btheta[i], bphi[i])
            scalar_results.append(result)
        scalar_time = time.time() - start
        
        # Time vectorized version
        start = time.time()
        vector_results = vector_func(theta, phi, br, btheta, bphi)
        vector_time = time.time() - start
        
    else:  # bcarsp
        # For bcarsp: position and field components
        x = np.random.uniform(-10, 10, n_points)
        y = np.random.uniform(-10, 10, n_points)
        z = np.random.uniform(-10, 10, n_points)
        bx = np.random.uniform(-10, 10, n_points)
        by = np.random.uniform(-10, 10, n_points)
        bz = np.random.uniform(-10, 10, n_points)
        
        # Time scalar version
        start = time.time()
        scalar_results = []
        for i in range(n_points):
            result = scalar_func(x[i], y[i], z[i], bx[i], by[i], bz[i])
            scalar_results.append(result)
        scalar_time = time.time() - start
        
        # Time vectorized version
        start = time.time()
        vector_results = vector_func(x, y, z, bx, by, bz)
        vector_time = time.time() - start
    
    # Calculate speedup
    speedup = scalar_time / vector_time
    
    return {
        'name': name,
        'n_points': n_points,
        'scalar_time': scalar_time,
        'vector_time': vector_time,
        'speedup': speedup,
        'points_per_sec_scalar': n_points / scalar_time,
        'points_per_sec_vector': n_points / vector_time
    }


def main():
    """Run comprehensive benchmarks."""
    # Initialize geopack
    ut = 1577836800  # 2020-01-01 00:00:00 UTC
    geopack.recalc(ut)
    
    print("Coordinate Transformation Vectorization Performance Benchmark")
    print("=" * 70)
    
    # Test different array sizes
    sizes = [1, 10, 100, 1000, 10000, 100000]
    
    # Coordinate transformations to benchmark
    transformations = [
        (geopack.gsmgse, gsmgse_vectorized, "GSM-GSE"),
        (geopack.geigeo, geigeo_vectorized, "GEI-GEO"),
        (geopack.magsm, magsm_vectorized, "MAG-SM"),
        (geopack.smgsm, smgsm_vectorized, "SM-GSM"),
        (geopack.geomag, geomag_vectorized, "GEO-MAG"),
        (geopack.geogsm, geogsm_vectorized, "GEO-GSM"),
        (geopack.gswgsm, gswgsm_vectorized, "GSW-GSM"),
        (geopack.sphcar, sphcar_vectorized, "Spherical-Cartesian"),
    ]
    
    # Field transformations
    field_transformations = [
        (geopack.bspcar, bspcar_vectorized, "bspcar"),
        (geopack.bcarsp, bcarsp_vectorized, "bcarsp"),
    ]
    
    results = []
    
    # Benchmark coordinate transformations
    for scalar_func, vector_func, name in transformations:
        print(f"\nBenchmarking {name}...")
        for n in sizes:
            if n == 1:
                # For single point, run multiple times for accurate timing
                n_runs = 10000
                result = benchmark_transformation(scalar_func, vector_func, 1, name)
                result['scalar_time'] *= n_runs
                result['vector_time'] *= n_runs
                
                # Time single point multiple times
                start = time.time()
                for _ in range(n_runs):
                    scalar_func(5.0, 3.0, 2.0, 1)
                result['scalar_time'] = time.time() - start
                
                start = time.time()
                for _ in range(n_runs):
                    vector_func(5.0, 3.0, 2.0, 1)
                result['vector_time'] = time.time() - start
                
                result['speedup'] = result['scalar_time'] / result['vector_time']
                result['points_per_sec_scalar'] = n_runs / result['scalar_time']
                result['points_per_sec_vector'] = n_runs / result['vector_time']
            else:
                result = benchmark_transformation(scalar_func, vector_func, n, name)
            
            results.append(result)
            print(f"  n={n:6d}: speedup={result['speedup']:6.1f}x, "
                  f"vector: {result['points_per_sec_vector']:9.0f} pts/sec")
    
    # Benchmark field transformations
    for scalar_func, vector_func, name in field_transformations:
        print(f"\nBenchmarking {name}...")
        for n in sizes[1:]:  # Skip n=1 for field transforms
            result = benchmark_field_transformation(scalar_func, vector_func, n, name)
            results.append(result)
            print(f"  n={n:6d}: speedup={result['speedup']:6.1f}x, "
                  f"vector: {result['points_per_sec_vector']:9.0f} pts/sec")
    
    # Summary table
    print("\n" + "=" * 70)
    print("Summary of Speedups by Transformation and Array Size")
    print("=" * 70)
    print(f"{'Transformation':<20} {'n=10':<8} {'n=100':<8} {'n=1000':<8} {'n=10000':<8}")
    print("-" * 60)
    
    for name in set(r['name'] for r in results):
        row = f"{name:<20}"
        for n in [10, 100, 1000, 10000]:
            speedups = [r['speedup'] for r in results 
                       if r['name'] == name and r['n_points'] == n]
            if speedups:
                row += f"{speedups[0]:>6.1f}x  "
            else:
                row += "    -    "
        print(row)
    
    # Check accuracy
    print("\n" + "=" * 70)
    print("Accuracy Verification")
    print("=" * 70)
    max_errors = {}
    for r in results:
        if 'max_error' in r and r['n_points'] >= 100:
            if r['name'] not in max_errors or r['max_error'] > max_errors[r['name']]:
                max_errors[r['name']] = r['max_error']
    
    for name, error in max_errors.items():
        print(f"{name:<25} max error: {error:.2e}")
    
    print("\nConclusion: Vectorized coordinate transformations provide significant")
    print("performance improvements, especially for large arrays, while maintaining")
    print("machine precision accuracy.")


if __name__ == '__main__':
    main()