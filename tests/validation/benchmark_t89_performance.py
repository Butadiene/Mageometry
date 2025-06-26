#!/usr/bin/env python3
"""
Comprehensive performance benchmark for T89 vectorized implementation.

This script tests various aspects of performance including:
- Single point vs batch processing
- Different array sizes
- Memory usage
- Various Kp levels and parameter combinations
"""

import numpy as np
import time
import sys
import gc
sys.path.append('../..')
from geopack import t89
from geopack.t89_vectorized import t89_vectorized


def benchmark_single_points(n_points=1000):
    """Benchmark single point calculations."""
    print("\n=== SINGLE POINT PERFORMANCE ===")
    
    # Generate random test points
    np.random.seed(42)
    x = np.random.uniform(-20, 20, n_points)
    y = np.random.uniform(-20, 20, n_points)
    z = np.random.uniform(-20, 20, n_points)
    iopt = np.random.randint(1, 8, n_points)
    ps = np.random.uniform(-0.5, 0.5, n_points)
    
    # Time scalar implementation
    gc.collect()
    t0 = time.perf_counter()
    for i in range(n_points):
        _ = t89.t89(iopt[i], ps[i], x[i], y[i], z[i])
    t_scalar = time.perf_counter() - t0
    
    # Time vectorized implementation (single points)
    gc.collect()
    t0 = time.perf_counter()
    for i in range(n_points):
        _ = t89_vectorized(iopt[i], ps[i], x[i], y[i], z[i])
    t_vec = time.perf_counter() - t0
    
    print(f"Points tested: {n_points}")
    print(f"Scalar time: {t_scalar:.3f}s ({n_points/t_scalar:.0f} points/sec)")
    print(f"Vector time: {t_vec:.3f}s ({n_points/t_vec:.0f} points/sec)")
    print(f"Overhead factor: {t_vec/t_scalar:.2f}x")


def benchmark_batch_sizes():
    """Benchmark different batch sizes."""
    print("\n=== BATCH SIZE PERFORMANCE ===")
    
    batch_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
    iopt = 3  # Middle Kp value
    ps = 0.1  # Small tilt
    
    print(f"{'Batch Size':>10} {'Scalar (s)':>12} {'Vector (s)':>12} {'Speedup':>10} {'Points/sec':>12}")
    print("-" * 70)
    
    for size in batch_sizes:
        # Generate test data
        x = np.random.uniform(-30, 30, size)
        y = np.random.uniform(-30, 30, size)
        z = np.random.uniform(-30, 30, size)
        
        # Time scalar
        gc.collect()
        t0 = time.perf_counter()
        bx_scalar = np.zeros(size)
        by_scalar = np.zeros(size)
        bz_scalar = np.zeros(size)
        for i in range(size):
            bx_scalar[i], by_scalar[i], bz_scalar[i] = t89.t89(iopt, ps, x[i], y[i], z[i])
        t_scalar = time.perf_counter() - t0
        
        # Time vectorized
        gc.collect()
        t0 = time.perf_counter()
        bx_vec, by_vec, bz_vec = t89_vectorized(iopt, ps, x, y, z)
        t_vec = time.perf_counter() - t0
        
        speedup = t_scalar / t_vec
        throughput = size / t_vec
        
        print(f"{size:10d} {t_scalar:12.4f} {t_vec:12.4f} {speedup:10.1f}x {throughput:12.0f}")


def benchmark_kp_levels():
    """Benchmark performance across different Kp levels."""
    print("\n=== Kp LEVEL PERFORMANCE ===")
    
    n_points = 1000
    ps = 0.2
    x = np.random.uniform(-30, 30, n_points)
    y = np.random.uniform(-30, 30, n_points)
    z = np.random.uniform(-30, 30, n_points)
    
    print(f"{'Kp Level':>10} {'Description':>15} {'Time (s)':>10} {'Points/sec':>12}")
    print("-" * 50)
    
    kp_descriptions = [
        (1, "Quiet"),
        (2, "Quiet+"),
        (3, "Unsettled"),
        (4, "Active"),
        (5, "Minor storm"),
        (6, "Major storm"),
        (7, "Severe storm")
    ]
    
    for iopt, desc in kp_descriptions:
        gc.collect()
        t0 = time.perf_counter()
        bx, by, bz = t89_vectorized(iopt, ps, x, y, z)
        t_elapsed = time.perf_counter() - t0
        throughput = n_points / t_elapsed
        
        print(f"{iopt:10d} {desc:>15} {t_elapsed:10.4f} {throughput:12.0f}")


def benchmark_spatial_regions():
    """Benchmark performance in different spatial regions."""
    print("\n=== SPATIAL REGION PERFORMANCE ===")
    
    n_points = 1000
    iopt = 3
    ps = 0.1
    
    regions = [
        ("Near-Earth", 1, 5),
        ("Mid-field", 5, 15),
        ("Far-field", 15, 30),
        ("Deep tail", 30, 70)
    ]
    
    print(f"{'Region':>15} {'Range (Re)':>12} {'Time (s)':>10} {'Points/sec':>12}")
    print("-" * 55)
    
    for name, r_min, r_max in regions:
        # Generate points in spherical coordinates
        r = np.random.uniform(r_min, r_max, n_points)
        theta = np.random.uniform(0, np.pi, n_points)
        phi = np.random.uniform(0, 2*np.pi, n_points)
        
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        gc.collect()
        t0 = time.perf_counter()
        bx, by, bz = t89_vectorized(iopt, ps, x, y, z)
        t_elapsed = time.perf_counter() - t0
        throughput = n_points / t_elapsed
        
        print(f"{name:>15} {f'{r_min}-{r_max}':>12} {t_elapsed:10.4f} {throughput:12.0f}")


def benchmark_memory_usage():
    """Benchmark memory usage for large arrays."""
    print("\n=== MEMORY USAGE ANALYSIS ===")
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    iopt = 3
    ps = 0.1
    
    sizes = [1000, 10000, 50000, 100000]
    
    print(f"{'Array Size':>12} {'Memory Before':>15} {'Memory After':>15} {'Memory Used':>15}")
    print("-" * 60)
    
    for size in sizes:
        # Generate test data
        x = np.random.uniform(-30, 30, size)
        y = np.random.uniform(-30, 30, size)
        z = np.random.uniform(-30, 30, size)
        
        gc.collect()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run vectorized calculation
        bx, by, bz = t89_vectorized(iopt, ps, x, y, z)
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before
        
        print(f"{size:12d} {mem_before:15.1f} MB {mem_after:15.1f} MB {mem_used:15.1f} MB")
        
        # Clean up
        del x, y, z, bx, by, bz
        gc.collect()


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n=== EDGE CASE TESTING ===")
    
    test_cases = [
        ("Zero position", 0.0, 0.0, 0.0),
        ("Very near Earth", 1.0, 0.0, 0.0),
        ("Far tail", -70.0, 0.0, 0.0),
        ("High latitude", 0.0, 0.0, 30.0),
        ("Equatorial", 20.0, 20.0, 0.0),
        ("Large Y", 0.0, 50.0, 0.0),
        ("Multiple zeros", 0.0, 0.0, 10.0),
        ("Very small values", 1e-6, 1e-6, 1e-6)
    ]
    
    iopt = 3
    ps = 0.2
    
    print(f"{'Test Case':>20} {'Position':>20} {'Scalar Result':>30} {'Vector Result':>30} {'Match':>8}")
    print("-" * 110)
    
    for name, x, y, z in test_cases:
        # Scalar calculation
        bx_s, by_s, bz_s = t89.t89(iopt, ps, x, y, z)
        
        # Vectorized calculation
        bx_v, by_v, bz_v = t89_vectorized(iopt, ps, x, y, z)
        
        # Check if results match
        match = np.allclose([bx_s, by_s, bz_s], [bx_v, by_v, bz_v], rtol=1e-10)
        
        pos_str = f"({x:.1f}, {y:.1f}, {z:.1f})"
        scalar_str = f"({bx_s:.3f}, {by_s:.3f}, {bz_s:.3f})"
        vector_str = f"({bx_v:.3f}, {by_v:.3f}, {bz_v:.3f})"
        
        print(f"{name:>20} {pos_str:>20} {scalar_str:>30} {vector_str:>30} {str(match):>8}")


def compare_accuracy_detailed():
    """Detailed accuracy comparison."""
    print("\n=== DETAILED ACCURACY COMPARISON ===")
    
    # Test different parameter combinations
    n_tests = 100
    np.random.seed(42)
    
    for kp in range(1, 8):
        errors = []
        
        for _ in range(n_tests):
            x = np.random.uniform(-30, 30)
            y = np.random.uniform(-30, 30)
            z = np.random.uniform(-30, 30)
            ps = np.random.uniform(-0.5, 0.5)
            
            bx_s, by_s, bz_s = t89.t89(kp, ps, x, y, z)
            bx_v, by_v, bz_v = t89_vectorized(kp, ps, x, y, z)
            
            b_mag = np.sqrt(bx_s**2 + by_s**2 + bz_s**2)
            if b_mag > 1e-10:
                error = np.sqrt((bx_v-bx_s)**2 + (by_v-by_s)**2 + (bz_v-bz_s)**2) / b_mag
                errors.append(error)
        
        errors = np.array(errors)
        print(f"Kp={kp}: Mean error={np.mean(errors):.2e}, Max error={np.max(errors):.2e}")


def profile_components():
    """Profile individual components of the vectorized implementation."""
    print("\n=== COMPONENT PROFILING ===")
    
    try:
        import cProfile
        import pstats
        from io import StringIO
        
        # Setup test data
        n_points = 10000
        x = np.random.uniform(-30, 30, n_points)
        y = np.random.uniform(-30, 30, n_points)
        z = np.random.uniform(-30, 30, n_points)
        iopt = 3
        ps = 0.1
        
        # Profile the vectorized function
        profiler = cProfile.Profile()
        profiler.enable()
        
        for _ in range(10):  # Run multiple times for better statistics
            _ = t89_vectorized(iopt, ps, x, y, z)
        
        profiler.disable()
        
        # Get statistics
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(15)  # Top 15 functions
        
        print("Top 15 time-consuming functions:")
        print(s.getvalue())
        
    except ImportError:
        print("cProfile not available, skipping component profiling")


def main():
    """Run all benchmarks."""
    print("T89 Vectorized Implementation Performance Verification")
    print("=" * 60)
    
    # Run all benchmarks
    benchmark_single_points(1000)
    benchmark_batch_sizes()
    benchmark_kp_levels()
    benchmark_spatial_regions()
    benchmark_memory_usage()
    test_edge_cases()
    compare_accuracy_detailed()
    profile_components()
    
    print("\n" + "=" * 60)
    print("Performance verification completed!")


if __name__ == "__main__":
    main()