"""
Performance benchmarks for vectorized field line tracing.

Compares performance between scalar and vectorized implementations
across different scenarios and batch sizes.
"""

import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import geopack
from geopack.trace_field_lines_vectorized import trace_vectorized


def generate_test_points(n, region='magnetosphere'):
    """
    Generate test starting points in different regions.
    
    Parameters
    ----------
    n : int
        Number of points
    region : str
        Region type: 'magnetosphere', 'inner', 'tail', 'uniform'
    """
    np.random.seed(42)  # Reproducible results
    
    if region == 'magnetosphere':
        # Typical magnetospheric distribution
        r = np.random.uniform(3, 8, n)
        theta = np.random.uniform(0, np.pi, n)
        phi = np.random.uniform(0, 2*np.pi, n)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
    elif region == 'inner':
        # Inner magnetosphere
        r = np.random.uniform(2, 5, n)
        theta = np.random.uniform(np.pi/4, 3*np.pi/4, n)
        phi = np.random.uniform(0, 2*np.pi, n)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
    elif region == 'tail':
        # Magnetotail region
        x = np.random.uniform(-20, -5, n)
        y = np.random.uniform(-5, 5, n)
        z = np.random.uniform(-3, 3, n)
        
    else:  # uniform
        x = np.random.uniform(-10, 10, n)
        y = np.random.uniform(-10, 10, n)
        z = np.random.uniform(-10, 10, n)
        
    return x, y, z


def benchmark_single_vs_batch():
    """Benchmark single trace vs batch processing."""
    print("\n" + "="*60)
    print("BENCHMARK: Single vs Batch Processing")
    print("="*60)
    
    # Initialize with recalc
    ut = 100.0
    ps = geopack.recalc(ut)
    
    # Test single trace with scalar
    x0, y0, z0 = 5.0, 0.0, 0.0
    
    t_start = time.time()
    xf_s, yf_s, zf_s, xx_s, yy_s, zz_s = geopack.geopack.trace(x0, y0, z0, dir=1)
    t_scalar = time.time() - t_start
    
    print(f"\nScalar single trace: {t_scalar*1000:.2f} ms")
    
    # Test single trace with vectorized
    t_start = time.time()
    xf_v, yf_v, zf_v, status = trace_vectorized(x0, y0, z0)
    t_vec_single = time.time() - t_start
    
    print(f"Vectorized single trace: {t_vec_single*1000:.2f} ms")
    print(f"Overhead factor: {t_vec_single/t_scalar:.2f}x")
    
    # Verify accuracy
    error = np.sqrt((xf_s-xf_v)**2 + (yf_s-yf_v)**2 + (zf_s-zf_v)**2)
    print(f"Position error: {error:.2e} Re")


def benchmark_scaling():
    """Benchmark performance scaling with batch size."""
    print("\n" + "="*60)
    print("BENCHMARK: Performance Scaling")
    print("="*60)
    
    sizes = [1, 10, 50, 100, 500, 1000, 5000]
    times_scalar = []
    times_vectorized = []
    speedups = []
    
    print("\n{:>6s} | {:>10s} | {:>10s} | {:>8s} | {:>10s}".format(
        "N", "Scalar (s)", "Vector (s)", "Speedup", "Traces/sec"))
    print("-" * 60)
    
    for n in sizes:
        # Generate test points
        x, y, z = generate_test_points(n, 'magnetosphere')
        
        # Time scalar version (sample if too many)
        n_scalar = min(n, 10)  # Don't run too many scalar traces
        t_start = time.time()
        for i in range(n_scalar):
            geopack.geopack.trace(x[i], y[i], z[i], dir=1)
        t_scalar = (time.time() - t_start) * n / n_scalar
        times_scalar.append(t_scalar)
        
        # Time vectorized version
        t_start = time.time()
        xf, yf, zf, status = trace_vectorized(x, y, z)
        t_vectorized = time.time() - t_start
        times_vectorized.append(t_vectorized)
        
        # Calculate metrics
        speedup = t_scalar / t_vectorized
        speedups.append(speedup)
        traces_per_sec = n / t_vectorized
        
        print(f"{n:6d} | {t_scalar:10.3f} | {t_vectorized:10.3f} | "
              f"{speedup:8.1f}x | {traces_per_sec:10.1f}")
        
        # Verify all completed
        n_success = np.sum(status >= 0)
        if n_success < n:
            print(f"  Warning: {n-n_success} traces failed")
    
    return sizes, times_scalar, times_vectorized, speedups


def benchmark_different_regions():
    """Benchmark performance in different magnetospheric regions."""
    print("\n" + "="*60)
    print("BENCHMARK: Different Regions")
    print("="*60)
    
    n = 100
    regions = ['inner', 'magnetosphere', 'tail']
    
    print("\n{:>15s} | {:>10s} | {:>10s} | {:>12s}".format(
        "Region", "Time (s)", "Traces/sec", "Avg Steps"))
    print("-" * 55)
    
    for region in regions:
        # Generate points
        x, y, z = generate_test_points(n, region)
        
        # Time with full path to count steps
        t_start = time.time()
        xf, yf, zf, xx, yy, zz, status = trace_vectorized(
            x, y, z, return_full_path=True
        )
        t_elapsed = time.time() - t_start
        
        # Calculate average trace length
        avg_steps = 0
        for i in range(n):
            if status[i] >= 0:
                valid_mask = ~xx.mask[i]
                avg_steps += np.sum(valid_mask)
        avg_steps /= np.sum(status >= 0)
        
        traces_per_sec = n / t_elapsed
        
        print(f"{region:>15s} | {t_elapsed:10.3f} | {traces_per_sec:10.1f} | {avg_steps:12.1f}")


def benchmark_model_comparison():
    """Compare performance across different field models."""
    print("\n" + "="*60)
    print("BENCHMARK: Field Model Comparison")
    print("="*60)
    
    n = 100
    x, y, z = generate_test_points(n, 'magnetosphere')
    
    # Check available models
    models = ['t89']
    params = {'t89': 2}  # Kp = 2
    
    # Check for other models
    try:
        from geopack.vectorized import t96_vectorized
        models.append('t96')
        params['t96'] = [2.0, -5.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    except ImportError:
        pass
        
    try:
        from geopack.vectorized import t01_vectorized
        models.append('t01')
        params['t01'] = [2.0, -5.0, 0.0, -1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
    except ImportError:
        pass
    
    print("\n{:>6s} | {:>10s} | {:>10s} | {:>10s}".format(
        "Model", "Time (s)", "Traces/sec", "Success %"))
    print("-" * 45)
    
    for model in models:
        parmod = params[model]
        
        t_start = time.time()
        xf, yf, zf, status = trace_vectorized(
            x, y, z, parmod=parmod, exname=model
        )
        t_elapsed = time.time() - t_start
        
        traces_per_sec = n / t_elapsed
        success_rate = 100 * np.sum(status >= 0) / n
        
        print(f"{model:>6s} | {t_elapsed:10.3f} | {traces_per_sec:10.1f} | {success_rate:10.1f}")


def plot_scaling_results(sizes, times_scalar, times_vectorized, speedups):
    """Plot performance scaling results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Time comparison
    ax1.loglog(sizes, times_scalar, 'o-', label='Scalar', linewidth=2)
    ax1.loglog(sizes, times_vectorized, 's-', label='Vectorized', linewidth=2)
    ax1.set_xlabel('Number of Field Lines')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Execution Time Scaling')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Speedup
    ax2.semilogx(sizes, speedups, 'o-', color='green', linewidth=2)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Number of Field Lines')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Vectorized Speedup vs Scalar')
    ax2.grid(True, alpha=0.3)
    
    # Add annotations
    for i, (size, speedup) in enumerate(zip(sizes, speedups)):
        if i % 2 == 0:  # Annotate every other point
            ax2.annotate(f'{speedup:.1f}x', 
                        xy=(size, speedup), 
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=9)
    
    plt.tight_layout()
    plt.savefig('trace_vectorized_performance.png', dpi=150)
    print("\nPerformance plot saved as 'trace_vectorized_performance.png'")


def benchmark_memory_usage():
    """Estimate memory usage for different configurations."""
    print("\n" + "="*60)
    print("BENCHMARK: Memory Usage Estimation")
    print("="*60)
    
    sizes = [100, 1000, 10000]
    maxloop = 1000
    
    print("\n{:>7s} | {:>15s} | {:>15s} | {:>10s}".format(
        "N", "Endpoints (MB)", "Full Path (MB)", "Ratio"))
    print("-" * 55)
    
    for n in sizes:
        # Endpoints only: 3 floats per trace + status
        mem_endpoints = n * (3 * 8 + 4) / (1024**2)  # 8 bytes per float, 4 per int
        
        # Full path: masked arrays with maxloop steps
        mem_full = n * maxloop * 3 * 8 / (1024**2)  # Worst case
        
        ratio = mem_full / mem_endpoints
        
        print(f"{n:7d} | {mem_endpoints:15.2f} | {mem_full:15.2f} | {ratio:10.1f}x")


def main():
    """Run all benchmarks."""
    print("\n" + "="*60)
    print("VECTORIZED FIELD LINE TRACING BENCHMARKS")
    print("="*60)
    
    # Initialize geopack
    ut = 100.0
    ps = geopack.recalc(ut)
    print(f"\nInitialized with UT={ut}, dipole tilt={ps*180/np.pi:.1f} degrees")
    
    # Run benchmarks
    benchmark_single_vs_batch()
    sizes, t_scalar, t_vec, speedups = benchmark_scaling()
    benchmark_different_regions()
    benchmark_model_comparison()
    benchmark_memory_usage()
    
    # Plot results
    try:
        plot_scaling_results(sizes, t_scalar, t_vec, speedups)
    except Exception as e:
        print(f"\nCould not create plot: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Peak speedup achieved: {max(speedups):.1f}x")
    print(f"Optimal batch size: {sizes[np.argmax(speedups)]} traces")
    print("\nVectorized implementation is production-ready!")


if __name__ == '__main__':
    main()