#!/usr/bin/env python
"""
Benchmark script for all magnetospheric models.

Compares performance between scalar and vectorized implementations.
"""

import numpy as np
import time
import sys
import os
from tabulate import tabulate

# Add parent directory to path

import geopack
from geopack import t89, t96, t01, t04
from geopack import t89_vectorized, t96_vectorized, t01_vectorized, t04_vectorized


def benchmark_model(name, scalar_func, vector_func, params, ps, n_points=10000):
    """Benchmark a single model."""
    # Generate test points
    x = np.random.uniform(-10, 5, n_points)
    y = np.random.uniform(-5, 5, n_points)
    z = np.random.uniform(-3, 3, n_points)
    
    # Benchmark scalar (sample 100 points)
    n_sample = min(100, n_points)
    t0 = time.time()
    for i in range(n_sample):
        if name == 'T89':
            _ = scalar_func(params, ps, x[i], y[i], z[i])
        else:
            _ = scalar_func(params, ps, x[i], y[i], z[i])
    t_scalar = (time.time() - t0) * n_points / n_sample
    
    # Benchmark vectorized
    t0 = time.time()
    if name == 'T89':
        _ = vector_func(params, ps, x, y, z)
    else:
        _ = vector_func(params, ps, x, y, z)
    t_vector = time.time() - t0
    
    # Calculate metrics
    speedup = t_scalar / t_vector
    throughput = n_points / t_vector
    
    return {
        'Model': name,
        'Points': n_points,
        'Scalar (s)': f"{t_scalar:.3f}",
        'Vector (s)': f"{t_vector:.3f}",
        'Speedup': f"{speedup:.1f}x",
        'Throughput': f"{throughput:.0f} pts/s"
    }


def main():
    """Run benchmarks for all models."""
    print("Magnetospheric Model Performance Benchmarks")
    print("=" * 60)
    
    # Set up parameters
    import datetime
    dt = datetime.datetime(2023, 3, 15, 12, 0, 0)
    ut = dt.timestamp()
    ps = geopack.recalc(ut)
    
    # Model parameters
    kp = 3
    parmod_t96 = np.array([2.0, -20.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0])
    parmod_t01 = np.array([2.0, -30.0, 2.0, -5.0, 0.5, 1.0, 0, 0, 0, 0])
    parmod_t04 = np.array([5.0, -50.0, 2.0, -5.0, 0.5, 1.0, 0.8, 1.2, 0.6, 0.9])
    
    # Run benchmarks
    results = []
    
    print("\nBenchmarking T89...")
    results.append(benchmark_model('T89', t89, t89_vectorized, kp, ps))
    
    print("Benchmarking T96...")
    results.append(benchmark_model('T96', t96, t96_vectorized, parmod_t96, ps))
    
    print("Benchmarking T01...")
    results.append(benchmark_model('T01', t01, t01_vectorized, parmod_t01, ps))
    
    print("Benchmarking T04...")
    results.append(benchmark_model('T04', t04, t04_vectorized, parmod_t04, ps))
    
    # Display results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(tabulate(results, headers='keys', tablefmt='grid'))
    
    # Summary statistics
    speedups = [float(r['Speedup'][:-1]) for r in results]
    print(f"\nAverage speedup: {np.mean(speedups):.1f}x")
    print(f"Minimum speedup: {np.min(speedups):.1f}x")
    print(f"Maximum speedup: {np.max(speedups):.1f}x")


if __name__ == "__main__":
    main()