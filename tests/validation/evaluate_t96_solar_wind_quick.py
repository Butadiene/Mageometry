#!/usr/bin/env python3
"""
Quick evaluation of T96 accuracy under various solar wind conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('../..')
from geopack import t96
from geopack.t96_vectorized import t96_vectorized


def quick_solar_wind_evaluation():
    """Quick test of T96 under different solar wind conditions."""
    
    print("T96 Accuracy Evaluation Under Various Solar Wind Conditions")
    print("=" * 60)
    
    # Define test conditions
    test_conditions = [
        # (name, pdyn, dst, byimf, bzimf)
        ("Quiet Northward", 1.0, -10.0, 0.0, 5.0),
        ("Moderate Activity", 3.0, -30.0, -5.0, 0.0),
        ("Storm Southward", 8.0, -100.0, 10.0, -10.0),
        ("Extreme Storm", 20.0, -200.0, -15.0, -20.0),
        ("Strong By", 5.0, -50.0, 20.0, -5.0),
        ("Recovery Phase", 2.0, -40.0, 5.0, 2.0),
    ]
    
    # Test points covering magnetosphere
    n_points = 500
    np.random.seed(42)
    r = np.random.uniform(2, 30, n_points)
    theta = np.random.uniform(0, np.pi, n_points)
    phi = np.random.uniform(0, 2*np.pi, n_points)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    ps = 0.2  # Dipole tilt
    
    results = []
    
    for name, pdyn, dst, byimf, bzimf in test_conditions:
        print(f"\n{name}:")
        print(f"  Pdyn={pdyn:.1f} nPa, Dst={dst:.0f} nT, By={byimf:.0f} nT, Bz={bzimf:.0f} nT")
        
        parmod = np.array([pdyn, dst, byimf, bzimf, 0, 0, 0, 0, 0, 0])
        
        # Calculate with scalar version
        t0 = time.perf_counter()
        bx_scalar = np.zeros(n_points)
        by_scalar = np.zeros(n_points)
        bz_scalar = np.zeros(n_points)
        for i in range(n_points):
            bx_scalar[i], by_scalar[i], bz_scalar[i] = t96.t96(
                parmod, ps, x[i], y[i], z[i]
            )
        t_scalar = time.perf_counter() - t0
        
        # Calculate with vectorized version
        t0 = time.perf_counter()
        bx_vector, by_vector, bz_vector = t96_vectorized(parmod, ps, x, y, z)
        t_vector = time.perf_counter() - t0
        
        # Calculate errors
        b_mag = np.sqrt(bx_scalar**2 + by_scalar**2 + bz_scalar**2)
        mask = b_mag > 1e-10
        
        if np.any(mask):
            errors = np.sqrt((bx_vector[mask] - bx_scalar[mask])**2 + 
                           (by_vector[mask] - by_scalar[mask])**2 + 
                           (bz_vector[mask] - bz_scalar[mask])**2) / b_mag[mask]
            
            mean_error = np.mean(errors)
            max_error = np.max(errors)
            percentile_99 = np.percentile(errors, 99)
        else:
            mean_error = max_error = percentile_99 = 0.0
        
        speedup = t_scalar / t_vector
        
        print(f"  Mean error: {mean_error:.2e}")
        print(f"  Max error: {max_error:.2e}")
        print(f"  99th percentile: {percentile_99:.2e}")
        print(f"  Speedup: {speedup:.1f}x")
        
        results.append({
            'name': name,
            'pdyn': pdyn,
            'dst': dst,
            'byimf': byimf,
            'bzimf': bzimf,
            'mean_error': mean_error,
            'max_error': max_error,
            'percentile_99': percentile_99,
            'speedup': speedup,
            'errors': errors if np.any(mask) else np.array([0])
        })
    
    return results


def plot_results(results):
    """Create visualization of results."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract data
    names = [r['name'] for r in results]
    max_errors = [r['max_error'] for r in results]
    mean_errors = [r['mean_error'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    # 1. Error comparison
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, mean_errors, width, label='Mean Error', alpha=0.7)
    bars2 = ax1.bar(x + width/2, max_errors, width, label='Max Error', alpha=0.7)
    
    ax1.set_yscale('log')
    ax1.set_ylabel('Relative Error')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Accuracy by Solar Wind Condition')
    ax1.axhline(y=1e-6, color='r', linestyle='--', alpha=0.5)
    
    # 2. Parameter visualization
    pdyn = [r['pdyn'] for r in results]
    dst = [abs(r['dst']) for r in results]
    
    ax2.scatter(pdyn, max_errors, s=[d*2 for d in dst], alpha=0.6, c=range(len(results)), cmap='viridis')
    ax2.set_xlabel('Pdyn (nPa)')
    ax2.set_ylabel('Max Error')
    ax2.set_yscale('log')
    ax2.set_title('Error vs Solar Wind Pressure (size = |Dst|)')
    ax2.grid(True, alpha=0.3)
    
    # 3. IMF effects
    bzimf = [r['bzimf'] for r in results]
    byimf = [r['byimf'] for r in results]
    
    scatter = ax3.scatter(byimf, bzimf, c=np.log10(max_errors), s=200, cmap='plasma')
    ax3.set_xlabel('By IMF (nT)')
    ax3.set_ylabel('Bz IMF (nT)')
    ax3.set_title('IMF Configuration (color = log10(error))')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Add condition labels
    for i, name in enumerate(names):
        ax3.annotate(name, (byimf[i], bzimf[i]), fontsize=8, 
                    xytext=(5, 5), textcoords='offset points')
    
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Log10(Max Error)')
    
    # 4. Speedup comparison
    bars = ax4.bar(names, speedups, color=plt.cm.viridis(np.linspace(0, 1, len(names))))
    ax4.set_ylabel('Speedup Factor')
    ax4.set_xticklabels(names, rotation=45, ha='right')
    ax4.set_title('Vectorization Performance')
    ax4.grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.1f}x', ha='center', va='bottom')
    
    plt.suptitle('T96 Vectorization Performance Across Solar Wind Conditions', fontsize=14)
    plt.tight_layout()
    plt.savefig('t96_solar_wind_conditions_quick.png', dpi=150, bbox_inches='tight')
    plt.show()


def test_magnetopause_boundary():
    """Test accuracy near the magnetopause under different conditions."""
    
    print("\n\nTesting Magnetopause Boundary Accuracy...")
    print("=" * 60)
    
    # Points near typical magnetopause locations
    test_points = [
        (10.0, 0.0, 0.0),   # Subsolar
        (8.0, 6.0, 0.0),    # Dawn-dusk
        (6.0, 0.0, 6.0),    # High latitude
        (5.0, 5.0, 5.0),    # Off-axis
    ]
    
    conditions = [
        ("Low pressure", 0.5, -20, 0, 2),
        ("High pressure", 10.0, -50, 5, -5),
        ("Strong southward", 3.0, -100, 0, -15),
    ]
    
    ps = 0.2
    
    for cond_name, pdyn, dst, byimf, bzimf in conditions:
        print(f"\n{cond_name}: Pdyn={pdyn}, Dst={dst}, By={byimf}, Bz={bzimf}")
        parmod = np.array([pdyn, dst, byimf, bzimf, 0, 0, 0, 0, 0, 0])
        
        max_error = 0
        for x, y, z in test_points:
            bx_s, by_s, bz_s = t96.t96(parmod, ps, x, y, z)
            bx_v, by_v, bz_v = t96_vectorized(parmod, ps, x, y, z)
            
            b_mag = np.sqrt(bx_s**2 + by_s**2 + bz_s**2)
            if b_mag > 1e-10:
                error = np.sqrt((bx_v-bx_s)**2 + (by_v-by_s)**2 + (bz_v-bz_s)**2) / b_mag
                max_error = max(max_error, error)
                print(f"  ({x:4.1f},{y:4.1f},{z:4.1f}): B={b_mag:6.1f} nT, error={error:.2e}")
        
        print(f"  Maximum error: {max_error:.2e}")


def main():
    """Run solar wind condition evaluation."""
    
    # Quick evaluation
    results = quick_solar_wind_evaluation()
    
    # Plot results
    plot_results(results)
    
    # Test magnetopause boundary
    test_magnetopause_boundary()
    
    # Summary statistics
    print("\n\nOVERALL SUMMARY")
    print("=" * 60)
    
    all_errors = []
    for r in results:
        all_errors.extend(r['errors'])
    all_errors = np.array(all_errors)
    
    print(f"Total conditions tested: {len(results)}")
    print(f"Total error measurements: {len(all_errors)}")
    print(f"Overall mean error: {np.mean(all_errors):.2e}")
    print(f"Overall max error: {np.max(all_errors):.2e}")
    print(f"Overall 99th percentile: {np.percentile(all_errors, 99):.2e}")
    
    errors_above_threshold = np.sum(all_errors > 1e-6)
    print(f"Points with error > 1e-6: {errors_above_threshold} ({100*errors_above_threshold/len(all_errors):.2f}%)")
    
    if np.max(all_errors) < 1e-6:
        print("\n✓ T96 vectorization maintains excellent accuracy across all tested solar wind conditions!")
    else:
        print(f"\n⚠ Maximum error {np.max(all_errors):.2e} exceeds 1e-6 threshold")
        
    # List conditions with highest errors
    print("\nConditions ranked by maximum error:")
    sorted_results = sorted(results, key=lambda x: x['max_error'], reverse=True)
    for i, r in enumerate(sorted_results[:3]):
        print(f"  {i+1}. {r['name']}: {r['max_error']:.2e}")


if __name__ == "__main__":
    main()