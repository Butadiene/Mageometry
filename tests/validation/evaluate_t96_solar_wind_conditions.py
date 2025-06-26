#!/usr/bin/env python3
"""
Comprehensive T96 accuracy evaluation under various solar wind conditions.

This script tests the vectorized T96 implementation across a wide range of
solar wind parameters to ensure accuracy under all conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
import sys
sys.path.append('../..')
from geopack import t96
from geopack.t96_vectorized import t96_vectorized


def evaluate_solar_wind_conditions():
    """Test T96 under various solar wind conditions."""
    
    # Define solar wind condition categories
    conditions = {
        'Quiet': {
            'pdyn': (0.5, 2.0),      # Low pressure
            'dst': (-20, 20),        # Quiet Dst
            'byimf': (-2, 2),        # Weak IMF
            'bzimf': (0, 5),         # Northward
            'description': 'Quiet solar wind, northward IMF'
        },
        'Moderate': {
            'pdyn': (2.0, 5.0),      # Moderate pressure
            'dst': (-50, -20),       # Moderate storm
            'byimf': (-5, 5),        # Moderate By
            'bzimf': (-5, 5),        # Variable Bz
            'description': 'Moderate activity'
        },
        'Storm': {
            'pdyn': (5.0, 10.0),     # High pressure
            'dst': (-100, -50),      # Storm conditions
            'byimf': (-10, 10),      # Strong By
            'bzimf': (-10, -5),      # Southward
            'description': 'Storm conditions, southward IMF'
        },
        'Extreme': {
            'pdyn': (10.0, 20.0),    # Very high pressure
            'dst': (-200, -100),     # Intense storm
            'byimf': (-15, 15),      # Very strong By
            'bzimf': (-20, -10),     # Strong southward
            'description': 'Extreme storm conditions'
        },
        'Recovery': {
            'pdyn': (1.0, 3.0),      # Decreasing pressure
            'dst': (-50, 0),         # Recovery phase
            'byimf': (-3, 3),        # Weakening By
            'bzimf': (-2, 5),        # Turning northward
            'description': 'Storm recovery phase'
        }
    }
    
    # Test parameters
    n_points_per_condition = 2000
    n_spatial_points = 100
    
    # Results storage
    results = {}
    
    # Fixed tilt angle
    ps = 0.2  # ~11.5 degrees
    
    print("Evaluating T96 accuracy under various solar wind conditions...")
    print("=" * 70)
    
    for condition_name, params in conditions.items():
        print(f"\n{condition_name} Conditions: {params['description']}")
        print("-" * 50)
        
        # Generate random parameters within ranges
        np.random.seed(42)  # For reproducibility
        pdyn = np.random.uniform(params['pdyn'][0], params['pdyn'][1], n_points_per_condition)
        dst = np.random.uniform(params['dst'][0], params['dst'][1], n_points_per_condition)
        byimf = np.random.uniform(params['byimf'][0], params['byimf'][1], n_points_per_condition)
        bzimf = np.random.uniform(params['bzimf'][0], params['bzimf'][1], n_points_per_condition)
        
        # Generate spatial test points
        r = np.random.uniform(2, 30, n_spatial_points)
        theta = np.random.uniform(0, np.pi, n_spatial_points)
        phi = np.random.uniform(0, 2*np.pi, n_spatial_points)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        errors = []
        exec_times_scalar = []
        exec_times_vector = []
        
        # Test each parameter combination
        for i in range(min(100, n_points_per_condition)):  # Limit for performance
            parmod = np.zeros(10)
            parmod[0] = pdyn[i]
            parmod[1] = dst[i]
            parmod[2] = byimf[i]
            parmod[3] = bzimf[i]
            
            # Time scalar calculation
            t0 = time.perf_counter()
            bx_scalar = np.zeros(n_spatial_points)
            by_scalar = np.zeros(n_spatial_points)
            bz_scalar = np.zeros(n_spatial_points)
            for j in range(n_spatial_points):
                bx_scalar[j], by_scalar[j], bz_scalar[j] = t96.t96(
                    parmod, ps, x[j], y[j], z[j]
                )
            t_scalar = time.perf_counter() - t0
            exec_times_scalar.append(t_scalar)
            
            # Time vectorized calculation
            t0 = time.perf_counter()
            bx_vector, by_vector, bz_vector = t96_vectorized(parmod, ps, x, y, z)
            t_vector = time.perf_counter() - t0
            exec_times_vector.append(t_vector)
            
            # Calculate errors
            b_mag = np.sqrt(bx_scalar**2 + by_scalar**2 + bz_scalar**2)
            mask = b_mag > 1e-10
            if np.any(mask):
                rel_error = np.sqrt((bx_vector[mask] - bx_scalar[mask])**2 + 
                                  (by_vector[mask] - by_scalar[mask])**2 + 
                                  (bz_vector[mask] - bz_scalar[mask])**2) / b_mag[mask]
                errors.extend(rel_error)
        
        errors = np.array(errors)
        
        # Store results
        results[condition_name] = {
            'errors': errors,
            'pdyn_range': params['pdyn'],
            'dst_range': params['dst'],
            'byimf_range': params['byimf'],
            'bzimf_range': params['bzimf'],
            'exec_times_scalar': exec_times_scalar,
            'exec_times_vector': exec_times_vector,
            'mean_error': np.mean(errors),
            'max_error': np.max(errors),
            'percentile_99': np.percentile(errors, 99)
        }
        
        # Print statistics
        print(f"  Parameter ranges:")
        print(f"    Pdyn:  {params['pdyn'][0]:.1f} - {params['pdyn'][1]:.1f} nPa")
        print(f"    Dst:   {params['dst'][0]:.0f} - {params['dst'][1]:.0f} nT")
        print(f"    ByIMF: {params['byimf'][0]:.0f} - {params['byimf'][1]:.0f} nT")
        print(f"    BzIMF: {params['bzimf'][0]:.0f} - {params['bzimf'][1]:.0f} nT")
        print(f"  Accuracy:")
        print(f"    Mean error: {np.mean(errors):.2e}")
        print(f"    Max error:  {np.max(errors):.2e}")
        print(f"    99th percentile: {np.percentile(errors, 99):.2e}")
        print(f"  Performance:")
        print(f"    Mean speedup: {np.mean(exec_times_scalar)/np.mean(exec_times_vector):.1f}x")
    
    return results, conditions


def plot_accuracy_by_condition(results, conditions):
    """Create visualization of accuracy across different solar wind conditions."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Error distribution by condition
    ax = axes[0, 0]
    condition_names = list(results.keys())
    errors_by_condition = [results[c]['errors'] for c in condition_names]
    
    bp = ax.boxplot(errors_by_condition, labels=condition_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], plt.cm.viridis(np.linspace(0, 1, len(condition_names)))):
        patch.set_facecolor(color)
    
    ax.set_yscale('log')
    ax.set_ylabel('Relative Error')
    ax.set_title('Error Distribution by Solar Wind Condition')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # 2. Parameter ranges visualization
    ax = axes[0, 1]
    y_pos = np.arange(len(condition_names))
    colors = plt.cm.viridis(np.linspace(0, 1, len(condition_names)))
    
    for i, (name, color) in enumerate(zip(condition_names, colors)):
        # Create stacked bars showing parameter ranges
        pdyn_range = conditions[name]['pdyn']
        dst_range = conditions[name]['dst']
        
        # Normalize to 0-1 for visualization
        pdyn_norm = pdyn_range[1] / 20.0  # Max pdyn = 20
        dst_norm = abs(dst_range[0]) / 200.0  # Max |Dst| = 200
        
        ax.barh(i, pdyn_norm, left=0, height=0.3, color=color, alpha=0.7, label='Pdyn' if i==0 else '')
        ax.barh(i-0.3, dst_norm, left=0, height=0.3, color=color, alpha=0.4, label='|Dst|' if i==0 else '')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(condition_names)
    ax.set_xlabel('Normalized Parameter Range')
    ax.set_title('Solar Wind Parameter Ranges')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Max error vs condition severity
    ax = axes[1, 0]
    max_errors = [results[c]['max_error'] for c in condition_names]
    mean_errors = [results[c]['mean_error'] for c in condition_names]
    
    x = np.arange(len(condition_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, mean_errors, width, label='Mean Error', alpha=0.7)
    bars2 = ax.bar(x + width/2, max_errors, width, label='Max Error', alpha=0.7)
    
    ax.set_yscale('log')
    ax.set_ylabel('Relative Error')
    ax.set_xticks(x)
    ax.set_xticklabels(condition_names, rotation=45)
    ax.set_title('Mean and Maximum Errors by Condition')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add reference line at 1e-6
    ax.axhline(y=1e-6, color='r', linestyle='--', alpha=0.5, label='1e-6 threshold')
    
    # 4. Performance comparison
    ax = axes[1, 1]
    speedups = []
    for name in condition_names:
        scalar_times = results[name]['exec_times_scalar']
        vector_times = results[name]['exec_times_vector']
        speedup = np.mean(scalar_times) / np.mean(vector_times)
        speedups.append(speedup)
    
    bars = ax.bar(condition_names, speedups, color=colors)
    ax.set_ylabel('Speedup Factor')
    ax.set_title('Vectorization Performance by Condition')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add speedup values on bars
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.1f}x', ha='center', va='bottom')
    
    plt.suptitle('T96 Vectorization Accuracy Across Solar Wind Conditions', fontsize=16)
    plt.tight_layout()
    plt.savefig('t96_solar_wind_accuracy.png', dpi=150, bbox_inches='tight')
    plt.show()


def test_extreme_parameters():
    """Test T96 with extreme parameter combinations."""
    print("\n\nTesting extreme parameter combinations...")
    print("=" * 70)
    
    extreme_cases = [
        {
            'name': 'Super Storm',
            'pdyn': 50.0,
            'dst': -400.0,
            'byimf': -20.0,
            'bzimf': -30.0
        },
        {
            'name': 'Extreme Pressure',
            'pdyn': 100.0,
            'dst': -50.0,
            'byimf': 0.0,
            'bzimf': 0.0
        },
        {
            'name': 'Strong Duskward IMF',
            'pdyn': 5.0,
            'dst': -100.0,
            'byimf': 30.0,
            'bzimf': -5.0
        },
        {
            'name': 'Near Zero Pressure',
            'pdyn': 0.1,
            'dst': 0.0,
            'byimf': 0.0,
            'bzimf': 5.0
        }
    ]
    
    # Test points
    test_points = [
        (6.6, 0, 0),    # Dayside
        (-15, 0, 0),    # Nightside
        (0, 10, 0),     # Dusk
        (0, -10, 0),    # Dawn
        (0, 0, 10),     # North
        (3, 3, 3),      # Off-axis
    ]
    
    ps = 0.2
    
    for case in extreme_cases:
        print(f"\n{case['name']}:")
        print(f"  Pdyn={case['pdyn']} nPa, Dst={case['dst']} nT")
        print(f"  ByIMF={case['byimf']} nT, BzIMF={case['bzimf']} nT")
        
        parmod = np.zeros(10)
        parmod[0] = case['pdyn']
        parmod[1] = case['dst']
        parmod[2] = case['byimf']
        parmod[3] = case['bzimf']
        
        errors = []
        for x, y, z in test_points:
            try:
                bx_s, by_s, bz_s = t96.t96(parmod, ps, x, y, z)
                bx_v, by_v, bz_v = t96_vectorized(parmod, ps, x, y, z)
                
                b_mag = np.sqrt(bx_s**2 + by_s**2 + bz_s**2)
                if b_mag > 1e-10:
                    error = np.sqrt((bx_v-bx_s)**2 + (by_v-by_s)**2 + (bz_v-bz_s)**2) / b_mag
                    errors.append(error)
                    
                    if error > 1e-6:
                        print(f"  WARNING: High error at ({x},{y},{z}): {error:.2e}")
            except Exception as e:
                print(f"  ERROR at ({x},{y},{z}): {str(e)}")
        
        if errors:
            print(f"  Max error: {max(errors):.2e}")
            print(f"  All errors < 1e-6: {all(e < 1e-6 for e in errors)}")


def analyze_imf_orientation_effects():
    """Analyze how IMF orientation affects accuracy."""
    print("\n\nAnalyzing IMF orientation effects...")
    print("=" * 70)
    
    # Fixed parameters
    pdyn = 2.0
    dst = -30.0
    ps = 0.2
    
    # IMF orientations (clock angles)
    n_angles = 16
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    b_total = 5.0  # nT
    
    # Test locations
    x = np.linspace(-20, 10, 50)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    
    errors_by_angle = []
    
    for angle in angles:
        byimf = b_total * np.sin(angle)
        bzimf = b_total * np.cos(angle)
        
        parmod = np.array([pdyn, dst, byimf, bzimf, 0, 0, 0, 0, 0, 0])
        
        # Calculate fields
        bx_scalar = np.zeros_like(x)
        by_scalar = np.zeros_like(x)
        bz_scalar = np.zeros_like(x)
        
        for i in range(len(x)):
            bx_scalar[i], by_scalar[i], bz_scalar[i] = t96.t96(parmod, ps, x[i], y[i], z[i])
        
        bx_vector, by_vector, bz_vector = t96_vectorized(parmod, ps, x, y, z)
        
        # Calculate errors
        b_mag = np.sqrt(bx_scalar**2 + by_scalar**2 + bz_scalar**2)
        mask = b_mag > 1e-10
        if np.any(mask):
            errors = np.sqrt((bx_vector[mask] - bx_scalar[mask])**2 + 
                           (by_vector[mask] - by_scalar[mask])**2 + 
                           (bz_vector[mask] - bz_scalar[mask])**2) / b_mag[mask]
            errors_by_angle.append(np.max(errors))
        else:
            errors_by_angle.append(0)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Polar plot of errors
    ax1 = plt.subplot(121, projection='polar')
    ax1.plot(angles, errors_by_angle, 'b-', linewidth=2)
    ax1.scatter(angles, errors_by_angle, c='red', s=50, zorder=5)
    ax1.set_rlabel_position(45)
    ax1.set_title('Max Error vs IMF Clock Angle', pad=20)
    ax1.grid(True)
    
    # Add labels for key directions
    ax1.text(0, ax1.get_ylim()[1]*1.1, 'Northward\n(Bz+)', ha='center')
    ax1.text(np.pi/2, ax1.get_ylim()[1]*1.1, 'Duskward\n(By+)', ha='center')
    ax1.text(np.pi, ax1.get_ylim()[1]*1.1, 'Southward\n(Bz-)', ha='center')
    ax1.text(3*np.pi/2, ax1.get_ylim()[1]*1.1, 'Dawnward\n(By-)', ha='center')
    
    # Cartesian plot
    ax2.plot(np.degrees(angles), np.array(errors_by_angle), 'b-', linewidth=2, marker='o')
    ax2.set_xlabel('IMF Clock Angle (degrees)')
    ax2.set_ylabel('Maximum Relative Error')
    ax2.set_title('Error Variation with IMF Orientation')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 360)
    ax2.set_yscale('log')
    
    # Add vertical lines for cardinal directions
    for angle, label in [(0, 'N'), (90, 'E'), (180, 'S'), (270, 'W')]:
        ax2.axvline(x=angle, color='gray', linestyle='--', alpha=0.5)
        ax2.text(angle, ax2.get_ylim()[1], label, ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('t96_imf_orientation_accuracy.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Maximum error across all IMF orientations: {max(errors_by_angle):.2e}")
    print(f"Minimum error across all IMF orientations: {min(errors_by_angle):.2e}")


def main():
    """Run comprehensive solar wind condition tests."""
    
    # Test accuracy under various conditions
    results, conditions = evaluate_solar_wind_conditions()
    
    # Create visualizations
    plot_accuracy_by_condition(results, conditions)
    
    # Test extreme parameters
    test_extreme_parameters()
    
    # Analyze IMF orientation effects
    analyze_imf_orientation_effects()
    
    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_errors = []
    for condition_name, result in results.items():
        all_errors.extend(result['errors'])
    
    all_errors = np.array(all_errors)
    
    print(f"Total test points: {len(all_errors)}")
    print(f"Overall mean error: {np.mean(all_errors):.2e}")
    print(f"Overall max error: {np.max(all_errors):.2e}")
    print(f"Overall 99th percentile: {np.percentile(all_errors, 99):.2e}")
    print(f"Points exceeding 1e-6: {np.sum(all_errors > 1e-6)} ({100*np.sum(all_errors > 1e-6)/len(all_errors):.2f}%)")
    
    if np.max(all_errors) < 1e-6:
        print("\n✓ T96 vectorization maintains excellent accuracy across all solar wind conditions!")
    else:
        print("\n✗ Some conditions show errors exceeding 1e-6 threshold")


if __name__ == "__main__":
    main()