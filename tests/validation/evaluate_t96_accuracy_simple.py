#!/usr/bin/env python3
"""
Comprehensive accuracy evaluation of T96 vectorized implementation.
Tests across various spatial regions, parameter sets, and field conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Direct imports to avoid module issues
import sys
sys.path.insert(0, 'geopack')
from t96 import t96
from t96_vectorized import t96_vectorized

def calculate_errors(scalar_b, vector_b):
    """Calculate absolute and relative errors."""
    abs_err = np.abs(vector_b - scalar_b)
    
    # Calculate relative error safely
    magnitude = np.sqrt(np.sum(scalar_b**2, axis=-1))
    rel_err = np.zeros_like(magnitude)
    mask = magnitude > 1e-10
    rel_err[mask] = np.sqrt(np.sum(abs_err[mask]**2, axis=-1)) / magnitude[mask]
    
    return abs_err, rel_err, magnitude

def test_spatial_regions():
    """Test accuracy across different magnetospheric regions."""
    print("\n" + "="*80)
    print("SPATIAL REGION ACCURACY TEST")
    print("="*80)
    
    # Define test regions
    regions = {
        'Near Earth': {
            'r_range': (1, 5),
            'n_points': 1000,
            'description': 'Inner magnetosphere'
        },
        'Ring Current': {
            'r_range': (3, 8),
            'n_points': 1000,
            'description': 'Ring current region'
        },
        'Tail Close': {
            'r_range': (10, 20),
            'n_points': 1000,
            'description': 'Near Earth tail'
        },
        'Tail Far': {
            'r_range': (20, 40),
            'n_points': 1000,
            'description': 'Distant tail'
        },
        'Magnetopause': {
            'r_range': (8, 12),
            'n_points': 1000,
            'description': 'Near magnetopause'
        }
    }
    
    # Standard parameters
    parmod = [2.0, -20.0, 3.0, -5.0, 0, 0, 0, 0, 0, 0]
    ps = 0.1
    
    results = {}
    
    for region_name, region_info in regions.items():
        # Generate random points in spherical coordinates
        r_min, r_max = region_info['r_range']
        n_pts = region_info['n_points']
        
        # Random spherical coordinates
        r = np.random.uniform(r_min, r_max, n_pts)
        theta = np.random.uniform(0, np.pi, n_pts)
        phi = np.random.uniform(0, 2*np.pi, n_pts)
        
        # Convert to Cartesian
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        # Calculate fields
        scalar_b = np.zeros((n_pts, 3))
        for i in range(n_pts):
            bx, by, bz = t96(parmod, ps, x[i], y[i], z[i])
            scalar_b[i] = [bx, by, bz]
        
        bx_vec, by_vec, bz_vec = t96_vectorized(parmod, ps, x, y, z)
        vector_b = np.stack([bx_vec, by_vec, bz_vec], axis=-1)
        
        # Calculate errors
        abs_err, rel_err, magnitude = calculate_errors(scalar_b, vector_b)
        
        results[region_name] = {
            'abs_err': abs_err,
            'rel_err': rel_err,
            'magnitude': magnitude,
            'x': x, 'y': y, 'z': z
        }
        
        # Print statistics
        print(f"\n{region_name} ({region_info['description']}):")
        print(f"  Points tested: {n_pts}")
        print(f"  R range: {r_min:.1f} - {r_max:.1f} Re")
        print(f"  Mean |B|: {np.mean(magnitude):.2f} nT")
        print(f"  Relative errors:")
        print(f"    Mean: {np.mean(rel_err)*100:.4f}%")
        print(f"    Median: {np.median(rel_err)*100:.4f}%")
        print(f"    95th percentile: {np.percentile(rel_err, 95)*100:.4f}%")
        print(f"    99th percentile: {np.percentile(rel_err, 99)*100:.4f}%")
        print(f"    Max: {np.max(rel_err)*100:.4f}%")
    
    return results

def test_parameter_space():
    """Test accuracy across different parameter combinations."""
    print("\n" + "="*80)
    print("PARAMETER SPACE ACCURACY TEST")
    print("="*80)
    
    # Define parameter ranges
    pdyn_values = [0.5, 1.0, 2.0, 5.0, 10.0]  # Solar wind pressure
    dst_values = [0, -20, -50, -100, -200]    # Dst index
    byimf_values = [-5, -2, 0, 2, 5]          # IMF By
    bzimf_values = [-10, -5, 0, 5, 10]        # IMF Bz
    
    # Test points
    test_points = [
        (5, 0, 0),    # Dayside
        (-10, 0, 0),  # Nightside
        (0, 8, 0),    # Flank
        (3, 3, 3),    # Off-equator
        (-15, 0, 5)   # Tail off-equator
    ]
    
    ps = 0.1
    results = []
    
    for pdyn in pdyn_values:
        for dst in dst_values:
            for byimf in byimf_values[:3]:  # Subset to reduce computation
                for bzimf in bzimf_values[:3]:
                    parmod = [pdyn, dst, byimf, bzimf, 0, 0, 0, 0, 0, 0]
                    
                    errors = []
                    for x, y, z in test_points:
                        # Scalar
                        bx_s, by_s, bz_s = t96(parmod, ps, x, y, z)
                        scalar_b = np.array([bx_s, by_s, bz_s])
                        
                        # Vector
                        bx_v, by_v, bz_v = t96_vectorized(parmod, ps, x, y, z)
                        vector_b = np.array([bx_v, by_v, bz_v])
                        
                        # Error
                        if np.linalg.norm(scalar_b) > 1e-10:
                            rel_err = np.linalg.norm(vector_b - scalar_b) / np.linalg.norm(scalar_b)
                            errors.append(rel_err)
                    
                    if errors:
                        mean_error = np.mean(errors) * 100
                        max_error = np.max(errors) * 100
                        results.append({
                            'pdyn': pdyn, 'dst': dst, 
                            'byimf': byimf, 'bzimf': bzimf,
                            'mean_error': mean_error,
                            'max_error': max_error
                        })
    
    # Sort by error and show worst cases
    results.sort(key=lambda x: x['max_error'], reverse=True)
    
    print("\nWorst parameter combinations (by max error):")
    print("Pdyn   Dst    ByIMF  BzIMF  Mean Err%  Max Err%")
    print("-" * 50)
    for r in results[:10]:
        print(f"{r['pdyn']:4.1f}  {r['dst']:5.0f}  {r['byimf']:5.1f}  {r['bzimf']:5.1f}  "
              f"{r['mean_error']:8.4f}  {r['max_error']:8.4f}")
    
    return results

def test_extreme_conditions():
    """Test accuracy under extreme conditions."""
    print("\n" + "="*80)
    print("EXTREME CONDITIONS TEST")
    print("="*80)
    
    test_cases = [
        {
            'name': 'Extreme storm',
            'parmod': [10.0, -400.0, 10.0, -20.0, 0, 0, 0, 0, 0, 0],
            'points': [(3, 0, 0), (5, 5, 0), (-10, 0, 0)]
        },
        {
            'name': 'Very quiet',
            'parmod': [0.5, 10.0, 0.0, 5.0, 0, 0, 0, 0, 0, 0],
            'points': [(10, 0, 0), (0, 15, 0), (-20, 0, 0)]
        },
        {
            'name': 'Strong northward IMF',
            'parmod': [5.0, -20.0, 0.0, 20.0, 0, 0, 0, 0, 0, 0],
            'points': [(8, 0, 2), (0, 10, 0), (-15, 0, 0)]
        },
        {
            'name': 'Large By component',
            'parmod': [3.0, -50.0, 20.0, -5.0, 0, 0, 0, 0, 0, 0],
            'points': [(5, 5, 0), (0, -8, 3), (-12, 5, 0)]
        }
    ]
    
    ps = 0.1
    
    for test in test_cases:
        print(f"\n{test['name']}:")
        print(f"  Parameters: Pdyn={test['parmod'][0]}, Dst={test['parmod'][1]}, "
              f"ByIMF={test['parmod'][2]}, BzIMF={test['parmod'][3]}")
        
        errors = []
        for x, y, z in test['points']:
            # Scalar
            bx_s, by_s, bz_s = t96(test['parmod'], ps, x, y, z)
            scalar_b = np.array([bx_s, by_s, bz_s])
            
            # Vector
            bx_v, by_v, bz_v = t96_vectorized(test['parmod'], ps, x, y, z)
            vector_b = np.array([bx_v, by_v, bz_v])
            
            # Error
            mag_s = np.linalg.norm(scalar_b)
            if mag_s > 1e-10:
                rel_err = np.linalg.norm(vector_b - scalar_b) / mag_s
                errors.append(rel_err)
                print(f"    Point ({x:3.0f},{y:3.0f},{z:3.0f}): "
                      f"|B|={mag_s:6.1f} nT, error={rel_err*100:.4f}%")
        
        if errors:
            print(f"  Mean error: {np.mean(errors)*100:.4f}%")
            print(f"  Max error: {np.max(errors)*100:.4f}%")

def test_component_contributions():
    """Analyze which T96 components contribute most to errors."""
    print("\n" + "="*80)
    print("COMPONENT CONTRIBUTION ANALYSIS")
    print("="*80)
    
    # Import component functions
    from t96 import (tailrc96, dipshld, birk1tot_02, birk2tot_02, intercon)
    from t96_vectorized import (tailrc96_vectorized, dipshld_vectorized,
                               birk1tot_02_vectorized, birk2tot_02_vectorized,
                               intercon_vectorized)
    
    # Test parameters
    parmod = [2.0, -20.0, 3.0, -5.0, 0, 0, 0, 0, 0, 0]
    ps = 0.1
    sps = np.sin(ps)
    
    # Test points spread across magnetosphere
    test_points = [
        (5, 0, 0), (0, 5, 0), (0, 0, 5),
        (3, 4, 2), (-10, 0, 0), (0, -8, 3),
        (-15, 5, 0), (8, -6, 4), (-20, 0, 5)
    ]
    
    components = {
        'Dipole Shield': (dipshld, dipshld_vectorized, (ps,)),
        'Tail Current': (tailrc96, tailrc96_vectorized, (sps,)),
        'Birkeland 1': (birk1tot_02, birk1tot_02_vectorized, (ps,)),
        'Birkeland 2': (birk2tot_02, birk2tot_02_vectorized, (ps,)),
        'Interconnection': (intercon, intercon_vectorized, ())
    }
    
    component_errors = {name: [] for name in components}
    
    for x, y, z in test_points:
        for comp_name, (scalar_func, vector_func, extra_args) in components.items():
            # Scalar
            bx_s, by_s, bz_s = scalar_func(*extra_args, x, y, z)
            
            # Vector
            bx_v, by_v, bz_v = vector_func(*extra_args, x, y, z)
            
            # Error
            scalar_b = np.array([bx_s, by_s, bz_s])
            vector_b = np.array([bx_v, by_v, bz_v])
            
            mag_s = np.linalg.norm(scalar_b)
            if mag_s > 1e-10:
                rel_err = np.linalg.norm(vector_b - scalar_b) / mag_s
                component_errors[comp_name].append(rel_err)
    
    # Print statistics
    print("\nComponent error statistics:")
    print("Component          Mean %    Max %     Median %")
    print("-" * 50)
    for comp_name in components:
        if component_errors[comp_name]:
            errors = np.array(component_errors[comp_name]) * 100
            print(f"{comp_name:17} {np.mean(errors):7.4f}  {np.max(errors):7.4f}  "
                  f"{np.median(errors):7.4f}")

def main():
    """Run comprehensive accuracy evaluation."""
    print("T96 VECTORIZED IMPLEMENTATION - COMPREHENSIVE ACCURACY EVALUATION")
    print("================================================================")
    
    # Run tests
    spatial_results = test_spatial_regions()
    param_results = test_parameter_space()
    test_extreme_conditions()
    test_component_contributions()
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL ACCURACY SUMMARY")
    print("="*80)
    
    # Collect all errors
    all_errors = []
    for data in spatial_results.values():
        all_errors.extend(data['rel_err'])
    
    all_errors = np.array(all_errors) * 100
    
    print(f"\nTotal points tested: {len(all_errors)}")
    print(f"Mean relative error: {np.mean(all_errors):.4f}%")
    print(f"Median relative error: {np.median(all_errors):.4f}%")
    print(f"95th percentile: {np.percentile(all_errors, 95):.4f}%")
    print(f"99th percentile: {np.percentile(all_errors, 99):.4f}%")
    print(f"99.9th percentile: {np.percentile(all_errors, 99.9):.4f}%")
    print(f"Maximum error: {np.max(all_errors):.4f}%")
    
    # Error distribution
    print(f"\nError distribution:")
    print(f"  < 0.01%: {np.sum(all_errors < 0.01) / len(all_errors) * 100:.1f}%")
    print(f"  < 0.1%: {np.sum(all_errors < 0.1) / len(all_errors) * 100:.1f}%")
    print(f"  < 1%: {np.sum(all_errors < 1) / len(all_errors) * 100:.1f}%")
    print(f"  < 5%: {np.sum(all_errors < 5) / len(all_errors) * 100:.1f}%")
    
    # Performance check
    print("\n" + "="*80)
    print("PERFORMANCE CHECK")
    print("="*80)
    
    import time
    n_test = 10000
    x_test = np.random.uniform(-20, 20, n_test)
    y_test = np.random.uniform(-20, 20, n_test)
    z_test = np.random.uniform(-20, 20, n_test)
    
    # Time vectorized version
    start = time.time()
    bx, by, bz = t96_vectorized(param_results[0]['parmod'] if param_results else [2.0, -20.0, 3.0, -5.0, 0, 0, 0, 0, 0, 0], 
                                0.1, x_test, y_test, z_test)
    vec_time = time.time() - start
    
    print(f"Vectorized: {n_test} points in {vec_time:.3f} seconds")
    print(f"Rate: {n_test/vec_time:.0f} points/second")
    
    # Estimate scalar time from a small sample
    n_sample = 100
    start = time.time()
    for i in range(n_sample):
        _ = t96(param_results[0]['parmod'] if param_results else [2.0, -20.0, 3.0, -5.0, 0, 0, 0, 0, 0, 0], 
                0.1, x_test[i], y_test[i], z_test[i])
    sample_time = time.time() - start
    est_scalar_time = sample_time * n_test / n_sample
    
    print(f"Scalar (estimated): {est_scalar_time:.1f} seconds")
    print(f"Speedup: {est_scalar_time/vec_time:.1f}x")

if __name__ == "__main__":
    main()