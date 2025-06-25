#!/usr/bin/env python3
"""
Detailed precision analysis of T96 vectorized implementation.
Compares each component and sub-component to identify precision issues.
"""

import numpy as np
import sys
sys.path.append('geopack')

from t96 import t96
from t96_vectorized import t96_vectorized

# Import individual components for detailed comparison
from t96 import (tailrc96, dipshld, birk1tot_02, birk2tot_02, intercon,
                 r2sheet, r2inner, r2outer, loops4, bconic, condip1)
from t96_vectorized import (tailrc96_vectorized, dipshld_vectorized, 
                           birk1tot_02_vectorized, birk2tot_02_vectorized, 
                           intercon_vectorized, r2sheet_vectorized, 
                           r2inner_vectorized, r2outer_vectorized,
                           loops4_vectorized, bconic_vectorized, condip1_vectorized)

def compare_function(name, scalar_func, vector_func, args, component_names=['Bx', 'By', 'Bz']):
    """Compare scalar and vectorized function outputs."""
    # Get scalar results
    scalar_result = scalar_func(*args)
    if not isinstance(scalar_result, tuple):
        scalar_result = (scalar_result,)
    
    # Get vectorized results - ensure we pass arrays
    vector_args = []
    for arg in args:
        if isinstance(arg, (int, float)):
            vector_args.append(np.array([arg]))
        else:
            vector_args.append(arg)
    
    vector_result = vector_func(*vector_args)
    if not isinstance(vector_result, tuple):
        vector_result = (vector_result,)
    
    print(f"\n{name}:")
    print("-" * 60)
    
    max_error = 0
    for i, (s, v, comp_name) in enumerate(zip(scalar_result, vector_result, component_names)):
        v_scalar = v.item() if hasattr(v, 'item') else v[0]
        
        if abs(s) > 1e-10:
            rel_error = abs(s - v_scalar) / abs(s)
        else:
            rel_error = abs(s - v_scalar)
        
        max_error = max(max_error, rel_error)
        
        print(f"  {comp_name}: scalar={s:12.8f}, vector={v_scalar:12.8f}, "
              f"diff={s-v_scalar:12.8e}, rel_err={rel_error:12.8e}")
    
    return max_error

def analyze_tailrc96_components():
    """Analyze tailrc96 subcomponents in detail."""
    # Parameters
    sps = np.sin(0.0)
    x, y, z = 0.0, 5.0, 0.0
    xx, yy, zz = x, y, z  # No scaling for this test
    
    # Get xappa and related parameters
    xappa = 1.0
    xappa3 = xappa**3
    sxc = np.sin(xappa * xx)
    cxc = np.cos(xappa * xx)
    
    print("\n\nTAILRC96 DETAILED ANALYSIS")
    print("=" * 80)
    
    # Test bconic
    nmax = 5
    print("\nTesting bconic (conical harmonics):")
    cbx_s, cby_s, cbz_s = bconic(x, y, z, nmax)
    cbx_v, cby_v, cbz_v = bconic_vectorized(x, y, z, nmax)
    
    print(f"  Shape: scalar=({nmax} values), vector={cbx_v.shape}")
    for m in range(nmax):
        s_val = [cbx_s[m], cby_s[m], cbz_s[m]]
        v_val = [cbx_v[m,0], cby_v[m,0], cbz_v[m,0]]
        
        print(f"  m={m}: scalar=[{s_val[0]:8.5f}, {s_val[1]:8.5f}, {s_val[2]:8.5f}]")
        print(f"       vector=[{v_val[0]:8.5f}, {v_val[1]:8.5f}, {v_val[2]:8.5f}]")
        for j in range(3):
            if abs(s_val[j]) > 1e-10:
                rel_err = abs(s_val[j] - v_val[j]) / abs(s_val[j])
                print(f"       rel_err[{j}]={rel_err:8.5e}")

def main():
    # Test parameters
    parmod = [2.0, -10.0, 1.0, -5.0, 0, 0, 0, 0, 0, 0]
    ps = 0.0
    
    # Test points with known issues
    test_points = [
        (0.0, 5.0, 0.0),    # Original problem point
        (5.0, 0.0, 0.0),    # 1.95% error
        (0.0, 0.0, 5.0),    # 2.04% error  
        (8.0, 0.0, 0.0),    # 5.80% error (worst case)
        (-5.0, -3.0, 4.0),  # 3.30% error
    ]
    
    print("T96 PRECISION ANALYSIS")
    print("=" * 80)
    print(f"Parameters: parmod = {parmod}")
    print(f"            ps = {ps}")
    
    # Test individual components
    print("\n\nCOMPONENT-BY-COMPONENT COMPARISON")
    print("=" * 80)
    
    for x, y, z in test_points:
        print(f"\n\nTesting point ({x}, {y}, {z})")
        print("-" * 80)
        
        # Calculate scaled coordinates (same as in t96)
        pdyn = parmod[0]
        xappa = (pdyn / 2.0) ** (1.0 / 6.0)
        xx, yy, zz = x * xappa, y * xappa, z * xappa
        
        # Test dipshld
        compare_function("dipshld", dipshld, dipshld_vectorized, 
                        (ps, xx, yy, zz))
        
        # Test tailrc96 with detailed parameters
        sps = np.sin(ps)
        rcampl = -30.659110  # From debug output
        tampl2 = 23.437104
        tampl3 = 6.903000
        
        compare_function("tailrc96", tailrc96, tailrc96_vectorized,
                        (sps, xx, yy, zz, rcampl, tampl2, tampl3))
        
        # Test birk1tot_02
        compare_function("birk1tot_02", birk1tot_02, birk1tot_02_vectorized,
                        (ps, xx, yy, zz))
        
        # Test birk2tot_02
        b1ampl = 0.766449
        b2ampl = 15.328985
        compare_function("birk2tot_02", birk2tot_02, birk2tot_02_vectorized,
                        (ps, xx, yy, zz, b1ampl, b2ampl))
        
        # Test intercon
        compare_function("intercon", intercon, intercon_vectorized,
                        (xx, yy, zz))
        
        # Test loops4 (part of birk2tot_02)
        if x == 0.0 and y == 5.0 and z == 0.0:
            print("\nTesting loops4 (key component):")
            # Parameters from r2inner
            xc, yc, zc = 6.982, 4.949, 0.0
            r = 4.0
            theta, phi = 1.571, 3.142
            compare_function("loops4", loops4, loops4_vectorized,
                            (xx, yy, zz, xc, yc, zc, r, theta, phi))
    
    # Detailed tailrc96 analysis for worst case
    print("\n\nDETAILED TAILRC96 ANALYSIS FOR WORST CASE")
    analyze_tailrc96_components()

if __name__ == "__main__":
    main()