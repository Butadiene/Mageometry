#!/usr/bin/env python
"""Test script to validate the fixes in 04_accuracy_validation.ipynb"""

import numpy as np
import geopack
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Initialize geopack
ut = datetime(2020, 1, 1, 12, 0, 0).timestamp()
ps = geopack.recalc(ut)

print("Testing key functionalities from the notebook fixes:")
print("=" * 80)

# Test 1: Test bspcar with spherical coordinates
print("\n1. Testing bspcar with spherical coordinates:")
print("-" * 40)
# Spherical test points (theta, phi)
test_cases = [
    (0.0, 0.0),           # North pole
    (np.pi, 0.0),         # South pole
    (np.pi/2, 0.0),       # Equator
    (np.pi/4, np.pi/6),   # 45 degrees, 30 degrees
]

for theta, phi in test_cases:
    br, btheta, bphi = 1.0, 0.5, 0.3
    
    # Scalar
    bx_s, by_s, bz_s = geopack.bspcar(theta, phi, br, btheta, bphi)
    
    # Vectorized
    theta_arr = np.array([theta])
    phi_arr = np.array([phi])
    br_arr = np.array([br])
    btheta_arr = np.array([btheta])
    bphi_arr = np.array([bphi])
    bx_v, by_v, bz_v = geopack.bspcar_vectorized(theta_arr, phi_arr, br_arr, btheta_arr, bphi_arr)
    
    error = np.sqrt((bx_s - bx_v[0])**2 + (by_s - by_v[0])**2 + (bz_s - bz_v[0])**2)
    status = 'PASS' if error < 1e-14 else 'FAIL'
    
    print(f"   theta={theta:.3f}, phi={phi:.3f}: error={error:.2e} [{status}]")

# Test 2: Test bcarsp handling NaN at origin
print("\n2. Testing bcarsp NaN handling at origin:")
print("-" * 40)
test_points = [
    (0.0, 0.0, 0.0),      # Origin - should produce NaN
    (1e-10, 1e-10, 1e-10), # Very small but not origin
    (1.0, 0.0, 0.0),      # On axis
]

for x, y, z in test_points:
    bx, by, bz = 1.0, 2.0, 3.0
    
    # Scalar
    bx_s, by_s, bz_s = geopack.bcarsp(x, y, z, bx, by, bz)
    
    # Vectorized
    x_arr = np.array([x])
    y_arr = np.array([y])
    z_arr = np.array([z])
    bx_arr = np.array([bx])
    by_arr = np.array([by])
    bz_arr = np.array([bz])
    bx_v, by_v, bz_v = geopack.bcarsp_vectorized(x_arr, y_arr, z_arr, bx_arr, by_arr, bz_arr)
    
    # Check NaN consistency
    scalar_has_nan = np.isnan(bx_s) or np.isnan(by_s) or np.isnan(bz_s)
    vector_has_nan = np.isnan(bx_v[0]) or np.isnan(by_v[0]) or np.isnan(bz_v[0])
    
    if scalar_has_nan and vector_has_nan:
        print(f"   ({x}, {y}, {z}): Both have NaN (expected) [PASS]")
    elif not scalar_has_nan and not vector_has_nan:
        error = np.sqrt((bx_s - bx_v[0])**2 + (by_s - by_v[0])**2 + (bz_s - bz_v[0])**2)
        status = 'PASS' if error < 1e-14 else 'FAIL'
        print(f"   ({x}, {y}, {z}): error={error:.2e} [{status}]")
    else:
        print(f"   ({x}, {y}, {z}): Inconsistent NaN behavior [FAIL]")

# Test 3: Test models with X > -15 Re constraint
print("\n3. Testing models with X > -15 Re constraint:")
print("-" * 40)

# Model parameters
parmod_t01 = np.zeros(10)
parmod_t01[0] = 3.0   # Pdyn
parmod_t01[1] = -20.0 # Dst
parmod_t01[2] = 0.0   # ByIMF
parmod_t01[3] = -5.0  # BzIMF
parmod_t01[4] = 2.0   # G1
parmod_t01[5] = 3.0   # G2

parmod_t04 = parmod_t01.copy()
parmod_t04[4:10] = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]  # W1-W6

# Test points around the X = -15 boundary
test_x_values = [-16.0, -15.5, -15.0, -14.9, -14.5, -10.0]
y, z = 5.0, 2.0

print("   T01 model:")
for x in test_x_values:
    try:
        # Scalar
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bx_s, by_s, bz_s = geopack.t01(parmod_t01, ps, x, y, z)
        
        # Vectorized
        x_arr = np.array([x])
        y_arr = np.array([y])
        z_arr = np.array([z])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bx_v, by_v, bz_v = geopack.t01_vectorized(parmod_t01, ps, x_arr, y_arr, z_arr)
        
        error = np.sqrt((bx_s - bx_v[0])**2 + (by_s - by_v[0])**2 + (bz_s - bz_v[0])**2)
        print(f"      X={x:6.1f}: error={error:.2e} [{'PASS' if error < 1e-10 else 'FAIL'}]")
    except Exception as e:
        print(f"      X={x:6.1f}: Error - {str(e)}")

print("\n   T04 model:")
for x in test_x_values:
    try:
        # Scalar
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bx_s, by_s, bz_s = geopack.t04(parmod_t04, ps, x, y, z)
        
        # Vectorized
        x_arr = np.array([x])
        y_arr = np.array([y])
        z_arr = np.array([z])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bx_v, by_v, bz_v = geopack.t04_vectorized(parmod_t04, ps, x_arr, y_arr, z_arr)
        
        error = np.sqrt((bx_s - bx_v[0])**2 + (by_s - by_v[0])**2 + (bz_s - bz_v[0])**2)
        print(f"      X={x:6.1f}: error={error:.2e} [{'PASS' if error < 1e-10 else 'FAIL'}]")
    except Exception as e:
        print(f"      X={x:6.1f}: Error - {str(e)}")

# Test 4: Test coordinate transformations with edge cases
print("\n4. Testing coordinate transformations with edge cases:")
print("-" * 40)

coord_funcs = [
    ('gsmgse', geopack.gsmgse, geopack.gsmgse_vectorized),
    ('sphcar', geopack.sphcar, geopack.sphcar_vectorized),
    ('bcarsp', geopack.bcarsp, geopack.bcarsp_vectorized),
]

edge_cases = [
    (0.0, 0.0, 0.0),      # Origin
    (1e-15, 1e-15, 1e-15), # Very small
    (1e10, 0, 0),         # Very large
    (1.0, 0.0, 0.0),      # On axis
]

for func_name, scalar_func, vector_func in coord_funcs:
    print(f"\n   {func_name}:")
    for x, y, z in edge_cases:
        try:
            if func_name == 'bcarsp':
                # Special handling for bcarsp
                bx_s, by_s, bz_s = scalar_func(x, y, z, 1.0, 2.0, 3.0)
                x_arr = np.array([x])
                y_arr = np.array([y])
                z_arr = np.array([z])
                bx_v, by_v, bz_v = vector_func(x_arr, y_arr, z_arr, 
                                              np.array([1.0]), np.array([2.0]), np.array([3.0]))
            else:
                # Regular coordinate transform
                bx_s, by_s, bz_s = scalar_func(x, y, z, 1)
                x_arr = np.array([x])
                y_arr = np.array([y])
                z_arr = np.array([z])
                bx_v, by_v, bz_v = vector_func(x_arr, y_arr, z_arr, 1)
            
            # Check for NaN
            scalar_nan = np.isnan(bx_s) or np.isnan(by_s) or np.isnan(bz_s)
            vector_nan = np.isnan(bx_v[0]) or np.isnan(by_v[0]) or np.isnan(bz_v[0])
            
            if scalar_nan and vector_nan:
                print(f"      ({x:.0e}, {y:.0e}, {z:.0e}): Both NaN [PASS]")
            elif not scalar_nan and not vector_nan:
                error = np.sqrt((bx_s - bx_v[0])**2 + (by_s - by_v[0])**2 + (bz_s - bz_v[0])**2)
                print(f"      ({x:.0e}, {y:.0e}, {z:.0e}): error={error:.2e} [{'PASS' if error < 1e-14 else 'FAIL'}]")
            else:
                print(f"      ({x:.0e}, {y:.0e}, {z:.0e}): Inconsistent NaN [FAIL]")
                
        except Exception as e:
            print(f"      ({x:.0e}, {y:.0e}, {z:.0e}): Error - {str(e)}")

print("\n" + "=" * 80)
print("Test validation complete!")