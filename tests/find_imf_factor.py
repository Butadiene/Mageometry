"""
Find the factor that's reducing IMF by 10x.
"""

import numpy as np


def find_factor():
    """Find what's causing the 10x reduction."""
    
    # The ratio we're seeing
    ratio = 0.102817
    
    print("FINDING THE 10x REDUCTION FACTOR")
    print("=" * 50)
    print(f"Observed ratio: {ratio:.6f}")
    print(f"This is approximately 1/{1/ratio:.3f}")
    
    # Model coefficients
    a = np.array([
        1.00000, 2.47341, 0.40791, 0.30429, -0.10637, -0.89108, 3.29350,
        -0.05413, -0.00696, 1.07869, -0.02314, -0.66173, -0.68018, -0.03246,
        0.02681, 0.28062, 0.16535, -0.02939, 0.02639, -0.24891, -0.08063,
        0.08900, -0.02475, 0.05887, 0.57691, 0.65256, -0.03230, 2.24733,
        4.10546, 1.13665, 0.05506, 0.97669, 0.21164, 0.64594, 1.12556, 0.01389,
        1.02978, 0.02968, 0.15821, 9.00519, 28.17582, 1.35285, 0.42279
    ])
    
    # Check various coefficients that might be the factor
    print("\nChecking coefficients that might be ~0.1:")
    for i, coef in enumerate(a):
        if 0.08 < abs(coef) < 0.12:
            print(f"  a[{i}] = {coef:.6f}")
    
    # Check if it's related to other parameters
    pdyn = 25.0
    xappa = (pdyn / 2.0) ** a[38]
    print(f"\nxappa = (pdyn/2)^a[38] = ({pdyn}/2)^{a[38]:.6f} = {xappa:.6f}")
    
    # Check various combinations
    print("\nChecking combinations:")
    print(f"  a[9] = {a[9]:.6f}")
    print(f"  a[10] = {a[10]:.6f}")
    print(f"  a[11] = {a[11]:.6f}")
    print(f"  a[22] = {a[22]:.6f}")
    
    # The key insight: a[9] = 0.107869 is very close to our ratio!
    print(f"\n*** a[9] = {a[9]:.6f} is very close to ratio = {ratio:.6f}! ***")
    print(f"Difference: {abs(a[9] - ratio):.6f}")
    
    # Let's check if this is the factor
    # a[9] might be a scaling factor for IMF penetration
    print(f"\nIf we multiply the expected IMF by a[9]:")
    expected_y = -4.580564
    expected_z = -5.725705
    print(f"  By: {expected_y:.6f} * {a[9]:.6f} = {expected_y * a[9]:.6f}")
    print(f"  Bz: {expected_z:.6f} * {a[9]:.6f} = {expected_z * a[9]:.6f}")
    
    # Compare to actual
    actual_y = -0.470960
    actual_z = -0.588700
    print(f"\nActual IMF contribution:")
    print(f"  By: {actual_y:.6f}")
    print(f"  Bz: {actual_z:.6f}")
    
    # The match isn't perfect but it's very close!
    # a[9] = 0.107869 vs ratio = 0.102817
    # This suggests a[9] might be an IMF penetration coefficient


if __name__ == "__main__":
    find_factor()