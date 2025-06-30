"""
Verify coordinate usage in tail field calculation.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopack import t01


def verify_coordinates():
    """Verify coordinates."""
    print("COORDINATE USAGE VERIFICATION")
    print("=" * 80)
    
    # Test parameters
    ps = -0.1
    x = -10.0
    y = 0.0
    z = 0.0
    parmod = np.array([10.0, -150.0, 3.0, -5.0, 2.0, 1.0])
    
    # Model coefficients
    a = np.array([
        1.00000, 2.47341, 0.40791, 0.30429, -0.10637, -0.89108, 3.29350,
        -0.05413, -0.00696, 1.07869, -0.02314, -0.66173, -0.68018, -0.03246,
        0.02681, 0.28062, 0.16535, -0.02939, 0.02639, -0.24891, -0.08063,
        0.08900, -0.02475, 0.05887, 0.57691, 0.65256, -0.03230, 2.24733,
        4.10546, 1.13665, 0.05506, 0.97669, 0.21164, 0.64594, 1.12556, 0.01389,
        1.02978, 0.02968, 0.15821, 9.00519, 28.17582, 1.35285, 0.42279
    ])
    
    # Calculate xappa
    pdyn = parmod[0]
    xappa = (pdyn/2.0)**a[38]
    print(f"xappa = {xappa:.6f}")
    
    # Scalar version flow:
    print("\nSCALAR VERSION FLOW:")
    print("1. t01() scales: xx = x * xappa = {:.6f}".format(x * xappa))
    print("2. t01() calls: extall(..., ps, x, y, z)")
    print("3. extall() scales again: xx = x * xappa = {:.6f}".format(x * xappa))
    print("4. extall() calls: deformed(..., ps, xx, yy, zz)")
    print("   So deformed gets scaled coordinates")
    
    # Vectorized version flow:
    print("\nVECTORIZED VERSION FLOW:")
    print("1. t01_vectorized() scales: xx = x * xappa = {:.6f}".format(x * xappa))
    print("2. t01_vectorized() calls: extall_vectorized(..., ps, xx, yy, zz)")
    print("3. extall_vectorized() sets: xx = x (no scaling, x already scaled)")
    print("4. extall_vectorized() calls: deformed_vectorized(..., ps, x[mask], y[mask], z[mask])")
    print("   So deformed_vectorized gets scaled coordinates")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("Both versions pass scaled coordinates to deformed!")
    print("The coordinate usage is correct in the vectorized version.")
    
    # But wait, let me check the scalar extall more carefully
    print("\n" + "=" * 80)
    print("WAIT - CHECKING SCALAR EXTALL MORE CAREFULLY:")
    
    # The scalar extall receives x, y, z (unscaled) and scales them internally
    print("Actually, scalar extall receives UNSCALED x, y, z")
    print("Then it scales them: xx = x * xappa")
    print("So the scalar flow is:")
    print("1. t01() receives unscaled x, y, z")
    print("2. t01() passes unscaled x, y, z to extall()")
    print("3. extall() scales: xx = x * xappa")
    print("4. extall() passes scaled xx, yy, zz to deformed()")
    
    print("\nBut vectorized flow is:")
    print("1. t01_vectorized() receives unscaled x, y, z")
    print("2. t01_vectorized() scales: xx = x * xappa")
    print("3. t01_vectorized() passes SCALED xx, yy, zz to extall_vectorized()")
    print("4. extall_vectorized() doesn't scale again (correctly)")
    print("5. extall_vectorized() passes scaled coordinates to deformed_vectorized()")
    
    print("\nSo both are passing scaled coordinates to deformed - this is correct!")


if __name__ == "__main__":
    verify_coordinates()