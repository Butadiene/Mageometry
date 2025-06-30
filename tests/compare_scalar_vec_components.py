"""
Compare scalar and vectorized component by component.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopack import t01
from geopack.t01_vectorized import t01_vectorized, extall_vectorized, calculate_parameters, shlcar3x3_vectorized_partial


def compare_components():
    """Compare components."""
    print("COMPONENT COMPARISON")
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
    
    params = calculate_parameters(parmod, ps, a, 1)
    xappa = params.xappa
    print(f"xappa = {xappa:.6f}")
    
    xx = x * xappa
    yy = y * xappa
    zz = z * xappa
    print(f"Scaled coords: xx={xx:.6f}, yy={yy:.6f}, zz={zz:.6f}")
    
    # First, let's directly compare shlcar3x3
    print("\n" + "=" * 80)
    print("SHLCAR3X3 COMPARISON:")
    
    # Scalar
    bx_s3_scalar, by_s3_scalar, bz_s3_scalar = t01.shlcar3x3(xx, yy, zz, ps)
    print(f"Scalar shlcar3x3: Bz={bz_s3_scalar:.6f}")
    
    # Vectorized
    bx_s3_vec, by_s3_vec, bz_s3_vec = shlcar3x3_vectorized_partial(
        np.array([xx]), np.array([yy]), np.array([zz]), ps
    )
    print(f"Vectorized shlcar3x3: Bz={bz_s3_vec[0]:.6f}")
    print(f"Difference: {bz_s3_vec[0] - bz_s3_scalar:.6f}")
    
    # Now let's check if the scalar T01 is doing something different
    print("\n" + "=" * 80)
    print("CHECKING SCALAR T01 CALCULATION:")
    
    # The scalar T01 calls extall directly
    bx_t01, by_t01, bz_t01 = t01.t01(parmod, ps, x, y, z)
    print(f"Scalar T01: Bz={bz_t01:.6f}")
    
    # What does scalar extall return?
    pdyn = parmod[0]
    dst_ast = parmod[1]*0.8 - 13.*np.sqrt(pdyn)
    
    bx_extall, by_extall, bz_extall = t01.extall(0, 0, 0, 0, a, 43, pdyn, dst_ast,
                                                 parmod[2], parmod[3], parmod[4], parmod[5],
                                                 ps, x, y, z)
    print(f"Scalar extall(0): Bz={bz_extall:.6f}")
    print(f"They match: {abs(bz_t01 - bz_extall) < 1e-6}")
    
    # Now let's check individual scalar components if we can
    print("\n" + "=" * 80)
    print("HYPOTHESIS:")
    print("The scalar version might be calculating something differently")
    print("Let me check the dipole shield amplitude...")
    
    # In the scalar code, dipole shield is multiplied by a[0]
    print(f"\na[0] = {a[0]}")
    print(f"Scalar dipole shield: {a[0]} * {bz_s3_scalar:.6f} = {a[0] * bz_s3_scalar:.6f}")
    print(f"Vectorized returns: {bz_s3_vec[0]:.6f}")
    
    # Wait, they should both multiply by a[0]...
    # Let me check what the vectorized extall does
    
    # Get vectorized dipole shield component
    _, _, bz_shield_vec = extall_vectorized(1, 0, 0, 0, a, 43, pdyn, dst_ast,
                                           parmod[2], parmod[3], parmod[4], parmod[5],
                                           ps, np.array([xx]), np.array([yy]), np.array([zz]), params)
    print(f"\nVectorized dipole shield from extall: {bz_shield_vec[0]:.6f}")
    print(f"Expected (a[0] * shlcar3x3): {a[0] * bz_s3_vec[0]:.6f}")
    
    # So the vectorized extall is already applying a[0], good
    
    # The issue might be in how the total is calculated
    print("\n" + "=" * 80)
    print("FINAL ANALYSIS:")
    print(f"Scalar T01 returns: {bz_t01:.6f}")
    print(f"Vectorized T01 returns: {bz_t01_vec:.6f}")
    print(f"Difference: {bz_t01_vec - bz_t01:.6f}")
    
    # Let's check if there's a pattern in the errors across components
    vec_sum = -23.529369
    scalar_result = -17.466888
    ratio = vec_sum / scalar_result
    print(f"\nRatio (vec/scalar): {ratio:.6f}")


if __name__ == "__main__":
    # Get vectorized result first
    ps = -0.1
    x = -10.0
    y = 0.0
    z = 0.0
    parmod = np.array([10.0, -150.0, 3.0, -5.0, 2.0, 1.0])
    bx_t01_vec, by_t01_vec, bz_t01_vec = t01_vectorized(parmod, ps, x, y, z)
    
    compare_components()