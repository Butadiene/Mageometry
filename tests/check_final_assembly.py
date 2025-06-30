"""
Check how field components are assembled in scalar vs vectorized.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopack import t01
from geopack.t01_vectorized import t01_vectorized, extall_vectorized, calculate_parameters


def check_assembly():
    """Check field assembly."""
    print("FIELD ASSEMBLY COMPARISON")
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
    
    # Get results
    bx_scalar, by_scalar, bz_scalar = t01.t01(parmod, ps, x, y, z)
    bx_vec, by_vec, bz_vec = t01_vectorized(parmod, ps, x, y, z)
    
    print(f"Scalar T01: Bz={bz_scalar:.6f}")
    print(f"Vectorized T01: Bz={bz_vec:.6f}")
    print(f"Error: {bz_vec - bz_scalar:.6f}")
    
    # Let's check the scalar formula from extall
    print("\n" + "=" * 80)
    print("SCALAR FIELD ASSEMBLY (from t01.py:extall):")
    print("bbz = a[0]*bzcf + tamp1*bzt1 + tamp2*bzt2 + a_src*bzsrc + a_prc*bzprc")
    print("      + a_r11*bzr11 + a_r12*bzr12 + a_r21*bzr21 + a_r22*bzr22")
    print("      + a[23]*hzimf + a[24]*hzimf*sthetah")
    
    # Get vectorized components
    params = calculate_parameters(parmod, ps, a, 1)
    pdyn = parmod[0]
    dst = parmod[1]
    dst_ast = dst * 0.8 - 13.0 * np.sqrt(pdyn)
    
    xx = x * params.xappa
    yy = y * params.xappa
    zz = z * params.xappa
    
    print("\n" + "=" * 80)
    print("VECTORIZED COMPONENTS:")
    
    components = {}
    names = ['dipole_shield', 'tail', 'birkeland', 'ring_current', 'interconnection']
    
    for i, name in enumerate(names, 1):
        _, _, bz = extall_vectorized(i, 0, 0, 0, a, 43, pdyn, dst_ast,
                                    parmod[2], parmod[3], parmod[4], parmod[5],
                                    ps, np.array([xx]), np.array([yy]), np.array([zz]), params)
        components[name] = bz[0]
        print(f"  {name}: {bz[0]:.6f}")
    
    total = sum(components.values())
    print(f"\nSum: {total:.6f}")
    
    # Check interconnection field calculation
    print("\n" + "=" * 80)
    print("INTERCONNECTION FIELD:")
    print(f"Value: {components['interconnection']:.6f}")
    
    # The interconnection field formula includes:
    # a[23]*hzimf + a[24]*hzimf*sthetah
    # where hzimf = bzimf * factimf
    
    byimf = parmod[2]
    bzimf = parmod[3]
    
    if byimf == 0 and bzimf == 0:
        theta = 0.0
    else:
        theta = np.arctan2(byimf, bzimf)
        if theta <= 0:
            theta += 2 * np.pi
    
    sthetah = np.sin(theta / 2.0) ** 2
    factimf = a[23] + a[24] * sthetah
    hzimf = bzimf * factimf
    
    print(f"\nManual calculation:")
    print(f"  byimf = {byimf}, bzimf = {bzimf}")
    print(f"  theta = {theta:.6f}")
    print(f"  sthetah = {sthetah:.6f}")
    print(f"  factimf = a[23] + a[24]*sthetah = {a[23]} + {a[24]}*{sthetah:.3f} = {factimf:.6f}")
    print(f"  hzimf = bzimf * factimf = {bzimf} * {factimf:.6f} = {hzimf:.6f}")
    
    # The interconnection contribution should be:
    # a[23]*hzimf + a[24]*hzimf*sthetah
    intercon_manual = a[23] * hzimf + a[24] * hzimf * sthetah
    print(f"  Interconnection = a[23]*hzimf + a[24]*hzimf*sthetah")
    print(f"                  = {a[23]}*{hzimf:.3f} + {a[24]}*{hzimf:.3f}*{sthetah:.3f}")
    print(f"                  = {intercon_manual:.6f}")
    
    print(f"\nVectorized reports: {components['interconnection']:.6f}")
    print(f"Difference: {components['interconnection'] - intercon_manual:.6f}")


if __name__ == "__main__":
    check_assembly()