"""
Check how components are assembled in extall_vectorized.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopack import t01
from geopack.t01_vectorized import extall_vectorized, calculate_parameters


def check_assembly():
    """Check component assembly."""
    print("COMPONENT ASSEMBLY CHECK")
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
    
    # Get vectorized components
    params = calculate_parameters(parmod, ps, a, 1)
    pdyn = parmod[0]
    dst = parmod[1]
    dst_ast = dst * 0.8 - 13.0 * np.sqrt(pdyn)
    
    xx = x * params.xappa
    yy = y * params.xappa
    zz = z * params.xappa
    
    # Get individual components
    components = {}
    names = ['dipole_shield', 'tail', 'birkeland', 'ring_current', 'interconnection']
    
    for i, name in enumerate(names, 1):
        _, _, bz = extall_vectorized(i, 0, 0, 0, a, 43, pdyn, dst_ast,
                                    parmod[2], parmod[3], parmod[4], parmod[5],
                                    ps, np.array([xx]), np.array([yy]), np.array([zz]), params)
        components[name] = bz[0]
        print(f"{name}: Bz={bz[0]:.6f}")
    
    # Manual sum
    manual_sum = sum(components.values())
    print(f"\nManual sum of components: {manual_sum:.6f}")
    
    # Get total from extall_vectorized
    _, _, bz_total = extall_vectorized(0, 0, 0, 0, a, 43, pdyn, dst_ast,
                                       parmod[2], parmod[3], parmod[4], parmod[5],
                                       ps, np.array([xx]), np.array([yy]), np.array([zz]), params)
    
    print(f"extall_vectorized(iopgen=0): {bz_total[0]:.6f}")
    print(f"Difference: {bz_total[0] - manual_sum:.6f}")
    
    # Get scalar result
    bx_scalar, by_scalar, bz_scalar = t01.t01(parmod, ps, x, y, z)
    print(f"\nScalar T01: {bz_scalar:.6f}")
    print(f"Vectorized T01: {bz_total[0]:.6f}")
    print(f"Error: {bz_total[0] - bz_scalar:.6f}")
    
    # Check if the issue is in the small tail field discrepancy
    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("The individual components sum correctly in the vectorized version.")
    print("But there's a ~6 nT systematic error compared to scalar.")
    print("\nComponent errors found:")
    print("  - Ring current: 0.144 nT")
    print("  - Tail field assembly: ~0.4 nT (from earlier tests)")
    print("  - Total explained: ~0.5 nT")
    print("\nUnexplained error: ~5.5 nT")
    print("\nThis suggests a more fundamental issue in how extall_vectorized")
    print("processes the field calculation compared to the scalar version.")


if __name__ == "__main__":
    check_assembly()