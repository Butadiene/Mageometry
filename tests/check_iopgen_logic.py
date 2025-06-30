"""
Check iopgen logic in extall_vectorized.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopack.t01_vectorized import extall_vectorized, calculate_parameters


def check_iopgen_logic():
    """Check how iopgen affects the output."""
    print("IOPGEN LOGIC CHECK")
    print("=" * 80)
    
    # Test parameters
    parmod = np.array([10.0, -150.0, 3.0, -5.0, 2.0, 1.0])
    ps = -0.1
    x = -10.0
    y = 0.0
    z = 0.0
    
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
    pdyn = parmod[0]
    dst = parmod[1]
    dst_ast = dst * 0.8 - 13.0 * np.sqrt(pdyn)
    
    xx = x * params.xappa
    yy = y * params.xappa
    zz = z * params.xappa
    
    xx_arr = np.array([xx])
    yy_arr = np.array([yy])
    zz_arr = np.array([zz])
    
    print(f"Test point: x={x}, y={y}, z={z}")
    print(f"Scaled: xx={xx:.6f}, yy={yy:.6f}, zz={zz:.6f}")
    
    # Test different iopgen values
    print("\n" + "=" * 80)
    print("TESTING DIFFERENT IOPGEN VALUES:")
    
    for iopgen in range(6):
        bx, by, bz = extall_vectorized(iopgen, 0, 0, 0, a, 43, pdyn, dst_ast,
                                       parmod[2], parmod[3], parmod[4], parmod[5],
                                       ps, xx_arr, yy_arr, zz_arr, params)
        
        if iopgen == 0:
            print(f"\niopgen={iopgen} (all components): Bz={bz[0]:.6f}")
        elif iopgen == 1:
            print(f"iopgen={iopgen} (dipole shield): Bz={bz[0]:.6f}")
        elif iopgen == 2:
            print(f"iopgen={iopgen} (tail field): Bz={bz[0]:.6f}")
        elif iopgen == 3:
            print(f"iopgen={iopgen} (Birkeland): Bz={bz[0]:.6f}")
        elif iopgen == 4:
            print(f"iopgen={iopgen} (ring current): Bz={bz[0]:.6f}")
        elif iopgen == 5:
            print(f"iopgen={iopgen} (interconnection): Bz={bz[0]:.6f}")
    
    # The issue might be that individual components don't have the amplitude factors applied
    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("For dipole shield, we expect a[0] * shlcar3x3 output")
    print(f"a[0] = {a[0]}")
    print("From earlier tests, shlcar3x3 with scaled coords gives Bz=5.424524")
    print(f"So we expect: {a[0]} * 5.424524 = {a[0] * 5.424524:.6f}")
    print(f"But extall_vectorized with iopgen=1 gives: 3.886294")
    
    # Maybe the issue is region handling?
    print("\nThe test point might be in the INSIDE region where external field is returned")
    print("Let me check if the sum of components equals the total...")
    
    # Get individual components
    _, _, bz_shield = extall_vectorized(1, 0, 0, 0, a, 43, pdyn, dst_ast,
                                        parmod[2], parmod[3], parmod[4], parmod[5],
                                        ps, xx_arr, yy_arr, zz_arr, params)
    _, _, bz_tail = extall_vectorized(2, 0, 0, 0, a, 43, pdyn, dst_ast,
                                      parmod[2], parmod[3], parmod[4], parmod[5],
                                      ps, xx_arr, yy_arr, zz_arr, params)
    _, _, bz_birk = extall_vectorized(3, 0, 0, 0, a, 43, pdyn, dst_ast,
                                      parmod[2], parmod[3], parmod[4], parmod[5],
                                      ps, xx_arr, yy_arr, zz_arr, params)
    _, _, bz_rc = extall_vectorized(4, 0, 0, 0, a, 43, pdyn, dst_ast,
                                    parmod[2], parmod[3], parmod[4], parmod[5],
                                    ps, xx_arr, yy_arr, zz_arr, params)
    _, _, bz_imf = extall_vectorized(5, 0, 0, 0, a, 43, pdyn, dst_ast,
                                     parmod[2], parmod[3], parmod[4], parmod[5],
                                     ps, xx_arr, yy_arr, zz_arr, params)
    
    _, _, bz_all = extall_vectorized(0, 0, 0, 0, a, 43, pdyn, dst_ast,
                                     parmod[2], parmod[3], parmod[4], parmod[5],
                                     ps, xx_arr, yy_arr, zz_arr, params)
    
    bz_sum = bz_shield[0] + bz_tail[0] + bz_birk[0] + bz_rc[0] + bz_imf[0]
    
    print(f"\nSum of components: {bz_sum:.6f}")
    print(f"All components (iopgen=0): {bz_all[0]:.6f}")
    print(f"Difference: {bz_all[0] - bz_sum:.6f}")


if __name__ == "__main__":
    check_iopgen_logic()