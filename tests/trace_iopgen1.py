"""
Trace through extall_vectorized with iopgen=1 to understand the calculation.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopack import t01
from geopack.t01_vectorized import shlcar3x3_vectorized_partial


def trace_iopgen1():
    """Trace iopgen=1 calculation."""
    print("TRACING IOPGEN=1 (DIPOLE SHIELD ONLY)")
    print("=" * 80)
    
    ps = -0.1
    x = -10.0
    y = 0.0
    z = 0.0
    xappa = 1.289983
    
    xx = x * xappa
    yy = y * xappa
    zz = z * xappa
    
    # Model coefficients
    a = np.array([
        1.00000, 2.47341, 0.40791, 0.30429, -0.10637, -0.89108, 3.29350,
        -0.05413, -0.00696, 1.07869, -0.02314, -0.66173, -0.68018, -0.03246,
        0.02681, 0.28062, 0.16535, -0.02939, 0.02639, -0.24891, -0.08063,
        0.08900, -0.02475, 0.05887, 0.57691, 0.65256, -0.03230, 2.24733,
        4.10546, 1.13665, 0.05506, 0.97669, 0.21164, 0.64594, 1.12556, 0.01389,
        1.02978, 0.02968, 0.15821, 9.00519, 28.17582, 1.35285, 0.42279
    ])
    
    print("When iopgen=1, extall_vectorized should:")
    print("1. Calculate dipole shield with shlcar3x3_vectorized_partial")
    print("2. Store in bxcf, bycf, bzcf arrays")
    print("3. Apply a[0] factor when calculating bbz")
    print("4. Return bbz for INSIDE region")
    
    print(f"\nExpected calculation:")
    print(f"1. shlcar3x3 with scaled coords ({xx:.6f}, {yy:.6f}, {zz:.6f})")
    
    bx_shl, by_shl, bz_shl = shlcar3x3_vectorized_partial(
        np.array([xx]), np.array([yy]), np.array([zz]), ps
    )
    print(f"   Result: Bz={bz_shl[0]:.6f}")
    
    print(f"\n2. Apply a[0]={a[0]} factor:")
    print(f"   Result: Bz={a[0] * bz_shl[0]:.6f}")
    
    print(f"\n3. This should be returned for INSIDE region")
    
    print("\nBut extall_vectorized returns 3.886295 instead of 5.424525")
    print(f"Ratio: {3.886295 / 5.424525:.6f}")
    
    # The ratio 0.716 suggests there's a missing factor
    # Let me check if it could be related to how components are summed
    
    print("\n" + "=" * 80)
    print("HYPOTHESIS:")
    print("When iopgen != 0, maybe the function is not going through the")
    print("normal summation path that applies the a[0] factor?")
    
    # Let's check what happens with iopgen=0
    print("\n" + "=" * 80)
    print("CHECKING IOPGEN=0 (ALL COMPONENTS):")
    
    from geopack.t01_vectorized import extall_vectorized, calculate_parameters
    
    parmod = np.array([10.0, -150.0, 3.0, -5.0, 2.0, 1.0])
    params = calculate_parameters(parmod, ps, a, 1)
    pdyn = parmod[0]
    dst = parmod[1]
    dst_ast = dst * 0.8 - 13.0 * np.sqrt(pdyn)
    
    xx_arr = np.array([xx])
    yy_arr = np.array([yy])
    zz_arr = np.array([zz])
    
    # Get all components
    _, _, bz_all = extall_vectorized(0, 0, 0, 0, a, 43, pdyn, dst_ast,
                                     parmod[2], parmod[3], parmod[4], parmod[5],
                                     ps, xx_arr, yy_arr, zz_arr, params)
    
    print(f"All components (iopgen=0): Bz={bz_all[0]:.6f}")
    
    # Get individual components and sum them
    components = {}
    for iopgen in range(1, 6):
        _, _, bz_comp = extall_vectorized(iopgen, 0, 0, 0, a, 43, pdyn, dst_ast,
                                          parmod[2], parmod[3], parmod[4], parmod[5],
                                          ps, xx_arr, yy_arr, zz_arr, params)
        components[iopgen] = bz_comp[0]
    
    print(f"\nIndividual components:")
    print(f"  Dipole shield (1): {components[1]:.6f}")
    print(f"  Tail field (2): {components[2]:.6f}")
    print(f"  Birkeland (3): {components[3]:.6f}")
    print(f"  Ring current (4): {components[4]:.6f}")
    print(f"  Interconnection (5): {components[5]:.6f}")
    print(f"  Sum: {sum(components.values()):.6f}")
    
    print(f"\nThe sum matches iopgen=0, so the issue is not in the summation")


if __name__ == "__main__":
    trace_iopgen1()