"""
Trace through the tail calculation in extall_vectorized.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopack.t01_vectorized import extall_vectorized, calculate_parameters


def trace_tail():
    """Trace tail calculation."""
    print("TRACING TAIL CALCULATION")
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
    pdyn = parmod[0]
    dst = parmod[1]
    dst_ast = dst * 0.8 - 13.0 * np.sqrt(pdyn)
    
    # t01_vectorized scales coordinates and passes to extall_vectorized
    xx = x * params.xappa
    yy = y * params.xappa
    zz = z * params.xappa
    
    print(f"In t01_vectorized:")
    print(f"  Original: x={x}, y={y}, z={z}")
    print(f"  Scaled: xx={xx:.6f}, yy={yy:.6f}, zz={zz:.6f}")
    print(f"  These scaled coords are passed to extall_vectorized")
    
    # Now let's trace what happens in extall_vectorized
    print("\n" + "=" * 80)
    print("IN EXTALL_VECTORIZED:")
    
    # In extall_vectorized, the input x, y, z are actually the scaled xx, yy, zz
    print(f"  Input x, y, z are the scaled coordinates")
    print(f"  They get assigned to xx, yy, zz for clarity")
    
    # Get tail field from extall
    bx_tail, by_tail, bz_tail = extall_vectorized(
        2, 0, 0, 0, a, 43, pdyn, dst_ast,
        parmod[2], parmod[3], parmod[4], parmod[5],
        ps, np.array([xx]), np.array([yy]), np.array([zz]), params
    )
    
    print(f"\nTail field from extall_vectorized(iopgen=2):")
    print(f"  Bx={bx_tail[0]:.6f}, By={by_tail[0]:.6f}, Bz={bz_tail[0]:.6f}")
    
    # Now let's check if the issue is with the iopt parameter
    print("\n" + "=" * 80)
    print("CHECKING IOPT PARAMETER:")
    
    # When iopgen=2 (tail only), what iopt is used?
    # Looking at the code, iopt is the second parameter to extall_vectorized
    # and it's passed as 0 in our test
    
    print(f"extall_vectorized is called with iopt=0 (both tail modes)")
    print(f"This should calculate both mode 1 and mode 2 and add them")
    
    # Let's also get the total field to see the overall picture
    bx_all, by_all, bz_all = extall_vectorized(
        0, 0, 0, 0, a, 43, pdyn, dst_ast,
        parmod[2], parmod[3], parmod[4], parmod[5],
        ps, np.array([xx]), np.array([yy]), np.array([zz]), params
    )
    
    print(f"\nTotal field from extall_vectorized(iopgen=0):")
    print(f"  Bz={bz_all[0]:.6f}")
    
    # Compare with scalar
    from geopack import t01
    bx_scalar, by_scalar, bz_scalar = t01.t01(parmod, ps, x, y, z)
    print(f"\nScalar T01:")
    print(f"  Bz={bz_scalar:.6f}")
    print(f"\nDifference: {bz_all[0] - bz_scalar:.6f}")


if __name__ == "__main__":
    trace_tail()