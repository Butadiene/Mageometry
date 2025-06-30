"""
Trace IMF handling issue step by step.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def trace_imf_handling():
    """Trace IMF handling in the vectorized code."""
    
    # Parameters
    parmod = np.array([25.0, -300.0, -8.0, -10.0, 4.0, 3.0])
    ps = -0.2
    x, y, z = 0.0, -6.6, 0.0
    
    # Extract IMF
    byimf = parmod[2]
    bzimf = parmod[3]
    
    # Model coefficients (just the ones we need)
    a23 = 0.058870
    a24 = 0.576910
    
    # Calculate theta 
    b2 = byimf**2 + bzimf**2
    theta = np.arctan2(byimf, bzimf)
    if byimf > 0:
        theta = theta - np.pi
    sthetah = np.sin(theta / 2.0) ** 2
    
    print("IMF HANDLING TRACE")
    print("=" * 70)
    print(f"IMF components: byimf={byimf}, bzimf={bzimf}")
    print(f"theta={theta:.6f}, sthetah={sthetah:.6f}")
    print(f"a[23]={a23}, a[24]={a24}")
    
    # What the scalar code does:
    # Inside magnetosphere: hyimf = byimf, hzimf = bzimf
    # Then adds: a[23]*hyimf + a[24]*hyimf*sthetah
    scalar_contrib_y = a23 * byimf + a24 * byimf * sthetah
    scalar_contrib_z = a23 * bzimf + a24 * bzimf * sthetah
    
    print(f"\nScalar calculation:")
    print(f"  hyimf = {byimf}")
    print(f"  hzimf = {bzimf}")
    print(f"  By contribution: {a23}*{byimf} + {a24}*{byimf}*{sthetah:.6f}")
    print(f"                 = {a23*byimf:.6f} + {a24*byimf*sthetah:.6f}")
    print(f"                 = {scalar_contrib_y:.6f}")
    print(f"  Bz contribution: {scalar_contrib_z:.6f}")
    
    # What the vectorized code should do (same as scalar)
    hyimf = byimf  # = -8.0
    hzimf = bzimf  # = -10.0
    vector_contrib_y = a23 * hyimf + a24 * hyimf * sthetah
    vector_contrib_z = a23 * hzimf + a24 * hzimf * sthetah
    
    print(f"\nVectorized calculation (expected):")
    print(f"  hyimf = {hyimf}")
    print(f"  hzimf = {hzimf}")
    print(f"  By contribution: {a23}*{hyimf} + {a24}*{hyimf}*{sthetah:.6f}")
    print(f"                 = {vector_contrib_y:.6f}")
    print(f"  Bz contribution: {vector_contrib_z:.6f}")
    
    # But if factimf is being applied somehow:
    factimf = a23 + a24 * sthetah
    wrong_contrib_y = factimf * byimf
    wrong_contrib_z = factimf * bzimf
    
    print(f"\nIf factimf is incorrectly applied:")
    print(f"  factimf = {a23} + {a24}*{sthetah:.6f} = {factimf:.6f}")
    print(f"  By contribution: {factimf:.6f} * {byimf} = {wrong_contrib_y:.6f}")
    print(f"  Bz contribution: {factimf:.6f} * {bzimf} = {wrong_contrib_z:.6f}")
    
    # Compare
    print(f"\nDifference (wrong - correct):")
    print(f"  ΔBy = {wrong_contrib_y:.6f} - {scalar_contrib_y:.6f} = {wrong_contrib_y - scalar_contrib_y:.6f}")
    print(f"  ΔBz = {wrong_contrib_z:.6f} - {scalar_contrib_z:.6f} = {wrong_contrib_z - scalar_contrib_z:.6f}")
    
    # From the debug output, we saw:
    # Scalar IMF: ΔBy=-0.471, ΔBz=-0.589
    # Vectorized IMF: ΔBy=-4.581, ΔBz=-5.726
    # Difference: ΔΔBy=-4.110, ΔΔBz=-5.137
    
    print(f"\nFrom actual test:")
    print(f"  Scalar IMF contribution: ΔBy=-0.471, ΔBz=-0.589")
    print(f"  Vectorized IMF contribution: ΔBy=-4.581, ΔBz=-5.726")
    print(f"  Error: ΔΔBy=-4.110, ΔΔBz=-5.137")


if __name__ == "__main__":
    trace_imf_handling()