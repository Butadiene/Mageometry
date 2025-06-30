"""
Trace scalar IMF calculation step by step.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the scalar t01 module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'geopack'))
import t01


def trace_scalar_imf():
    """Add debug to scalar t01 to trace IMF handling."""
    
    # Monkey patch the t01 function to add debug
    original_t01 = t01.t01
    
    def debug_t01(iopt, parmod, ps, x, y, z):
        # Extract parameters
        pdyn = parmod[0]
        dst = parmod[1]
        byimf = parmod[2]
        bzimf = parmod[3]
        
        # Call original but capture intermediate values
        # We'll need to look at the code directly
        
        # For now, just call the original
        return original_t01(iopt, parmod, ps, x, y, z)
    
    # Test parameters
    parmod = np.array([25.0, -300.0, -8.0, -10.0, 4.0, 3.0])
    ps = -0.2
    x, y, z = 0.0, -6.6, 0.0
    
    print("SCALAR IMF TRACE")
    print("=" * 50)
    
    # Model coefficients we care about
    a23 = 0.05887
    a24 = 0.57691
    
    # Calculate theta
    byimf = parmod[2]
    bzimf = parmod[3]
    b2 = byimf**2 + bzimf**2
    theta = np.arctan2(byimf, bzimf)
    if byimf > 0:
        theta = theta - np.pi
    sthetah = np.sin(theta / 2.0) ** 2
    
    print(f"IMF: byimf={byimf}, bzimf={bzimf}")
    print(f"theta={theta:.6f}, sthetah={sthetah:.6f}")
    
    # The expected contribution
    expected_y = a23 * byimf + a24 * byimf * sthetah
    expected_z = a23 * bzimf + a24 * bzimf * sthetah
    
    print(f"\nExpected contributions:")
    print(f"  By: {a23} * {byimf} + {a24} * {byimf} * {sthetah:.6f} = {expected_y:.6f}")
    print(f"  Bz: {a23} * {bzimf} + {a24} * {bzimf} * {sthetah:.6f} = {expected_z:.6f}")
    
    # But we know the actual is only ~10% of this
    # Let's check if there's a coefficient issue
    
    # Look at coefficient a[23] and a[24] from the full array
    a = np.array([
        1.00000, 2.47341, 0.40791, 0.30429, -0.10637, -0.89108, 3.29350,
        -0.05413, -0.00696, 1.07869, -0.02314, -0.66173, -0.68018, -0.03246,
        0.02681, 0.28062, 0.16535, -0.02939, 0.02639, -0.24891, -0.08063,
        0.08900, -0.02475, 0.05887, 0.57691, 0.65256, -0.03230, 2.24733,
        4.10546, 1.13665, 0.05506, 0.97669, 0.21164, 0.64594, 1.12556, 0.01389,
        1.02978, 0.02968, 0.15821, 9.00519, 28.17582, 1.35285, 0.42279
    ])
    
    print(f"\nCoefficients from array:")
    print(f"  a[22] = {a[22]:.6f}")
    print(f"  a[23] = {a[23]:.6f}")
    print(f"  a[24] = {a[24]:.6f}")
    print(f"  a[25] = {a[25]:.6f}")
    
    # Wait! Let me check if the issue is the coefficient indices
    # The scalar code might be using different indices
    
    # Let's also check what happens with tail field amplitude
    tail1_amp = a[15] + a[16] * sthetah
    tail2_amp = a[17] + a[18] * sthetah
    
    print(f"\nTail field amplitudes:")
    print(f"  tail1: {a[15]} + {a[16]} * {sthetah:.6f} = {tail1_amp:.6f}")
    print(f"  tail2: {a[17]} + {a[18]} * {sthetah:.6f} = {tail2_amp:.6f}")
    
    # The actual scalar result
    bx, by, bz = t01.t01(parmod, ps, x, y, z)
    print(f"\nScalar result: Bx={bx:.6f}, By={by:.6f}, Bz={bz:.6f}")
    
    # Without IMF
    parmod_no_imf = parmod.copy()
    parmod_no_imf[2] = 0.0
    parmod_no_imf[3] = 0.0
    bx2, by2, bz2 = t01.t01(parmod_no_imf, ps, x, y, z)
    
    print(f"Without IMF: Bx={bx2:.6f}, By={by2:.6f}, Bz={bz2:.6f}")
    print(f"IMF contribution: ΔBx={bx-bx2:.6f}, ΔBy={by-by2:.6f}, ΔBz={bz-bz2:.6f}")
    
    # The 10% factor suggests there might be an issue with how the coefficients
    # are being used or there's another scaling factor we're missing


if __name__ == "__main__":
    trace_scalar_imf()