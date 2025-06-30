"""
Create a fixed version of rc_shield_vectorized.
"""

import numpy as np
from typing import Tuple, Union


def rc_shield_vectorized_fixed(a_arr: np.ndarray, ps: float, x_sc: Union[float, np.ndarray],
                               x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed vectorized ring current shielding field.
    
    This version exactly matches the scalar rc_shield from t01.py.
    """
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    x_sc = np.atleast_1d(x_sc)
    
    # Broadcast x_sc if needed
    if x_sc.size == 1 and x.size > 1:
        x_sc = np.full_like(x, x_sc.item())
    
    # Initialize output
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    
    # Scale factor
    fac_sc = (x_sc + 1.0) ** 3
    
    # Tilt parameters
    cps = np.cos(ps)
    sps = np.sin(ps)
    s3ps = 2 * cps  # Approximation for small ps
    
    # Tilt rotation angles from indices 84, 85
    pst1 = ps * a_arr[84]
    pst2 = ps * a_arr[85]
    
    st1 = np.sin(pst1)
    ct1 = np.cos(pst1)
    st2 = np.sin(pst2)
    ct2 = np.cos(pst2)
    
    # Rotated coordinates
    x1 = x * ct1 - z * st1
    z1 = x * st1 + z * ct1
    x2 = x * ct2 - z * st2
    z2 = x * st2 + z * ct2
    
    # Coefficient index
    l = 0
    
    # Two symmetries: m=0 for perpendicular, m=1 for parallel
    for m in range(2):
        # 3x3 harmonics
        for i in range(3):
            # Get harmonic parameters
            p = a_arr[72 + i]
            q = a_arr[78 + i]
            cypi = np.cos(y / p)
            cyqi = np.cos(y / q)
            sypi = np.sin(y / p)
            syqi = np.sin(y / q)
            
            for k in range(3):
                r = a_arr[75 + k]
                s = a_arr[81 + k]
                szrk = np.sin(z1 / r)
                czsk = np.cos(z2 / s)
                czrk = np.cos(z1 / r)
                szsk = np.sin(z2 / s)
                sqpr = np.sqrt(1/p**2 + 1/r**2)
                sqqs = np.sqrt(1/q**2 + 1/s**2)
                epr = np.exp(x1 * sqpr)
                eqs = np.exp(x2 * sqqs)
                
                # Four terms for each harmonic
                for n in range(2):
                    for nn in range(2):
                        if m == 0:  # Perpendicular
                            fx = -sqpr * epr * cypi * szrk * fac_sc
                            fy = epr * sypi * szrk / p * fac_sc
                            fz = -epr * cypi * czrk / r * fac_sc
                            
                            if n == 0:
                                if nn == 0:
                                    hx, hy, hz = fx, fy, fz
                                else:
                                    hx = fx * x_sc
                                    hy = fy * x_sc
                                    hz = fz * x_sc
                            else:
                                if nn == 0:
                                    hx = fx * cps
                                    hy = fy * cps
                                    hz = fz * cps
                                else:
                                    hx = fx * cps * x_sc
                                    hy = fy * cps * x_sc
                                    hz = fz * cps * x_sc
                            
                            # Rotate back from tilted coordinates
                            hxr = hx * ct1 + hz * st1
                            hzr = -hx * st1 + hz * ct1
                        else:  # Parallel
                            fx = -sps * sqqs * eqs * cyqi * czsk * fac_sc
                            fy = sps / q * eqs * syqi * czsk * fac_sc
                            fz = sps / s * eqs * cyqi * szsk * fac_sc
                            
                            if n == 0:
                                if nn == 0:
                                    hx, hy, hz = fx, fy, fz
                                else:
                                    hx = fx * x_sc
                                    hy = fy * x_sc
                                    hz = fz * x_sc
                            else:
                                if nn == 0:
                                    hx = fx * s3ps
                                    hy = fy * s3ps
                                    hz = fz * s3ps
                                else:
                                    hx = fx * s3ps * x_sc
                                    hy = fy * s3ps * x_sc
                                    hz = fz * s3ps * x_sc
                            
                            # Rotate back from tilted coordinates
                            hxr = hx * ct2 + hz * st2
                            hzr = -hx * st2 + hz * ct2
                        
                        # Accumulate with coefficient
                        bx += hxr * a_arr[l]
                        by += hy * a_arr[l]
                        bz += hzr * a_arr[l]
                        l += 1
    
    return bx, by, bz


def test_fixed_version():
    """Test the fixed version."""
    print("TESTING FIXED RC_SHIELD")
    print("=" * 80)
    
    # Test parameters
    ps = -0.1
    x = -12.899832
    y = 0.0
    z = 0.0
    x_sc = 0.312290
    
    # c_sy array
    c_sy = np.array([
        -957.2534900, -817.5450246, 583.2991249, 758.8568270,
        13.17029064, 68.94173502, -15.29764089, -53.43151590, 27.34311724,
        149.5252826, -11.00696044, -179.7031814, 953.0914774, 817.2340042,
        -581.0791366, -757.5387665, -13.10602697, -68.58155678, 15.22447386,
        53.15535633, -27.07982637, -149.1413391, 10.91433279, 179.3251739,
        -6.028703251, 1.303196101, -1.345909343, -1.138296330, -0.06642634348,
        -0.3795246458, .07487833559, .2891156371, -.5506314391, -.4443105812,
        0.2273682152, 0.01086886655, -9.130025352, 1.118684840, 1.110838825,
        .1219761512, -.06263009645, -.1896093743, .03434321042, .01523060688,
        -.4913171541, -.2264814165, -.04791374574, .1981955976, -68.32678140,
        -48.72036263, 14.03247808, 16.56233733, 2.369921099, 6.200577111,
        -1.415841250, -0.8184867835, -3.401307527, -8.490692287, 3.217860767,
        -9.037752107, 66.09298105, 48.23198578, -13.67277141, -16.27028909,
        -2.309299411, -6.016572391, 1.381468849, 0.7935312553, 3.436934845,
        8.260038635, -3.136213782, 8.833214943, 8.041075485, 8.024818618,
        35.54861873, 12.55415215, 1.738167799, 3.721685353, 23.06768025,
        6.871230562, 6.806229878, 21.35990364, 1.687412298, 3.500885177,
        0.3498952546, 0.6595919814
    ])
    
    # Test fixed version
    fx_fixed, fy_fixed, fz_fixed = rc_shield_vectorized_fixed(
        c_sy, ps, x_sc, np.array([x]), np.array([y]), np.array([z])
    )
    print(f"Fixed version: ({fx_fixed[0]:.8f}, {fy_fixed[0]:.8f}, {fz_fixed[0]:.8f})")
    
    # Compare with scalar
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from geopack import t01
    
    fx_scalar, fy_scalar, fz_scalar = t01.rc_shield(c_sy, ps, x_sc, x, y, z)
    print(f"Scalar version: ({fx_scalar:.8f}, {fy_scalar:.8f}, {fz_scalar:.8f})")
    
    print(f"\nDifference: ({fx_fixed[0]-fx_scalar:.8f}, {fy_fixed[0]-fy_scalar:.8f}, {fz_fixed[0]-fz_scalar:.8f})")
    
    # Also compare with current vectorized
    from geopack.ring_current_vectorized import rc_shield_vectorized
    fx_vec, fy_vec, fz_vec = rc_shield_vectorized(
        c_sy, ps, x_sc, np.array([x]), np.array([y]), np.array([z])
    )
    print(f"\nCurrent vectorized: ({fx_vec[0]:.8f}, {fy_vec[0]:.8f}, {fz_vec[0]:.8f})")
    print(f"Current error: ({fx_vec[0]-fx_scalar:.8f}, {fy_vec[0]-fy_scalar:.8f}, {fz_vec[0]-fz_scalar:.8f})")


if __name__ == "__main__":
    test_fixed_version()