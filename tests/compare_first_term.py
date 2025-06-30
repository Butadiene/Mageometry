"""
Compare just the first term calculation between scalar and vectorized.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compare_first_term():
    """Compare first term only."""
    print("FIRST TERM COMPARISON")
    print("=" * 80)
    
    # Parameters
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
    
    # Basic calculations
    fac_sc = (x_sc + 1.0) ** 3
    cps = np.cos(ps)
    sps = np.sin(ps)
    
    # Rotations
    pst1 = ps * c_sy[84]
    st1 = np.sin(pst1)
    ct1 = np.cos(pst1)
    
    x1 = x * ct1 - z * st1
    z1 = x * st1 + z * ct1
    
    # First term: m=0, i=0, k=0, n=0, nn=0
    p = c_sy[72]  # p1
    r = c_sy[75]  # r1
    
    cypi = np.cos(y / p)
    sypi = np.sin(y / p)
    szrk = np.sin(z1 / r)
    czrk = np.cos(z1 / r)
    sqpr = np.sqrt(1/p**2 + 1/r**2)
    epr = np.exp(x1 * sqpr)
    
    # Field components
    fx = -sqpr * epr * cypi * szrk * fac_sc
    fy = epr * sypi * szrk / p * fac_sc
    fz = -epr * cypi * czrk / r * fac_sc
    
    # For m=0, n=0, nn=0: no additional factors
    hx = fx
    hy = fy
    hz = fz
    
    # Rotate back
    hx_final = hx * ct1 + hz * st1
    hy_final = hy
    hz_final = -hx * st1 + hz * ct1
    
    # Apply first coefficient
    coeff = c_sy[0]
    contribution_x = hx_final * coeff
    contribution_y = hy_final * coeff
    contribution_z = hz_final * coeff
    
    print(f"First term contribution:")
    print(f"  Bx = {contribution_x:.8f}")
    print(f"  By = {contribution_y:.8f}")
    print(f"  Bz = {contribution_z:.8f}")
    
    # Now let's check what the vectorized version would give
    # The issue might be in how we index the coefficients
    print(f"\nDebugging coefficient indexing:")
    print(f"  For m=0, i=0, k=0, n=0, nn=0: l should be 0")
    print(f"  c_sy[0] = {c_sy[0]}")
    
    # Check if issue is with fac_sc application
    print(f"\nChecking fac_sc:")
    print(f"  fac_sc = (x_sc + 1)^3 = ({x_sc} + 1)^3 = {fac_sc}")
    print(f"  This should be applied to fx, fy, fz BEFORE rotation")


if __name__ == "__main__":
    compare_first_term()