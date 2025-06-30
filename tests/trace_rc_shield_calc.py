"""
Trace through rc_shield calculation step by step.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def trace_calculation():
    """Trace rc_shield calculation."""
    print("RC_SHIELD CALCULATION TRACE")
    print("=" * 80)
    
    # Test parameters
    ps = -0.1
    x = -12.899832
    y = 0.0
    z = 0.0
    x_sc = 0.312290
    
    # c_sy array (only showing relevant indices)
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
    
    # Extract harmonic parameters
    print("Harmonic parameters:")
    print(f"  p1, p2, p3 = {c_sy[72]}, {c_sy[73]}, {c_sy[74]}")
    print(f"  r1, r2, r3 = {c_sy[75]}, {c_sy[76]}, {c_sy[77]}")
    print(f"  q1, q2, q3 = {c_sy[78]}, {c_sy[79]}, {c_sy[80]}")
    print(f"  s1, s2, s3 = {c_sy[81]}, {c_sy[82]}, {c_sy[83]}")
    
    # Basic calculations
    fac_sc = (x_sc + 1.0) ** 3
    cps = np.cos(ps)
    sps = np.sin(ps)
    s3ps = 2 * cps  # Small angle approximation
    
    print(f"\nBasic values:")
    print(f"  fac_sc = {fac_sc:.6f}")
    print(f"  cps = {cps:.6f}")
    print(f"  sps = {sps:.6f}")
    print(f"  s3ps = {s3ps:.6f}")
    
    # Rotations
    pst1 = ps * c_sy[84]
    pst2 = ps * c_sy[85]
    st1 = np.sin(pst1)
    ct1 = np.cos(pst1)
    st2 = np.sin(pst2)
    ct2 = np.cos(pst2)
    
    x1 = x * ct1 - z * st1
    z1 = x * st1 + z * ct1
    x2 = x * ct2 - z * st2
    z2 = x * st2 + z * ct2
    
    print(f"\nRotated coordinates:")
    print(f"  x1 = {x1:.6f}, z1 = {z1:.6f}")
    print(f"  x2 = {x2:.6f}, z2 = {z2:.6f}")
    
    # First harmonic calculation (m=0, i=0, k=0, n=0, nn=0)
    print(f"\nFirst harmonic (m=0, i=0, k=0, n=0, nn=0):")
    p = c_sy[72]  # p1
    r = c_sy[75]  # r1
    print(f"  p = {p}, r = {r}")
    
    cypi = np.cos(y / p)
    sypi = np.sin(y / p)
    szrk = np.sin(z1 / r)
    czrk = np.cos(z1 / r)
    sqpr = np.sqrt(1/p**2 + 1/r**2)
    epr = np.exp(x1 * sqpr)
    
    print(f"  cypi = cos({y}/{p}) = {cypi}")
    print(f"  szrk = sin({z1}/{r}) = {szrk}")
    print(f"  sqpr = sqrt(1/{p}^2 + 1/{r}^2) = {sqpr}")
    print(f"  epr = exp({x1} * {sqpr}) = {epr}")
    
    fx = -sqpr * epr * cypi * szrk * fac_sc
    fy = epr * sypi * szrk / p * fac_sc
    fz = -epr * cypi * czrk / r * fac_sc
    
    print(f"  fx = {fx}")
    print(f"  fy = {fy}")
    print(f"  fz = {fz}")
    
    # For n=0, nn=0: hx = fx, hy = fy, hz = fz
    hx = fx
    hy = fy
    hz = fz
    
    # Rotate back
    hx_rot = hx * ct1 + hz * st1
    hz_rot = -hx * st1 + hz * ct1
    hy_rot = hy
    
    print(f"  After rotation: hx={hx_rot}, hy={hy_rot}, hz={hz_rot}")
    
    # Apply coefficient
    coeff = c_sy[0]  # First coefficient
    print(f"  Coefficient c_sy[0] = {coeff}")
    print(f"  Contribution: ({hx_rot * coeff}, {hy_rot * coeff}, {hz_rot * coeff})")


if __name__ == "__main__":
    trace_calculation()