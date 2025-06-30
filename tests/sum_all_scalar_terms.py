"""
Sum all terms manually to match scalar exactly.
"""

import numpy as np


def sum_all_terms():
    """Sum all rc_shield terms manually."""
    print("SUMMING ALL RC_SHIELD TERMS")
    print("=" * 80)
    
    # Parameters
    ps = -0.1
    x = -12.899832
    y = 0.0
    z = 0.0
    x_sc = 0.312290
    
    # c_sy array
    a = np.array([
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
    
    # Reproduce scalar calculation exactly
    fac_sc = (x_sc + 1) ** 3
    cps = np.cos(ps)
    sps = np.sin(ps)
    s3ps = 2 * cps
    
    pst1 = ps * a[84]
    pst2 = ps * a[85]
    st1 = np.sin(pst1)
    ct1 = np.cos(pst1)
    st2 = np.sin(pst2)
    ct2 = np.cos(pst2)
    
    x1 = x * ct1 - z * st1
    z1 = x * st1 + z * ct1
    x2 = x * ct2 - z * st2
    z2 = x * st2 + z * ct2
    
    l = 0
    bx, by, bz = 0.0, 0.0, 0.0
    
    # Reproduce the exact scalar loop
    for m in range(2):
        for i in range(3):
            p = a[72 + i]
            q = a[78 + i]
            cypi = np.cos(y / p)
            cyqi = np.cos(y / q)
            sypi = np.sin(y / p)
            syqi = np.sin(y / q)
            
            for k in range(3):
                r = a[75 + k]
                s = a[81 + k]
                szrk = np.sin(z1 / r)
                czsk = np.cos(z2 / s)
                czrk = np.cos(z1 / r)
                szsk = np.sin(z2 / s)
                sqpr = np.sqrt(1/p**2 + 1/r**2)
                sqqs = np.sqrt(1/q**2 + 1/s**2)
                epr = np.exp(x1 * sqpr)
                eqs = np.exp(x2 * sqqs)
                
                for n in range(2):
                    for nn in range(2):
                        if m == 0:
                            fx = -sqpr * epr * cypi * szrk * fac_sc
                            fy = epr * sypi * szrk / p * fac_sc
                            fz = -epr * cypi * czrk / r * fac_sc
                            if n == 0:
                                if nn == 0:
                                    hx, hy, hz = fx, fy, fz
                                else:
                                    hx, hy, hz = fx * x_sc, fy * x_sc, fz * x_sc
                            else:
                                if nn == 0:
                                    hx, hy, hz = fx * cps, fy * cps, fz * cps
                                else:
                                    hx, hy, hz = fx * cps * x_sc, fy * cps * x_sc, fz * cps * x_sc
                        else:  # m == 1
                            fx = -sps * sqqs * eqs * cyqi * czsk * fac_sc
                            fy = sps / q * eqs * syqi * czsk * fac_sc
                            fz = sps / s * eqs * cyqi * szsk * fac_sc
                            if n == 0:
                                if nn == 0:
                                    hx, hy, hz = fx, fy, fz
                                else:
                                    hx, hy, hz = fx * x_sc, fy * x_sc, fz * x_sc
                            else:
                                if nn == 0:
                                    hx, hy, hz = fx * s3ps, fy * s3ps, fz * s3ps
                                else:
                                    hx, hy, hz = fx * s3ps * x_sc, fy * s3ps * x_sc, fz * s3ps * x_sc
                        
                        if m == 0:
                            hxr = hx * ct1 + hz * st1
                            hzr = -hx * st1 + hz * ct1
                        else:
                            hxr = hx * ct2 + hz * st2
                            hzr = -hx * st2 + hz * ct2
                        
                        bx += hxr * a[l]
                        by += hy * a[l]
                        bz += hzr * a[l]
                        
                        if l < 4:
                            print(f"Term {l}: coeff={a[l]:10.6f}, hxr={hxr:10.6f}, contrib={hxr*a[l]:10.6f}, sum={bx:10.6f}")
                        
                        l += 1
    
    print(f"\nFinal result: ({bx:.8f}, {by:.8f}, {bz:.8f})")
    
    # Import and test scalar
    from geopack import t01
    fx_s, fy_s, fz_s = t01.rc_shield(a, ps, x_sc, x, y, z)
    print(f"Scalar result: ({fx_s:.8f}, {fy_s:.8f}, {fz_s:.8f})")
    print(f"\nDifference: ({bx-fx_s:.8f}, {by-fy_s:.8f}, {bz-fz_s:.8f})")


if __name__ == "__main__":
    sum_all_terms()