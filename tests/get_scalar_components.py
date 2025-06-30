"""
Get individual components from scalar version.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopack import t01


def get_scalar_components():
    """Get scalar components."""
    print("SCALAR COMPONENT VALUES")
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
    
    # First get the total
    bx_total, by_total, bz_total = t01.t01(parmod, ps, x, y, z)
    print(f"Total field: Bz={bz_total:.6f}")
    
    # Now let's manually calculate what scalar does
    # From the scalar extall, we know it calculates:
    # bbz = a[0]*bzcf + tamp1*bzt1 + tamp2*bzt2 + a_src*bzsrc + a_prc*bzprc
    #       + a_r11*bzr11 + a_r12*bzr12 + a_r21*bzr21 + a_r22*bzr22
    #       + a[23]*hzimf + a[24]*hzimf*sthetah
    
    # Get parameters
    pdyn = parmod[0]
    dst = parmod[1]
    xappa = (pdyn/2.)**a[38]
    xx = x * xappa
    yy = y * xappa
    zz = z * xappa
    
    print(f"\nParameters:")
    print(f"  xappa = {xappa:.6f}")
    print(f"  Scaled coords: ({xx:.6f}, {yy:.6f}, {zz:.6f})")
    
    # Dipole shield
    _, _, bzcf = t01.shlcar3x3(xx, yy, zz, ps)
    dipole_shield = a[0] * bzcf
    print(f"\nDipole shield: {dipole_shield:.6f}")
    
    # Tail field
    _, _, bzt1, _, _, bzt2 = t01.deformed(0, ps, xx, yy, zz)
    dlp1 = (pdyn/2)**a[41]
    dlp2 = (pdyn/2)**a[42]
    tamp1 = a[1] + a[2]*dlp1 + a[3]*parmod[4] + a[4]*dst
    tamp2 = a[5] + a[6]*dlp2 + a[7]*parmod[4] + a[8]*dst
    tail_field = tamp1*bzt1 + tamp2*bzt2
    print(f"Tail field: {tail_field:.6f}")
    
    # Ring current
    _, _, bzsrc, _, _, bzprc = t01.full_rc(0, ps, xx, yy, zz)
    a_src = a[9] + a[10]*dst + a[11]*np.sqrt(pdyn)
    a_prc = a[12] + a[13]*dst + a[14]*np.sqrt(pdyn)
    ring_current = a_src*bzsrc + a_prc*bzprc
    print(f"Ring current: {ring_current:.6f}")
    
    # Birkeland
    _, _, bzr11, _, _, bzr12, _, _, bzr21, _, _, bzr22 = t01.birk_tot(0, ps, xx, yy, zz)
    g2 = parmod[5]
    a_r11 = a[15] + a[16]*g2
    a_r12 = a[17] + a[18]*g2
    a_r21 = a[19] + a[20]*g2
    a_r22 = a[21] + a[22]*g2
    birkeland = a_r11*bzr11 + a_r12*bzr12 + a_r21*bzr21 + a_r22*bzr22
    print(f"Birkeland: {birkeland:.6f}")
    
    # Interconnection
    byimf = parmod[2]
    bzimf = parmod[3]
    if byimf == 0 and bzimf == 0:
        theta = 0.0
    else:
        theta = np.arctan2(byimf, bzimf)
        if theta <= 0:
            theta += 2 * np.pi
    sthetah = np.sin(theta / 2.0) ** 2
    hzimf = bzimf
    interconnection = a[23]*hzimf + a[24]*hzimf*sthetah
    print(f"Interconnection: {interconnection:.6f}")
    
    # Sum
    manual_sum = dipole_shield + tail_field + ring_current + birkeland + interconnection
    print(f"\nManual sum: {manual_sum:.6f}")
    print(f"Actual scalar result: {bz_total:.6f}")
    print(f"Difference: {bz_total - manual_sum:.6f}")
    
    print("\n" + "=" * 80)
    print("COMPARISON WITH VECTORIZED:")
    print("Component        Scalar      Vectorized   Difference")
    print("-" * 50)
    print(f"Dipole shield    {dipole_shield:10.6f}  {5.424524:10.6f}  {5.424524-dipole_shield:10.6f}")
    print(f"Tail field       {tail_field:10.6f}  {-33.377771:10.6f}  {-33.377771-tail_field:10.6f}")
    print(f"Ring current     {ring_current:10.6f}  {8.277758:10.6f}  {8.277758-ring_current:10.6f}")
    print(f"Birkeland        {birkeland:10.6f}  {-0.880515:10.6f}  {-0.880515-birkeland:10.6f}")
    print(f"Interconnection  {interconnection:10.6f}  {-2.973366:10.6f}  {-2.973366-interconnection:10.6f}")
    print("-" * 50)
    print(f"Total            {bz_total:10.6f}  {-23.529369:10.6f}  {-23.529369-bz_total:10.6f}")


if __name__ == "__main__":
    get_scalar_components()