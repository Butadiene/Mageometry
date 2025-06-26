"""
Fixed exact vectorized implementation of condip1 that matches the original.
"""

import numpy as np


def condip1_exact_vectorized(x, y, z, ps):
    """
    Exact vectorized implementation matching the original condip1.
    Returns the field directly (with c2 coefficients applied).
    """
    # Global constants from original T96 
    dx = -0.16
    scalein = 0.08
    scaleout = 0.4
    
    # Dipole positions from original T96
    xx2 = np.array([-10.,-7,-4,-4,0,4,4,7,10,0,0,0,0,0])
    yy2 = np.array([3.,6,3,9,6,3,9,6,3,0,0,0,0,0])
    zz2 = np.array([20.,20,4,20,4,4,20,20,20,2,3,4.5,7,10])
    
    # Model coefficients - exact copy from original T96
    c2 = np.array([
        6.04133, .305415, .606066e-02, .128379e-03, -.179406e-04,
        1.41714, -27.2586, -4.28833, -1.30675, 35.5607, 8.95792, .961617E-03,
        -.801477E-03, -.782795E-03, -1.65242, -16.5242, -5.33798, .424878E-03,
        .331787E-03, -.704305E-03, .844342E-03, .953682E-04, .886271E-03,
        25.1120, 20.9299, 5.14569, -44.1670, -51.0672, -1.87725, 20.2998,
        48.7505, -2.97415, 3.35184, -54.2921, -.838712, -10.5123, 70.7594,
        -4.94104, .106166E-03, .465791E-03, -.193719E-03, 10.8439, -29.7968,
        8.08068, .463507E-03, -.224475E-04, .177035E-03, -.317581E-03,
        -.264487E-03, .102075E-03, 7.71390, 10.1915, -4.99797, -23.1114,
        -29.2043, 12.2928, 10.9542, 33.6671, -9.3851, .174615E-03, -.789777E-06,
        .686047E-03, .460104E-04, -.345216E-02, .221871E-02, .110078E-01,
        -.661373E-02, .249201E-02, .343978E-01, -.193145E-05, .493963E-05,
        -.535748E-04, .191833E-04, -.100496E-03, -.210103E-03, -.232195E-02,
        .315335E-02, -.134320E-01, -.263222E-01
    ])
    
    # Ensure arrays
    scalar_input = np.isscalar(x)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y) 
    z = np.atleast_1d(z)
    
    cps = np.cos(ps)
    sps = np.sin(ps)
    
    # Initialize field
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    
    # Part 1: Conical harmonics (indices 0-4)
    xsm = x * cps - z * sps - dx
    zsm = z * cps + x * sps
    ro2 = xsm**2 + y**2
    ro = np.sqrt(ro2)
    ro_safe = np.where(ro < 1e-9, 1e-9, ro)
    
    # Calculate phi multiples
    cf = np.zeros((5,) + x.shape)
    sf = np.zeros((5,) + x.shape)
    cf[0] = xsm / ro_safe
    sf[0] = y / ro_safe
    
    for m in range(1, 5):
        cf[m] = cf[m-1] * cf[0] - sf[m-1] * sf[0]
        sf[m] = sf[m-1] * cf[0] + cf[m-1] * sf[0]
    
    r2 = ro2 + zsm**2
    r = np.sqrt(r2)
    r_safe = np.where(r < 1e-9, 1e-9, r)
    c = zsm / r_safe
    s = ro / r_safe
    ch = np.sqrt(0.5 * (1 + c))
    sh = np.sqrt(0.5 * (1 - c))
    ch_safe = np.where(ch < 1e-9, 1e-9, ch)
    sh_safe = np.where(sh < 1e-9, 1e-9, sh)
    tnh = sh / ch_safe
    cnh = ch_safe / sh_safe
    
    # Process conical harmonics - matching scalar exactly
    for m in range(5):
        m1 = m + 1
        
        # Safe division for r*s
        rs_safe = np.where(r_safe * s < 1e-9, 1e-9, r_safe * s)
        
        bt = m1 * cf[m] / rs_safe * (tnh**m1 + cnh**m1)
        bf = -0.5 * m1 * sf[m] / r_safe * (tnh**m / ch_safe**2 - cnh**m / sh_safe**2)
        
        bxsm = bt * c * cf[0] - bf * sf[0]
        bysm = bt * c * sf[0] + bf * cf[0]
        bzsm = -bt * s
        
        bx += c2[m] * (bxsm * cps + bzsm * sps)
        by += c2[m] * bysm
        bz += c2[m] * (bzsm * cps - bxsm * sps)
    
    # Part 2: Dipole terms
    xsm = x * cps - z * sps
    zsm = z * cps + x * sps
    
    # Process 9 dipole configurations
    for i in range(9):
        if i in [2, 4, 5]:
            xd = xx2[i] * scalein
            yd = yy2[i] * scalein
        else:
            xd = xx2[i] * scaleout
            yd = yy2[i] * scaleout
        zd = zz2[i]
        
        # Calculate dipole contributions
        bxx1, byx1, bzx1, bxy1, byy1, bzy1, bxz1, byz1, bzz1 = dipxyz_exact(xsm - xd, y - yd, zsm - zd)
        bxx2, byx2, bzx2, bxy2, byy2, bzy2, bxz2, byz2, bzz2 = dipxyz_exact(xsm - xd, y + yd, zsm - zd)
        bxx3, byx3, bzx3, bxy3, byy3, bzy3, bxz3, byz3, bzz3 = dipxyz_exact(xsm - xd, y - yd, zsm + zd)
        bxx4, byx4, bzx4, bxy4, byy4, bzy4, bxz4, byz4, bzz4 = dipxyz_exact(xsm - xd, y + yd, zsm + zd)
        
        # X-derivative
        ix = i * 3 + 5
        bxsm = bxx1 + bxx2 - bxx3 - bxx4
        bysm = byx1 + byx2 - byx3 - byx4
        bzsm = bzx1 + bzx2 - bzx3 - bzx4
        bx += c2[ix] * (bxsm * cps + bzsm * sps)
        by += c2[ix] * bysm
        bz += c2[ix] * (bzsm * cps - bxsm * sps)
        
        # Y-derivative
        iy = ix + 1
        bxsm = bxy1 - bxy2 - bxy3 + bxy4
        bysm = byy1 - byy2 - byy3 + byy4
        bzsm = bzy1 - bzy2 - bzy3 + bzy4
        bx += c2[iy] * (bxsm * cps + bzsm * sps)
        by += c2[iy] * bysm
        bz += c2[iy] * (bzsm * cps - bxsm * sps)
        
        # Z-derivative
        iz = iy + 1
        bxsm = bxz1 + bxz2 + bxz3 + bxz4
        bysm = byz1 + byz2 + byz3 + byz4
        bzsm = bzz1 + bzz2 + bzz3 + bzz4
        bx += c2[iz] * (bxsm * cps + bzsm * sps)
        by += c2[iz] * bysm
        bz += c2[iz] * (bzsm * cps - bxsm * sps)
        
        # Same with sps factor (indices 32-58)
        ix2 = ix + 27
        iy2 = iy + 27
        iz2 = iz + 27
        
        # X-derivative with sps
        bxsm = bxx1 + bxx2 + bxx3 + bxx4
        bysm = byx1 + byx2 + byx3 + byx4
        bzsm = bzx1 + bzx2 + bzx3 + bzx4
        bx += c2[ix2] * sps * (bxsm * cps + bzsm * sps)
        by += c2[ix2] * sps * bysm
        bz += c2[ix2] * sps * (bzsm * cps - bxsm * sps)
        
        # Y-derivative with sps
        bxsm = bxy1 - bxy2 + bxy3 - bxy4
        bysm = byy1 - byy2 + byy3 - byy4
        bzsm = bzy1 - bzy2 + bzy3 - bzy4
        bx += c2[iy2] * sps * (bxsm * cps + bzsm * sps)
        by += c2[iy2] * sps * bysm
        bz += c2[iy2] * sps * (bzsm * cps - bxsm * sps)
        
        # Z-derivative with sps
        bxsm = bxz1 + bxz2 - bxz3 - bxz4
        bysm = byz1 + byz2 - byz3 - byz4
        bzsm = bzz1 + bzz2 - bzz3 - bzz4
        bx += c2[iz2] * sps * (bxsm * cps + bzsm * sps)
        by += c2[iz2] * sps * bysm
        bz += c2[iz2] * sps * (bzsm * cps - bxsm * sps)
    
    # Part 3: Special dipoles (indices 59-78)
    for i in range(5):
        zd = zz2[i + 9]
        bxx1, byx1, bzx1, bxy1, byy1, bzy1, bxz1, byz1, bzz1 = dipxyz_exact(xsm, y, zsm - zd)
        bxx2, byx2, bzx2, bxy2, byy2, bzy2, bxz2, byz2, bzz2 = dipxyz_exact(xsm, y, zsm + zd)
        
        # X-derivative
        ix = 59 + i * 2
        bxsm = bxx1 - bxx2
        bysm = byx1 - byx2
        bzsm = bzx1 - bzx2
        bx += c2[ix] * (bxsm * cps + bzsm * sps)
        by += c2[ix] * bysm
        bz += c2[ix] * (bzsm * cps - bxsm * sps)
        
        # Z-derivative
        iz = ix + 1
        bxsm = bxz1 + bxz2
        bysm = byz1 + byz2
        bzsm = bzz1 + bzz2
        bx += c2[iz] * (bxsm * cps + bzsm * sps)
        by += c2[iz] * bysm
        bz += c2[iz] * (bzsm * cps - bxsm * sps)
        
        # With sps factor
        ix2 = ix + 10
        iz2 = iz + 10
        
        # X-derivative with sps
        bxsm = bxx1 + bxx2
        bysm = byx1 + byx2
        bzsm = bzx1 + bzx2
        bx += c2[ix2] * sps * (bxsm * cps + bzsm * sps)
        by += c2[ix2] * sps * bysm
        bz += c2[ix2] * sps * (bzsm * cps - bxsm * sps)
        
        # Z-derivative with sps
        bxsm = bxz1 - bxz2
        bysm = byz1 - byz2
        bzsm = bzz1 - bzz2
        bx += c2[iz2] * sps * (bxsm * cps + bzsm * sps)
        by += c2[iz2] * sps * bysm
        bz += c2[iz2] * sps * (bzsm * cps - bxsm * sps)
    
    if scalar_input:
        return float(bx), float(by), float(bz)
    else:
        return bx, by, bz


def dipxyz_exact(x, y, z):
    """Exact vectorized dipole field calculation."""
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    x2 = x**2
    y2 = y**2
    z2 = z**2
    r2 = x2 + y2 + z2
    
    # Safe division
    r2_safe = np.where(r2 < 1e-15, 1e-15, r2)
    xmr5 = 30574.0 / (r2_safe * r2_safe * np.sqrt(r2_safe))
    xmr53 = 3 * xmr5
    
    bxx = xmr5 * (3 * x2 - r2)
    byx = xmr53 * x * y
    bzx = xmr53 * x * z
    
    bxy = byx
    byy = xmr5 * (3 * y2 - r2)
    bzy = xmr53 * y * z
    
    bxz = bzx
    byz = bzy
    bzz = xmr5 * (3 * z2 - r2)
    
    return bxx, byx, bzx, bxy, byy, bzy, bxz, byz, bzz