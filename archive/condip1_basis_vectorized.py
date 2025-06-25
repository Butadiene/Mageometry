"""
Fixed vectorized condip1 that returns basis functions like the original.
This version exactly matches the scalar implementation.
"""

import numpy as np


def condip1_basis_vectorized(x, y, z, ps):
    """
    Vectorized condip1 that returns basis functions (not final field).
    
    Returns d array of shape (3, 79, n_points) where:
    - First dimension is field component (x, y, z)
    - Second dimension is basis function index (0-78)
    - Third dimension is point index
    
    The final field is computed as: bx = sum(c2[i] * d[0, i, :])
    """
    # Global constants from original T96 
    dx = -0.16
    scalein = 0.08
    scaleout = 0.4
    
    # Dipole positions from original T96
    xx2 = np.array([-10.,-7,-4,-4,0,4,4,7,10,0,0,0,0,0])
    yy2 = np.array([3.,6,3,9,6,3,9,6,3,0,0,0,0,0])
    zz2 = np.array([20.,20,4,20,4,4,20,20,20,2,3,4.5,7,10])
    
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y) 
    z = np.atleast_1d(z)
    n_points = x.shape[0]
    
    # Initialize output array
    d = np.zeros((3, 79, n_points))
    
    cps = np.cos(ps)
    sps = np.sin(ps)
    
    # Part 1: Conical harmonics (indices 0-4)
    xsm = x * cps - z * sps - dx
    zsm = z * cps + x * sps
    ro2 = xsm**2 + y**2
    ro = np.sqrt(ro2)
    
    # Safe division
    ro_safe = np.where(ro < 1e-9, 1e-9, ro)
    
    # Calculate phi multiples
    cf = np.zeros((5, n_points))
    sf = np.zeros((5, n_points))
    cf[0] = xsm / ro_safe
    sf[0] = y / ro_safe
    cf[1] = cf[0]**2 - sf[0]**2
    sf[1] = 2 * sf[0] * cf[0]
    cf[2] = cf[1] * cf[0] - sf[1] * sf[0]
    sf[2] = sf[1] * cf[0] + cf[1] * sf[0]
    cf[3] = cf[2] * cf[0] - sf[2] * sf[0]
    sf[3] = sf[2] * cf[0] + cf[2] * sf[0]
    cf[4] = cf[3] * cf[0] - sf[3] * sf[0]
    sf[4] = sf[3] * cf[0] + cf[3] * sf[0]
    
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
    
    # Process m=0 to 4 - matching scalar implementation exactly
    for m in range(5):
        m1 = m + 1  # m1 = 1, 2, 3, 4, 5
        
        # Safe division for r*s
        rs_safe = np.where(r_safe * s < 1e-9, 1e-9, r_safe * s)
        
        bt = m1 * cf[m] / rs_safe * (tnh**m1 + cnh**m1)
        bf = -0.5 * m1 * sf[m] / r_safe * (tnh**m / ch_safe**2 - cnh**m / sh_safe**2)
        
        bxsm = bt * c * cf[0] - bf * sf[0]
        by = bt * c * sf[0] + bf * cf[0]
        bzsm = -bt * s
        
        d[0, m] = bxsm * cps + bzsm * sps
        d[1, m] = by
        d[2, m] = -bxsm * sps + bzsm * cps
    
    # Part 2: Dipole terms (indices 5-31 and 32-58)
    xsm = x * cps - z * sps
    zsm = z * cps + x * sps
    
    # Process 9 dipole configurations
    for i in range(9):
        # Determine scaling
        if i in [2, 4, 5]:
            xd = xx2[i] * scalein
            yd = yy2[i] * scalein
        else:
            xd = xx2[i] * scaleout
            yd = yy2[i] * scaleout
        zd = zz2[i]
        
        # Four dipole positions
        x1 = xsm - xd
        y1 = y - yd
        z1 = zsm - zd
        x2 = xsm - xd
        y2 = y + yd
        z2 = zsm - zd
        x3 = xsm - xd
        y3 = y - yd
        z3 = zsm + zd
        x4 = xsm - xd
        y4 = y + yd
        z4 = zsm + zd
        
        # Get dipole derivatives
        bxx1, byx1, bzx1, bxy1, byy1, bzy1, bxz1, byz1, bzz1 = dipxyz_vec(x1, y1, z1)
        bxx2, byx2, bzx2, bxy2, byy2, bzy2, bxz2, byz2, bzz2 = dipxyz_vec(x2, y2, z2)
        bxx3, byx3, bzx3, bxy3, byy3, bzy3, bxz3, byz3, bzz3 = dipxyz_vec(x3, y3, z3)
        bxx4, byx4, bzx4, bxy4, byy4, bzy4, bxz4, byz4, bzz4 = dipxyz_vec(x4, y4, z4)
        
        # Index for first set
        ix = i * 3 + 5
        iy = ix + 1
        iz = iy + 1
        
        # X-derivative
        d[0, ix] = (bxx1 + bxx2 - bxx3 - bxx4) * cps + (bzx1 + bzx2 - bzx3 - bzx4) * sps
        d[1, ix] = byx1 + byx2 - byx3 - byx4
        d[2, ix] = (bzx1 + bzx2 - bzx3 - bzx4) * cps - (bxx1 + bxx2 - bxx3 - bxx4) * sps
        
        # Y-derivative  
        d[0, iy] = (bxy1 - bxy2 - bxy3 + bxy4) * cps + (bzy1 - bzy2 - bzy3 + bzy4) * sps
        d[1, iy] = byy1 - byy2 - byy3 + byy4
        d[2, iy] = (bzy1 - bzy2 - bzy3 + bzy4) * cps - (bxy1 - bxy2 - bxy3 + bxy4) * sps
        
        # Z-derivative - note the sign pattern here!
        d[0, iz] = (bxz1 + bxz2 + bxz3 + bxz4) * cps + (bzz1 + bzz2 + bzz3 + bzz4) * sps
        d[1, iz] = byz1 + byz2 + byz3 + byz4
        d[2, iz] = (bzz1 + bzz2 + bzz3 + bzz4) * cps - (bxz1 + bxz2 + bxz3 + bxz4) * sps
        
        # Index for second set (with sps factor)
        ix2 = ix + 27
        iy2 = iy + 27
        iz2 = iz + 27
        
        # Same but with different sign patterns and multiplied by sps
        d[0, ix2] = sps * ((bxx1 + bxx2 + bxx3 + bxx4) * cps + (bzx1 + bzx2 + bzx3 + bzx4) * sps)
        d[1, ix2] = sps * (byx1 + byx2 + byx3 + byx4)
        d[2, ix2] = sps * ((bzx1 + bzx2 + bzx3 + bzx4) * cps - (bxx1 + bxx2 + bxx3 + bxx4) * sps)
        
        d[0, iy2] = sps * ((bxy1 - bxy2 + bxy3 - bxy4) * cps + (bzy1 - bzy2 + bzy3 - bzy4) * sps)
        d[1, iy2] = sps * (byy1 - byy2 + byy3 - byy4)
        d[2, iy2] = sps * ((bzy1 - bzy2 + bzy3 - bzy4) * cps - (bxy1 - bxy2 + bxy3 - bxy4) * sps)
        
        d[0, iz2] = sps * ((bxz1 + bxz2 - bxz3 - bxz4) * cps + (bzz1 + bzz2 - bzz3 - bzz4) * sps)
        d[1, iz2] = sps * (byz1 + byz2 - byz3 - byz4)
        d[2, iz2] = sps * ((bzz1 + bzz2 - bzz3 - bzz4) * cps - (bxz1 + bxz2 - bxz3 - bxz4) * sps)
    
    # Part 3: Special dipoles (indices 59-78)
    # Process 5 dipoles with z-derivatives only (indices 59-68)
    for i in range(5):
        zd = zz2[i + 9]  # Start from zz2[9]
        bxx1, byx1, bzx1, bxy1, byy1, bzy1, bxz1, byz1, bzz1 = dipxyz_vec(xsm, y, zsm - zd)
        bxx2, byx2, bzx2, bxy2, byy2, bzy2, bxz2, byz2, bzz2 = dipxyz_vec(xsm, y, zsm + zd)
        
        # X-derivative (index 59+i*2)
        ix = 59 + i * 2
        d[0, ix] = (bxx1 - bxx2) * cps + (bzx1 - bzx2) * sps
        d[1, ix] = byx1 - byx2
        d[2, ix] = (bzx1 - bzx2) * cps - (bxx1 - bxx2) * sps
        
        # Z-derivative (index 59+i*2+1)
        iz = ix + 1
        d[0, iz] = (bxz1 + bxz2) * cps + (bzz1 + bzz2) * sps
        d[1, iz] = byz1 + byz2
        d[2, iz] = (bzz1 + bzz2) * cps - (bxz1 + bxz2) * sps
        
        # With sps factor (indices 69-78)
        ix2 = ix + 10
        iz2 = iz + 10
        d[0, ix2] = sps * ((bxx1 + bxx2) * cps + (bzx1 + bzx2) * sps)
        d[1, ix2] = sps * (byx1 + byx2)
        d[2, ix2] = sps * ((bzx1 + bzx2) * cps - (bxx1 + bxx2) * sps)
        
        d[0, iz2] = sps * ((bxz1 - bxz2) * cps + (bzz1 - bzz2) * sps)
        d[1, iz2] = sps * (byz1 - byz2)
        d[2, iz2] = sps * ((bzz1 - bzz2) * cps - (bxz1 - bxz2) * sps)
    
    return d


def dipxyz_vec(x, y, z):
    """
    Vectorized dipole field calculation.
    Returns 9 components: bxx,byx,bzx,bxy,byy,bzy,bxz,byz,bzz
    """
    # Ensure arrays
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