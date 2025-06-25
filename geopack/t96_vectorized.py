"""
Vectorized implementation of the T96 magnetospheric magnetic field model.

This implementation follows the principles outlined in direction_vectorize.md:
1. All functions accept NumPy arrays for x, y, z coordinates
2. Conditional logic uses np.where instead of if/else
3. Safe division using np.divide with where parameter
4. No global variables - all parameters passed explicitly
5. Proper array initialization with np.zeros_like()

The vectorized version provides significant performance improvements
for processing multiple points simultaneously.
"""

import numpy as np
from scipy import special


def t96_vectorized(parmod, ps, x, y, z):
    """
    Vectorized version of the T96 magnetic field model.
    
    Parameters
    ----------
    parmod : array_like
        10-element array containing model parameters:
        [0] - solar wind pressure pdyn (nanopascals)
        [1] - dst (nanotesla)
        [2] - byimf (nanotesla)
        [3] - bzimf (nanotesla)
        [4-9] - unused
    ps : float
        Geodipole tilt angle in radians
    x, y, z : array_like
        GSM coordinates in Re (Earth radii)
        
    Returns
    -------
    bx, by, bz : ndarray
        Magnetic field components in GSM system (nT)
    """
    # Convert inputs to numpy arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    # Extract parameters
    pdyn, dst, byimf, bzimf = parmod[0:4]
    
    # Constants
    pdyn0, eps10 = 2.0, 3630.7
    a = np.array([1.162, 22.344, 18.50, 2.602, 6.903, 5.287, 0.5790, 0.4462, 0.7850])
    am0, s0, x00, dsig = 70.0, 1.08, 5.48, 0.005
    delimfx, delimfy = 20.0, 10.0
    
    sps = np.sin(ps)
    cps = np.cos(ps)
    
    # Calculate IMF-related quantities
    depr = 0.8 * dst - 13.0 * np.sqrt(pdyn)
    bt = np.sqrt(byimf**2 + bzimf**2)
    
    # Handle theta calculation
    if (byimf == 0) and (bzimf == 0):
        theta = 0
    else:
        theta = np.arctan2(byimf, bzimf)
        if theta < 0:
            theta += 2 * np.pi
    
    ct = np.cos(theta)
    st = np.sin(theta)
    eps = 718.5 * np.sqrt(pdyn) * bt * np.sin(theta / 2.0)
    
    facteps = eps / eps10 - 1.0
    factpd = np.sqrt(pdyn / pdyn0) - 1.0
    rcampl = -a[0] * depr
    tampl2 = a[1] + a[2] * factpd + a[3] * facteps
    tampl3 = a[4] + a[5] * factpd
    b1ampl = a[6] + a[7] * facteps
    b2ampl = 20.0 * b1ampl
    reconn = a[8]
    
    xappa = (pdyn / pdyn0)**0.14
    xappa3 = xappa**3
    
    # Coordinate transformations
    ys = y * ct - z * st
    zs = z * ct + y * st
    
    # IMF penetration factor
    factimf = np.exp(x / delimfx - (ys / delimfy)**2)
    
    # External IMF components
    oimfx = np.zeros_like(x)
    oimfy = reconn * byimf * factimf
    oimfz = reconn * bzimf * factimf
    
    rimfampl = reconn * bt
    
    # Scale coordinates
    xx = x * xappa
    yy = y * xappa
    zz = z * xappa
    
    # Magnetopause parameters
    x0 = x00 / xappa
    am = am0 / xappa
    rho2 = y**2 + z**2
    asq = am**2
    xmxm = am + x - x0
    xmxm = np.maximum(xmxm, 0)  # Vectorized version of if xmxm < 0: xmxm = 0
    axx0 = xmxm**2
    aro = asq + rho2
    sqrt_arg = (aro + axx0)**2 - 4.0 * asq * axx0
    sqrt_arg = np.maximum(sqrt_arg, 0)  # Ensure non-negative
    sigma = np.sqrt((aro + axx0 + np.sqrt(sqrt_arg)) / (2.0 * asq))
    
    # Initialize output arrays
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    
    # Define masks for three regions
    mask_inside = sigma < (s0 - dsig)
    mask_layer = (sigma >= (s0 - dsig)) & (sigma < (s0 + dsig))
    mask_outside = sigma >= (s0 + dsig)
    
    # Case 1: Inside magnetosphere
    if np.any(mask_inside):
        idx = mask_inside
        bx_in, by_in, bz_in = calculate_internal_field(
            parmod, ps, xx[idx], yy[idx], zz[idx], x[idx], y[idx], z[idx],
            ys[idx], zs[idx], xappa, xappa3, rcampl, tampl2, tampl3,
            b1ampl, b2ampl, rimfampl, ct, st
        )
        bx[idx] = bx_in
        by[idx] = by_in
        bz[idx] = bz_in
    
    # Case 2: Boundary layer
    if np.any(mask_layer):
        idx = mask_layer
        sigma_layer = sigma[idx]
        
        # Internal field contribution
        bx_int, by_int, bz_int = calculate_internal_field(
            parmod, ps, xx[idx], yy[idx], zz[idx], x[idx], y[idx], z[idx],
            ys[idx], zs[idx], xappa, xappa3, rcampl, tampl2, tampl3,
            b1ampl, b2ampl, rimfampl, ct, st
        )
        
        # Dipole field
        qx, qy, qz = dipole_vectorized(ps, x[idx], y[idx], z[idx])
        
        # Interpolation factors
        fint = 0.5 * (1.0 - (sigma_layer - s0) / dsig)
        fext = 1.0 - fint
        
        # Blend internal and external fields
        bx[idx] = (bx_int + qx) * fint + oimfx[idx] * fext - qx
        by[idx] = (by_int + qy) * fint + oimfy[idx] * fext - qy
        bz[idx] = (bz_int + qz) * fint + oimfz[idx] * fext - qz
    
    # Case 3: Outside magnetosphere
    if np.any(mask_outside):
        idx = mask_outside
        qx, qy, qz = dipole_vectorized(ps, x[idx], y[idx], z[idx])
        bx[idx] = oimfx[idx] - qx
        by[idx] = oimfy[idx] - qy
        bz[idx] = oimfz[idx] - qz
    
    return bx, by, bz


def calculate_internal_field(parmod, ps, xx, yy, zz, x, y, z, ys, zs,
                            xappa, xappa3, rcampl, tampl2, tampl3,
                            b1ampl, b2ampl, rimfampl, ct, st):
    """Calculate internal magnetospheric field."""
    sps = np.sin(ps)
    
    # Dipole shielding
    cfx, cfy, cfz = dipshld_vectorized(ps, xx, yy, zz)
    
    # Tail and ring current
    bxrc, byrc, bzrc, bxt2, byt2, bzt2, bxt3, byt3, bzt3 = tailrc96_vectorized(
        sps, xx, yy, zz
    )
    
    # Birkeland currents
    r1x, r1y, r1z = birk1tot_02_vectorized(ps, xx, yy, zz)
    r2x, r2y, r2z = birk2tot_02_vectorized(ps, xx, yy, zz)
    
    # Interconnection field
    rimfx, rimfys, rimfzs = intercon_vectorized(xx, ys * xappa, zs * xappa)
    rimfy = rimfys * ct + rimfzs * st
    rimfz = rimfzs * ct - rimfys * st
    
    # Total internal field
    fx = (cfx * xappa3 + rcampl * bxrc + tampl2 * bxt2 + tampl3 * bxt3 + 
          b1ampl * r1x + b2ampl * r2x + rimfampl * rimfx)
    fy = (cfy * xappa3 + rcampl * byrc + tampl2 * byt2 + tampl3 * byt3 + 
          b1ampl * r1y + b2ampl * r2y + rimfampl * rimfy)
    fz = (cfz * xappa3 + rcampl * bzrc + tampl2 * bzt2 + tampl3 * bzt3 + 
          b1ampl * r1z + b2ampl * r2z + rimfampl * rimfz)
    
    return fx, fy, fz


def dipole_vectorized(ps, x, y, z):
    """Vectorized Earth's dipole field."""
    sps = np.sin(ps)
    cps = np.cos(ps)
    
    p = x**2
    u = z**2
    v = 3 * z * x
    t = y**2
    q = 30574.0 / np.power(p + t + u + 1e-15, 2.5)
    
    bx = q * ((t + u - 2 * p) * sps - v * cps)
    by = -3 * y * q * (x * sps + z * cps)
    bz = q * ((p + t - 2 * u) * cps - v * sps)
    
    return bx, by, bz


def dipshld_vectorized(ps, x, y, z):
    """Vectorized dipole shielding field."""
    cps = np.cos(ps)
    sps = np.sin(ps)
    
    a1 = np.array([0.24777, -27.003, -0.46815, 7.0637, -1.5918, -0.090317,
                   57.522, 13.757, 2.0100, 10.458, 4.5798, 2.1695])
    a2 = np.array([-0.65385, -18.061, -0.40457, -5.0995, 1.2846, 0.078231,
                   39.592, 13.291, 1.9970, 10.062, 4.5140, 2.1558])
    
    hx, hy, hz = cylharm_vectorized(a1, x, y, z)
    fx, fy, fz = cylhar1_vectorized(a2, x, y, z)
    
    bx = hx * cps + fx * sps
    by = hy * cps + fy * sps
    bz = hz * cps + fz * sps
    
    return bx, by, bz


def cylharm_vectorized(a, x, y, z):
    """Vectorized cylindrical harmonics expansion."""
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    rho = np.sqrt(y**2 + z**2)
    
    # Safe division for angles
    sinfi = np.divide(z, rho, out=np.ones_like(z), where=rho > 1e-8)
    cosfi = np.divide(y, rho, out=np.zeros_like(y), where=rho > 1e-8)
    
    # Handle rho=0 case
    mask_zero = rho < 1e-8
    if np.any(mask_zero):
        sinfi = np.where(mask_zero, 1.0, sinfi)
        cosfi = np.where(mask_zero, 0.0, cosfi)
        rho = np.where(mask_zero, 1e-8, rho)
    
    sinfi2 = sinfi**2
    si2co2 = sinfi2 - cosfi**2
    
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    
    # First 3 harmonics
    for i in range(3):
        dzeta = rho / a[i + 6]
        xksi = x / a[i + 6]
        xj0 = special.j0(dzeta)
        xj1 = special.j1(dzeta)
        xexp = np.exp(xksi)
        
        # Safe division for j1/dzeta
        j1_over_dzeta = np.divide(xj1, dzeta, 
                                  out=0.5 * np.ones_like(dzeta),
                                  where=dzeta > 1e-8)
        
        bx = bx - a[i] * xj1 * xexp * sinfi
        by = by + a[i] * (2 * j1_over_dzeta - xj0) * xexp * sinfi * cosfi
        bz = bz + a[i] * (j1_over_dzeta * si2co2 - xj0 * sinfi2) * xexp
    
    # Next 3 harmonics
    for i in range(3, 6):
        dzeta = rho / a[i + 6]
        xksi = x / a[i + 6]
        xj0 = special.j0(dzeta)
        xj1 = special.j1(dzeta)
        xexp = np.exp(xksi)
        
        j1_over_dzeta = np.divide(xj1, dzeta,
                                  out=0.5 * np.ones_like(dzeta),
                                  where=dzeta > 1e-8)
        
        brho = (xksi * xj0 - (dzeta**2 + xksi - 1) * j1_over_dzeta) * xexp * sinfi
        bphi = (xj0 + j1_over_dzeta * (xksi - 1)) * xexp * cosfi
        
        bx = bx + a[i] * (dzeta * xj0 + xksi * xj1) * xexp * sinfi
        by = by + a[i] * (brho * cosfi - bphi * sinfi)
        bz = bz + a[i] * (brho * sinfi + bphi * cosfi)
    
    return bx, by, bz


def cylhar1_vectorized(a, x, y, z):
    """Vectorized cylindrical harmonics (variant 1)."""
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    rho = np.sqrt(y**2 + z**2)
    
    # Safe division for angles
    sinfi = np.divide(z, rho, out=np.ones_like(z), where=rho > 1e-8)
    cosfi = np.divide(y, rho, out=np.zeros_like(y), where=rho > 1e-8)
    
    # Handle rho=0 case
    mask_zero = rho < 1e-8
    if np.any(mask_zero):
        sinfi = np.where(mask_zero, 1.0, sinfi)
        cosfi = np.where(mask_zero, 0.0, cosfi)
        rho = np.where(mask_zero, 1e-8, rho)
    
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    
    # First 3 terms
    for i in range(3):
        dzeta = rho / a[i + 6]
        xksi = x / a[i + 6]
        xj0 = special.j0(dzeta)
        xj1 = special.j1(dzeta)
        xexp = np.exp(xksi)
        brho = xj1 * xexp
        
        bx = bx - a[i] * xj0 * xexp
        by = by + a[i] * brho * cosfi
        bz = bz + a[i] * brho * sinfi
    
    # Next 3 terms
    for i in range(3, 6):
        dzeta = rho / a[i + 6]
        xksi = x / a[i + 6]
        xj0 = special.j0(dzeta)
        xj1 = special.j1(dzeta)
        xexp = np.exp(xksi)
        brho = (dzeta * xj0 + xksi * xj1) * xexp
        
        bx = bx + a[i] * (dzeta * xj1 - xj0 * (xksi + 1)) * xexp
        by = by + a[i] * brho * cosfi
        bz = bz + a[i] * brho * sinfi
    
    return bx, by, bz


# Placeholder functions for the complex tail/Birkeland current calculations
# These would need full vectorization following the same principles

def tailrc96_vectorized(sps, x, y, z):
    """
    Vectorized implementation of tail and ring current calculations.
    
    Includes contributions from:
    - Ring current (via shlcar3x3 and ringcurr96)
    - Tail sheet current (via shlcar3x3 and taildisk)
    - Tail current (via shlcar3x3 and tail87)
    """
    # Constants
    rh, dr = 9.0, 4.0
    g, d0, deltady = 10.0, 2.0, 10.0
    dr2 = dr * dr
    c11 = np.sqrt((1 + rh)**2 + dr2)
    c12 = np.sqrt((1 - rh)**2 + dr2)
    c1 = c11 - c12
    spsc1 = sps / c1
    rps = 0.5 * (c11 + c12) * sps
    
    # Calculate warping parameters
    r = np.sqrt(x**2 + y**2 + z**2)
    sq1 = np.sqrt((r + rh)**2 + dr2)
    sq2 = np.sqrt((r - rh)**2 + dr2)
    c = sq1 - sq2
    cs = (r + rh) / sq1 - (r - rh) / sq2
    
    # Safe division for r
    r_safe = np.where(r < 1e-8, 1e-8, r)
    spss = spsc1 / r_safe * c
    
    # Ensure spss is in valid range [-1, 1]
    spss = np.clip(spss, -1.0, 1.0)
    cpss = np.sqrt(1 - spss**2)
    
    # Calculate dpsrr safely
    spss_arg = (r * c1)**2 - (c * sps)**2
    spss_arg = np.maximum(spss_arg, 1e-8)
    dpsrr = sps / (r_safe**2 * np.sqrt(spss_arg)) * (cs * r - c)
    
    # Warping factor
    wfac = y / (y**4 + 1e4)
    w = wfac * y**3
    ws = 4e4 * y * wfac**2
    warp = g * sps * w
    
    # Warped coordinates
    xs = x * cpss - z * spss
    zsww = z * cpss + x * spss
    zs = zsww + warp
    
    # Derivatives for warped coordinates
    dxsx = cpss - x * zsww * dpsrr
    dxsy = -y * zsww * dpsrr
    dxsz = -spss - z * zsww * dpsrr
    dzsx = spss + x * xs * dpsrr
    dzsy = xs * y * dpsrr + g * sps * ws
    dzsz = cpss + xs * z * dpsrr
    
    # D parameter
    d = d0 + deltady * (y / 20.0)**2
    dddy = deltady * y * 0.005
    dzetas = np.sqrt(zs**2 + d**2)
    ddzetadx = zs * dzsx / dzetas
    ddzetady = (zs * dzsy + d * dddy) / dzetas
    ddzetadz = zs * dzsz / dzetas
    
    # Pack warped params for subfunctions
    warp_params = {
        'xs': xs, 'zs': zs, 'zsww': zsww,
        'dxsx': dxsx, 'dxsy': dxsy, 'dxsz': dxsz,
        'dzsx': dzsx, 'dzsy': dzsy, 'dzsz': dzsz,
        'cpss': cpss, 'spss': spss, 'dpsrr': dpsrr,
        'rps': rps, 'warp': warp,
        'd': d, 'dddy': dddy, 'dzetas': dzetas,
        'ddzetadx': ddzetadx, 'ddzetady': ddzetady, 'ddzetadz': ddzetadz
    }
    
    # Coefficient arrays
    arc = np.array([-3.087, 3.516, 18.81, -13.95, -5.497, 0.171, 2.392, -2.728, 
                    -14.79, 11.08, 4.388, 0.0249, 0.703, -0.796, -3.835, 2.642, 
                    -0.240, -0.729, -0.368, 0.133, 2.795, -1.078, 0.801, 0.124, 
                    0.615, -0.220, -4.424, 1.730, -1.716, -0.230, -0.245, 0.086, 
                    1.547, -0.656, -0.653, 0.207, 12.75, 11.37, 636.4, 1.752, 
                    3.604, 12.83, 7.412, 9.434, 676.7, 1.701, 3.580, 14.64])
    
    atail2 = np.array([.874, -.911, 2.209, -2.159, -7.059, 5.924, -1.916, 1.996, 
                       -3.877, 3.947, 11.38, -8.343, 1.194, -1.244, 3.738, -4.406, 
                       -20.66, 3.020, .218, -.099, -.927, .155, .699, -.081, 
                       -.756, .468, 4.266, -.371, -3.920, .022, .703, -.549, 
                       -6.675, .827, -2.234, -1.622, 5.187, 6.802, 39.13, 2.784, 
                       6.979, 25.71, 4.495, 8.068, 93.47, 4.158, 9.313, 57.18])
    
    atail3 = np.array([-19091., -3011., 20582., 4242., -2377., -1504., 19884., 2725., 
                       -21389., -3990., 2401., 1548., -946., 490., 986., -489., 
                       -67.9, 8.71, -45.1, -10.7, 210.7, 11.41, -178.0, .755, 
                       339.3, 9.90, 69.5, -118.0, 22.8, 45.9, -425., 15.4, 
                       118.2, 65.5, -201., -14.5, 19.6, 20.3, 86.4, 22.5, 
                       23.4, 48.4, 24.6, 123.5, 223.5, 39.5, 65.8, 266.2])
    
    # Ring current
    wx, wy, wz = shlcar3x3_vectorized(arc, x, y, z, sps)
    hx, hy, hz = ringcurr96_vectorized(x, y, z, warp_params)
    bxrc = wx + hx
    byrc = wy + hy
    bzrc = wz + hz
    
    # Tail disk
    wx, wy, wz = shlcar3x3_vectorized(atail2, x, y, z, sps)
    hx, hy, hz = taildisk_vectorized(x, y, z, warp_params)
    bxt2 = wx + hx
    byt2 = wy + hy
    bzt2 = wz + hz
    
    # Tail current
    wx, wy, wz = shlcar3x3_vectorized(atail3, x, y, z, sps)
    hx, hz = tail87_vectorized(x, z, warp_params)
    bxt3 = wx + hx
    byt3 = wy
    bzt3 = wz + hz
    
    return bxrc, byrc, bzrc, bxt2, byt2, bzt2, bxt3, byt3, bzt3


def shlcar3x3_vectorized(a, x, y, z, sps):
    """Vectorized shielded cartesian 3x3 harmonic expansion."""
    cps = np.sqrt(1 - sps**2)
    s3ps = 4 * cps**2 - 1
    
    hx = np.zeros_like(x)
    hy = np.zeros_like(y)
    hz = np.zeros_like(z)
    
    l = 0
    for m in range(2):
        for i in range(3):
            p = a[36 + i]
            q = a[42 + i]
            
            for k in range(3):
                r = a[39 + k]
                s = a[45 + k]
                
                for n in range(2):
                    if m == 0:
                        cypi = np.cos(y / p)
                        sypi = np.sin(y / p)
                        szrk = np.sin(z / r)
                        czrk = np.cos(z / r)
                        sqpr = np.sqrt(1 / p**2 + 1 / r**2)
                        epr = np.exp(x * sqpr)
                        
                        dx_base = -sqpr * epr * cypi * szrk
                        dy_base = epr / p * sypi * szrk
                        dz_base = -epr / r * cypi * czrk
                        
                        if n == 0:
                            hx += a[l] * dx_base
                            hy += a[l] * dy_base
                            hz += a[l] * dz_base
                        else:
                            hx += a[l] * dx_base * cps
                            hy += a[l] * dy_base * cps
                            hz += a[l] * dz_base * cps
                    else:  # m == 1
                        cyqi = np.cos(y / q)
                        syqi = np.sin(y / q)
                        czsk = np.cos(z / s)
                        szsk = np.sin(z / s)
                        sqqs = np.sqrt(1 / q**2 + 1 / s**2)
                        eqs = np.exp(x * sqqs)
                        
                        dx_base = -sps * sqqs * eqs * cyqi * czsk
                        dy_base = sps * eqs / q * syqi * czsk
                        dz_base = sps * eqs / s * cyqi * szsk
                        
                        if n == 0:
                            hx += a[l] * dx_base
                            hy += a[l] * dy_base
                            hz += a[l] * dz_base
                        else:
                            hx += a[l] * dx_base * s3ps
                            hy += a[l] * dy_base * s3ps
                            hz += a[l] * dz_base * s3ps
                    
                    l += 1
    
    return hx, hy, hz


def ringcurr96_vectorized(x, y, z, warp_params):
    """Vectorized ring current contribution."""
    # Constants
    d0, deltadx, xd, xldx = 2.0, 0.0, 0.0, 4.0
    # Original values are F multiplied by BETA and by -0.43
    f = np.array([569.895366, -1603.386993])
    beta = np.array([2.722188, 3.766875])
    
    # Extract warped params
    xs = warp_params['xs']
    dxsx = warp_params['dxsx']
    dxsy = warp_params['dxsy']
    dxsz = warp_params['dxsz']
    spss = warp_params['spss']
    cpss = warp_params['cpss']
    dpsrr = warp_params['dpsrr']
    zsww = warp_params['zsww']
    dzsx = warp_params['dzsx']
    dzsz = warp_params['dzsz']
    
    # Recalculate some parameters for ring current
    dzsy = xs * y * dpsrr  # No warping in Y-Z plane for ring current
    xxd = x - xd
    fdx = 0.5 * (1 + xxd / np.sqrt(xxd**2 + xldx**2))
    dddx = deltadx * 0.5 * xldx**2 / np.power(xxd**2 + xldx**2, 1.5)
    d = d0 + deltadx * fdx
    
    # Spread out the sheet
    zs = zsww
    dzetas = np.sqrt(zs**2 + d**2)
    rhos = np.sqrt(xs**2 + y**2)
    ddzetadx = (zs * dzsx + d * dddx) / dzetas
    ddzetady = zs * dzsy / dzetas
    ddzetadz = zs * dzsz / dzetas
    
    # Safe division for derivatives
    rhos_safe = np.where(rhos < 1e-5, 1e-5, rhos)
    drhosdx = xs * dxsx / rhos_safe
    drhosdy = (xs * dxsy + y) / rhos_safe
    drhosdz = xs * dxsz / rhos_safe
    
    # Handle rhos = 0 case
    mask_zero = rhos < 1e-5
    drhosdx = np.where(mask_zero, 0.0, drhosdx)
    drhosdy = np.where(mask_zero, np.sign(y), drhosdy)
    drhosdz = np.where(mask_zero, 0.0, drhosdz)
    
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    
    for i in range(2):
        bi = beta[i]
        
        s1 = np.sqrt((dzetas + bi)**2 + (rhos + bi)**2)
        s2 = np.sqrt((dzetas + bi)**2 + (rhos - bi)**2)
        ds1ddz = (dzetas + bi) / s1
        ds2ddz = (dzetas + bi) / s2
        ds1drhos = (rhos + bi) / s1
        ds2drhos = (rhos - bi) / s2
        
        ds1dx = ds1ddz * ddzetadx + ds1drhos * drhosdx
        ds1dy = ds1ddz * ddzetady + ds1drhos * drhosdy
        ds1dz = ds1ddz * ddzetadz + ds1drhos * drhosdz
        
        ds2dx = ds2ddz * ddzetadx + ds2drhos * drhosdx
        ds2dy = ds2ddz * ddzetady + ds2drhos * drhosdy
        ds2dz = ds2ddz * ddzetadz + ds2drhos * drhosdz
        
        s1ts2 = s1 * s2
        s1ps2 = s1 + s2
        s1ps2sq = s1ps2**2
        fac1 = np.sqrt(s1ps2sq - (2 * bi)**2)
        as0 = fac1 / (s1ts2 * s1ps2sq)
        term1 = 1 / (s1ts2 * s1ps2 * fac1)
        fac2 = as0 / s1ps2sq
        dasds1 = term1 - fac2 / s1 * (s2**2 + s1 * (3 * s1 + 4 * s2))
        dasds2 = term1 - fac2 / s2 * (s1**2 + s2 * (3 * s2 + 4 * s1))
        
        dasdx = dasds1 * ds1dx + dasds2 * ds2dx
        dasdy = dasds1 * ds1dy + dasds2 * ds2dy
        dasdz = dasds1 * ds1dz + dasds2 * ds2dz
        
        bx += f[i] * ((2 * as0 + y * dasdy) * spss - xs * dasdz + 
                      as0 * dpsrr * (y**2 * cpss + z * zs))
        by += -f[i] * y * (as0 * dpsrr * xs + dasdz * cpss + dasdx * spss)
        bz += f[i] * ((2 * as0 + y * dasdy) * cpss + xs * dasdx - 
                      as0 * dpsrr * (x * zs + y**2 * spss))
    
    return bx, by, bz


def taildisk_vectorized(x, y, z, warp_params):
    """Vectorized tail disk contribution - similar to ringcurr96 but with different params."""
    xshift = 4.5
    # Original F values multiplied by BETA to economize calculations
    f = np.array([-745796.7338, 1176470.141, -444610.529, -57508.01028])
    beta = np.array([7.9250000, 8.0850000, 8.4712500, 27.89500])
    
    # Extract warped params
    xs = warp_params['xs']
    dxsx = warp_params['dxsx']
    dxsy = warp_params['dxsy']
    dxsz = warp_params['dxsz']
    dzetas = warp_params['dzetas']
    ddzetadx = warp_params['ddzetadx']
    ddzetady = warp_params['ddzetady']
    ddzetadz = warp_params['ddzetadz']
    spss = warp_params['spss']
    cpss = warp_params['cpss']
    dpsrr = warp_params['dpsrr']
    zs = warp_params['zs']
    zsww = warp_params['zsww']
    
    rhos = np.sqrt((xs - xshift)**2 + y**2)
    
    # Safe division for derivatives
    rhos_safe = np.where(rhos < 1e-5, 1e-5, rhos)
    drhosdx = (xs - xshift) * dxsx / rhos_safe
    drhosdy = ((xs - xshift) * dxsy + y) / rhos_safe
    drhosdz = (xs - xshift) * dxsz / rhos_safe
    
    # Handle rhos = 0 case
    mask_zero = rhos < 1e-5
    drhosdx = np.where(mask_zero, 0.0, drhosdx)
    drhosdy = np.where(mask_zero, np.sign(y), drhosdy)
    drhosdz = np.where(mask_zero, 0.0, drhosdz)
    
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    
    for i in range(4):
        bi = beta[i]
        
        s1 = np.sqrt((dzetas + bi)**2 + (rhos + bi)**2)
        s2 = np.sqrt((dzetas + bi)**2 + (rhos - bi)**2)
        ds1ddz = (dzetas + bi) / s1
        ds2ddz = (dzetas + bi) / s2
        ds1drhos = (rhos + bi) / s1
        ds2drhos = (rhos - bi) / s2
        
        ds1dx = ds1ddz * ddzetadx + ds1drhos * drhosdx
        ds1dy = ds1ddz * ddzetady + ds1drhos * drhosdy
        ds1dz = ds1ddz * ddzetadz + ds1drhos * drhosdz
        
        ds2dx = ds2ddz * ddzetadx + ds2drhos * drhosdx
        ds2dy = ds2ddz * ddzetady + ds2drhos * drhosdy
        ds2dz = ds2ddz * ddzetadz + ds2drhos * drhosdz
        
        s1ts2 = s1 * s2
        s1ps2 = s1 + s2
        s1ps2sq = s1ps2**2
        fac1 = np.sqrt(s1ps2sq - (2 * bi)**2)
        as0 = fac1 / (s1ts2 * s1ps2sq)
        term1 = 1 / (s1ts2 * s1ps2 * fac1)
        fac2 = as0 / s1ps2sq
        dasds1 = term1 - fac2 / s1 * (s2**2 + s1 * (3 * s1 + 4 * s2))
        dasds2 = term1 - fac2 / s2 * (s1**2 + s2 * (3 * s2 + 4 * s1))
        
        dasdx = dasds1 * ds1dx + dasds2 * ds2dx
        dasdy = dasds1 * ds1dy + dasds2 * ds2dy
        dasdz = dasds1 * ds1dz + dasds2 * ds2dz
        
        bx += f[i] * ((2 * as0 + y * dasdy) * spss - (xs - xshift) * dasdz + 
                      as0 * dpsrr * (y**2 * cpss + z * zsww))
        by += -f[i] * y * (as0 * dpsrr * xs + dasdz * cpss + dasdx * spss)
        bz += f[i] * ((2 * as0 + y * dasdy) * cpss + (xs - xshift) * dasdx - 
                      as0 * dpsrr * (x * zsww + y**2 * spss))
    
    return bx, by, bz


def tail87_vectorized(x, z, warp_params):
    """Vectorized 1987 tail model."""
    # Extract warped params
    rps = warp_params['rps']
    warp = warp_params['warp']
    
    # Constants
    dd = 3.0
    hpi = 1.5707963
    rt = 40.0
    xn = -10.0
    tscale = 1.0
    
    b0 = 0.391734
    b1 = 5.89715 * tscale
    b2 = 24.6833 * tscale**2
    
    x1 = -1.261
    x2 = -0.663
    xn21 = (xn - x1)**2
    xnr = 1.0 / (xn - x2)
    adln = -np.log(xnr**2 * xn21)
    
    # Warped z coordinates
    zs = z - rps + warp
    zp = z - rt
    zm = z + rt
    
    # X-related calculations
    xnx = xn - x
    xnx2 = xnx**2
    xc1 = x - x1
    xc2 = x - x2
    xc22 = xc2**2
    xr2 = xc2 * xnr
    xc12 = xc1**2
    
    # B-field components
    d2 = dd**2
    b20 = zs**2 + d2
    b2p = zp**2 + d2
    b2m = zm**2 + d2
    b = np.sqrt(b20)
    bp = np.sqrt(b2p)
    bm = np.sqrt(b2m)
    
    xa1 = xc12 + b20
    xap1 = xc12 + b2p
    xam1 = xc12 + b2m
    xa2 = 1.0 / (xc22 + b20)
    xap2 = 1.0 / (xc22 + b2p)
    xam2 = 1.0 / (xc22 + b2m)
    xna = xnx2 + b20
    xnap = xnx2 + b2p
    xnam = xnx2 + b2m
    
    f = b20 - xc22
    fp = b2p - xc22
    fm = b2m - xc22
    
    xln1 = np.log(xn21 / xna)
    xlnp1 = np.log(xn21 / xnap)
    xlnm1 = np.log(xn21 / xnam)
    xln2 = xln1 + adln
    xlnp2 = xlnp1 + adln
    xlnm2 = xlnm1 + adln
    
    aln = 0.25 * (xlnp1 + xlnm1 - 2.0 * xln1)
    
    s0 = (np.arctan(xnx / b) + hpi) / b
    s0p = (np.arctan(xnx / bp) + hpi) / bp
    s0m = (np.arctan(xnx / bm) + hpi) / bm
    
    s1 = (xln1 * 0.5 + xc1 * s0) / xa1
    s1p = (xlnp1 * 0.5 + xc1 * s0p) / xap1
    s1m = (xlnm1 * 0.5 + xc1 * s0m) / xam1
    
    s2 = (xc2 * xa2 * xln2 - xnr - f * xa2 * s0) * xa2
    s2p = (xc2 * xap2 * xlnp2 - xnr - fp * xap2 * s0p) * xap2
    s2m = (xc2 * xam2 * xlnm2 - xnr - fm * xam2 * s0m) * xam2
    
    g1 = (b20 * s0 - 0.5 * xc1 * xln1) / xa1
    g1p = (b2p * s0p - 0.5 * xc1 * xlnp1) / xap1
    g1m = (b2m * s0m - 0.5 * xc1 * xlnm1) / xam1
    
    g2 = ((0.5 * f * xln2 + 2.0 * s0 * b20 * xc2) * xa2 + xr2) * xa2
    g2p = ((0.5 * fp * xlnp2 + 2.0 * s0p * b2p * xc2) * xap2 + xr2) * xap2
    g2m = ((0.5 * fm * xlnm2 + 2.0 * s0m * b2m * xc2) * xam2 + xr2) * xam2
    
    bx = (b0 * (zs * s0 - 0.5 * (zp * s0p + zm * s0m)) +
          b1 * (zs * s1 - 0.5 * (zp * s1p + zm * s1m)) +
          b2 * (zs * s2 - 0.5 * (zp * s2p + zm * s2m)))
    
    bz = (b0 * aln +
          b1 * (g1 - 0.5 * (g1p + g1m)) +
          b2 * (g2 - 0.5 * (g2p + g2m)))
    
    return bx, bz


def birk1tot_02_vectorized(ps, x, y, z):
    """Vectorized Birkeland field region 1 (placeholder)."""
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    return bx, by, bz


def birk2tot_02_vectorized(ps, x, y, z):
    """Vectorized Birkeland field region 2 (placeholder)."""
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    return bx, by, bz


def intercon_vectorized(x, y, z):
    """Vectorized interconnection field (placeholder)."""
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    return bx, by, bz


# Utility function to handle scalar inputs
def t96(parmod, ps, x, y, z):
    """
    Wrapper for scalar inputs - maintains compatibility with original interface.
    """
    scalar_input = np.isscalar(x)
    
    bx, by, bz = t96_vectorized(parmod, ps, x, y, z)
    
    if scalar_input:
        return float(bx.item()), float(by.item()), float(bz.item())
    else:
        return bx, by, bz


if __name__ == '__main__':
    # Test the vectorized implementation
    print("Testing T96 vectorized implementation...")
    
    # Test parameters
    parmod = [2.0, -10.0, 0.5, -3.0, 0, 0, 0, 0, 0, 0]
    ps = 0.1
    
    # Test with scalar inputs
    x, y, z = 5.0, 0.0, 0.0
    bx, by, bz = t96(parmod, ps, x, y, z)
    print(f"Scalar input: B = ({bx:.3f}, {by:.3f}, {bz:.3f}) nT")
    
    # Test with array inputs
    x_arr = np.array([5.0, -10.0, 0.0])
    y_arr = np.array([0.0, 0.0, 5.0])
    z_arr = np.array([0.0, 0.0, 0.0])
    
    bx_arr, by_arr, bz_arr = t96_vectorized(parmod, ps, x_arr, y_arr, z_arr)
    print("\nArray input:")
    for i in range(len(x_arr)):
        print(f"  Point ({x_arr[i]}, {y_arr[i]}, {z_arr[i]}): "
              f"B = ({bx_arr[i]:.3f}, {by_arr[i]:.3f}, {bz_arr[i]:.3f}) nT")