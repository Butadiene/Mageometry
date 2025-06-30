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
try:
    from .condip1_exact_vectorized import condip1_exact_vectorized
except ImportError:
    from condip1_exact_vectorized import condip1_exact_vectorized


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
    bx, by, bz : ndarray or float
        Magnetic field components in GSM system (nT).
        Returns scalars if all inputs were scalars.
    """
    # Track if all inputs were scalar
    scalar_input = np.isscalar(x) and np.isscalar(y) and np.isscalar(z)
    
    # Convert inputs to numpy arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    # Broadcast arrays to same shape
    x, y, z = np.broadcast_arrays(x, y, z)
    
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
    
    # Return scalar if input was scalar
    if scalar_input:
        return bx.item(), by.item(), bz.item()
    else:
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
    arc = np.array([
        -3.087699646,3.516259114,18.81380577,-13.95772338,-5.497076303,0.1712890838,
        2.392629189,-2.728020808,-14.79349936,11.08738083,4.388174084,0.2492163197E-01,
        0.7030375685,-.7966023165,-3.835041334,2.642228681,-0.2405352424,-0.7297705678,
        -0.3680255045,0.1333685557,2.795140897,-1.078379954,0.8014028630,0.1245825565,
        0.6149982835,-0.2207267314,-4.424578723,1.730471572,-1.716313926,-0.2306302941,
        -0.2450342688,0.8617173961E-01,1.54697858,-0.6569391113,-0.6537525353,0.2079417515,
        12.75434981,11.37659788,636.4346279,1.752483754,3.604231143,12.83078674,
        7.412066636,9.434625736,676.7557193,1.701162737,3.580307144,14.64298662])
    
    atail2 = np.array([
        .8747515218,-.9116821411,2.209365387,-2.159059518,-7.059828867,5.924671028,
        -1.916935691,1.996707344,-3.877101873,3.947666061,11.38715899,-8.343210833,
        1.194109867,-1.244316975,3.73895491,-4.406522465,-20.66884863,3.020952989,
        .2189908481,-.09942543549,-.927225562,.1555224669,.6994137909,-.08111721003,
        -.7565493881,.4686588792,4.266058082,-.3717470262,-3.920787807,.02298569870,
        .7039506341,-.5498352719,-6.675140817,.8279283559,-2.234773608,-1.622656137,
        5.187666221,6.802472048,39.13543412,2.784722096,6.979576616,25.71716760,
        4.495005873,8.068408272,93.47887103,4.158030104,9.313492566,57.18240483])
    
    atail3 = np.array([
        -19091.95061,-3011.613928,20582.16203,4242.918430,-2377.091102,-1504.820043,
        19884.04650,2725.150544,-21389.04845,-3990.475093,2401.610097,1548.171792,
        -946.5493963,490.1528941,986.9156625,-489.3265930,-67.99278499,8.711175710,
        -45.15734260,-10.76106500,210.7927312,11.41764141,-178.0262808,.7558830028,
        339.3806753,9.904695974,69.50583193,-118.0271581,22.85935896,45.91014857,
        -425.6607164,15.47250738,118.2988915,65.58594397,-201.4478068,-14.57062940,
        19.69877970,20.30095680,86.45407420,22.50403727,23.41617329,48.48140573,
        24.61031329,123.5395974,223.5367692,39.50824342,65.83385762,266.2948657])
    
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
                
                if m == 0:
                    cypi = np.cos(y / p)
                    sypi = np.sin(y / p)
                    szrk = np.sin(z / r)
                    czrk = np.cos(z / r)
                    sqpr = np.sqrt(1 / p**2 + 1 / r**2)
                    epr = np.exp(x * sqpr)
                    
                    dx = -sqpr * epr * cypi * szrk
                    dy = epr / p * sypi * szrk
                    dz = -epr / r * cypi * czrk
                else:  # m == 1
                    cyqi = np.cos(y / q)
                    syqi = np.sin(y / q)
                    czsk = np.cos(z / s)
                    szsk = np.sin(z / s)
                    sqqs = np.sqrt(1 / q**2 + 1 / s**2)
                    eqs = np.exp(x * sqqs)
                    
                    dx = -sps * sqqs * eqs * cyqi * czsk
                    dy = sps * eqs / q * syqi * czsk
                    dz = sps * eqs / s * cyqi * szsk
                
                for n in range(2):
                    if n == 0:
                        hx += a[l] * dx
                        hy += a[l] * dy
                        hz += a[l] * dz
                    else:
                        if m == 0:
                            # For n=1, reuse dx,dy,dz and multiply by cps
                            hx += a[l] * dx * cps
                            hy += a[l] * dy * cps
                            hz += a[l] * dz * cps
                        else:
                            # For n=1, reuse dx,dy,dz and multiply by s3ps
                            hx += a[l] * dx * s3ps
                            hy += a[l] * dy * s3ps
                            hz += a[l] * dz * s3ps
                    
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
    """
    Vectorized Birkeland field region 1.
    This is the most complex function with 4 different regions and interpolation.
    """
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    # Model constants
    rh, dr = 9.0, 4.0
    xltday, xltnght = 78.0, 70.0
    dtet0 = 0.034906
    tnoonn = (90 - xltday) * 0.01745329
    tnoons = np.pi - tnoonn
    dtetdn = (xltday - xltnght) * 0.01745329
    dr2 = dr * dr
    sps = np.sin(ps)
    
    # Calculate tet0 for all points
    r2 = x**2 + y**2 + z**2
    r = np.sqrt(r2)
    r3 = r * r2
    
    # Safe division
    r_safe = np.where(r < 1e-9, 1e-9, r)
    
    rmrh = r - rh
    rprh = r + rh
    sqm = np.sqrt(rmrh**2 + dr2)
    sqp = np.sqrt(rprh**2 + dr2)
    c = sqp - sqm
    q = np.sqrt((rh + 1)**2 + dr2) - np.sqrt((rh - 1)**2 + dr2)
    spsas = sps / r_safe * c / q
    
    # Ensure spsas is in valid range
    spsas = np.clip(spsas, -1.0, 1.0)
    cpsas = np.sqrt(1 - spsas**2)
    
    xas = x * cpsas - z * spsas
    zas = x * spsas + z * cpsas
    
    # Calculate angles
    pas = np.arctan2(y, xas)
    tas = np.arctan2(np.sqrt(xas**2 + y**2), zas)
    stas = np.sin(tas)
    
    # Calculate f with safe division
    f_denom = (stas**6 * (1 - r3) + r3)**(1.0/6.0)
    f_denom_safe = np.where(f_denom < 1e-9, 1e-9, f_denom)
    f = stas / f_denom_safe
    
    # Ensure f is in valid range for arcsin
    f = np.clip(f, -1.0, 1.0)
    tet0 = np.arcsin(f)
    tet0 = np.where(tas > np.pi/2, np.pi - tet0, tet0)
    
    # Calculate region boundaries
    dtet = dtetdn * np.sin(pas * 0.5)**2
    tetr1n = tnoonn + dtet
    tetr1s = tnoons - dtet
    
    # Determine location for all points
    loc1 = (tet0 < tetr1n - dtet0) | (tet0 > tetr1s + dtet0)  # High latitude
    loc2 = (tet0 > tetr1n + dtet0) & (tet0 < tetr1s - dtet0)  # Plasma sheet
    loc3 = (tet0 >= tetr1n - dtet0) & (tet0 <= tetr1n + dtet0)  # North PSBL
    loc4 = (tet0 >= tetr1s - dtet0) & (tet0 <= tetr1s + dtet0)  # South PSBL
    
    # Initialize output arrays
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    
    # Process each region
    # Region 1: High latitude - use diploop1
    if np.any(loc1):
        idx = loc1
        bx1, by1, bz1 = diploop1_vectorized(x[idx], y[idx], z[idx], ps)
        bx[idx] = bx1
        by[idx] = by1
        bz[idx] = bz1
    
    # Region 2: Plasma sheet - use condip1
    if np.any(loc2):
        idx = loc2
        bx2, by2, bz2 = condip1_exact_vectorized(x[idx], y[idx], z[idx], ps)
        bx[idx] = bx2
        by[idx] = by2
        bz[idx] = bz2
    
    # Region 3: North PSBL - interpolate
    if np.any(loc3):
        idx = loc3
        bx3, by3, bz3 = interpolate_region3(
            x[idx], y[idx], z[idx], r[idx], r3[idx], ps, sps,
            cpsas[idx], spsas[idx], pas[idx], tetr1n[idx], dtet0
        )
        bx[idx] = bx3
        by[idx] = by3
        bz[idx] = bz3
    
    # Region 4: South PSBL - interpolate
    if np.any(loc4):
        idx = loc4
        bx4, by4, bz4 = interpolate_region4(
            x[idx], y[idx], z[idx], r[idx], r3[idx], ps, sps,
            cpsas[idx], spsas[idx], pas[idx], tetr1s[idx], dtet0
        )
        bx[idx] = bx4
        by[idx] = by4
        bz[idx] = bz4
    
    # Add shielding field
    bsx, bsy, bsz = birk1shld_vectorized(ps, x, y, z)
    
    return bx + bsx, by + bsy, bz + bsz


def interpolate_region3(x, y, z, r, r3, ps, sps, cpsas, spsas, pas, tetr1n, dtet0):
    """Interpolate between high-lat (diploop1) and plasma sheet (condip1) for north PSBL."""
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    r = np.atleast_1d(r)
    r3 = np.atleast_1d(r3)
    cpsas = np.atleast_1d(cpsas)
    spsas = np.atleast_1d(spsas)
    pas = np.atleast_1d(pas)
    tetr1n = np.atleast_1d(tetr1n)
    
    # Constants
    rh, dr = 9.0, 4.0
    cps = np.cos(ps)
    
    # Calculate boundary points
    t01 = tetr1n - dtet0
    t02 = tetr1n + dtet0
    sqr = np.sqrt(r)
    st01as = sqr / (r3 + 1/np.sin(t01)**6 - 1)**(1.0/6.0)
    st02as = sqr / (r3 + 1/np.sin(t02)**6 - 1)**(1.0/6.0)
    ct01as = np.sqrt(1 - st01as**2)
    ct02as = np.sqrt(1 - st02as**2)
    
    # Northern boundary point (high-lat)
    xas1 = r * st01as * np.cos(pas)
    y1 = r * st01as * np.sin(pas)
    zas1 = r * ct01as
    
    x1 = xas1 * cpsas + zas1 * spsas
    z1 = -xas1 * spsas + zas1 * cpsas
    
    # Get field at northern boundary using diploop1
    bx1, by1, bz1 = diploop1_vectorized(x1, y1, z1, ps)
    
    # Southern boundary point (plasma sheet)
    xas2 = r * st02as * np.cos(pas)
    y2 = r * st02as * np.sin(pas)
    zas2 = r * ct02as
    x2 = xas2 * cpsas + zas2 * spsas
    z2 = -xas2 * spsas + zas2 * cpsas
    
    # Get field at southern boundary using condip1
    bx2, by2, bz2 = condip1_exact_vectorized(x2, y2, z2, ps)
    
    # Interpolate
    ss = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    ds = np.sqrt((x-x1)**2 + (y-y1)**2 + (z-z1)**2)
    frac = ds / ss
    
    bx = bx1 * (1 - frac) + bx2 * frac
    by = by1 * (1 - frac) + by2 * frac
    bz = bz1 * (1 - frac) + bz2 * frac
    
    return bx, by, bz


def interpolate_region4(x, y, z, r, r3, ps, sps, cpsas, spsas, pas, tetr1s, dtet0):
    """Interpolate between plasma sheet (condip1) and high-lat (diploop1) for south PSBL."""
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    r = np.atleast_1d(r)
    r3 = np.atleast_1d(r3)
    cpsas = np.atleast_1d(cpsas)
    spsas = np.atleast_1d(spsas)
    pas = np.atleast_1d(pas)
    tetr1s = np.atleast_1d(tetr1s)
    
    # Constants
    rh, dr = 9.0, 4.0
    cps = np.cos(ps)
    
    # Calculate boundary points
    t01 = tetr1s - dtet0
    t02 = tetr1s + dtet0
    sqr = np.sqrt(r)
    st01as = sqr / (r3 + 1/np.sin(t01)**6 - 1)**(1.0/6.0)
    st02as = sqr / (r3 + 1/np.sin(t02)**6 - 1)**(1.0/6.0)
    ct01as = -np.sqrt(1 - st01as**2)  # Note negative for southern hemisphere
    ct02as = -np.sqrt(1 - st02as**2)
    
    # Northern boundary point (plasma sheet)
    xas1 = r * st01as * np.cos(pas)
    y1 = r * st01as * np.sin(pas)
    zas1 = r * ct01as
    
    x1 = xas1 * cpsas + zas1 * spsas
    z1 = -xas1 * spsas + zas1 * cpsas
    
    # Get field at northern boundary using condip1
    bx1, by1, bz1 = condip1_exact_vectorized(x1, y1, z1, ps)
    
    # Southern boundary point (high-lat)
    xas2 = r * st02as * np.cos(pas)
    y2 = r * st02as * np.sin(pas)
    zas2 = r * ct02as
    x2 = xas2 * cpsas + zas2 * spsas
    z2 = -xas2 * spsas + zas2 * cpsas
    
    # Get field at southern boundary using diploop1
    bx2, by2, bz2 = diploop1_vectorized(x2, y2, z2, ps)
    
    # Interpolate
    ss = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    ds = np.sqrt((x-x1)**2 + (y-y1)**2 + (z-z1)**2)
    frac = ds / ss
    
    bx = bx1 * (1 - frac) + bx2 * frac
    by = by1 * (1 - frac) + by2 * frac
    bz = bz1 * (1 - frac) + bz2 * frac
    
    return bx, by, bz


def birk2tot_02_vectorized(ps, x, y, z):
    """Vectorized Birkeland field region 2."""
    # Get shielding contribution
    wx, wy, wz = birk2shl_vectorized(x, y, z, ps)
    
    # Get main field contribution
    hx, hy, hz = r2_birk_vectorized(x, y, z, ps)
    
    return wx + hx, wy + hy, wz + hz


def intercon_vectorized(x, y, z):
    """
    Vectorized interconnection field inside the magnetosphere.
    Calculates the potential interconnection field using Fourier expansion.
    """
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    # Model coefficients
    a = np.array([
        -8.411078731, 5932254.951, -9073284.93, -11.68794634, 6027598.824,
        -9218378.368, -6.508798398, -11824.42793, 18015.66212, 7.99754043,
        13.9669886, 90.24475036, 16.75728834, 1015.645781, 1553.493216
    ])
    
    # Extract scale parameters
    p = a[9:12]
    r = a[12:15]
    rp = 1.0 / p
    rr = 1.0 / r
    
    # Initialize output
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    
    # Calculate Fourier components
    l = 0
    for i in range(3):
        cypi = np.cos(y * rp[i])
        sypi = np.sin(y * rp[i])
        
        for k in range(3):
            szrk = np.sin(z * rr[k])
            czrk = np.cos(z * rr[k])
            sqpr = np.sqrt(rp[i]**2 + rr[k]**2)
            epr = np.exp(x * sqpr)
            
            hx = -sqpr * epr * cypi * szrk
            hy = rp[i] * epr * sypi * szrk
            hz = -rr[k] * epr * cypi * czrk
            
            bx += a[l] * hx
            by += a[l] * hy
            bz += a[l] * hz
            l += 1
    
    return bx, by, bz


# Supporting functions for Birkeland currents

def diploop1_vectorized(x, y, z, ps):
    """Vectorized dipole loop for high latitude region."""
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    # Model coefficients for region 1
    c1 = np.array([
        -0.911582e-03, -0.376654e-02, -0.727423e-02, -0.270084e-02, -0.123899e-02,
        -0.154387e-02, -0.340040e-02, -0.191858e-01, -0.518979e-01, 0.635061e-01,
        0.440680, -0.396570, 0.561238e-02, 0.160938e-02, -0.451229e-02,
        -0.251810e-02, -0.151599e-02, -0.133665e-02, -0.962089e-03, -0.272085e-01,
        -0.524319e-01, 0.717024e-01, 0.523439, -0.405015, -89.5587, 23.2806
    ])
    
    # Constants
    xx1 = np.array([-11., -7, -7, -3, -3, 1, 1, 1, 5, 5, 9, 9])
    yy1 = np.array([2., 0, 4, 2, 6, 0, 4, 8, 2, 6, 0, 4])
    tilt = 1.00891
    xcentre = np.array([2.28397, -5.60831])
    radius = np.array([1.86106, 7.83281])
    dipx = 1.12541
    dipy = 0.945719
    rh = 9.0
    dr = 4.0
    
    sps = np.sin(ps)
    
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    
    # Dipole contributions
    for i in range(12):
        r2 = (xx1[i] * dipx)**2 + (yy1[i] * dipy)**2
        r_dip = np.sqrt(r2)
        rmrh = r_dip - rh
        rprh = r_dip + rh
        sqm = np.sqrt(rmrh**2 + dr**2)
        sqp = np.sqrt(rprh**2 + dr**2)
        c = sqp - sqm
        q = np.sqrt((rh + 1)**2 + dr**2) - np.sqrt((rh - 1)**2 + dr**2)
        spsas = sps / r_dip * c / q
        cpsas = np.sqrt(1 - spsas**2)
        
        xd = xx1[i] * dipx
        yd = yy1[i] * dipy
        xd_rot = xd * cpsas
        zd_rot = -xd * spsas
        
        # Calculate dipole field using dipxyz_vectorized
        # First dipole at (xd_rot, yd, zd_rot)
        dx = x - xd_rot
        dy = y - yd
        dz = z - zd_rot
        
        # Get Z-dipole field
        bx1z, by1z, bz1z = dipxyz_vectorized(dx, dy, dz, 2)
        
        # Get X-dipole field
        bx1x, by1x, bz1x = dipxyz_vectorized(dx, dy, dz, 0)
        
        # Handle symmetric y contribution if needed
        if np.abs(yd) > 1e-10:
            dy2 = y + yd
            bx2z, by2z, bz2z = dipxyz_vectorized(dx, dy2, dz, 2)
            bx2x, by2x, bz2x = dipxyz_vectorized(dx, dy2, dz, 0)
        else:
            bx2z = by2z = bz2z = 0.0
            bx2x = by2x = bz2x = 0.0
        
        # Z-component contribution (indices 0-11)
        bx += c1[i] * (bx1z + bx2z)
        by += c1[i] * (by1z + by2z)
        bz += c1[i] * (bz1z + bz2z)
        
        # X-component contribution (indices 12-23, scaled by sps)
        bx += c1[i + 12] * (bx1x + bx2x) * sps
        by += c1[i + 12] * (by1x + by2x) * sps
        bz += c1[i + 12] * (bz1x + bz2x) * sps
    
    # Loop contributions
    # First loop uses crosslp
    r2 = (xcentre[0] + radius[0])**2
    r = np.sqrt(r2)
    rmrh = r - rh
    rprh = r + rh
    sqm = np.sqrt(rmrh**2 + dr**2)
    sqp = np.sqrt(rprh**2 + dr**2)
    c = sqp - sqm
    q = np.sqrt((rh + 1)**2 + dr**2) - np.sqrt((rh - 1)**2 + dr**2)
    spsas = sps / r * c / q
    cpsas = np.sqrt(1 - spsas**2)
    xoct1 = x * cpsas - z * spsas
    yoct1 = y
    zoct1 = x * spsas + z * cpsas
    
    bxoct1, byoct1, bzoct1 = crosslp_vectorized(xoct1, yoct1, zoct1, xcentre[0], radius[0], tilt)
    bx += c1[24] * (bxoct1 * cpsas + bzoct1 * spsas)
    by += c1[24] * byoct1
    bz += c1[24] * (-bxoct1 * spsas + bzoct1 * cpsas)
    
    # Second loop uses circle
    r2 = (radius[1] - xcentre[1])**2
    r = np.sqrt(r2)
    rmrh = r - rh
    rprh = r + rh
    sqm = np.sqrt(rmrh**2 + dr**2)
    sqp = np.sqrt(rprh**2 + dr**2)
    c = sqp - sqm
    q = np.sqrt((rh + 1)**2 + dr**2) - np.sqrt((rh - 1)**2 + dr**2)
    spsas = sps / r * c / q
    cpsas = np.sqrt(1 - spsas**2)
    xoct2 = x * cpsas - z * spsas - xcentre[1]
    yoct2 = y
    zoct2 = x * spsas + z * cpsas
    
    bx_circ, by_circ, bz_circ = circle_vectorized(xoct2, yoct2, zoct2, radius[1])
    bx += c1[25] * (bx_circ * cpsas + bz_circ * spsas)
    by += c1[25] * by_circ
    bz += c1[25] * (-bx_circ * spsas + bz_circ * cpsas)
    
    return bx, by, bz


def dipxyz_vectorized(x, y, z, comp):
    """
    Vectorized version of dipxyz.
    Returns the field component for a dipole oriented along a specific axis.
    comp: 0 for X-dipole, 1 for Y-dipole, 2 for Z-dipole
    """
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    x2 = x**2
    y2 = y**2
    z2 = z**2
    r2 = x2 + y2 + z2
    
    # Earth's dipole moment constant
    xmr5 = 30574 / (r2 * r2 * np.sqrt(r2))
    xmr53 = 3 * xmr5
    
    if comp == 0:  # X-dipole
        bx = xmr5 * (3 * x2 - r2)
        by = xmr53 * x * y
        bz = xmr53 * x * z
    elif comp == 1:  # Y-dipole
        bx = xmr53 * x * y
        by = xmr5 * (3 * y2 - r2)
        bz = xmr53 * y * z
    else:  # Z-dipole
        bx = xmr53 * x * z
        by = xmr53 * y * z
        bz = xmr5 * (3 * z2 - r2)
    
    return bx, by, bz


def crosslp_vectorized(x, y, z, xc, rl, al):
    """
    Vectorized version of crosslp.
    Returns field components of a pair of loops with a common center and diameter,
    coinciding with the x axis. The loops are inclined to the equatorial plane by
    the angle al (radians) and shifted in the positive x-direction by the distance xc.
    """
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    cal = np.cos(al)
    sal = np.sin(al)
    
    y1 = y * cal - z * sal
    z1 = y * sal + z * cal
    y2 = y * cal + z * sal
    z2 = -y * sal + z * cal
    
    bx1, by1, bz1 = circle_vectorized(x - xc, y1, z1, rl)
    bx2, by2, bz2 = circle_vectorized(x - xc, y2, z2, rl)
    
    bx = bx1 + bx2
    by = (by1 + by2) * cal + (bz1 - bz2) * sal
    bz = -(by1 - by2) * sal + (bz1 + bz2) * cal
    
    return bx, by, bz


def condip1_vectorized(x, y, z, ps):
    """Vectorized confined dipole for plasma sheet region."""
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
    
    # Model coefficients for region 2
    c2 = np.array([
        6.04133, .305415, .606066e-02, .128379e-03, -.179406e-04,
        1.41714, -27.2586, -4.28833, -1.30675, 35.5607, 8.95792, .961617e-03,
        -.801477e-03, -.782795e-03, -1.65242, -16.5242, -5.33798, .424878e-03,
        .331787e-03, -.704305e-03, .844342e-03, .953682e-04, .886271e-03,
        25.1120, 20.9299, 5.14569, -44.1670, -51.0672, -1.87725, 20.2998,
        48.7505, -2.97415, 3.35184, -54.2921, -.838712, -10.5123, 70.7594,
        -4.94104, .106166e-03, .465791e-03, -.193719e-03, 10.8439, -29.7968,
        8.08068, .463507e-03, -.224475e-04, .177035e-03, -.317581e-03,
        -.264487e-03, .102075e-03, 7.71390, 10.1915, -4.99797, -23.1114,
        -29.2043, 12.2928, 10.9542, 33.6671, -9.3851, .174615e-03, -.789777e-06,
        .686047e-03, .460104e-04, -.345216e-02, .221871e-02, .110078e-01,
        -.661373e-02, .249201e-02, .343978e-01, -.193145e-05, .493963e-05,
        -.535748e-04, .191833e-04, -.100496e-03, -.210103e-03, -.232195e-02,
        .315335e-02, -.134320e-01, -.263222e-01
    ])
    
    # Constants
    xx2 = np.array([-10., -7, -4, -4, 0, 4, 4, 7, 10, 0, 0, 0, 0, 0])
    yy2 = np.array([3., 6, 3, 9, 6, 3, 9, 6, 3, 0, 0, 0, 0, 0])
    zz2 = np.array([20., 20, 4, 20, 4, 4, 20, 20, 20, 2, 3, 4.5, 7, 10])
    
    scalein = 0.08
    scaleout = 0.4
    dx = -0.16
    
    cps = np.cos(ps)
    sps = np.sin(ps)
    
    # Initialize result with conical harmonics
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    
    # First add the conical harmonics (indices 0-4)
    xsm = x * cps - z * sps - dx
    zsm = z * cps + x * sps
    ro2 = xsm**2 + y**2
    ro = np.sqrt(ro2)
    ro_safe = np.where(ro < 1e-9, 1e-9, ro)
    
    # Calculate cos/sin multiples
    cf0 = xsm / ro_safe
    sf0 = y / ro_safe
    cf1 = cf0**2 - sf0**2
    sf1 = 2 * sf0 * cf0
    cf2 = cf1 * cf0 - sf1 * sf0
    sf2 = sf1 * cf0 + cf1 * sf0
    cf3 = cf2 * cf0 - sf2 * sf0
    sf3 = sf2 * cf0 + cf2 * sf0
    cf4 = cf3 * cf0 - sf3 * sf0
    sf4 = sf3 * cf0 + cf3 * sf0
    
    # Stack arrays properly for vectorized computation
    cf = np.stack([cf0, cf1, cf2, cf3, cf4], axis=0)
    sf = np.stack([sf0, sf1, sf2, sf3, sf4], axis=0)
    
    r2 = ro2 + zsm**2
    r = np.sqrt(r2)
    r_safe = np.where(r < 1e-9, 1e-9, r)
    c = zsm / r_safe
    s = ro / r_safe
    s_safe = np.where(s < 1e-9, 1e-9, s)
    ch = np.sqrt(0.5 * (1 + c))
    sh = np.sqrt(0.5 * (1 - c))
    ch_safe = np.where(ch < 1e-9, 1e-9, ch)
    sh_safe = np.where(sh < 1e-9, 1e-9, sh)
    tnh = sh / ch_safe
    cnh = ch_safe / sh_safe
    
    # Conical harmonics (indices 0-4)
    for m in range(5):
        m1 = m + 1
        tnhm = tnh**m1
        cnhm = cnh**m1
        bt = m1 * cf[m] / (r_safe * s_safe) * (tnhm + cnhm)
        
        if m == 0:
            bf = 0.0
        else:
            tnhm_prev = tnh**m
            cnhm_prev = cnh**m
            bf = -0.5 * m1 * sf[m] / r_safe * (tnhm_prev / ch_safe**2 - cnhm_prev / sh_safe**2)
        
        bxsm = bt * c * cf0 - bf * sf0
        bysm = bt * c * sf0 + bf * cf0
        bzsm = -bt * s
        
        bx += c2[m] * (bxsm * cps + bzsm * sps)
        by += c2[m] * bysm
        bz += c2[m] * (bzsm * cps - bxsm * sps)
    
    # Now process the dipole terms (indices 5-31 and 32-58)
    xsm = x * cps - z * sps
    zsm = z * cps + x * sps
    
    # Indices 5-31: First set of dipoles
    for i in range(9):
        xd = xx2[i] * (scalein if i in [2, 4, 5] else scaleout)
        yd = yy2[i] * (scalein if i in [2, 4, 5] else scaleout)
        zd = zz2[i]
        
        # Four symmetric dipoles
        x1, y1, z1 = xsm - xd, y - yd, zsm - zd
        x2, y2, z2 = xsm - xd, y + yd, zsm - zd
        x3, y3, z3 = xsm - xd, y - yd, zsm + zd
        x4, y4, z4 = xsm - xd, y + yd, zsm + zd
        
        # Get dipole fields and their derivatives
        bx1x, by1x, bz1x = dipxyz_vectorized(x1, y1, z1, 0)
        bx2x, by2x, bz2x = dipxyz_vectorized(x2, y2, z2, 0)
        bx3x, by3x, bz3x = dipxyz_vectorized(x3, y3, z3, 0)
        bx4x, by4x, bz4x = dipxyz_vectorized(x4, y4, z4, 0)
        
        bx1y, by1y, bz1y = dipxyz_vectorized(x1, y1, z1, 1)
        bx2y, by2y, bz2y = dipxyz_vectorized(x2, y2, z2, 1)
        bx3y, by3y, bz3y = dipxyz_vectorized(x3, y3, z3, 1)
        bx4y, by4y, bz4y = dipxyz_vectorized(x4, y4, z4, 1)
        
        bx1z, by1z, bz1z = dipxyz_vectorized(x1, y1, z1, 2)
        bx2z, by2z, bz2z = dipxyz_vectorized(x2, y2, z2, 2)
        bx3z, by3z, bz3z = dipxyz_vectorized(x3, y3, z3, 2)
        bx4z, by4z, bz4z = dipxyz_vectorized(x4, y4, z4, 2)
        
        # X-derivative terms
        ix = i * 3 + 5
        bxsm = (bx1x + bx2x - bx3x - bx4x)
        bysm = (by1x + by2x - by3x - by4x)
        bzsm = (bz1x + bz2x - bz3x - bz4x)
        bx += c2[ix] * (bxsm * cps + bzsm * sps)
        by += c2[ix] * bysm
        bz += c2[ix] * (bzsm * cps - bxsm * sps)
        
        # Y-derivative terms
        iy = ix + 1
        bxsm = (bx1y - bx2y - bx3y + bx4y)
        bysm = (by1y - by2y - by3y + by4y)
        bzsm = (bz1y - bz2y - bz3y + bz4y)
        bx += c2[iy] * (bxsm * cps + bzsm * sps)
        by += c2[iy] * bysm
        bz += c2[iy] * (bzsm * cps - bxsm * sps)
        
        # Z-derivative terms
        iz = iy + 1
        bxsm = (bx1z + bx2z + bx3z + bx4z)
        bysm = (by1z + by2z + by3z + by4z)
        bzsm = (bz1z + bz2z + bz3z + bz4z)
        bx += c2[iz] * (bxsm * cps + bzsm * sps)
        by += c2[iz] * bysm
        bz += c2[iz] * (bzsm * cps - bxsm * sps)
        
        # Indices 32-58: Second set with sps factor
        ix2 = ix + 27
        iy2 = iy + 27
        iz2 = iz + 27
        
        # X-derivative with sps
        bxsm = (bx1x + bx2x + bx3x + bx4x)
        bysm = (by1x + by2x + by3x + by4x)
        bzsm = (bz1x + bz2x + bz3x + bz4x)
        bx += c2[ix2] * sps * (bxsm * cps + bzsm * sps)
        by += c2[ix2] * sps * bysm
        bz += c2[ix2] * sps * (bzsm * cps - bxsm * sps)
        
        # Y-derivative with sps
        bxsm = (bx1y - bx2y + bx3y - bx4y)
        bysm = (by1y - by2y + by3y - by4y)
        bzsm = (bz1y - bz2y + bz3y - bz4y)
        bx += c2[iy2] * sps * (bxsm * cps + bzsm * sps)
        by += c2[iy2] * sps * bysm
        bz += c2[iy2] * sps * (bzsm * cps - bxsm * sps)
        
        # Z-derivative with sps
        bxsm = (bx1z + bx2z - bx3z - bx4z)
        bysm = (by1z + by2z - by3z - by4z)
        bzsm = (bz1z + bz2z - bz3z - bz4z)
        bx += c2[iz2] * sps * (bxsm * cps + bzsm * sps)
        by += c2[iz2] * sps * bysm
        bz += c2[iz2] * sps * (bzsm * cps - bxsm * sps)
    
    # Indices 59-68 and 69-78: Vertical dipoles
    for i in range(5):
        zd = zz2[i + 9]
        
        # Two symmetric dipoles
        x1, y1, z1 = xsm, y, zsm - zd
        x2, y2, z2 = xsm, y, zsm + zd
        
        # Get dipole fields
        bx1x, by1x, bz1x = dipxyz_vectorized(x1, y1, z1, 0)
        bx2x, by2x, bz2x = dipxyz_vectorized(x2, y2, z2, 0)
        
        bx1z, by1z, bz1z = dipxyz_vectorized(x1, y1, z1, 2)
        bx2z, by2z, bz2z = dipxyz_vectorized(x2, y2, z2, 2)
        
        # X-derivative terms (indices 59-68)
        ix = 59 + i * 2
        bxsm = (bx1x - bx2x)
        bysm = (by1x - by2x)
        bzsm = (bz1x - bz2x)
        bx += c2[ix] * (bxsm * cps + bzsm * sps)
        by += c2[ix] * bysm
        bz += c2[ix] * (bzsm * cps - bxsm * sps)
        
        # Z-derivative terms
        iz = ix + 1
        bxsm = (bx1z + bx2z)
        bysm = (by1z + by2z)
        bzsm = (bz1z + bz2z)
        bx += c2[iz] * (bxsm * cps + bzsm * sps)
        by += c2[iz] * bysm
        bz += c2[iz] * (bzsm * cps - bxsm * sps)
        
        # With sps factor (indices 69-78)
        ix2 = ix + 10
        iz2 = iz + 10
        
        # X-derivative with sps
        bxsm = (bx1x + bx2x)
        bysm = (by1x + by2x)
        bzsm = (bz1x + bz2x)
        bx += c2[ix2] * sps * (bxsm * cps + bzsm * sps)
        by += c2[ix2] * sps * bysm
        bz += c2[ix2] * sps * (bzsm * cps - bxsm * sps)
        
        # Z-derivative with sps
        bxsm = (bx1z - bx2z)
        bysm = (by1z - by2z)
        bzsm = (bz1z - bz2z)
        bx += c2[iz2] * sps * (bxsm * cps + bzsm * sps)
        by += c2[iz2] * sps * bysm
        bz += c2[iz2] * sps * (bzsm * cps - bxsm * sps)
    
    return bx, by, bz


def dipxyz_vectorized(x, y, z, mode):
    """Vectorized dipole field and derivatives."""
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    r2 = x**2 + y**2 + z**2
    r5_inv = np.power(r2 + 1e-15, -2.5)
    
    if mode == 0:  # X-derivative
        bxx = 30574.0 * r5_inv * (3 * x**2 - r2)
        byx = 30574.0 * r5_inv * (3 * x * y)
        bzx = 30574.0 * r5_inv * (3 * x * z)
        return bxx, byx, bzx
    elif mode == 1:  # Y-derivative
        bxy = 30574.0 * r5_inv * (3 * x * y)
        byy = 30574.0 * r5_inv * (3 * y**2 - r2)
        bzy = 30574.0 * r5_inv * (3 * y * z)
        return bxy, byy, bzy
    else:  # Z-derivative
        bxz = 30574.0 * r5_inv * (3 * x * z)
        byz = 30574.0 * r5_inv * (3 * y * z)
        bzz = 30574.0 * r5_inv * (3 * z**2 - r2)
        return bxz, byz, bzz




def birk1shld_vectorized(ps, x, y, z):
    """Vectorized shielding field for Birkeland region 1."""
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    # Model coefficients
    a = np.array([
        1.174198045,-1.463820502,4.840161537,-3.674506864,82.18368896,
        -94.94071588,-4122.331796,4670.278676,-21.54975037,26.72661293,
        -72.81365728,44.09887902,40.08073706,-51.23563510,1955.348537,
        -1940.971550,794.0496433,-982.2441344,1889.837171,-558.9779727,
        -1260.543238,1260.063802,-293.5942373,344.7250789,-773.7002492,
        957.0094135,-1824.143669,520.7994379,1192.484774,-1192.184565,
        89.15537624,-98.52042999,-0.8168777675E-01,0.4255969908E-01,0.3155237661,
        -0.3841755213,2.494553332,-0.6571440817E-01,-2.765661310,0.4331001908,
        0.1099181537,-0.6154126980E-01,-0.3258649260,0.6698439193,-5.542735524,
        0.1604203535,5.854456934,-0.8323632049,3.732608869,-3.130002153,
        107.0972607,-32.28483411,-115.2389298,54.45064360,-0.5826853320,
        -3.582482231,-4.046544561,3.311978102,-104.0839563,30.26401293,
        97.29109008,-50.62370872,-296.3734955,127.7872523,5.303648988,
        10.40368955,69.65230348,466.5099509,1.645049286,3.825838190,
        11.66675599,558.9781177,1.826531343,2.066018073,25.40971369,
        990.2795225,2.319489258,4.555148484,9.691185703,591.8280358
    ])
    
    p1 = a[64:68]
    r1 = a[68:72]
    q1 = a[72:76]
    s1 = a[76:80]
    
    cps = np.cos(ps)
    sps = np.sin(ps)
    s3ps = 4 * cps**2 - 1
    
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    
    l = 0
    for m in range(2):
        for i in range(4):
            for k in range(4):
                for n in range(2):
                    if m == 0:
                        # First harmonic type
                        rp = 1.0 / p1[i]
                        rr = 1.0 / r1[k]
                        cypi = np.cos(y * rp)
                        sypi = np.sin(y * rp)
                        szrk = np.sin(z * rr)
                        czrk = np.cos(z * rr)
                        sqpr = np.sqrt(rp**2 + rr**2)
                        epr = np.exp(x * sqpr)
                        
                        hx_base = -sqpr * epr * cypi * szrk
                        hy_base = rp * epr * sypi * szrk
                        hz_base = -rr * epr * cypi * czrk
                        factor = cps if n == 1 else 1.0
                    else:
                        # Second harmonic type
                        rq = 1.0 / q1[i]
                        rs = 1.0 / s1[k]
                        cyqi = np.cos(y * rq)
                        syqi = np.sin(y * rq)
                        czsk = np.cos(z * rs)
                        szsk = np.sin(z * rs)
                        sqqs = np.sqrt(rq**2 + rs**2)
                        eqs = np.exp(x * sqqs)
                        
                        hx_base = -sps * sqqs * eqs * cyqi * czsk
                        hy_base = sps * rq * eqs * syqi * czsk
                        hz_base = sps * rs * eqs * cyqi * szsk
                        factor = s3ps if n == 1 else 1.0
                    
                    bx += a[l] * hx_base * factor
                    by += a[l] * hy_base * factor
                    bz += a[l] * hz_base * factor
                    l += 1
    
    return bx, by, bz


def circle_vectorized(x, y, z, radius):
    """Vectorized circular current loop field using same approximation as scalar version."""
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y) 
    z = np.atleast_1d(z)
    
    rho2 = x**2 + y**2
    rho = np.sqrt(rho2)
    r22 = z**2 + (rho + radius)**2
    r2 = np.sqrt(r22)
    r12 = r22 - 4*rho*radius
    r32 = 0.5 * (r12 + r22)
    xk2 = 1 - r12/r22
    xk2s = 1 - xk2
    
    # Use same approximations as scalar version for elliptic integrals
    dl = np.log(1/xk2s)
    k = (1.38629436112 + xk2s*(0.09666344259 + xk2s*(0.03590092383 + 
         xk2s*(0.03742563713 + xk2s*0.01451196212))) +
         dl*(0.5 + xk2s*(0.12498593597 + xk2s*(0.06880248576 + 
         xk2s*(0.03328355346 + xk2s*0.00441787012)))))
    e = (1 + xk2s*(0.44325141463 + xk2s*(0.0626060122 + 
         xk2s*(0.04757383546 + xk2s*0.01736506451))) +
         dl*xk2s*(0.2499836831 + xk2s*(0.09200180037 + 
         xk2s*(0.04069697526 + xk2s*0.00526449639))))
    
    # Field components
    brho = np.where(rho > 1e-6,
                    z/(rho2*r2)*(r32/r12*e - k),
                    np.pi*radius/r2*(radius-rho)/r12*z/(r32-rho2))
    
    bx = brho * x
    by = brho * y
    bz = (k - e*(r32 - 2*radius*radius)/r12) / r2
    
    return bx, by, bz


def birk2shl_vectorized(x, y, z, ps):
    """Vectorized shielding for Birkeland region 2."""
    # Model coefficients
    a = np.array([
        -111.6371348, 124.5402702, 110.3735178, -122.0095905, 111.9448247, -129.1957743,
        -110.7586562, 126.5649012, -0.7865034384, -0.2483462721, 0.8026023894, 0.2531397188,
        10.72890902, 0.8483902118, -10.96884315, -0.8583297219, 13.85650567, 14.90554500,
        10.21914434, 10.09021632, 6.340382460, 14.40432686, 12.71023437, 12.83966657
    ])
    
    p = a[16:18]
    r = a[18:20]
    q = a[20:22]
    s = a[22:24]
    
    cps = np.cos(ps)
    sps = np.sin(ps)
    s3ps = 4 * cps**2 - 1
    
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    
    l = 0
    for m in range(2):
        for i in range(2):
            for k in range(2):
                for n in range(2):
                    if m == 0:
                        # First harmonic type
                        rp = 1.0 / p[i]
                        rr = 1.0 / r[k]
                        sqpr = np.sqrt(rp**2 + rr**2)
                        epr = np.exp(x * sqpr)
                        cypi = np.cos(y * rp)
                        sypi = np.sin(y * rp)
                        szrk = np.sin(z * rr)
                        czrk = np.cos(z * rr)
                        
                        hx_base = -sqpr * epr * cypi * szrk
                        hy_base = rp * epr * sypi * szrk
                        hz_base = -rr * epr * cypi * czrk
                        factor = cps if n == 1 else 1.0
                    else:
                        # Second harmonic type
                        rq = 1.0 / q[i]
                        rs = 1.0 / s[k]
                        sqqs = np.sqrt(rq**2 + rs**2)
                        eqs = np.exp(x * sqqs)
                        cyqi = np.cos(y * rq)
                        syqi = np.sin(y * rq)
                        czsk = np.cos(z * rs)
                        szsk = np.sin(z * rs)
                        
                        hx_base = -sps * sqqs * eqs * cyqi * czsk
                        hy_base = sps * rq * eqs * syqi * czsk
                        hz_base = sps * rs * eqs * cyqi * szsk
                        factor = s3ps if n == 1 else 1.0
                    
                    bx += a[l] * hx_base * factor
                    by += a[l] * hy_base * factor
                    bz += a[l] * hz_base * factor
                    l += 1
    
    return bx, by, bz


def r2_birk_vectorized(x, y, z, ps):
    """Vectorized R2 Birkeland current system."""
    delarg = 0.03
    delarg1 = 0.015
    
    cps = np.cos(ps)
    sps = np.sin(ps)
    
    # Transform to SM coordinates
    xsm = x * cps - z * sps
    zsm = z * cps + x * sps
    
    # Calculate xksi parameter
    xks = xksi_vectorized(xsm, y, zsm)
    
    # Calculate fields for all regions
    bout_x, bout_y, bout_z = r2outer_vectorized(xsm, y, zsm)
    bsht_x, bsht_y, bsht_z = r2sheet_vectorized(xsm, y, zsm)
    binn_x, binn_y, binn_z = r2inner_vectorized(xsm, y, zsm)
    
    # Determine regions and interpolate
    conditions = [
        xks < -(delarg + delarg1),
        (xks >= -(delarg + delarg1)) & (xks < -delarg + delarg1),
        (xks >= -delarg + delarg1) & (xks < delarg - delarg1),
        (xks >= delarg - delarg1) & (xks < delarg + delarg1)
    ]
    
    # Region 1: outer
    bxsm1 = bout_x * -0.02
    by1 = bout_y * -0.02
    bzsm1 = bout_z * -0.02
    
    # Region 2: transition outer-sheet
    tksi2 = tksi_vectorized(xks, -delarg, delarg1)
    f2 = -0.02 * tksi2
    f1 = -0.02 - f2
    bxsm2 = bout_x * f1 + bsht_x * f2
    by2 = bout_y * f1 + bsht_y * f2
    bzsm2 = bout_z * f1 + bsht_z * f2
    
    # Region 3: sheet
    bxsm3 = bsht_x * -0.02
    by3 = bsht_y * -0.02
    bzsm3 = bsht_z * -0.02
    
    # Region 4: transition sheet-inner
    tksi3 = tksi_vectorized(xks, delarg, delarg1)
    f1_2 = -0.02 * tksi3
    f2_2 = -0.02 - f1_2
    bxsm4 = binn_x * f1_2 + bsht_x * f2_2
    by4 = binn_y * f1_2 + bsht_y * f2_2
    bzsm4 = binn_z * f1_2 + bsht_z * f2_2
    
    # Default: inner
    bxsm5 = binn_x * -0.02
    by5 = binn_y * -0.02
    bzsm5 = binn_z * -0.02
    
    # Select based on conditions
    choices_x = [bxsm1, bxsm2, bxsm3, bxsm4]
    choices_y = [by1, by2, by3, by4]
    choices_z = [bzsm1, bzsm2, bzsm3, bzsm4]
    
    bxsm = np.select(conditions, choices_x, default=bxsm5)
    by = np.select(conditions, choices_y, default=by5)
    bzsm = np.select(conditions, choices_z, default=bzsm5)
    
    # Transform back to GSM
    bx = bxsm * cps + bzsm * sps
    bz = bzsm * cps - bxsm * sps
    
    return bx, by, bz


def xksi_vectorized(x, y, z):
    """Calculate xksi parameter for R2 current system."""
    # Model parameters
    a = np.array([0.305662, -0.383593, 0.2677733, -0.097656, -0.636034, 
                  -0.359862, 0.424706, -0.126366, 0.292578, 1.21563, 7.50937])
    
    tnoon = 0.3665191
    dteta = 0.09599309
    r0 = a[9]
    dr = a[10]
    
    r2 = x**2 + y**2 + z**2
    r = np.sqrt(r2)
    r_safe = np.where(r < 1e-9, 1e-9, r)
    
    xr = x / r_safe
    yr = y / r_safe
    zr = z / r_safe
    
    # Calculate pr
    pr = np.sqrt((r - r0)**2 + dr**2) - dr
    pr = np.where(r < r0, 0.0, pr)
    
    # Deformed coordinates
    f = x + pr * (a[0] + a[1]*xr + a[2]*xr**2 + a[3]*yr**2 + a[4]*zr**2)
    g = y + pr * (a[5]*yr + a[6]*xr*yr)
    h = z + pr * (a[7]*zr + a[8]*xr*zr)
    
    fgh2 = f**2 + g**2 + h**2
    fgh_safe = np.where(fgh2 < 1e-9, 1e-9, np.sqrt(fgh2))
    
    fchsg2 = f**2 + g**2
    sqfchsg2 = np.sqrt(fchsg2)
    sqfchsg2_safe = np.where(sqfchsg2 < 1e-9, 1e-9, sqfchsg2)
    
    alpha = fchsg2 / (fgh_safe**3)
    theta = tnoon + 0.5 * dteta * (1 - f / sqfchsg2_safe)
    phi = np.sin(theta)**2
    
    return np.where(fchsg2 < 1e-5, -1.0, alpha - phi)


def tksi_vectorized(xksi, xks0, dxksi):
    """Transition function for smooth region boundaries."""
    tdz3 = 2.0 * dxksi**3
    br3_1 = (xksi - xks0 + dxksi)**3
    br3_2 = (xksi - xks0 - dxksi)**3
    
    conditions = [
        xksi - xks0 < -dxksi,
        xksi < xks0,
        xksi - xks0 < dxksi,
    ]
    
    choices = [
        0.0,
        1.5 * br3_1 / (tdz3 + br3_1),
        1.0 + 1.5 * br3_2 / (tdz3 - br3_2),
    ]
    
    return np.select(conditions, choices, default=1.0)


def r2outer_vectorized(x, y, z):
    """R2 outer region field."""
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    # Model coefficients
    pl = np.array([-34.105, -2.00019, 628.639, 73.4847, 12.5162])
    pn = np.array([0.55, 0.694, 0.0031, 1.55, 2.8, 0.1375, -0.7, 0.2, 0.9625,
                   -2.994, 2.925, -1.775, 4.3, -0.275, 2.7, 0.4312, 1.55])
    
    # Three pairs of crossed loops
    dbx1, dby1, dbz1 = crosslp_vectorized(x, y, z, pn[0], pn[1], pn[2])
    dbx2, dby2, dbz2 = crosslp_vectorized(x, y, z, pn[3], pn[4], pn[5])
    dbx3, dby3, dbz3 = crosslp_vectorized(x, y, z, pn[6], pn[7], pn[8])
    
    # Equatorial loop on the nightside
    dbx4, dby4, dbz4 = circle_vectorized(x - pn[9], y, z, pn[10])
    
    # 4-loop system on the nightside
    dbx5, dby5, dbz5 = loops4_vectorized(x, y, z, pn[11], pn[12], pn[13], pn[14], pn[15], pn[16])
    
    # Compute field components
    bx = pl[0]*dbx1 + pl[1]*dbx2 + pl[2]*dbx3 + pl[3]*dbx4 + pl[4]*dbx5
    by = pl[0]*dby1 + pl[1]*dby2 + pl[2]*dby3 + pl[3]*dby4 + pl[4]*dby5
    bz = pl[0]*dbz1 + pl[1]*dbz2 + pl[2]*dbz3 + pl[3]*dbz4 + pl[4]*dbz5
    
    return bx, by, bz


def r2sheet_vectorized(x, y, z):
    """R2 sheet region field."""
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    # Model parameters
    pnonx = np.array([-19.0969, -9.28828, -0.129687, 5.58594, 22.5055, 0.0483750, 0.0396953, 0.0579023])
    pnony = np.array([-13.6750, -6.70625, 2.31875, 11.4062, 20.4562, 0.0478750, 0.0363750, 0.0567500])
    pnonz = np.array([-16.7125, -16.4625, -0.1625, 5.1, 23.7125, 0.0355625, 0.0318750, 0.0538750])
    
    # Large coefficient arrays (80 elements each)
    a_coef = np.array([
        8.07190,-7.39582,-7.62341,0.684671,-13.5672,11.6681,13.1154,-0.890217,7.78726,-5.38346,
        -8.08738,0.609385,-2.70410, 3.53741,3.15549,-1.11069,-8.47555,0.278122,2.73514,4.55625,
        13.1134,1.15848,-3.52648,-8.24698,-6.85710,-2.81369,2.03795,4.64383,2.49309,-1.22041,
        -1.67432,-0.422526,-5.39796,7.10326,5.53730,-13.1918,4.67853,-7.60329,-2.53066,7.76338,
        5.60165,5.34816,-4.56441,7.05976,-2.62723,-0.529078,1.42019,-2.93919,55.6338,-1.55181,
        39.8311,-80.6561,-46.9655,32.8925,-6.32296,19.7841,124.731,10.4347,-30.7581,102.680,
        -47.4037,-3.31278,9.37141,-50.0268,-533.319,110.426,1000.20,-1051.40,1619.48,589.855,
        -1462.73,1087.10,-1994.73,-1654.12,1263.33,-260.210,1424.84,1255.71,-956.733,219.946
    ])
    
    b_coef = np.array([
        -9.08427,10.6777,10.3288,-0.969987,6.45257,-8.42508,-7.97464,1.41996,-1.92490,3.93575,
        2.83283,-1.48621,0.244033,-0.757941,-0.386557,0.344566,9.56674,-2.5365,-3.32916,-5.86712,
        -6.19625,1.83879,2.52772,4.34417,1.87268,-2.13213,-1.69134,-.176379,-.261359,.566419,
        0.3138,-0.134699,-3.83086,-8.4154,4.77005,-9.31479,37.5715,19.3992,-17.9582,36.4604,
        -14.9993,-3.1442,6.17409,-15.5519,2.28621,-0.891549e-2,-.462912,2.47314,41.7555,208.614,
        -45.7861,-77.8687,239.357,-67.9226,66.8743,238.534,-112.136,16.2069,-40.4706,-134.328,
        21.56,-0.201725,2.21,32.5855,-108.217,-1005.98,585.753,323.668,-817.056,235.750,
        -560.965,-576.892,684.193,85.0275,168.394,477.776,-289.253,-123.216,75.6501,-178.605
    ])
    
    c_coef = np.array([
        1167.61,-917.782,-1253.2,-274.128,-1538.75,1257.62,1745.07,113.479,393.326,-426.858,
        -641.1,190.833,-29.9435,-1.04881,117.125,-25.7663,-1168.16,910.247,1239.31,289.515,
        1540.56,-1248.29,-1727.61,-131.785,-394.577,426.163,637.422,-187.965,30.0348,0.221898,
        -116.68,26.0291,12.6804,4.84091,1.18166,-2.75946,-17.9822,-6.80357,-1.47134,3.02266,
        4.79648,0.665255,-0.256229,-0.857282e-1,-0.588997,0.634812e-1,0.164303,-0.15285,22.2524,-22.4376,
        -3.85595,6.07625,-105.959,-41.6698,0.378615,1.55958,44.3981,18.8521,3.19466,5.89142,
        -8.63227,-2.36418,-1.027,-2.31515,1035.38,2040.66,-131.881,-744.533,-3274.93,-4845.61,
        482.438,1567.43,1354.02,2040.47,-151.653,-845.012,-111.723,-265.343,-26.1171,216.632
    ])
    
    # Calculate xksi parameter
    xks = xksi_vectorized(x, y, z)
    
    # Calculate transition functions
    t1x = xks / np.sqrt(xks**2 + pnonx[5]**2)
    t2x = pnonx[6]**3 / np.sqrt(xks**2 + pnonx[6]**2)**3
    t3x = xks / np.sqrt(xks**2 + pnonx[7]**2)**5 * 3.493856 * pnonx[7]**4
    
    t1y = xks / np.sqrt(xks**2 + pnony[5]**2)
    t2y = pnony[6]**3 / np.sqrt(xks**2 + pnony[6]**2)**3
    t3y = xks / np.sqrt(xks**2 + pnony[7]**2)**5 * 3.493856 * pnony[7]**4
    
    t1z = xks / np.sqrt(xks**2 + pnonz[5]**2)
    t2z = pnonz[6]**3 / np.sqrt(xks**2 + pnonz[6]**2)**3
    t3z = xks / np.sqrt(xks**2 + pnonz[7]**2)**5 * 3.493856 * pnonz[7]**4
    
    # Calculate geometric factors
    rho2 = x**2 + y**2
    r = np.sqrt(rho2 + z**2)
    rho = np.sqrt(rho2)
    
    # Safe division
    rho_safe = np.where(rho < 1e-9, 1e-9, rho)
    r_safe = np.where(r < 1e-9, 1e-9, r)
    
    c1p = x / rho_safe
    s1p = y / rho_safe
    s2p = 2 * s1p * c1p
    c2p = c1p**2 - s1p**2
    s3p = s2p * c1p + c2p * s1p
    c3p = c2p * c1p - s2p * s1p
    s4p = s3p * c1p + c3p * s1p
    ct = z / r_safe
    st = rho / r_safe
    
    # Calculate exponential factors
    s1 = fexp_vectorized(ct, pnonx[0])
    s2 = fexp_vectorized(ct, pnonx[1])
    s3 = fexp_vectorized(ct, pnonx[2])
    s4 = fexp_vectorized(ct, pnonx[3])
    s5 = fexp_vectorized(ct, pnonx[4])
    
    s1y = fexp_vectorized(ct, pnony[0])
    s2y = fexp_vectorized(ct, pnony[1])
    s3y = fexp_vectorized(ct, pnony[2])
    s4y = fexp_vectorized(ct, pnony[3])
    s5y = fexp_vectorized(ct, pnony[4])
    
    s1z = fexp1_vectorized(ct, pnonz[0])
    s2z = fexp1_vectorized(ct, pnonz[1])
    s3z = fexp1_vectorized(ct, pnonz[2])
    s4z = fexp1_vectorized(ct, pnonz[3])
    s5z = fexp1_vectorized(ct, pnonz[4])
    
    # Calculate field components - full expansion
    bx = (s1 * ((a_coef[0] + a_coef[1]*t1x + a_coef[2]*t2x + a_coef[3]*t3x) + 
                c1p * (a_coef[4] + a_coef[5]*t1x + a_coef[6]*t2x + a_coef[7]*t3x) +
                c2p * (a_coef[8] + a_coef[9]*t1x + a_coef[10]*t2x + a_coef[11]*t3x) +
                c3p * (a_coef[12] + a_coef[13]*t1x + a_coef[14]*t2x + a_coef[15]*t3x)) +
          s2 * ((a_coef[16] + a_coef[17]*t1x + a_coef[18]*t2x + a_coef[19]*t3x) +
                c1p * (a_coef[20] + a_coef[21]*t1x + a_coef[22]*t2x + a_coef[23]*t3x) +
                c2p * (a_coef[24] + a_coef[25]*t1x + a_coef[26]*t2x + a_coef[27]*t3x) +
                c3p * (a_coef[28] + a_coef[29]*t1x + a_coef[30]*t2x + a_coef[31]*t3x)) +
          s3 * ((a_coef[32] + a_coef[33]*t1x + a_coef[34]*t2x + a_coef[35]*t3x) +
                c1p * (a_coef[36] + a_coef[37]*t1x + a_coef[38]*t2x + a_coef[39]*t3x) +
                c2p * (a_coef[40] + a_coef[41]*t1x + a_coef[42]*t2x + a_coef[43]*t3x) +
                c3p * (a_coef[44] + a_coef[45]*t1x + a_coef[46]*t2x + a_coef[47]*t3x)) +
          s4 * ((a_coef[48] + a_coef[49]*t1x + a_coef[50]*t2x + a_coef[51]*t3x) +
                c1p * (a_coef[52] + a_coef[53]*t1x + a_coef[54]*t2x + a_coef[55]*t3x) +
                c2p * (a_coef[56] + a_coef[57]*t1x + a_coef[58]*t2x + a_coef[59]*t3x) +
                c3p * (a_coef[60] + a_coef[61]*t1x + a_coef[62]*t2x + a_coef[63]*t3x)) +
          s5 * ((a_coef[64] + a_coef[65]*t1x + a_coef[66]*t2x + a_coef[67]*t3x) +
                c1p * (a_coef[68] + a_coef[69]*t1x + a_coef[70]*t2x + a_coef[71]*t3x) +
                c2p * (a_coef[72] + a_coef[73]*t1x + a_coef[74]*t2x + a_coef[75]*t3x) +
                c3p * (a_coef[76] + a_coef[77]*t1x + a_coef[78]*t2x + a_coef[79]*t3x)))
    
    by = (s1y * (s1p * (b_coef[0] + b_coef[1]*t1y + b_coef[2]*t2y + b_coef[3]*t3y) +
                 s2p * (b_coef[4] + b_coef[5]*t1y + b_coef[6]*t2y + b_coef[7]*t3y) +
                 s3p * (b_coef[8] + b_coef[9]*t1y + b_coef[10]*t2y + b_coef[11]*t3y) +
                 s4p * (b_coef[12] + b_coef[13]*t1y + b_coef[14]*t2y + b_coef[15]*t3y)) +
          s2y * (s1p * (b_coef[16] + b_coef[17]*t1y + b_coef[18]*t2y + b_coef[19]*t3y) +
                 s2p * (b_coef[20] + b_coef[21]*t1y + b_coef[22]*t2y + b_coef[23]*t3y) +
                 s3p * (b_coef[24] + b_coef[25]*t1y + b_coef[26]*t2y + b_coef[27]*t3y) +
                 s4p * (b_coef[28] + b_coef[29]*t1y + b_coef[30]*t2y + b_coef[31]*t3y)) +
          s3y * (s1p * (b_coef[32] + b_coef[33]*t1y + b_coef[34]*t2y + b_coef[35]*t3y) +
                 s2p * (b_coef[36] + b_coef[37]*t1y + b_coef[38]*t2y + b_coef[39]*t3y) +
                 s3p * (b_coef[40] + b_coef[41]*t1y + b_coef[42]*t2y + b_coef[43]*t3y) +
                 s4p * (b_coef[44] + b_coef[45]*t1y + b_coef[46]*t2y + b_coef[47]*t3y)) +
          s4y * (s1p * (b_coef[48] + b_coef[49]*t1y + b_coef[50]*t2y + b_coef[51]*t3y) +
                 s2p * (b_coef[52] + b_coef[53]*t1y + b_coef[54]*t2y + b_coef[55]*t3y) +
                 s3p * (b_coef[56] + b_coef[57]*t1y + b_coef[58]*t2y + b_coef[59]*t3y) +
                 s4p * (b_coef[60] + b_coef[61]*t1y + b_coef[62]*t2y + b_coef[63]*t3y)) +
          s5y * (s1p * (b_coef[64] + b_coef[65]*t1y + b_coef[66]*t2y + b_coef[67]*t3y) +
                 s2p * (b_coef[68] + b_coef[69]*t1y + b_coef[70]*t2y + b_coef[71]*t3y) +
                 s3p * (b_coef[72] + b_coef[73]*t1y + b_coef[74]*t2y + b_coef[75]*t3y) +
                 s4p * (b_coef[76] + b_coef[77]*t1y + b_coef[78]*t2y + b_coef[79]*t3y)))
    
    bz = (s1z * ((c_coef[0] + c_coef[1]*t1z + c_coef[2]*t2z + c_coef[3]*t3z) +
                 c1p * (c_coef[4] + c_coef[5]*t1z + c_coef[6]*t2z + c_coef[7]*t3z) +
                 c2p * (c_coef[8] + c_coef[9]*t1z + c_coef[10]*t2z + c_coef[11]*t3z) +
                 c3p * (c_coef[12] + c_coef[13]*t1z + c_coef[14]*t2z + c_coef[15]*t3z)) +
          s2z * ((c_coef[16] + c_coef[17]*t1z + c_coef[18]*t2z + c_coef[19]*t3z) +
                 c1p * (c_coef[20] + c_coef[21]*t1z + c_coef[22]*t2z + c_coef[23]*t3z) +
                 c2p * (c_coef[24] + c_coef[25]*t1z + c_coef[26]*t2z + c_coef[27]*t3z) +
                 c3p * (c_coef[28] + c_coef[29]*t1z + c_coef[30]*t2z + c_coef[31]*t3z)) +
          s3z * ((c_coef[32] + c_coef[33]*t1z + c_coef[34]*t2z + c_coef[35]*t3z) +
                 c1p * (c_coef[36] + c_coef[37]*t1z + c_coef[38]*t2z + c_coef[39]*t3z) +
                 c2p * (c_coef[40] + c_coef[41]*t1z + c_coef[42]*t2z + c_coef[43]*t3z) +
                 c3p * (c_coef[44] + c_coef[45]*t1z + c_coef[46]*t2z + c_coef[47]*t3z)) +
          s4z * ((c_coef[48] + c_coef[49]*t1z + c_coef[50]*t2z + c_coef[51]*t3z) +
                 c1p * (c_coef[52] + c_coef[53]*t1z + c_coef[54]*t2z + c_coef[55]*t3z) +
                 c2p * (c_coef[56] + c_coef[57]*t1z + c_coef[58]*t2z + c_coef[59]*t3z) +
                 c3p * (c_coef[60] + c_coef[61]*t1z + c_coef[62]*t2z + c_coef[63]*t3z)) +
          s5z * ((c_coef[64] + c_coef[65]*t1z + c_coef[66]*t2z + c_coef[67]*t3z) +
                 c1p * (c_coef[68] + c_coef[69]*t1z + c_coef[70]*t2z + c_coef[71]*t3z) +
                 c2p * (c_coef[72] + c_coef[73]*t1z + c_coef[74]*t2z + c_coef[75]*t3z) +
                 c3p * (c_coef[76] + c_coef[77]*t1z + c_coef[78]*t2z + c_coef[79]*t3z)))
    
    return bx, by, bz


def r2inner_vectorized(x, y, z):
    """R2 inner region field."""
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    # Model coefficients
    pl = np.array([154.185, -2.12446, 0.601735e-01, -0.153954e-02, 0.355077e-04, 29.9996, 262.886, 99.9132])
    pn = np.array([-8.1902, 6.5239, 5.504, 7.7815, 0.8573, 3.0986, 0.0774, -0.038])
    
    # Conical harmonics
    cbx, cby, cbz = bconic_vectorized(x, y, z, 5)
    
    # 4-loop system
    dbx8, dby8, dbz8 = loops4_vectorized(x, y, z, pn[0], pn[1], pn[2], pn[3], pn[4], pn[5])
    
    # Dipolar distributions
    dbx6, dby6, dbz6 = dipdistr_vectorized(x - pn[6], y, z, 0)
    dbx7, dby7, dbz7 = dipdistr_vectorized(x - pn[7], y, z, 1)
    
    # Compute field components
    bx = (pl[0]*cbx[0] + pl[1]*cbx[1] + pl[2]*cbx[2] + pl[3]*cbx[3] + pl[4]*cbx[4] +
          pl[5]*dbx6 + pl[6]*dbx7 + pl[7]*dbx8)
    by = (pl[0]*cby[0] + pl[1]*cby[1] + pl[2]*cby[2] + pl[3]*cby[3] + pl[4]*cby[4] +
          pl[5]*dby6 + pl[6]*dby7 + pl[7]*dby8)
    bz = (pl[0]*cbz[0] + pl[1]*cbz[1] + pl[2]*cbz[2] + pl[3]*cbz[3] + pl[4]*cbz[4] +
          pl[5]*dbz6 + pl[6]*dbz7 + pl[7]*dbz8)
    
    return bx, by, bz


def crosslp_vectorized(x, y, z, xc, rl, al):
    """Vectorized crossed loop pair contribution.
    Two loops with common center, inclined by angle al to equatorial plane."""
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    cal = np.cos(al)
    sal = np.sin(al)
    
    # First loop (rotated by +al)
    y1 = y * cal - z * sal
    z1 = y * sal + z * cal
    bx1, by1, bz1 = circle_vectorized(x - xc, y1, z1, rl)
    
    # Second loop (rotated by -al)
    y2 = y * cal + z * sal
    z2 = -y * sal + z * cal
    bx2, by2, bz2 = circle_vectorized(x - xc, y2, z2, rl)
    
    # Rotate back
    by1_rot = by1 * cal + bz1 * sal
    bz1_rot = -by1 * sal + bz1 * cal
    
    by2_rot = by2 * cal - bz2 * sal
    bz2_rot = by2 * sal + bz2 * cal
    
    return bx1 + bx2, by1_rot + by2_rot, bz1_rot + bz2_rot


def loops4_vectorized(x, y, z, xc, yc, zc, r, theta, phi):
    """Vectorized system of 4 current loops."""
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    
    bx_total = np.zeros_like(x)
    by_total = np.zeros_like(y)
    bz_total = np.zeros_like(z)
    
    # 1st quadrant:
    xs = (x - xc) * cp + (y - yc) * sp
    yss = (y - yc) * cp - (x - xc) * sp
    zs = z - zc
    xss = xs * ct - zs * st
    zss = zs * ct + xs * st
    
    bxss, bys, bzss = circle_vectorized(xss, yss, zss, r)
    bxs = bxss * ct + bzss * st
    bz1 = bzss * ct - bxss * st
    bx1 = bxs * cp - bys * sp
    by1 = bxs * sp + bys * cp
    
    # 2nd quadrant:
    xs = (x - xc) * cp - (y + yc) * sp
    yss = (y + yc) * cp + (x - xc) * sp
    zs = z - zc
    xss = xs * ct - zs * st
    zss = zs * ct + xs * st
    
    bxss, bys, bzss = circle_vectorized(xss, yss, zss, r)
    bxs = bxss * ct + bzss * st
    bz2 = bzss * ct - bxss * st
    bx2 = bxs * cp + bys * sp
    by2 = -bxs * sp + bys * cp
    
    # 3rd quadrant:
    xs = -(x - xc) * cp + (y + yc) * sp
    yss = -(y + yc) * cp - (x - xc) * sp
    zs = z + zc
    xss = xs * ct - zs * st
    zss = zs * ct + xs * st
    
    bxss, bys, bzss = circle_vectorized(xss, yss, zss, r)
    bxs = bxss * ct + bzss * st
    bz3 = bzss * ct - bxss * st
    bx3 = -bxs * cp - bys * sp
    by3 = bxs * sp - bys * cp
    
    # 4th quadrant:
    xs = -(x - xc) * cp - (y - yc) * sp
    yss = -(y - yc) * cp + (x - xc) * sp
    zs = z + zc
    xss = xs * ct - zs * st
    zss = zs * ct + xs * st
    
    bxss, bys, bzss = circle_vectorized(xss, yss, zss, r)
    bxs = bxss * ct + bzss * st
    bz4 = bzss * ct - bxss * st
    bx4 = -bxs * cp + bys * sp
    by4 = -bxs * sp - bys * cp
    
    return bx1 + bx2 + bx3 + bx4, by1 + by2 + by3 + by4, bz1 + bz2 + bz3 + bz4


def bconic_vectorized(x, y, z, nmax):
    """Vectorized conical harmonics."""
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    ro2 = x**2 + y**2
    ro = np.sqrt(ro2)
    
    # Handle z-axis case (when ro = 0)
    on_z_axis = ro < 1e-9
    ro_safe = np.where(on_z_axis, 1e-9, ro)
    
    # For points on z-axis, set cf=1, sf=0 (arbitrary but consistent)
    cf = np.where(on_z_axis, 1.0, x / ro_safe)
    sf = np.where(on_z_axis, 0.0, y / ro_safe)
    
    r2 = ro2 + z**2
    r = np.sqrt(r2)
    r_safe = np.where(r < 1e-9, 1e-9, r)
    
    c = z / r_safe
    s = ro / r_safe
    ch = np.sqrt(0.5 * (1 + c))
    sh = np.sqrt(0.5 * (1 - c))
    
    # Safe division for tanh/coth and s
    ch_safe = np.where(ch < 1e-9, 1e-9, ch)
    sh_safe = np.where(sh < 1e-9, 1e-9, sh)
    s_safe = np.where(s < 1e-9, 1e-9, s)
    
    tnh = sh / ch_safe
    cnh = ch_safe / sh_safe
    
    # Initialize output arrays
    num_points = x.shape[0] if x.ndim > 0 else 1
    cbx = np.zeros((nmax, num_points))
    cby = np.zeros((nmax, num_points))
    cbz = np.zeros((nmax, num_points))
    
    cfm1 = np.ones_like(x)
    sfm1 = np.zeros_like(x)
    tnhm1 = np.ones_like(x)
    cnhm1 = np.ones_like(x)
    
    for m in range(nmax):
        m1 = m + 1
        
        # Update cos/sin multiples
        cfm = cfm1 * cf - sfm1 * sf
        sfm = cfm1 * sf + sfm1 * cf
        cfm1 = cfm
        sfm1 = sfm
        
        # Update hyperbolic functions
        tnhm = tnhm1 * tnh
        cnhm = cnhm1 * cnh
        
        # Calculate field components
        # When on z-axis (s=0), bt calculation would be 0/0 -> set to 0
        bt = np.where(on_z_axis, 0.0, m1 * cfm / (r_safe * s_safe) * (tnhm + cnhm))
        bf = -0.5 * m1 * sfm / r_safe * (tnhm1 / ch_safe**2 - cnhm1 / sh_safe**2)
        
        tnhm1 = tnhm
        cnhm1 = cnhm
        
        cbx[m] = bt * c * cf - bf * sf
        cby[m] = bt * c * sf + bf * cf
        cbz[m] = -bt * s
    
    return cbx, cby, cbz


def dipdistr_vectorized(x, y, z, mode):
    """Vectorized dipolar distribution."""
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    x2 = x**2
    rho2 = x2 + y**2
    r2 = rho2 + z**2
    r3 = r2 * np.sqrt(r2)
    
    # Safe division
    rho2_safe = np.where(rho2 < 1e-9, 1e-9, rho2)
    r3_safe = np.where(r3 < 1e-15, 1e-15, r3)
    
    if mode == 0:
        # Step function mode
        bx = z / rho2_safe**2 * (r2 * (y**2 - x2) - rho2 * x2) / r3_safe
        by = -x * y * z / rho2_safe**2 * (2 * r2 + rho2) / r3_safe
        bz = x / r3_safe
    else:
        # Linear variation mode
        bx = z / rho2_safe**2 * (y**2 - x2)
        by = -2 * x * y * z / rho2_safe**2
        bz = x / rho2_safe
    
    return bx, by, bz


def fexp_vectorized(s, a):
    """Vectorized fexp function."""
    # Ensure arrays for proper broadcasting
    s = np.atleast_1d(s)
    
    if np.isscalar(a):
        if a < 0:
            return np.sqrt(-2 * a * np.e) * s * np.exp(a * s**2)
        else:
            return s * np.exp(a * (s**2 - 1))
    else:
        return np.where(a < 0, 
                        np.sqrt(-2 * a * np.e) * s * np.exp(a * s**2),
                        s * np.exp(a * (s**2 - 1)))


def fexp1_vectorized(s, a):
    """Vectorized fexp1 function."""
    # Ensure arrays for proper broadcasting
    s = np.atleast_1d(s)
    
    if np.isscalar(a):
        if a <= 0:
            return np.exp(a * s**2)
        else:
            return np.exp(a * (s**2 - 1))
    else:
        return np.where(a <= 0, np.exp(a * s**2), np.exp(a * (s**2 - 1)))


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