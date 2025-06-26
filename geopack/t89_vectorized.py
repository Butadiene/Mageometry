"""
Vectorized implementation of the T89 magnetospheric magnetic field model.

This implementation follows the principles established in the T96 vectorization policy:
1. All functions accept NumPy arrays for x, y, z coordinates
2. Conditional logic uses np.where instead of if/else
3. Safe division using proper guards against division by zero
4. No global variables - all parameters passed explicitly
5. Proper array initialization with np.zeros_like()
6. Maintains scalar compatibility (scalar in → scalar out)

The vectorized version provides significant performance improvements
for processing multiple points simultaneously while maintaining
numerical accuracy.
"""

import numpy as np


def t89_vectorized(iopt, ps, x, y, z):
    """
    Vectorized version of the T89 magnetic field model.
    
    Computes GSM components of the magnetic field produced by extra-
    terrestrial current systems in the geomagnetosphere. The model is
    valid up to geocentric distances of 70 Re.
    
    Parameters
    ----------
    iopt : int
        Specifies the ground disturbance level:
        iopt= 1       2        3        4        5        6      7
        correspond to:
        kp=  0,0+  1-,1,1+  2-,2,2+  3-,3,3+  4-,4,4+  5-,5,5+  >=6-
    ps : float
        Geo-dipole tilt angle in radians
    x, y, z : array_like
        GSM coordinates in Re (1 Re = 6371.2 km)
        
    Returns
    -------
    bx, by, bz : ndarray or float
        Magnetic field components in GSM system (nT)
        Returns scalars if all inputs were scalars
    """
    # Track if all inputs were scalar
    scalar_input = np.isscalar(x) and np.isscalar(y) and np.isscalar(z)
    
    # Convert inputs to numpy arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    # Broadcast arrays to same shape
    x, y, z = np.broadcast_arrays(x, y, z)
    
    # Parameter array (30x7) for different Kp levels
    param = np.array([
        -116.53,-10719.,42.375,59.753,-11363.,1.7844,30.268,
        -0.35372E-01,-0.66832E-01,0.16456E-01,-1.3024,0.16529E-02,
        0.20293E-02,20.289,-0.25203E-01,224.91,-9234.8,22.788,7.8813,
        1.8362,-0.27228,8.8184,2.8714,14.468,32.177,0.01,0.0,
        7.0459,4.0,20.0,-55.553,-13198.,60.647,61.072,-16064.,
        2.2534,34.407,-0.38887E-01,-0.94571E-01,0.27154E-01,-1.3901,
        0.13460E-02,0.13238E-02,23.005,-0.30565E-01,55.047,-3875.7,
        20.178,7.9693,1.4575,0.89471,9.4039,3.5215,14.474,36.555,
        0.01,0.0,7.0787,4.0,20.0,-101.34,-13480.,111.35,12.386,-24699.,
        2.6459,38.948,-0.34080E-01,-0.12404,0.29702E-01,-1.4052,
        0.12103E-02,0.16381E-02,24.49,-0.37705E-01,-298.32,4400.9,18.692,
        7.9064,1.3047,2.4541,9.7012,7.1624,14.288,33.822,0.01,0.0,6.7442,
        4.0,20.0,-181.69,-12320.,173.79,-96.664,-39051.,3.2633,44.968,
        -0.46377E-01,-0.16686,0.048298,-1.5473,0.10277E-02,0.31632E-02,
        27.341,-0.50655E-01,-514.10,12482.,16.257,8.5834,1.0194,3.6148,
        8.6042,5.5057,13.778,32.373,0.01,0.0,7.3195,4.0,20.0,-436.54,
        -9001.0,323.66,-410.08,-50340.,3.9932,58.524,-0.38519E-01,
        -0.26822,0.74528E-01,-1.4268,-0.10985E-02,0.96613E-02,27.557,
        -0.56522E-01,-867.03,20652.,14.101,8.3501,0.72996,3.8149,9.2908,
         6.4674,13.729,28.353,0.01,0.0,7.4237,4.0,20.0,-707.77,-4471.9,
        432.81,-435.51,-60400.,4.6229,68.178,-0.88245E-01,-0.21002,
        0.11846,-2.6711,0.22305E-02,0.10910E-01,27.547,-0.54080E-01,
        -424.23,1100.2,13.954,7.5337,0.89714,3.7813,8.2945,5.174,14.213,
        25.237,0.01,0.0,7.0037,4.0,20.0,-1190.4,2749.9,742.56,-1110.3,
        -77193.,7.6727,102.05,-0.96015E-01,-0.74507,0.11214,-1.3614,
        0.15157E-02,0.22283E-01,23.164,-0.74146E-01,-2219.1,48253.,
        12.714,7.6777,0.57138,2.9633,9.3909,9.7263,11.123,21.558,0.01,
        0.0,4.4518,4.0,20.0])
    param = param.reshape((30, 7), order='F')
    
    # Select parameters for the given Kp level
    a = param[:, iopt-1]
    
    # Calculate external field
    bx, by, bz = extern_vectorized(a, x, y, z, ps)
    
    # Return scalars if input was scalar
    if scalar_input:
        return bx.item(), by.item(), bz.item()
    else:
        return bx, by, bz


def extern_vectorized(a, x, y, z, tilt):
    """
    Vectorized calculation of external magnetic field components.
    
    Parameters
    ----------
    a : array_like
        30-element array containing model parameters
    x, y, z : ndarray
        GSM coordinates
    tilt : float
        Dipole tilt angle in radians
        
    Returns
    -------
    fx, fy, fz : ndarray
        Magnetic field components
    """
    # Constants
    a02, xlw2, yn, rpi, rt = 25., 170., 30.0, 0.318309890, 30.
    xd, xld2 = 0., 40.
    sxc, xlwc2 = 4., 50.
    dxl = 20.
    
    # Extract parameters
    dyc = a[29]
    dyc2 = dyc**2
    dx = a[17]
    ha02 = 0.5 * a02
    rdx2m = -1. / dx**2
    rdx2 = -rdx2m
    rdyc2 = 1 / dyc2
    hlwc2m = -0.5 * xlwc2
    drdyc2 = -2 * rdyc2
    drdyc3 = 2 * rdyc2 * np.sqrt(rdyc2)
    hxlw2m = -0.5 * xlw2
    
    adr = a[18]  # Ring current radius
    d0 = a[19]   # Basic tail sheet half-thickness
    dd = a[20]   # Ring current thickening rate
    rc = a[21]   # Hinging distance
    g = a[22]    # Tail warping amplitude
    at = a[23]   # Tail current radius
    dt = d0
    p = a[24]    # Y-scale distance
    delt = a[25] # Y-direction thickening rate
    q = a[26]    # Q parameter
    sx = a[27]   # Sx parameter
    gam = a[28]  # Tail sheet thickening rate
    hxld2m = -0.5 * xld2
    dbldel = 2. * delt
    
    # Derived parameters
    w1 = -0.5 / dx
    w2 = w1 * 2.
    w4 = -1. / 3.
    w3 = w4 / dx
    w5 = -0.5
    w6 = -3.
    
    # Linear parameters
    ak1, ak2, ak3, ak4, ak5, ak6, ak7, ak8, ak9, ak10, ak11, ak12, ak13, ak14, ak15, ak16, ak17 = a[0:17]
    
    # Tilt-dependent quantities
    tlt2 = tilt**2
    sps = np.sin(tilt)
    cps = np.cos(tilt)
    
    # Coordinate transformations
    x2 = x * x
    y2 = y * y
    z2 = z * z
    tps = np.divide(sps, cps, out=np.zeros_like(sps), where=cps!=0)
    htp = tps * 0.5
    gsp = g * sps
    xsm = x * cps - z * sps
    zsm = x * sps + z * cps
    
    # Tail current sheet shape function
    xrc = xsm + rc
    xrc16 = xrc**2 + 16
    sxrc = np.sqrt(xrc16)
    y4 = y2 * y2
    y410 = y4 + 1e4
    sy4 = sps / y410
    gsy4 = g * sy4
    zs1 = htp * (xrc - sxrc)
    dzsx = -zs1 / sxrc
    zs = zs1 - gsy4 * y4
    d2zsgy = -sy4 / y410 * 4e4 * y2 * y
    dzsy = g * d2zsgy
    
    # Ring current contribution
    xsm2 = xsm**2
    dsqt = np.sqrt(xsm2 + a02)
    fa0 = 0.5 * (1 + xsm / dsqt)
    ddr = d0 + dd * fa0
    dfa0 = ha02 / dsqt**3
    zr = zsm - zs
    tr = np.sqrt(zr**2 + ddr**2)
    rtr = 1 / tr
    ro2 = xsm2 + y2
    adrt = adr + tr
    adrt2 = adrt**2
    fk = 1 / (adrt2 + ro2)
    dsfc = np.sqrt(fk)
    fc = fk**2 * dsfc
    facxy = 3 * adrt * fc * rtr
    xzr = xsm * zr
    yzr = y * zr
    dbxdp = facxy * xzr
    dbydp = facxy * yzr
    xzyz = xsm * dzsx + y * dzsy
    faq = zr * xzyz - ddr * dd * dfa0 * xsm
    dbzdp = fc * (2 * adrt2 - ro2) + facxy * faq
    bxr = ak5 * (dbxdp * cps + dbzdp * sps)
    byr = ak5 * dbydp
    bzr = ak5 * (dbzdp * cps - dbxdp * sps)
    
    # Tail current sheet contribution
    dely2 = delt * y2
    d = dt + dely2
    
    # Handle gamma-dependent thickness modification
    use_gamma = np.abs(gam) >= 1e-6
    if use_gamma:
        xxd = xsm - xd
        rqd = 1 / (xxd**2 + xld2)
        rqds = np.sqrt(rqd)
        h = 0.5 * (1 + xxd * rqds)
        hs = -hxld2m * rqd * rqds
        gamh = gam * h
        d = d + gamh
        xghs = xsm * gam * hs
        adsl = -d * xghs
    else:
        adsl = 0.
    
    d2 = d**2
    t = np.sqrt(zr**2 + d2)
    xsmx = xsm - sx
    rdsq2 = 1 / (xsmx**2 + xlw2)
    rdsq = np.sqrt(rdsq2)
    v = 0.5 * (1 - xsmx * rdsq)
    dvx = hxlw2m * rdsq * rdsq2
    om = np.sqrt(np.sqrt(xsm2 + 16) - xsm)
    oms = -om / (om * om + xsm) * 0.5
    rdy = 1 / (p + q * om)
    omsv = oms * v
    rdy2 = rdy**2
    fy = 1 / (1 + y2 * rdy2)
    w = v * fy
    yfy1 = 2 * fy * y2 * rdy2
    fypr = yfy1 * rdy
    fydy = fypr * fy
    dwx = dvx * fy + fydy * q * omsv
    ydwy = -v * yfy1 * fy
    ddy = dbldel * y
    att = at + t
    s1 = np.sqrt(att**2 + ro2)
    f5 = 1 / s1
    f7 = 1 / (s1 + att)
    f1 = f5 * f7
    f3 = f5**3
    f9 = att * f3
    fs = zr * xzyz - d * y * ddy + adsl
    xdwx = xsm * dwx + ydwy
    rtt = 1 / t
    wt = w * rtt
    brrz1 = wt * f1
    brrz2 = wt * f3
    dbxc1 = brrz1 * xzr
    dbxc2 = brrz2 * xzr
    dbyc1 = brrz1 * yzr
    dbyc2 = brrz2 * yzr
    wtfs = wt * fs
    dbzc1 = w * f5 + xdwx * f7 + wtfs * f1
    dbzc2 = w * f9 + xdwx * f1 + wtfs * f3
    
    # Tail current field
    bxt1 = ak1 * (dbxc1 * cps + dbzc1 * sps)
    bxt2 = ak2 * (dbxc2 * cps + dbzc2 * sps)
    bxt15 = ak16 * (dbxc1 * cps + dbzc1 * sps) * tlt2
    bxt16 = ak17 * (dbxc2 * cps + dbzc2 * sps) * tlt2
    byt1 = ak1 * dbyc1
    byt2 = ak2 * dbyc2
    byt15 = ak16 * dbyc1 * tlt2
    byt16 = ak17 * dbyc2 * tlt2
    bzt1 = ak1 * (dbzc1 * cps - dbxc1 * sps)
    bzt2 = ak2 * (dbzc2 * cps - dbxc2 * sps)
    bzt15 = ak16 * (dbzc1 * cps - dbxc1 * sps) * tlt2
    bzt16 = ak17 * (dbzc2 * cps - dbxc2 * sps) * tlt2
    
    bxt = bxt1 + bxt2 + bxt15 + bxt16
    byt = byt1 + byt2 + byt15 + byt16
    bzt = bzt1 + bzt2 + bzt15 + bzt16
    
    # Closure currents contribution
    zpl = z + rt
    zmn = z - rt
    rogsm2 = x2 + y2
    spl = np.sqrt(zpl**2 + rogsm2)
    smn = np.sqrt(zmn**2 + rogsm2)
    xsxc = x - sxc
    rqc2 = 1 / (xsxc**2 + xlwc2)
    rqc = np.sqrt(rqc2)
    fyc = 1 / (1 + y2 * rdyc2)
    wc = 0.5 * (1 - xsxc * rqc) * fyc
    dwcx = hlwc2m * rqc2 * rqc * fyc
    dwcy = drdyc2 * wc * fyc * y
    # Safe division to avoid divide by zero
    szrp = np.divide(1.0, spl + zpl, out=np.zeros_like(spl), where=(spl + zpl) != 0)
    szrm = np.divide(1.0, smn - zmn, out=np.zeros_like(smn), where=(smn - zmn) != 0)
    xywc = x * dwcx + y * dwcy
    wcsp = np.divide(wc, spl, out=np.zeros_like(wc), where=spl != 0)
    wcsm = np.divide(wc, smn, out=np.zeros_like(wc), where=smn != 0)
    fxyp = wcsp * szrp
    fxym = wcsm * szrm
    fxpl = x * fxyp
    fxmn = -x * fxym
    fypl = y * fxyp
    fymn = -y * fxym
    fzpl = wcsp + xywc * szrp
    fzmn = wcsm + xywc * szrm
    
    bxcl = ak3 * (fxpl + fxmn) + ak4 * (fxpl - fxmn) * sps
    bycl = ak3 * (fypl + fymn) + ak4 * (fypl - fymn) * sps
    bzcl = ak3 * (fzpl + fzmn) + ak4 * (fzpl - fzmn) * sps
    
    # Chapman-Ferraro field
    ex = np.exp(x / dx)
    ec = ex * cps
    es = ex * sps
    ecz = ec * z
    esz = es * z
    eszy2 = esz * y2
    eszz2 = esz * z2
    ecz2 = ecz * z
    esy = es * y
    
    # Combined parameters
    ak610 = ak6 * w1 + ak10 * w5
    ak711 = ak7 * w2 - ak11
    ak812 = ak8 * w2 + ak12 * w6
    ak913 = ak9 * w3 + ak13 * w4
    
    # Chapman-Ferraro field components
    sx1 = ak6 * ecz + ak7 * es + ak8 * esy * y + ak9 * esz * z
    sy1 = ak10 * ecz * y + ak11 * esy + ak12 * esy * y2 + ak13 * esy * z2
    sz1 = ak14 * ec + ak15 * ec * y2 + ak610 * ecz2 + ak711 * esz + ak812 * eszy2 + ak913 * eszz2
    
    # Total external field
    fx = bxt + bxcl + bxr + sx1
    fy = byt + bycl + byr + sy1
    fz = bzt + bzcl + bzr + sz1
    
    return fx, fy, fz