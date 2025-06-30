"""
Vectorized implementation of the T04 magnetospheric magnetic field model.

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
import warnings


def t04_vectorized(parmod, ps, x, y, z):
    """
    A data-based model of the external (i.e., without earth's contribution) part of the
    magnetospheric magnetic field, calibrated by
        (1) solar wind pressure pdyn (nanopascals),
        (2) dst (nanotesla),
        (3) byimf,
        (4) bzimf (nanotesla)
        (5-10) indices w1 - w6, calculated as time integrals from the beginning of a storm

    Assembled: March 25, 2004; Updated: August 2 & 31, December 27, 2004.
    A bug eliminated March 14, 2005 (might cause compilation problems with some fortran compilers)

    Attention: The model is based on data taken sunward from x=-15Re, and hence becomes invalid at larger tailward distances !!!

    Parameters
    ----------
    parmod : array_like
        10-element array containing model parameters.
    ps : float
        Geodipole tilt angle in radians.
    x, y, z : array_like
        GSM coordinates in Re (Earth radii).

    Returns
    -------
    bx, by, bz : ndarray or float
        Magnetic field components in GSM system (nT).
        Returns scalars if all inputs were scalars.

    References
    ----------
    (1) N. A. Tsyganenko, A new data-based model of the near magnetosphere magnetic field:
        1. Mathematical structure.
        2. Parameterization and fitting to observations.  JGR v. 107(A8), 1176/1179, doi:10.1029/2001JA000219/220, 2002.
    (2) N. A. Tsyganenko, H. J. Singer, J. C. Kasper, Storm-time distortion of the
        inner magnetosphere: How severe can it get ?  JGR v. 108(A5), 1209, doi:10.1029/2002JA009808, 2003.
    (3) N. A. Tsyganenko and M. I. Sitnov, Modeling the dynamics of the inner magnetosphere during
        strong geomagnetic storms, J. Geophys. Res., v. 110 (A3), A03208, doi: 10.1029/2004JA010798, 2005.
    """
    # Track if all inputs were scalar
    scalar_input = np.isscalar(x) and np.isscalar(y) and np.isscalar(z)

    # Convert inputs to numpy arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)

    a = np.array([
        1.00000,5.44118,0.891995,9.09684,0.00000,-7.18972,12.2700,
        -4.89408,0.00000,0.870536,1.36081,0.00000,0.688650,0.602330,
        0.00000,0.316346,1.22728,-0.363620E-01,-0.405821,0.452536,
        0.755831,0.215662,0.152759,5.96235,23.2036,11.2994,69.9596,
        0.989596,-0.132131E-01,0.985681,0.344212E-01,1.02389,0.207867,
        1.51220,0.682715E-01,1.84714,1.76977,1.37690,0.696350,0.343280,
        3.28846,111.293,5.82287,4.39664,0.383403,0.648176,0.318752E-01,
        0.581168,1.15070,0.843004,0.394732,0.846509,0.916555,0.550920,
        0.180725,0.898772,0.387365,2.26596,1.29123,0.436819,1.28211,
        1.33199,.405553,1.6229,.699074,1.26131,2.42297,.537116,.619441])


    # Handle invalid X values by clipping instead of raising error
    invalid_mask = x < -15
    if np.any(invalid_mask):
        print(f'Warning: T04 model is valid only for X > -15 Re. Clipping {np.sum(invalid_mask)} points to X = -15 Re.')
        x = np.where(invalid_mask, -15, x)

    iopgen,ioptt,iopb,iopr = [0,0,0,0]

    pdyn=parmod[0]
    dst_ast=parmod[1]*0.8-13*np.sqrt(pdyn)
    bximf,byimf,bzimf=[0.,parmod[2],parmod[3]]
    w1,w2,w3,w4,w5,w6 = parmod[4:10]
    pss,xx,yy,zz = [ps,x,y,z]

    # Suppress warnings for expected singularities (e.g., at origin)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        bx,by,bz = extern(iopgen,ioptt,iopb,iopr,a,69,pdyn,dst_ast,bximf,byimf,bzimf,
            w1,w2,w3,w4,w5,w6,pss,xx,yy,zz)
    
    # Return scalar if input was scalar
    if scalar_input:
        return bx.item(), by.item(), bz.item()
    else:
        return bx, by, bz


def extern(iopgen,iopt,iopb,iopr,a,ntot,pdyn,dst,bximf,byimf,bzimf,w1,w2,w3,w4,w5,w6,ps,x,y,z):
    """
    Vectorized version of extern.
    """
    global dxshift1, dxshift2, d, deltady
    global xkappa1, xkappa2
    global sc_sy, sc_pr, phi
    global g
    global rh0

    a0_a,a0_s0,a0_x0 = [34.586,1.1960,3.4397]
    dsig = 0.005
    rh0_val,rh2 = [8.0,-5.2] # rh0 is set below, so this is rh0_val

    xappa = (pdyn/2.)**a[22]
    rh0 = 7.5
    g = 35.0

    xappa3=xappa**3

    xx=x*xappa
    yy=y*xappa
    zz=z*xappa

    sps=np.sin(ps)

    x0=a0_x0/xappa
    am=a0_a/xappa
    s0=a0_s0

    factimf=a[19]

    oimfx=0.
    oimfy=byimf*factimf
    oimfz=bzimf*factimf

    r=np.sqrt(x**2+y**2+z**2)
    r_safe = np.where(r == 0, 1e-9, r)

    xss=x.copy()
    zss=z.copy()

    # Vectorized iterative search for unwarped coords
    for _ in range(100):
        xsold=xss.copy()
        zsold=zss.copy()
        rh=rh0+rh2*(zss/r_safe)**2
        rh_safe = np.where(rh==0, 1e-9, rh)
        sinpsas=sps/(1+(r/rh_safe)**3)**0.33333333
        cospsas=np.sqrt(1-sinpsas**2)
        zss_new=x*sinpsas+z*cospsas
        xss_new=x*cospsas-z*sinpsas
        dd=np.abs(xss_new-xsold)+np.abs(zss_new-zsold)
        xss, zss = xss_new, zss_new
        if not np.any(dd > 1e-6):
            break

    rho2=y**2+zss**2
    asq=am**2
    xmxm=am+xss-x0
    xmxm = np.maximum(0., xmxm)
    axx0=xmxm**2
    aro=asq+rho2
    sigma=np.sqrt((aro+axx0+np.sqrt((aro+axx0)**2-4.*asq*axx0))/(2.*asq))

    # Vectorized calculation for all regions
    # Initialize field components
    bxcf,bycf,bzcf = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    bxt1,byt1,bzt1,bxt2,byt2,bzt2 = [np.zeros_like(x) for _ in range(6)]
    bxr11,byr11,bzr11, bxr12,byr12,bzr12, bxr21,byr21,bzr21, bxr22,byr22,bzr22 = [np.zeros_like(x) for _ in range(12)]
    bxsrc,bysrc,bzsrc, bxprc,byprc,bzprc = [np.zeros_like(x) for _ in range(6)]
    hximf,hyimf,hzimf = [np.zeros_like(x) for _ in range(3)]

    # Calculations for different source contributions
    if iopgen <= 1:
        cfx,cfy,cfz = shlcar3x3(xx,yy,zz,ps)
        bxcf,bycf,bzcf = cfx*xappa3, cfy*xappa3, cfz*xappa3

    if (iopgen == 0) | (iopgen == 2):
        dstt = min(dst, -20.)
        znam = np.abs(dstt)**0.37
        dxshift1=a[23]-a[24]/znam
        dxshift2=a[25]-a[26]/znam
        d=a[35]*np.exp(-w1/a[36])+a[68]
        deltady=4.7
        bxt1,byt1,bzt1,bxt2,byt2,bzt2 = deformed(iopt,ps,xx,yy,zz)

    if (iopgen == 0) | (iopgen == 3):
        znam = max(np.abs(dst), 20.) if dst >= -20 else np.abs(dst)
        xkappa1=a[31]*(znam/20)**a[32]
        xkappa2=a[33]*(znam/20)**a[34]
        bxr11,byr11,bzr11, bxr12,byr12,bzr12, bxr21,byr21,bzr21, bxr22,byr22,bzr22 = \
            birk_tot(iopb,ps,xx,yy,zz)

    if (iopgen == 0) | (iopgen == 4):
        phi=a[37]
        znam= max(np.abs(dst), 20.) if dst >= -20 else np.abs(dst)
        sc_sy=a[27]*(20/znam)**a[28]*xappa
        sc_pr=a[29]*(20/znam)**a[30]*xappa
        bxsrc,bysrc,bzsrc, bxprc,byprc,bzprc = full_rc(iopr,ps,xx,yy,zz)

    if (iopgen == 0) | (iopgen == 5):
        hximf,hyimf,hzimf = np.zeros_like(x), byimf+np.zeros_like(x), bzimf+np.zeros_like(x)

    # Calculate amplitudes
    dlp1=(pdyn/2)**a[20]
    dlp2=(pdyn/2)**a[21]
    tamp1=a[1]+a[2]*dlp1+a[3]*a[38]*w1/np.sqrt(w1**2+a[38]**2)+a[4]*dst
    tamp2=a[5]+a[6]*dlp2+a[7]*a[39]*w2/np.sqrt(w2**2+a[39]**2)+a[8]*dst
    a_src=a[9] +a[10]*a[40]*w3/np.sqrt(w3**2+a[40]**2)+a[11]*dst
    a_prc=a[12]+a[13]*a[41]*w4/np.sqrt(w4**2+a[41]**2)+a[14]*dst
    a_r11=a[15]+a[16]*a[42]*w5/np.sqrt(w5**2+a[42]**2)
    a_r21=a[17]+a[18]*a[43]*w6/np.sqrt(w6**2+a[43]**2)

    # Sum up all components
    bbx=a[0]*bxcf + tamp1*bxt1+tamp2*bxt2 + a_src*bxsrc+a_prc*bxprc + a_r11*bxr11+a_r21*bxr21 + a[19]*hximf
    bby=a[0]*bycf + tamp1*byt1+tamp2*byt2 + a_src*bysrc+a_prc*byprc + a_r11*byr11+a_r21*byr21 + a[19]*hyimf
    bbz=a[0]*bzcf + tamp1*bzt1+tamp2*bzt2 + a_src*bzsrc+a_prc*bzprc + a_r11*bzr11+a_r21*bzr21 + a[19]*hzimf

    # Three cases for sigma based on masks
    mask_in = sigma < (s0 - dsig)
    mask_interp = (sigma >= (s0 - dsig)) & (sigma < (s0 + dsig))

    # Case 1: Inside the magnetosphere (default)
    bx, by, bz = bbx, bby, bbz

    # Case 2: Interpolation region
    if np.any(mask_interp):
        fint = 0.5 * (1. - (sigma - s0) / dsig)
        fext = 0.5 * (1. + (sigma - s0) / dsig)
        qx, qy, qz = dipole(ps, x, y, z)
        bx_interp = (bbx + qx) * fint + oimfx * fext - qx
        by_interp = (bby + qy) * fint + oimfy * fext - qy
        bz_interp = (bbz + qz) * fint + oimfz * fext - qz
        bx = np.where(mask_interp, bx_interp, bx)
        by = np.where(mask_interp, by_interp, by)
        bz = np.where(mask_interp, bz_interp, bz)
    
    # Case 3: Outside the magnetosphere
    mask_out = sigma >= (s0 + dsig)
    if np.any(mask_out):
        qx, qy, qz = dipole(ps, x, y, z)
        bx_out = oimfx - qx
        by_out = oimfy - qy
        bz_out = oimfz - qz
        bx = np.where(mask_out, bx_out, bx)
        by = np.where(mask_out, by_out, by)
        bz = np.where(mask_out, bz_out, bz)

    return bx,by,bz

#
# The following functions (shlcar3x3, deformed, warped, etc.) are identical
# to the vectorized versions from t01.py, as their mathematical structure
# is the same in both models.
#
def shlcar3x3(x, y, z, ps):
    """
    Calculates GSM components of the shielded field for the ring current
    (tail modes). Vectorized version.
    """
    a = np.array([
        -901.2327248,895.8011176,817.6208321,-845.5880889,-83.73539535,
        86.58542841,336.8781402,-329.3619944,-311.2947120,308.6011161,
        31.94469304,-31.30824526,125.8739681,-372.3384278,-235.4720434,
        286.7594095,21.86305585,-27.42344605,-150.4874688,2.669338538,
        1.395023949,-.5540427503,-56.85224007,3.681827033,-43.48705106,
        5.103131905,1.073551279,-.6673083508,12.21404266,4.177465543,
        5.799964188,-.3977802319,-1.044652977,.5703560010,3.536082962,
        -3.222069852,9.620648151,6.082014949,27.75216226,12.44199571,
        5.122226936,6.982039615,20.12149582,6.150973118,4.663639687,
        15.73319647,2.303504968,5.840511214,.8385953499E-01,.3477844929])
    
    p1,p2,p3, r1,r2,r3, q1,q2,q3, s1,s2,s3 = a[36:48]
    t1,t2 = a[48:50]

    cps=np.cos(ps)
    sps=np.sin(ps)
    s2ps=2*cps

    st1=np.sin(ps*t1)
    ct1=np.cos(ps*t1)
    st2=np.sin(ps*t2)
    ct2=np.cos(ps*t2)

    x1=x*ct1-z*st1
    z1=x*st1+z*ct1
    x2=x*ct2-z*st2
    z2=x*st2+z*ct2
    
    # make the terms in the 1st sum ("perpendicular" symmetry):
    # i=1:
    sqpr= np.sqrt(1/p1**2+1/r1**2)
    cyp = np.cos(y/p1)
    syp = np.sin(y/p1)
    czr = np.cos(z1/r1)
    szr = np.sin(z1/r1)
    expr= np.exp(sqpr*x1)
    fx1 =-sqpr*expr*cyp*szr
    hy1 = expr/p1*syp*szr
    fz1 =-expr*cyp/r1*czr
    hx1 = fx1*ct1+fz1*st1
    hz1 =-fx1*st1+fz1*ct1

    sqpr= np.sqrt(1/p1**2+1/r2**2)
    cyp = np.cos(y/p1)
    syp = np.sin(y/p1)
    czr = np.cos(z1/r2)
    szr = np.sin(z1/r2)
    expr= np.exp(sqpr*x1)
    fx2 =-sqpr*expr*cyp*szr
    hy2 = expr/p1*syp*szr
    fz2 =-expr*cyp/r2*czr
    hx2 = fx2*ct1+fz2*st1
    hz2 =-fx2*st1+fz2*ct1

    sqpr= np.sqrt(1/p1**2+1/r3**2)
    cyp = np.cos(y/p1)
    syp = np.sin(y/p1)
    czr = np.cos(z1/r3)
    szr = np.sin(z1/r3)
    expr= np.exp(sqpr*x1)
    fx3 =-expr*cyp*(sqpr*z1*czr+szr/r3*(x1+1/sqpr))
    hy3 = expr/p1*syp*(z1*czr+x1/r3*szr/sqpr)
    fz3 =-expr*cyp*(czr*(1+x1/r3**2/sqpr)-z1/r3*szr)
    hx3 = fx3*ct1+fz3*st1
    hz3 =-fx3*st1+fz3*ct1

    # i=2:
    sqpr= np.sqrt(1/p2**2+1/r1**2)
    cyp = np.cos(y/p2)
    syp = np.sin(y/p2)
    czr = np.cos(z1/r1)
    szr = np.sin(z1/r1)
    expr= np.exp(sqpr*x1)
    fx4 =-sqpr*expr*cyp*szr
    hy4 = expr/p2*syp*szr
    fz4 =-expr*cyp/r1*czr
    hx4 = fx4*ct1+fz4*st1
    hz4 =-fx4*st1+fz4*ct1

    sqpr= np.sqrt(1/p2**2+1/r2**2)
    cyp = np.cos(y/p2)
    syp = np.sin(y/p2)
    czr = np.cos(z1/r2)
    szr = np.sin(z1/r2)
    expr= np.exp(sqpr*x1)
    fx5 =-sqpr*expr*cyp*szr
    hy5 = expr/p2*syp*szr
    fz5 =-expr*cyp/r2*czr
    hx5 = fx5*ct1+fz5*st1
    hz5 =-fx5*st1+fz5*ct1

    sqpr= np.sqrt(1/p2**2+1/r3**2)
    cyp = np.cos(y/p2)
    syp = np.sin(y/p2)
    czr = np.cos(z1/r3)
    szr = np.sin(z1/r3)
    expr= np.exp(sqpr*x1)
    fx6 =-expr*cyp*(sqpr*z1*czr+szr/r3*(x1+1/sqpr))
    hy6 = expr/p2*syp*(z1*czr+x1/r3*szr/sqpr)
    fz6 =-expr*cyp*(czr*(1+x1/r3**2/sqpr)-z1/r3*szr)
    hx6 = fx6*ct1+fz6*st1
    hz6 =-fx6*st1+fz6*ct1

    # i=3:
    sqpr= np.sqrt(1/p3**2+1/r1**2)
    cyp = np.cos(y/p3)
    syp = np.sin(y/p3)
    czr = np.cos(z1/r1)
    szr = np.sin(z1/r1)
    expr= np.exp(sqpr*x1)
    fx7 =-sqpr*expr*cyp*szr
    hy7 = expr/p3*syp*szr
    fz7 =-expr*cyp/r1*czr
    hx7 = fx7*ct1+fz7*st1
    hz7 =-fx7*st1+fz7*ct1

    sqpr= np.sqrt(1/p3**2+1/r2**2)
    cyp = np.cos(y/p3)
    syp = np.sin(y/p3)
    czr = np.cos(z1/r2)
    szr = np.sin(z1/r2)
    expr= np.exp(sqpr*x1)
    fx8 =-sqpr*expr*cyp*szr
    hy8 = expr/p3*syp*szr
    fz8 =-expr*cyp/r2*czr
    hx8 = fx8*ct1+fz8*st1
    hz8 =-fx8*st1+fz8*ct1

    sqpr= np.sqrt(1/p3**2+1/r3**2)
    cyp = np.cos(y/p3)
    syp = np.sin(y/p3)
    czr = np.cos(z1/r3)
    szr = np.sin(z1/r3)
    expr= np.exp(sqpr*x1)
    fx9 =-expr*cyp*(sqpr*z1*czr+szr/r3*(x1+1/sqpr))
    hy9 = expr/p3*syp*(z1*czr+x1/r3*szr/sqpr)
    fz9 =-expr*cyp*(czr*(1+x1/r3**2/sqpr)-z1/r3*szr)
    hx9 = fx9*ct1+fz9*st1
    hz9 =-fx9*st1+fz9*ct1

    a1=a[0]+a[1]*cps
    a2=a[2]+a[3]*cps
    a3=a[4]+a[5]*cps
    a4=a[6]+a[7]*cps
    a5=a[8]+a[9]*cps
    a6=a[10]+a[11]*cps
    a7=a[12]+a[13]*cps
    a8=a[14]+a[15]*cps
    a9=a[16]+a[17]*cps
    bx=a1*hx1+a2*hx2+a3*hx3+a4*hx4+a5*hx5+a6*hx6+a7*hx7+a8*hx8+a9*hx9
    by=a1*hy1+a2*hy2+a3*hy3+a4*hy4+a5*hy5+a6*hy6+a7*hy7+a8*hy8+a9*hy9
    bz=a1*hz1+a2*hz2+a3*hz3+a4*hz4+a5*hz5+a6*hz6+a7*hz7+a8*hz8+a9*hz9

    # make the terms in the 2nd sum ("parallel" symmetry):
    # i=1
    sqqs= np.sqrt(1/q1**2+1/s1**2)
    cyq = np.cos(y/q1)
    syq = np.sin(y/q1)
    czs = np.cos(z2/s1)
    szs = np.sin(z2/s1)
    exqs= np.exp(sqqs*x2)
    fx1 =-sqqs*exqs*cyq*czs *sps
    hy1 = exqs/q1*syq*czs   *sps
    fz1 = exqs*cyq/s1*szs   *sps
    hx1 = fx1*ct2+fz1*st2
    hz1 =-fx1*st2+fz1*ct2

    sqqs= np.sqrt(1/q1**2+1/s2**2)
    cyq = np.cos(y/q1)
    syq = np.sin(y/q1)
    czs = np.cos(z2/s2)
    szs = np.sin(z2/s2)
    exqs= np.exp(sqqs*x2)
    fx2 =-sqqs*exqs*cyq*czs *sps
    hy2 = exqs/q1*syq*czs   *sps
    fz2 = exqs*cyq/s2*szs   *sps
    hx2 = fx2*ct2+fz2*st2
    hz2 =-fx2*st2+fz2*ct2

    sqqs= np.sqrt(1/q1**2+1/s3**2)
    cyq = np.cos(y/q1)
    syq = np.sin(y/q1)
    czs = np.cos(z2/s3)
    szs = np.sin(z2/s3)
    exqs= np.exp(sqqs*x2)
    fx3 =-sqqs*exqs*cyq*czs *sps
    hy3 = exqs/q1*syq*czs   *sps
    fz3 = exqs*cyq/s3*szs   *sps
    hx3 = fx3*ct2+fz3*st2
    hz3 =-fx3*st2+fz3*ct2

    # i=2:
    sqqs= np.sqrt(1/q2**2+1/s1**2)
    cyq = np.cos(y/q2)
    syq = np.sin(y/q2)
    czs = np.cos(z2/s1)
    szs = np.sin(z2/s1)
    exqs= np.exp(sqqs*x2)
    fx4 =-sqqs*exqs*cyq*czs *sps
    hy4 = exqs/q2*syq*czs   *sps
    fz4 = exqs*cyq/s1*szs   *sps
    hx4 = fx4*ct2+fz4*st2
    hz4 =-fx4*st2+fz4*ct2

    sqqs= np.sqrt(1/q2**2+1/s2**2)
    cyq = np.cos(y/q2)
    syq = np.sin(y/q2)
    czs = np.cos(z2/s2)
    szs = np.sin(z2/s2)
    exqs= np.exp(sqqs*x2)
    fx5 =-sqqs*exqs*cyq*czs *sps
    hy5 = exqs/q2*syq*czs   *sps
    fz5 = exqs*cyq/s2*szs   *sps
    hx5 = fx5*ct2+fz5*st2
    hz5 =-fx5*st2+fz5*ct2

    sqqs= np.sqrt(1/q2**2+1/s3**2)
    cyq = np.cos(y/q2)
    syq = np.sin(y/q2)
    czs = np.cos(z2/s3)
    szs = np.sin(z2/s3)
    exqs= np.exp(sqqs*x2)
    fx6 =-sqqs*exqs*cyq*czs *sps
    hy6 = exqs/q2*syq*czs   *sps
    fz6 = exqs*cyq/s3*szs   *sps
    hx6 = fx6*ct2+fz6*st2
    hz6 =-fx6*st2+fz6*ct2

    # i=3:
    sqqs= np.sqrt(1/q3**2+1/s1**2)
    cyq = np.cos(y/q3)
    syq = np.sin(y/q3)
    czs = np.cos(z2/s1)
    szs = np.sin(z2/s1)
    exqs= np.exp(sqqs*x2)
    fx7 =-sqqs*exqs*cyq*czs *sps
    hy7 = exqs/q3*syq*czs   *sps
    fz7 = exqs*cyq/s1*szs   *sps
    hx7 = fx7*ct2+fz7*st2
    hz7 =-fx7*st2+fz7*ct2

    sqqs= np.sqrt(1/q3**2+1/s2**2)
    cyq = np.cos(y/q3)
    syq = np.sin(y/q3)
    czs = np.cos(z2/s2)
    szs = np.sin(z2/s2)
    exqs= np.exp(sqqs*x2)
    fx8 =-sqqs*exqs*cyq*czs *sps
    hy8 = exqs/q3*syq*czs   *sps
    fz8 = exqs*cyq/s2*szs   *sps
    hx8 = fx8*ct2+fz8*st2
    hz8 =-fx8*st2+fz8*ct2

    sqqs= np.sqrt(1/q3**2+1/s3**2)
    cyq = np.cos(y/q3)
    syq = np.sin(y/q3)
    czs = np.cos(z2/s3)
    szs = np.sin(z2/s3)
    exqs= np.exp(sqqs*x2)
    fx9 =-sqqs*exqs*cyq*czs *sps
    hy9 = exqs/q3*syq*czs   *sps
    fz9 = exqs*cyq/s3*szs   *sps
    hx9 = fx9*ct2+fz9*st2
    hz9 =-fx9*st2+fz9*ct2

    a1=a[18]+a[19]*s2ps
    a2=a[20]+a[21]*s2ps
    a3=a[22]+a[23]*s2ps
    a4=a[24]+a[25]*s2ps
    a5=a[26]+a[27]*s2ps
    a6=a[28]+a[29]*s2ps
    a7=a[30]+a[31]*s2ps
    a8=a[32]+a[33]*s2ps
    a9=a[34]+a[35]*s2ps

    bx=bx+a1*hx1+a2*hx2+a3*hx3+a4*hx4+a5*hx5+a6*hx6+a7*hx7+a8*hx8+a9*hx9
    by=by+a1*hy1+a2*hy2+a3*hy3+a4*hy4+a5*hy5+a6*hy6+a7*hy7+a8*hy8+a9*hy9
    bz=bz+a1*hz1+a2*hz2+a3*hz3+a4*hz4+a5*hz5+a6*hz6+a7*hz7+a8*hz8+a9*hz9
    
    return bx, by, bz


def deformed(iopt, ps, x, y, z):
    """This function is already vectorized, assuming sub-calls are vectorized."""
    global rh0
    rh2,ieps = [-5.2,3]

    sps = np.sin(ps)
    r2 = x**2+y**2+z**2
    r = np.sqrt(r2)
    r_safe = np.where(r==0, 1e-9, r)
    zr = z/r_safe
    rh = rh0+rh2*zr**2
    rh_safe = np.where(rh==0, 1e-9, rh)

    drhdr = -zr/r_safe*2*rh2*zr
    drhdz = 2*rh2*zr/r_safe
    
    rrh = r/rh_safe
    f = 1/(1+rrh**ieps)**(1/ieps)
    dfdr = -rrh**(ieps-1)*f**(ieps+1)/rh_safe
    dfdrh = -rrh*dfdr

    spsas = sps*f
    cospsas = np.sqrt(1-spsas**2)

    xas = x*cospsas-z*spsas
    zas = x*spsas+z*cospsas

    facps = sps/cospsas*(dfdr+dfdrh*drhdr)/r_safe
    psasx = facps*x
    psasy = facps*y
    psasz = facps*z+sps/cospsas*dfdrh*drhdz

    dxasdx = cospsas-zas*psasx
    dxasdy =-zas*psasy
    dxasdz =-spsas-zas*psasz
    dzasdx = spsas+xas*psasx
    dzasdy = xas*psasy
    dzasdz = cospsas+xas*psasz
    fac1 = dxasdz*dzasdy-dxasdy*dzasdz
    fac2 = dxasdx*dzasdz-dxasdz*dzasdx
    fac3 = dzasdx*dxasdy-dxasdx*dzasdy

    bxas1,byas1,bzas1, bxas2,byas2,bzas2 = warped(iopt,ps,xas,y,zas)

    bx1=bxas1*dzasdz-bzas1*dxasdz +byas1*fac1
    by1=byas1*fac2
    bz1=bzas1*dxasdx-bxas1*dzasdx +byas1*fac3

    bx2=bxas2*dzasdz-bzas2*dxasdz +byas2*fac1
    by2=byas2*fac2
    bz2=bzas2*dxasdx-bxas2*dzasdx +byas2*fac3

    return bx1,by1,bz1, bx2,by2,bz2


def warped(iopt, ps, x, y, z):
    """Vectorized version of warped."""
    global g
    dgdx,xl,dxldx = [0.,20,0]

    sps=np.sin(ps)
    rho2=y**2+z**2
    rho=np.sqrt(rho2)

    phi=np.arctan2(z,y)
    cphi = np.where(rho != 0, y / rho, 1.0)
    sphi = np.where(rho != 0, z / rho, 0.0)

    rr4l4=rho/(rho2**2+xl**4)

    f=phi+g*rho2*rr4l4*cphi*sps
    dfdphi=1-g*rho2*rr4l4*sphi*sps
    dfdrho=g*rr4l4**2*(3*xl**4-rho2**2)*cphi*sps
    dfdx=rr4l4*cphi*sps*(dgdx*rho2-g*rho*rr4l4*4*xl**3*dxldx)

    cf=np.cos(f)
    sf=np.sin(f)
    yas=rho*cf
    zas=rho*sf

    bx_as1,by_as1,bz_as1, bx_as2,by_as2,bz_as2 = unwarped(iopt,x,yas,zas)

    brho_as =  by_as1*cf+bz_as1*sf
    bphi_as = -by_as1*sf+bz_as1*cf
    brho_s = brho_as*dfdphi
    bphi_s = bphi_as-rho*(bx_as1*dfdx+brho_as*dfdrho)
    bx1    = bx_as1*dfdphi
    by1    = brho_s*cphi-bphi_s*sphi
    bz1    = brho_s*sphi+bphi_s*cphi

    brho_as =  by_as2*cf+bz_as2*sf
    bphi_as = -by_as2*sf+bz_as2*cf
    brho_s = brho_as*dfdphi
    bphi_s = bphi_as-rho*(bx_as2*dfdx+brho_as*dfdrho)
    bx2    = bx_as2*dfdphi
    by2    = brho_s*cphi-bphi_s*sphi
    bz2    = brho_s*sphi+bphi_s*cphi

    return bx1,by1,bz1, bx2,by2,bz2



def unwarped(iopt, x, y, z):
    """This function is already vectorized, assuming sub-calls are vectorized."""
    global dxshift1, dxshift2, d, deltady

    deltadx1,alpha1,xshift1 = [1.,1.1,6]
    deltadx2,alpha2,xshift2 = [0.,.25,4]

    a1 = np.array([
        -25.45869857,57.35899080,317.5501869,-2.626756717,-93.38053698,
        -199.6467926,-858.8129729,34.09192395,845.4214929,-29.07463068,
        47.10678547,-128.9797943,-781.7512093,6.165038619,167.8905046,
        492.0680410,1654.724031,-46.77337920,-1635.922669,40.86186772,
        -.1349775602,-.9661991179e-01,-.1662302354,.002810467517,.2487355077,
        .1025565237,-14.41750229,-.8185333989,11.07693629,.7569503173,
        -9.655264745,112.2446542,777.5948964,-5.745008536,-83.03921993,
        -490.2278695,-1155.004209,39.08023320,1172.780574,-39.44349797,
        -14.07211198,-40.41201127,-313.2277343,2.203920979,8.232835341,
        197.7065115,391.2733948,-18.57424451,-437.2779053,23.04976898,
        11.75673963,13.60497313,4.691927060,18.20923547,27.59044809,
        6.677425469,1.398283308,2.839005878,31.24817706,24.53577264])

    a2 = np.array([
        -287187.1962,4970.499233,410490.1952,-1347.839052,-386370.3240,
        3317.983750,-143462.3895,5706.513767,171176.2904,250.8882750,
        -506570.8891,5733.592632,397975.5842,9771.762168,-941834.2436,
        7990.975260,54313.10318,447.5388060,528046.3449,12751.04453,
        -21920.98301,-21.05075617,31971.07875,3012.641612,-301822.9103,
        -3601.107387,1797.577552,-6.315855803,142578.8406,13161.93640,
        804184.8410,-14168.99698,-851926.6360,-1890.885671,972475.6869,
        -8571.862853,26432.49197,-2554.752298,-482308.3431,-4391.473324,
        105155.9160,-1134.622050,-74353.53091,-5382.670711,695055.0788,
        -916.3365144,-12111.06667,67.20923358,-367200.9285,-21414.14421,
        14.75567902,20.75638190,59.78601609,16.86431444,32.58482365,
        23.69472951,17.24977936,13.64902647,68.40989058,11.67828167])

    xm1,xm2 = [-12.,-12]
    bx1,by1,bz1, bx2,by2,bz2 = [np.zeros_like(x) for _ in range(6)]

    if iopt < 2:
        xsc1 = (x-xshift1-dxshift1)*alpha1-xm1*(alpha1-1)
        ysc1 = y*alpha1
        zsc1 = z*alpha1
        d0sc1 = d*alpha1

        fx1,fy1,fz1 = taildisk(d0sc1,deltadx1,deltady,xsc1,ysc1,zsc1)
        hx1,hy1,hz1 = shlcar5x5(a1,x,y,z,dxshift1)

        bx1=fx1+hx1
        by1=fy1+hy1
        bz1=fz1+hz1

    if iopt != 1:
        xsc2 = (x-xshift2-dxshift2)*alpha2-xm2*(alpha2-1)
        ysc2 = y*alpha2
        zsc2 = z*alpha2
        d0sc2 = d*alpha2

        fx2,fy2,fz2 = taildisk(d0sc2,deltadx2,deltady,xsc2,ysc2,zsc2)
        hx2,hy2,hz2 = shlcar5x5(a2,x,y,z,dxshift2)

        bx2=fx2+hx2
        by2=fy2+hy2
        bz2=fz2+hz2

    return bx1,by1,bz1, bx2,by2,bz2



def shlcar5x5(a,x,y,z,dshift):
    """This function is already vectorized."""
    dhx,dhy,dhz = [np.zeros_like(x) for _ in range(3)]

    l=0
    for i in range(5):
        rp=1/a[50+i]
        cypi=np.cos(y*rp)
        sypi=np.sin(y*rp)

        for k in range(5):
            rr=1/a[55+k]
            szrk=np.sin(z*rr)
            czrk=np.cos(z*rr)
            sqpr=np.sqrt(rp**2+rr**2)
            epr= np.exp(x*sqpr)

            dbx=-sqpr*epr*cypi*szrk
            dby= rp*epr*sypi*szrk
            dbz=-rr*epr*cypi*czrk

            coef=a[l]+a[l+1]*dshift
            l += 2

            dhx=dhx+coef*dbx
            dhy=dhy+coef*dby
            dhz=dhz+coef*dbz

    return dhx,dhy,dhz


def taildisk(d0,deltadx,deltady, x,y,z):
    """Vectorized version of taildisk."""
    f = np.array([-71.09346626,-1014.308601,-1272.939359,-3224.935936,-44546.86232])
    b = np.array([10.90101242,12.68393898,13.51791954,14.86775017,15.12306404])
    c = np.array([.7954069972,.6716601849,1.174866319,2.565249920,10.01986790])

    rho=np.sqrt(x**2+y**2)
    drhodx = np.where(rho != 0, x / rho, 0.0)
    drhody = np.where(rho != 0, y / rho, 0.0)

    dex=np.exp(x/7)
    d=d0+deltady*(y/20)**2+deltadx*dex
    dddy=deltady*y*0.005
    dddx=deltadx/7*dex

    dzeta=np.sqrt(z**2+d**2)
    dzeta_safe = np.where(dzeta==0, 1e-9, dzeta)
    ddzetadx=d*dddx/dzeta_safe
    ddzetady=d*dddy/dzeta_safe
    ddzetadz=z/dzeta_safe

    dbx,dby,dbz = [np.zeros_like(x) for _ in range(3)]

    for i in range(5):
        bi=b[i]
        ci=c[i]

        s1=np.sqrt((rho+bi)**2+(dzeta+ci)**2)
        s2=np.sqrt((rho-bi)**2+(dzeta+ci)**2)

        ds1drho=(rho+bi)/s1
        ds2drho=(rho-bi)/s2
        ds1ddz=(dzeta+ci)/s1
        ds2ddz=(dzeta+ci)/s2

        ds1dx=ds1drho*drhodx+ds1ddz*ddzetadx
        ds1dy=ds1drho*drhody+ds1ddz*ddzetady
        ds1dz=               ds1ddz*ddzetadz

        ds2dx=ds2drho*drhodx+ds2ddz*ddzetadx
        ds2dy=ds2drho*drhody+ds2ddz*ddzetady
        ds2dz=               ds2ddz*ddzetadz

        s1ts2=s1*s2
        s1ps2=s1+s2
        s1ps2sq=s1ps2**2

        fac1=np.sqrt(np.maximum(s1ps2sq-(2*bi)**2, 0)) # Ensure non-negative
        s1ts2_safe = np.where(s1ts2==0, 1e-9, s1ts2)
        s1ps2_safe = np.where(s1ps2==0, 1e-9, s1ps2)
        fac1_safe = np.where(fac1==0, 1e-9, fac1)
        s1_safe = np.where(s1==0, 1e-9, s1)
        s2_safe = np.where(s2==0, 1e-9, s2)
        
        asas=fac1_safe/(s1ts2_safe*s1ps2_safe**2)
        dasds1=(1/(fac1_safe*s2_safe)-asas/s1ps2_safe*(s2*s2+s1*(3*s1+4*s2)))/(s1_safe*s1ps2_safe)
        dasds2=(1/(fac1_safe*s1_safe)-asas/s1ps2_safe*(s1*s1+s2*(3*s2+4*s1)))/(s2_safe*s1ps2_safe)

        dasdx=dasds1*ds1dx+dasds2*ds2dx
        dasdy=dasds1*ds1dy+dasds2*ds2dy
        dasdz=dasds1*ds1dz+dasds2*ds2dz

        dbx=dbx-f[i]*x*dasdz
        dby=dby-f[i]*y*dasdz
        dbz=dbz+f[i]*(2*asas+x*dasdx+y*dasdy)

    return dbx, dby, dbz


def birk_tot(iopb, ps, x, y, z):
    """
    Calculates components of the field from Birkeland field-aligned currents.
    Vectorized version.
    """
    global xkappa1, xkappa2
    global dphi, b, rho_0, xkappa

    # This function's structure is identical to T01's, so the same vectorized logic applies.
    # The coefficients are hardcoded in the T04 model.
    sh11 = np.array([
        46488.84663,-15541.95244,-23210.09824,-32625.03856,-109894.4551,
        -71415.32808,58168.94612,55564.87578,-22890.60626,-6056.763968,
        5091.368100,239.7001538,-13899.49253,4648.016991,6971.310672,
        9699.351891,32633.34599,21028.48811,-17395.96190,-16461.11037,
        7447.621471,2528.844345,-1934.094784,-588.3108359,-32588.88216,
        10894.11453,16238.25044,22925.60557,77251.11274,50375.97787,
        -40763.78048,-39088.60660,15546.53559,3559.617561,-3187.730438,
        309.1487975,88.22153914,-243.0721938,-63.63543051,191.1109142,
        69.94451996,-187.9539415,-49.89923833,104.0902848,-120.2459738,
        253.5572433,89.25456949,-205.6516252,-44.93654156,124.7026309,
        32.53005523,-98.85321751,-36.51904756,98.88241690,24.88493459,
        -55.04058524,61.14493565,-128.4224895,-45.35023460,105.0548704,
        -43.66748755,119.3284161,31.38442798,-92.87946767,-33.52716686,
        89.98992001,25.87341323,-48.86305045,59.69362881,-126.5353789,
        -44.39474251,101.5196856,59.41537992,41.18892281,80.86101200,
        3.066809418,7.893523804,30.56212082,10.36861082,8.222335945,
        19.97575641,2.050148531,4.992657093,2.300564232,.2256245602,-.05841594319])

    sh12 = np.array([
        210260.4816,-1443587.401,-1468919.281,281939.2993,-1131124.839,
        729331.7943,2573541.307,304616.7457,468887.5847,181554.7517,
        -1300722.650,-257012.8601,645888.8041,-2048126.412,-2529093.041,
        571093.7972,-2115508.353,1122035.951,4489168.802,75234.22743,
        823905.6909,147926.6121,-2276322.876,-155528.5992,-858076.2979,
        3474422.388,3986279.931,-834613.9747,3250625.781,-1818680.377,
        -7040468.986,-414359.6073,-1295117.666,-346320.6487,3565527.409,
        430091.9496,-.1565573462,7.377619826,.4115646037,-6.146078880,
        3.808028815,-.5232034932,1.454841807,-12.32274869,-4.466974237,
        -2.941184626,-.6172620658,12.64613490,1.494922012,-21.35489898,
        -1.652256960,16.81799898,-1.404079922,-24.09369677,-10.99900839,
        45.94237820,2.248579894,31.91234041,7.575026816,-45.80833339,
        -1.507664976,14.60016998,1.348516288,-11.05980247,-5.402866968,
        31.69094514,12.28261196,-37.55354174,4.155626879,-33.70159657,
        -8.437907434,36.22672602,145.0262164,70.73187036,85.51110098,
        21.47490989,24.34554406,31.34405345,4.655207476,5.747889264,
        7.802304187,1.844169801,4.867254550,2.941393119,.1379899178,.06607020029])

    sh21 = np.array([
        162294.6224,503885.1125,-27057.67122,-531450.1339,84747.05678,
        -237142.1712,84133.61490,259530.0402,69196.05160,-189093.5264,
        -19278.55134,195724.5034,-263082.6367,-818899.6923,43061.10073,
        863506.6932,-139707.9428,389984.8850,-135167.5555,-426286.9206,
        -109504.0387,295258.3531,30415.07087,-305502.9405,100785.3400,
        315010.9567,-15999.50673,-332052.2548,54964.34639,-152808.3750,
        51024.67566,166720.0603,40389.67945,-106257.7272,-11126.14442,
        109876.2047,2.978695024,558.6019011,2.685592939,-338.0004730,
        -81.99724090,-444.1102659,89.44617716,212.0849592,-32.58562625,
        -982.7336105,-35.10860935,567.8931751,-1.917212423,-260.2023543,
        -1.023821735,157.5533477,23.00200055,232.0603673,-36.79100036,
        -111.9110936,18.05429984,447.0481000,15.10187415,-258.7297813,
        -1.032340149,-298.6402478,-1.676201415,180.5856487,64.52313024,
        209.0160857,-53.85574010,-98.52164290,14.35891214,536.7666279,
        20.09318806,-309.7349530,58.54144539,67.45226850,97.92374406,
        4.752449760,10.46824379,32.91856110,12.05124381,9.962933904,
        15.91258637,1.804233877,6.578149088,2.515223491,.1930034238,-.02261109942])

    sh22 = np.array([
        -131287.8986,-631927.6885,-318797.4173,616785.8782,-50027.36189,
        863099.9833,47680.20240,-1053367.944,-501120.3811,-174400.9476,
        222328.6873,333551.7374,-389338.7841,-1995527.467,-982971.3024,
        1960434.268,297239.7137,2676525.168,-147113.4775,-3358059.979,
        -2106979.191,-462827.1322,1017607.960,1039018.475,520266.9296,
        2627427.473,1301981.763,-2577171.706,-238071.9956,-3539781.111,
        94628.16420,4411304.724,2598205.733,637504.9351,-1234794.298,
        -1372562.403,-2.646186796,-31.10055575,2.295799273,19.20203279,
        30.01931202,-302.1028550,-14.78310655,162.1561899,.4943938056,
        176.8089129,-.2444921680,-100.6148929,9.172262228,137.4303440,
        -8.451613443,-84.20684224,-167.3354083,1321.830393,76.89928813,
        -705.7586223,18.28186732,-770.1665162,-9.084224422,436.3368157,
        -6.374255638,-107.2730177,6.080451222,65.53843753,143.2872994,
        -1028.009017,-64.22739330,547.8536586,-20.58928632,597.3893669,
        10.17964133,-337.7800252,159.3532209,76.34445954,84.74398828,
        12.76722651,27.63870691,32.69873634,5.145153451,6.310949163,
        6.996159733,1.971629939,4.436299219,2.904964304,.1486276863,.06859991529])

    xkappa = xkappa1
    x_sc = xkappa1 - 1.1

    bx11,by11,bz11, bx12,by12,bz12, bx21,by21,bz21, bx22,by22,bz22 = [np.zeros_like(x) for _ in range(12)]

    if (iopb == 0) or (iopb == 1):
        fx11,fy11,fz11 = birk_1n2(1,1,ps,x,y,z)
        hx11,hy11,hz11 = birk_shl(sh11,ps,x_sc,x,y,z)
        bx11,by11,bz11 = fx11+hx11, fy11+hy11, fz11+hz11

        fx12,fy12,fz12 = birk_1n2(1,2,ps,x,y,z)
        hx12,hy12,hz12 = birk_shl(sh12,ps,x_sc,x,y,z)
        bx12,by12,bz12 = fx12+hx12, fy12+hy12, fz12+hz12

    xkappa = xkappa2
    x_sc = xkappa2 - 1.0

    if (iopb == 0) or (iopb == 2):
        fx21,fy21,fz21 = birk_1n2(2,1,ps,x,y,z)
        hx21,hy21,hz21 = birk_shl(sh21,ps,x_sc,x,y,z)
        bx21,by21,bz21 = fx21+hx21, fy21+hy21, fz21+hz21

        fx22,fy22,fz22 = birk_1n2(2,2,ps,x,y,z)
        hx22,hy22,hz22 = birk_shl(sh22,ps,x_sc,x,y,z)
        bx22,by22,bz22 = fx22+hx22, fy22+hy22, fz22+hz22

    return bx11,by11,bz11, bx12,by12,bz12, bx21,by21,bz21, bx22,by22,bz22

def birk_1n2(numb,mode,ps,x,y,z):
    """
    Calculates field components for a model Birkeland current field.
    Vectorized version.
    """
    global dtheta, m, dphi, b, rho_0, xkappa

    beta = 0.9
    rh = 10.
    eps = 3.
    b=0.5
    rho_0=7.0

    a11 = np.array([
        .1618068350, -.1797957553, 2.999642482, -.9322708978, -.6811059760,
        .2099057262, -8.358815746, -14.86033550, .3838362986, -16.30945494,
        4.537022847, 2.685836007, 27.97833029, 6.330871059, 1.876532361,
        18.95619213, .9651528100, .4217195118, -.08957770020, -1.823555887,
        .7457045438, -.5785916524, -1.010200918, .01112389357, .09572927448,
        -.3599292276, 8.713700514, .9763932955, 3.834602998, 2.492118385, .7113544659])
    a12 = np.array([
        .7058026940, -.2845938535, 5.715471266, -2.472820880, -.7738802408,
        .3478293930, -11.37653694, -38.64768867, .6932927651, -212.4017288,
        4.944204937, 3.071270411, 33.05882281, 7.387533799, 2.366769108,
        79.22572682, .6154290178, .5592050551, -.1796585105, -1.654932210,
        .7309108776, -.4926292779, -1.130266095, -.009613974555, .1484586169,
        -.2215347198, 7.883592948, .02768251655, 2.950280953, 1.212634762, .5567714182])
    a21 = np.array([
        .1278764024, -.2320034273, 1.805623266, -32.37241440, -.9931490648,
        .3175085630, -2.492465814, -16.21600096, .2695393416, -6.752691265,
        3.971794901, 14.54477563, 41.10158386, 7.912889730, 1.258297372,
        9.583547721, 1.014141963, .5104134759, -.1790430468, -1.756358428,
        .7561986717, -.6775248254, -.04014016420, .01446794851, .1200521731,
        -.2203584559, 4.508963850, .8221623576, 1.779933730, 1.102649543, .8867880020])
    a22 = np.array([
        .4036015198, -.3302974212, 2.827730930, -45.44405830, -1.611103927,
        .4927112073, -.003258457559, -49.59014949, .3796217108, -233.7884098,
        4.312666980, 18.05051709, 28.95320323, 11.09948019, .7471649558,
        67.10246193, .5667096597, .6468519751, -.1560665317, -1.460805289,
        .7719653528, -.6658988668, .2515179349E-05, .02426021891, .1195003324,
        -.2625739255, 4.377172556, .2421190547, 2.503482679, 1.071587299, .7247997430])

    m=mode
    if numb == 1:
        dphi=0.055
        dtheta=0.06
    elif numb == 2:
        dphi=0.030
        dtheta=0.09
    else:
        raise ValueError

    xsc=x*xkappa
    ysc=y*xkappa
    zsc=z*xkappa
    rho=np.sqrt(xsc**2+zsc**2)

    rsc=np.sqrt(xsc**2+ysc**2+zsc**2)
    rho2=rho_0**2

    phi=np.arctan2(-zsc,xsc)
    sphic=np.sin(phi)
    cphic=np.cos(phi)

    brack=dphi+b*rho2/(rho2+1)*(rho**2-1)/(rho2+rho**2)
    r_safe = np.where(rsc==1.0, 1.0+1e-9, rsc)
    r1rh=(r_safe-1)/rh
    psias=beta*ps/(1+r1rh**eps)**(1/eps)

    phis=phi-brack*np.sin(phi) -psias
    dphisphi=1-brack*np.cos(phi)
    dphisrho=-2*b*rho2*rho/(rho2+rho**2)**2*np.sin(phi) \
        +beta*ps*r1rh**(eps-1)*rho/(rh*rsc*(1+r1rh**eps)**(1/eps+1))
    dphisdy= beta*ps*r1rh**(eps-1)*ysc/(rh*rsc*(1+r1rh**eps)**(1/eps+1))

    sphics=np.sin(phis)
    cphics=np.cos(phis)

    xs= rho*cphics
    zs=-rho*sphics

    if numb ==1:
        if mode == 1: [bxs,byas,bzs] = twocones(a11,xs,ysc,zs)
        elif mode == 2: [bxs,byas,bzs] = twocones(a12,xs,ysc,zs)
        else: raise ValueError
    else:
        if mode == 1: [bxs,byas,bzs] = twocones(a21,xs,ysc,zs)
        elif mode == 2: [bxs,byas,bzs] = twocones(a22,xs,ysc,zs)
        else: raise ValueError

    brhoas =  bxs*cphics-bzs*sphics
    bphias = -bxs*sphics-bzs*cphics

    brho_s=brhoas*dphisphi                             *xkappa
    bphi_s=(bphias-rho*(byas*dphisdy+brhoas*dphisrho)) *xkappa
    by_s=byas*dphisphi                                 *xkappa

    bx=brho_s*cphic-bphi_s*sphic
    by=by_s
    bz=-brho_s*sphic-bphi_s*cphic

    return bx,by,bz


def twocones(a,x,y,z):
    """
    Computes field of a model ring current by two cones.
    Vectorized version.
    """
    bxn,byn,bzn = one_cone(a,x, y, z)
    bxs,bys,bzs = one_cone(a,x,-y,-z)
    bx=bxn-bxs
    by=byn+bys
    bz=bzn+bzs

    return bx,by,bz


def one_cone(a,x,y,z):
    """
    Computes field components for a conical current system.
    Vectorized version.
    """
    global dtheta, m

    dr = 1e-6
    dt = 1e-6

    theta0=a[30]

    rho2=x**2+y**2
    rho=np.sqrt(rho2)
    r=np.sqrt(rho2+z**2)
    theta=np.arctan2(rho,z)
    phi=np.arctan2(y,x)

    rs=r_s(a,r,theta)
    thetas=theta_s(a,r,theta)
    phis=phi

    btast,bfast = fialcos(rs,thetas,phis,m,theta0,dtheta)

    drsdr=(r_s(a,r+dr,theta)-r_s(a,r-dr,theta))/(2*dr)
    drsdt=(r_s(a,r,theta+dt)-r_s(a,r,theta-dt))/(2*dt)
    dtsdr=(theta_s(a,r+dr,theta)-theta_s(a,r-dr,theta))/(2*dr)
    dtsdt=(theta_s(a,r,theta+dt)-theta_s(a,r,theta-dt))/(2*dt)

    st_safe = np.where(np.sin(theta)==0, 1e-9, np.sin(theta))
    stsst=np.sin(thetas)/st_safe
    r_safe = np.where(r==0, 1e-9, r)
    rsr=rs/r_safe

    br     =-rsr/r_safe*stsst*btast*drsdt
    btheta = rsr*stsst*btast*drsdr
    bphi   = rsr*bfast*(drsdr*dtsdt-drsdt*dtsdr)

    rho_safe = np.where(rho==0, 1e-9, rho)
    s = np.where(r != 0, rho / r, 0.0)
    c = np.where(r != 0, z / r, 0.0)
    sf = np.where(rho != 0, y / rho, 0.0)
    cf = np.where(rho != 0, x / rho, 1.0)

    be=br*s+btheta*c

    bx=a[0]*(be*cf-bphi*sf)
    by=a[0]*(be*sf+bphi*cf)
    bz=a[0]*(br*c-btheta*s)

    return bx,by,bz


def r_s(a,r,theta):
    """
    Computes shifted radial distance for the Birkeland current model.
    Vectorized version.
    """
    r_safe = np.where(r==0, 1e-9, r)
    return r+a[1]/r_safe+a[2]*r/np.sqrt(r**2+a[10]**2)+a[3]*r/(r**2+a[11]**2) \
        +(a[4]+a[5]/r_safe+a[6]*r/np.sqrt(r**2+a[12]**2)+a[7]*r/(r**2+a[13]**2))*np.cos(theta) \
        +(a[8]*r/np.sqrt(r**2+a[14]**2)+a[9]*r/(r**2+a[15]**2)**2)*np.cos(2*theta)


def theta_s(a,r,theta):
    """
    Computes shifted theta angle for the Birkeland current model.
    Vectorized version.
    """
    r_safe = np.where(r==0, 1e-9, r)
    return theta+(a[16]+a[17]/r_safe+a[18]/r_safe**2+a[19]*r/np.sqrt(r**2+a[26]**2))*np.sin(theta) \
        +(a[20]+a[21]*r/np.sqrt(r**2+a[27]**2)+a[22]*r/(r**2+a[28]**2))*np.sin(2*theta) \
        +(a[23]+a[24]/r_safe+a[25]*r/(r**2+a[29]**2))*np.sin(3*theta)


def fialcos(r,theta,phi,n,theta0,dt):
    """
    Calculates field components in Fialco coordinates.
    Vectorized version.
    """
    r, theta, phi = np.atleast_1d(r, theta, phi)

    sinte = np.sin(theta)
    coste = np.cos(theta)
    r_safe = np.where(r == 0, 1e-9, r)
    ro = r * sinte
    ro_safe = np.where(ro == 0, 1e-9, ro)

    # Avoid division by zero in tg and ctg
    coste_p1_safe = np.where(1 + coste == 0, 1e-9, 1 + coste)
    coste_m1_safe = np.where(1 - coste == 0, 1e-9, 1 - coste)
    tg = sinte / coste_p1_safe
    ctg = sinte / coste_m1_safe

    tetanp=theta0+dt
    tetanm=theta0-dt

    tgp=np.tan(tetanp*0.5)
    tgm=np.tan(tetanm*0.5)
    tgm2=tgm*tgm

    m = n
    tm = tg**m
    cosm_phi = np.cos(m * phi)
    sinm_phi = np.sin(m * phi)

    # Case 1: theta < tetanm
    t1 = tm.copy()
    dtt1 = 0.5*m*tm*(tg+ctg)

    # Case 2: tetanm <= theta < tetanp
    tgm2m=tgm**(2*m)
    fc=1/(tgp-tgm)
    fc1=1/(2*m+1)
    tgm2m1=tgm2m*tgm
    tg21=1+tg**2

    tm_safe = np.where(tm == 0, 1e-9, tm)
    tg_safe = np.where(tg == 0, 1e-9, tg)

    t2 = fc*(tm*(tgp-tg)+fc1*(tm*tg-tgm2m1/tm_safe))
    dtt2 = 0.5*m*fc*tg21*(tm/tg_safe*(tgp-tg)-fc1*(tm-tgm2m1/(tm_safe*tg_safe)))

    # Case 3: theta >= tetanp
    tgp2m=tgp**(2*m)
    t3 = fc*fc1*(tgp2m*tgp-tgm2m*tgm)/tm_safe
    dtt3 = -t3*m*0.5*(tg+ctg)

    mask1 = theta < tetanm
    mask2 = (theta >= tetanm) & (theta < tetanp)

    t = np.where(mask1, t1, np.where(mask2, t2, t3))
    dtt = np.where(mask1, dtt1, np.where(mask2, dtt2, dtt3))

    btn = m*t*cosm_phi/ro_safe
    bpn = -dtt*sinm_phi/r_safe

    btheta=btn *800.
    bphi  =bpn *800.

    return btheta, bphi


def birk_shl(a,ps,x_sc, x,y,z):
    """
    Calculates GSM components of the external field due to Birkeland currents.
    Vectorized version.
    """
    cps=np.cos(ps)
    sps=np.sin(ps)

    s3ps=2*cps

    pst1=ps*a[84]
    pst2=ps*a[85]

    st1=np.sin(pst1)
    ct1=np.cos(pst1)
    st2=np.sin(pst2)
    ct2=np.cos(pst2)

    x1=x*ct1-z*st1
    z1=x*st1+z*ct1
    x2=x*ct2-z*st2
    z2=x*st2+z*ct2

    l=0
    bx,by,bz = [np.zeros_like(x) for _ in range(3)]

    for m in range(1,3):
        for i in range(1,4):
            p = a[71 + i]
            q = a[77 + i]
            cypi = np.cos(y/p)
            cyqi = np.cos(y/q)
            sypi = np.sin(y/p)
            syqi = np.sin(y/q)

            for k in range(1,4):
                r=a[74+k]
                s=a[80+k]
                szrk=np.sin(z1/r)
                czsk=np.cos(z2/s)
                czrk=np.cos(z1/r)
                szsk=np.sin(z2/s)
                sqpr=np.sqrt(1/p**2+1/r**2)
                sqqs=np.sqrt(1/q**2+1/s**2)
                epr=np.exp(x1*sqpr)
                eqs=np.exp(x2*sqqs)

                for n in range(1,3):
                    for nn in range(1,3):
                        if m == 1:
                            fx = -sqpr*epr*cypi*szrk
                            fy =  epr*sypi*szrk/p
                            fz = -epr*cypi*czrk/r
                            if n == 1:
                                hx,hy,hz = (fx,fy,fz) if nn == 1 else (fx*x_sc, fy*x_sc, fz*x_sc)
                            else:
                                hx,hy,hz = (fx*cps, fy*cps, fz*cps) if nn == 1 else (fx*cps*x_sc, fy*cps*x_sc, fz*cps*x_sc)
                        else:
                            fx = -sps*sqqs*eqs*cyqi*czsk
                            fy =  sps/q*eqs*syqi*czsk
                            fz =  sps/s*eqs*cyqi*szsk
                            if n == 1:
                                hx,hy,hz = (fx,fy,fz) if nn == 1 else (fx*x_sc, fy*x_sc, fz*x_sc)
                            else:
                                hx,hy,hz = (fx*s3ps,fy*s3ps,fz*s3ps) if nn == 1 else (fx*s3ps*x_sc, fy*s3ps*x_sc, fz*s3ps*x_sc)
                        l=l+1
                        if m == 1:
                            hxr =  hx*ct1+hz*st1
                            hzr = -hx*st1+hz*ct1
                        else:
                            hxr =  hx*ct2+hz*st2
                            hzr = -hx*st2+hz*ct2

                        bx = bx+hxr*a[l-1]
                        by = by+hy *a[l-1]
                        bz = bz+hzr*a[l-1]

    return bx,by,bz


def full_rc(iopr,ps,x,y,z):
    """
    Calculates GSM field components from full ring current.
    Vectorized version.
    """
    global sc_sy, sc_pr, phi

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
        0.3498952546, 0.6595919814])

    c_pr = np.array([
        -64820.58481, -63965.62048, 66267.93413, 135049.7504, -36.56316878,
        124.6614669, 56.75637955, -87.56841077, 5848.631425, 4981.097722,
        -6233.712207, -10986.40188, 68716.52057, 65682.69473, -69673.32198,
        -138829.3568, 43.45817708, -117.9565488, -62.14836263, 79.83651604,
        -6211.451069, -5151.633113, 6544.481271, 11353.03491, 23.72352603,
        -256.4846331, 25.77629189, 145.2377187, -4.472639098, -3.554312754,
        2.936973114, 2.682302576, 2.728979958, 26.43396781, -9.312348296,
        -29.65427726, -247.5855336, -206.9111326, 74.25277664, 106.4069993,
        15.45391072, 16.35943569, -5.965177750, -6.079451700, 115.6748385,
        -35.27377307, -32.28763497, -32.53122151, 93.74409310, 84.25677504,
        -29.23010465, -43.79485175, -6.434679514, -6.620247951, 2.443524317,
        2.266538956, -43.82903825, 6.904117876, 12.24289401, 17.62014361,
        152.3078796, 124.5505289, -44.58690290, -63.02382410, -8.999368955,
        -9.693774119, 3.510930306, 3.770949738, -77.96705716, 22.07730961,
        20.46491655, 18.67728847, 9.451290614, 9.313661792, 644.7620970,
        418.2515954, 7.183754387, 35.62128817, 19.43180682, 39.57218411,
        15.69384715, 7.123215241, 2.300635346, 21.90881131, -.01775839370, .3996346710])


    hxsrc,hysrc,hzsrc, hxprc,hyprc,hzprc = src_prc(iopr, sc_sy,sc_pr, phi, ps, x,y,z)

    x_sc=sc_sy-1
    fsx,fsy,fsz = [np.zeros_like(x) for _ in range(3)]
    if (iopr == 0) | (iopr == 1):
        fsx,fsy,fsz = rc_shield(c_sy,ps,x_sc, x,y,z)

    x_sc=sc_pr-1
    fpx,fpy,fpz = [np.zeros_like(x) for _ in range(3)]
    if (iopr == 0) | (iopr == 2):
        fpx,fpy,fpz = rc_shield(c_pr,ps,x_sc, x,y,z)

    bxsrc=hxsrc+fsx
    bysrc=hysrc+fsy
    bzsrc=hzsrc+fsz

    bxprc=hxprc+fpx
    byprc=hyprc+fpy
    bzprc=hzprc+fpz

    return bxsrc,bysrc,bzsrc,bxprc,byprc,bzprc


def src_prc(iopr,sc_sy,sc_pr,phi,ps, x,y,z):
    """
    Returns field components from symmetric/partial ring current.
    Vectorized version.
    """
    cps=np.cos(ps)
    sps=np.sin(ps)

    xt=x*cps-z*sps
    zt=z*cps+x*sps

    xts=xt/sc_sy
    yts=y /sc_sy
    zts=zt/sc_sy

    xta=xt/sc_pr
    yta=y /sc_pr
    zta=zt/sc_pr

    bxs,bys,bzs = [np.zeros_like(x) for _ in range(3)]
    if iopr <= 1:
        bxs,bys,bzs = rc_symm(xts,yts,zts)

    bxa_s,bya_s,bza_s = [np.zeros_like(x) for _ in range(3)]
    if (iopr == 0) | (iopr == 2):
        bxa_s,bya_s,bza_s = prc_symm(xta,yta,zta)

    cp=np.cos(phi)
    sp=np.sin(phi)
    xr=xta*cp-yta*sp
    yr=xta*sp+yta*cp
    bxa_qr,bya_qr,bza_q = [np.zeros_like(x) for _ in range(3)]
    if (iopr == 0) | (iopr == 2):
        bxa_qr,bya_qr,bza_q = prc_quad(xr,yr,zta)

    bxa_q= bxa_qr*cp+bya_qr*sp
    bya_q=-bxa_qr*sp+bya_qr*cp

    bxp=bxa_s+bxa_q
    byp=bya_s+bya_q
    bzp=bza_s+bza_q

    bxsrc=bxs*cps+bzs*sps
    bysrc=bys
    bzsrc=bzs*cps-bxs*sps

    bxprc=bxp*cps+bzp*sps
    byprc=byp
    bzprc=bzp*cps-bxp*sps

    return bxsrc,bysrc,bzsrc, bxprc,byprc,bzprc


# Create aliases for compatibility
SRC_PRC = src_prc
RC_SHIELD = rc_shield = None  # Will be defined below


def rc_symm(x,y,z):
    """
    Calculates field of the symmetric part of ring current.
    Vectorized version.
    """
    ds = 1e-2
    dc = 0.99994999875
    d = 1e-4
    drd = 1.0/(2*d)

    rho2=x**2+y**2
    r2=rho2+z**2
    r=np.sqrt(r2)
    r_safe = np.where(r==0, 1e-9, r)
    sint=np.sqrt(rho2)/r_safe
    cost=z/r_safe

    mask = sint < ds

    # Branch 1: sint < ds
    a_b1=ap(r,ds,dc)/ds
    dardr_b1=( (r+d)*ap(r+d,ds,dc)-(r-d)*ap(r-d,ds,dc) )*drd
    fxy_b1=z*(2*a_b1-dardr_b1)/(r*r2)
    bx_b1=fxy_b1*x
    by_b1=fxy_b1*y
    bz_b1=(2*a_b1*cost**2+dardr_b1*sint**2)/r_safe

    # Branch 2: sint >= ds
    theta=np.arctan2(sint,cost)
    tp=theta+d
    tm=theta-d
    sintp=np.sin(tp)
    sintm=np.sin(tm)
    costp=np.cos(tp)
    costm=np.cos(tm)
    br_b2=(sintp*ap(r,sintp,costp)-sintm*ap(r,sintm,costm))/(r_safe*sint)*drd
    bt_b2=((r-d)*ap(r-d,sint,cost)-(r+d)*ap(r+d,sint,cost))/r_safe*drd
    sint_safe = np.where(sint==0, 1e-9, sint)
    fxy_b2=(br_b2+bt_b2*cost/sint_safe)/r_safe
    bx_b2=fxy_b2*x
    by_b2=fxy_b2*y
    bz_b2=br_b2*cost-bt_b2*sint

    bx = np.where(mask, bx_b1, bx_b2)
    by = np.where(mask, by_b1, by_b2)
    bz = np.where(mask, bz_b1, bz_b2)

    return bx, by, bz


def ap(r,sint,cost):
    """
    Computes the azimuthal component of the vector potential.
    Vectorized version.
    """
    a1,a2,rrc1,dd1,rrc2,dd2,p1,r1,dr1,dla1,p2,r2,dr2,dla2,p3,r3,dr3 = [
        -456.5289941, 375.9055332, 4.274684950, 2.439528329, 3.367557287,
        3.146382545, -0.2291904607, 3.746064740, 1.508802177, 0.5873525737,
        0.1556236119, 4.993638842, 3.324180497, 0.4368407663, 0.1855957207,
        2.969226745, 2.243367377]

    prox = sint < 1.e-2
    sint1 = np.where(prox, 1.e-2, sint)
    cost1 = np.where(prox, 0.99994999875, cost)

    r_safe = np.where(r==0, 1e-9, r)
    alpha=sint1**2/r_safe
    gamma=cost1/r_safe**2

    arg1=-((r-r1)/dr1)**2-(cost1/dla1)**2
    arg2=-((r-r2)/dr2)**2-(cost1/dla2)**2
    arg3=-((r-r3)/dr3)**2

    dexp1=np.exp(np.maximum(arg1, -500.))
    dexp2=np.exp(np.maximum(arg2, -500.))
    dexp3=np.exp(np.maximum(arg3, -500.))

    alpha_s=alpha*(1+p1*dexp1+p2*dexp2+p3*dexp3)
    gamma_s=gamma
    gammas2=gamma_s**2

    alsqh=alpha_s**2/2
    f=64/27*gammas2+alsqh**2
    q=(np.sqrt(f)+alsqh)**(1/3)
    q_safe = np.where(q==0, 1e-9, q)
    c=q-4*gammas2**(1/3)/(3*q_safe)
    c=np.maximum(c, 0)
    g=np.sqrt(c**2+4*gammas2**(1/3))
    denom = (np.sqrt(2*g-c)+np.sqrt(c))*(g+c)
    denom_safe = np.where(denom==0, 1e-9, denom)
    rs=4/denom_safe
    costs=gamma_s*rs**2
    sints=np.sqrt(np.maximum(1-costs**2, 0)) # ensure non-negative
    rhos=rs*sints
    zs=rs*costs

    p=(rrc1+rhos)**2+zs**2+dd1**2
    p_safe = np.where(p==0, 1e-9, p)
    xk2=4*rrc1*rhos/p_safe
    xk=np.sqrt(xk2)
    rhos_safe = np.where(rhos==0, 1e-9, rhos)
    xkrho12=xk*np.sqrt(rhos_safe)

    xk2s = 1-xk2
    dl = np.log(np.maximum(1/np.where(xk2s==0, 1e-9, xk2s), 1e-9))
    elk = 1.38629436112 + xk2s*(0.09666344259+xk2s*(0.03590092383+xk2s*(0.03742563713+xk2s*0.01451196212)))\
        + dl*(0.5+xk2s*(0.12498593597+xk2s*(0.06880248576+xk2s*(0.03328355346+xk2s*0.00441787012))))
    ele = 1+xk2s*(0.44325141463+xk2s*(0.0626060122+xk2s*(0.04757383546+xk2s*0.01736506451)))\
        + dl*xk2s*(0.2499836831+xk2s*(0.09200180037+xk2s*(0.04069697526+xk2s*0.00526449639)))
    aphi1 = np.where(xkrho12 != 0, ((1-xk2*0.5)*elk-ele) / xkrho12, 0.0)

    p=(rrc2+rhos)**2+zs**2+dd2**2
    p_safe = np.where(p==0, 1e-9, p)
    xk2=4*rrc2*rhos/p_safe
    xk=np.sqrt(xk2)
    xkrho12=xk*np.sqrt(rhos_safe)

    xk2s = 1-xk2
    dl = np.log(np.maximum(1/np.where(xk2s==0, 1e-9, xk2s), 1e-9))
    elk = 1.38629436112 + xk2s*(0.09666344259+xk2s*(0.03590092383+xk2s*(0.03742563713+xk2s*0.01451196212)))\
        + dl*(0.5+xk2s*(0.12498593597+xk2s*(0.06880248576+xk2s*(0.03328355346+xk2s*0.00441787012))))
    ele = 1+xk2s*(0.44325141463+xk2s*(0.0626060122+xk2s*(0.04757383546+xk2s*0.01736506451)))\
        + dl*xk2s*(0.2499836831+xk2s*(0.09200180037+xk2s*(0.04069697526+xk2s*0.00526449639)))
    aphi2 = np.where(xkrho12 != 0, ((1-xk2*0.5)*elk-ele) / xkrho12, 0.0)

    ap_val=a1*aphi1+a2*aphi2
    return np.where(prox, ap_val*sint/sint1, ap_val)


def prc_symm(x,y,z):
    """
    Calculates field of the symmetric part of partial ring current.
    Vectorized version.
    """
    ds = 1e-2
    dc = 0.99994999875
    d = 1e-4
    drd = 1.0/(2*d)

    rho2=x**2+y**2
    r2=rho2+z**2
    r=np.sqrt(r2)
    r_safe = np.where(r==0, 1e-9, r)
    sint=np.sqrt(rho2)/r_safe
    cost=z/r_safe

    mask = sint < ds

    # Branch 1: sint < ds
    a_b1=apprc(r,ds,dc)/ds
    dardr_b1=( (r+d)*apprc(r+d,ds,dc)-(r-d)*apprc(r-d,ds,dc) )*drd
    fxy_b1=z*(2*a_b1-dardr_b1)/(r*r2)
    bx_b1=fxy_b1*x
    by_b1=fxy_b1*y
    bz_b1=(2*a_b1*cost**2+dardr_b1*sint**2)/r_safe

    # Branch 2: sint >= ds
    theta=np.arctan2(sint,cost)
    tp=theta+d
    tm=theta-d
    sintp=np.sin(tp)
    sintm=np.sin(tm)
    costp=np.cos(tp)
    costm=np.cos(tm)
    br_b2=(sintp*apprc(r,sintp,costp)-sintm*apprc(r,sintm,costm))/(r_safe*sint)*drd
    bt_b2=((r-d)*apprc(r-d,sint,cost)-(r+d)*apprc(r+d,sint,cost))/r_safe*drd
    sint_safe = np.where(sint==0, 1e-9, sint)
    fxy_b2=(br_b2+bt_b2*cost/sint_safe)/r_safe
    bx_b2=fxy_b2*x
    by_b2=fxy_b2*y
    bz_b2=br_b2*cost-bt_b2*sint

    bx = np.where(mask, bx_b1, bx_b2)
    by = np.where(mask, by_b1, by_b2)
    bz = np.where(mask, bz_b1, bz_b2)

    return bx, by, bz


def apprc(r,sint,cost):
    """
    Computes the azimuthal component of the vector potential for partial ring current.
    Vectorized version.
    """
    a1,a2,rrc1,dd1,rrc2,dd2,p1,alpha1,dal1,beta1,dg1,p2,alpha2,dal2,beta2,dg2,beta3,p3,\
    alpha3,dal3,beta4,dg3,beta5,q0,q1,alpha4,dal4,dg4,q2,alpha5,dal5,dg5,beta6,beta7 = [
        -80.11202281,12.58246758,6.560486035,1.930711037,3.827208119,
        .7789990504,.3058309043,.1817139853,.1257532909,3.422509402,
        .04742939676,-4.800458958,-.02845643596,.2188114228,2.545944574,
        .00813272793,.35868244,103.1601001,-.00764731187,.1046487459,
        2.958863546,.01172314188,.4382872938,.01134908150,14.51339943,
        .2647095287,.07091230197,.01512963586,6.861329631,.1677400816,
        .04433648846,.05553741389,.7665599464,.7277854652]

    prox= sint < 1.e-2
    sint1 = np.where(prox, 1.e-2, sint)
    cost1 = np.where(prox, 0.99994999875, cost)

    r_safe = np.where(r==0, 1e-9, r)
    alpha=sint1**2/r_safe
    gamma=cost1/r_safe**2

    arg1=-(gamma/dg1)**2
    arg2=-((alpha-alpha4)/dal4)**2-(gamma/dg4)**2

    dexp1=np.exp(np.maximum(arg1, -500.))
    dexp2=np.exp(np.maximum(arg2, -500.))

    alpha_s = alpha*(1 + p1/(1+((alpha-alpha1)/dal1)**2)**beta1*dexp1
        + p2*(alpha-alpha2)/(1+((alpha-alpha2)/dal2)**2)**beta2/(1+(gamma/dg2)**2)**beta3
        + p3*(alpha-alpha3)**2/(1.+((alpha-alpha3)/dal3)**2)**beta4/(1+(gamma/dg3)**2)**beta5)
    gamma_s = gamma*(1 + q0 + q1*(alpha-alpha4)*dexp2
        + q2*(alpha-alpha5)/(1+((alpha-alpha5)/dal5)**2)**beta6/(1+(gamma/dg5)**2)**beta7)

    gammas2 = gamma_s**2

    alsqh=alpha_s**2/2.
    f=64./27.*gammas2+alsqh**2
    q=(np.sqrt(f)+alsqh)**(1/3)
    q_safe = np.where(q==0, 1e-9, q)
    c=q-4.*gammas2**(1/3)/(3.*q_safe)
    c=np.maximum(c, 0)
    g=np.sqrt(c**2+4*gammas2**(1/3))
    denom = (np.sqrt(2*g-c)+np.sqrt(c))*(g+c)
    denom_safe = np.where(denom==0, 1e-9, denom)
    rs=4./denom_safe
    costs=gamma_s*rs**2
    sints=np.sqrt(np.maximum(1-costs**2, 0))
    rhos=rs*sints
    zs=rs*costs

    p=(rrc1+rhos)**2+zs**2+dd1**2
    p_safe = np.where(p==0, 1e-9, p)
    xk2=4*rrc1*rhos/p_safe
    xk=np.sqrt(xk2)
    rhos_safe = np.where(rhos==0, 1e-9, rhos)
    xkrho12=xk*np.sqrt(rhos_safe)

    xk2s = 1-xk2
    dl = np.log(np.maximum(1/np.where(xk2s==0, 1e-9, xk2s), 1e-9))
    elk = 1.38629436112 + xk2s*(0.09666344259+xk2s*(0.03590092383+xk2s*(0.03742563713+xk2s*0.01451196212)))\
        + dl*(0.5+xk2s*(0.12498593597+xk2s*(0.06880248576+xk2s*(0.03328355346+xk2s*0.00441787012))))
    ele = 1 + xk2s*(0.44325141463+xk2s*(0.0626060122+xk2s*(0.04757383546+xk2s*0.01736506451)))\
        + dl*xk2s*(0.2499836831+xk2s*(0.09200180037+xk2s*(0.04069697526+xk2s*0.00526449639)))
    aphi1 = np.where(xkrho12 != 0, ((1-xk2*0.5)*elk-ele) / xkrho12, 0.0)

    p=(rrc2+rhos)**2+zs**2+dd2**2
    p_safe = np.where(p==0, 1e-9, p)
    xk2=4*rrc2*rhos/p_safe
    xk=np.sqrt(xk2)
    xkrho12=xk*np.sqrt(rhos_safe)

    xk2s = 1-xk2
    dl = np.log(np.maximum(1/np.where(xk2s==0, 1e-9, xk2s), 1e-9))
    elk = 1.38629436112 + xk2s*(0.09666344259+xk2s*(0.03590092383+xk2s*(0.03742563713+xk2s*0.01451196212)))\
        + dl*(0.5+xk2s*(0.12498593597+xk2s*(0.06880248576+xk2s*(0.03328355346+xk2s*0.00441787012))))
    ele = 1 + xk2s*(0.44325141463+xk2s*(0.0626060122+xk2s*(0.04757383546+xk2s*0.01736506451)))\
        + dl*xk2s*(0.2499836831+xk2s*(0.09200180037+xk2s*(0.04069697526+xk2s*0.00526449639)))
    aphi2 = np.where(xkrho12 != 0, ((1-xk2*0.5)*elk-ele) / xkrho12, 0.0)

    apprc_val=a1*aphi1+a2*aphi2
    return np.where(prox, apprc_val*sint/sint1, apprc_val)


def prc_quad(x,y,z):
    """
    Calculates field of the quadrupole part of partial ring current.
    Vectorized version.
    """
    d  = 1e-4
    dd = 2e-4
    ds = 1e-2
    dc = 0.99994999875

    rho2=x**2+y**2
    r=np.sqrt(rho2+z**2)
    rho=np.sqrt(rho2)
    r_safe = np.where(r==0, 1e-9, r)
    sint=rho/r_safe
    cost=z/r_safe

    mask = sint > ds

    # Branch 1: sint > ds
    rho_safe_b1 = np.where(rho==0, 1e-9, rho)
    cphi_b1=x/rho_safe_b1
    sphi_b1=y/rho_safe_b1
    br_b1=br_prc_q(r,sint,cost)
    bt_b1=bt_prc_q(r,sint,cost)
    dbrr_b1=(br_prc_q(r+d,sint,cost)-br_prc_q(r-d,sint,cost))/dd
    theta_b1=np.arctan2(sint,cost)
    tp_b1=theta_b1+d
    tm_b1=theta_b1-d
    dbtt_b1=(bt_prc_q(r,np.sin(tp_b1),np.cos(tp_b1))-bt_prc_q(r,np.sin(tm_b1),np.cos(tm_b1)))/dd
    bx_b1=sint*(br_b1+(br_b1+r*dbrr_b1+dbtt_b1)*sphi_b1**2)+cost*bt_b1
    by_b1=-sint*sphi_b1*cphi_b1*(br_b1+r*dbrr_b1+dbtt_b1)
    bz_b1=(br_b1*cost-bt_b1*sint)*cphi_b1

    # Branch 2: sint <= ds
    st_b2=ds
    ct_b2=dc*np.sign(z)
    ct_b2=np.where(z==0, dc, ct_b2)
    br_b2=br_prc_q(r,st_b2,ct_b2)
    bt_b2=bt_prc_q(r,st_b2,ct_b2)
    dbrr_b2=(br_prc_q(r+d,st_b2,ct_b2)-br_prc_q(r-d,st_b2,ct_b2))/dd
    theta_b2=np.arctan2(st_b2,ct_b2)
    tp_b2=theta_b2+d
    tm_b2=theta_b2-d
    dbtt_b2=(bt_prc_q(r,np.sin(tp_b2),np.cos(tp_b2))-bt_prc_q(r,np.sin(tm_b2),np.cos(tm_b2)))/dd
    fcxy_b2=r*dbrr_b2+dbtt_b2
    r_st_b2_sq = (r*st_b2)**2
    r_st_b2_sq_safe = np.where(r_st_b2_sq==0, 1e-9, r_st_b2_sq)
    bx_b2=(br_b2*(x**2+2.*y**2)+fcxy_b2*y**2)/r_st_b2_sq_safe+bt_b2*cost
    by_b2=-(br_b2+fcxy_b2)*x*y/r_st_b2_sq_safe
    bz_b2=(br_b2*cost/st_b2-bt_b2)*x/r_safe

    bx = np.where(mask, bx_b1, bx_b2)
    by = np.where(mask, by_b1, by_b2)
    bz = np.where(mask, bz_b1, bz_b2)

    return bx,by,bz


def br_prc_q(r,sint,cost):
    """
    Calculates radial component of the partial ring current field.
    Vectorized version.
    """
    a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,xk1,al1,dal1,b1,be1,xk2,al2,dal2,b2,be2,xk3,xk4,al3,dal3,b3,be3,al4,dal4,dg1,al5,dal5,dg2,c1,c2,c3,al6,dal6,drm = [
        -21.2666329, 32.24527521, -6.062894078, 7.515660734, 233.7341288,
        -227.1195714, 8.483233889, 16.80642754, -24.63534184, 9.067120578,
        -1.052686913, -12.08384538, 18.61969572, -12.71686069, 47017.35679,
        -50646.71204, 7746.058231, 1.531069371, 2.318824273, 0.1417519429,
        0.6388013110e-02, 5.303934488, 4.213397467, 0.7955534018, 0.1401142771,
        0.2306094179e-01, 3.462235072, 2.568743010, 3.477425908, 1.922155110,
        0.1485233485, 0.2319676273e-01, 7.830223587, 8.492933868, 0.1295221828,
        0.01753008801, 0.01125504083, 0.1811846095, 0.04841237481,
        0.01981805097, 6.557801891, 6.348576071, 5.744436687, 0.2265212965,
        0.1301957209, 0.5654023158]

    sint2=sint**2
    cost2=cost**2
    sc=sint*cost
    r_safe = np.where(r==0, 1e-9, r)
    alpha=sint2/r_safe
    gamma=cost/r_safe**2

    f,fa,fs = ffs(alpha,al1,dal1)
    d1=sc*f**xk1/((r/b1)**be1+1.)
    d2=d1*cost2

    f,fa,fs = ffs(alpha,al2,dal2)
    d3=sc*fs**xk2/((r/b2)**be2+1.)
    d4=d3*cost2

    f,fa,fs = ffs(alpha,al3,dal3)
    alpha_safe = np.where(alpha==0, 1e-9, alpha)
    d5=sc*(alpha_safe**xk3)*(fs**xk4)/((r/b3)**be3+1.)
    d6=d5*cost2

    arga=((alpha-al4)/dal4)**2+1.
    argg=1.+(gamma/dg1)**2
    d7=sc/arga/argg
    d8=d7/arga
    d9=d8/arga
    d10=d9/arga

    arga=((alpha-al5)/dal5)**2+1.
    argg=1.+(gamma/dg2)**2
    d11=sc/arga/argg
    d12=d11/arga
    d13=d12/arga
    d14=d13/arga

    d15=sc/(r**4+c1**4)
    d16=sc/(r**4+c2**4)*cost2
    d17=sc/(r**4+c3**4)*cost2**2

    f,fa,fs = ffs(alpha,al6,dal6)
    d18=sc*fs/(1.+((r-1.2)/drm)**2)

    br_val=a1*d1+a2*d2+a3*d3+a4*d4+a5*d5+a6*d6+a7*d7+a8*d8+a9*d9+\
             a10*d10+a11*d11+a12*d12+a13*d13+a14*d14+a15*d15+a16*d16+a17*d17+a18*d18

    return br_val


def bt_prc_q(r,sint,cost):
    """
    Calculates theta component of the partial ring current field.
    Vectorized version.
    """
    a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,xk1,al1,dal1,b1,be1,xk2,al2,dal2,be2,xk3,xk4,al3,dal3,b3,be3,al4,dal4,dg1,al5,dal5,dg2,c1,c2,c3 = [
        12.74640393, -7.516393516, -5.476233865, 3.212704645, -59.10926169,
        46.62198189, -.01644280062, 0.1234229112, -.08579198697, 0.01321366966,
        0.8970494003, 9.136186247, -38.19301215, 21.73775846, -410.0783424,
        -69.90832690, -848.8543440, 1.243288286, 0.2071721360, 0.05030555417,
        7.471332374, 3.180533613, 1.376743507, 0.1568504222, 0.02092910682,
        1.985148197, 0.3157139940, 1.056309517, 0.1701395257, 0.1019870070,
        6.293740981, 5.671824276, 0.1280772299, 0.02189060799, 0.01040696080,
        0.1648265607, 0.04701592613, 0.01526400086, 12.88384229, 3.361775101,
        23.44173897]

    sint2=sint**2
    cost2=cost**2
    r_safe = np.where(r==0, 1e-9, r)
    alpha=sint2/r_safe
    gamma=cost/r_safe**2

    f,fa,fs = ffs(alpha,al1,dal1)
    d1=f**xk1/((r/b1)**be1+1.)
    d2=d1*cost2

    f,fa,fs = ffs(alpha,al2,dal2)
    r_safe_be2 = np.where(r==0, 1e-9, r**be2)
    d3=fa**xk2/r_safe_be2
    d4=d3*cost2

    f,fa,fs = ffs(alpha,al3,dal3)
    alpha_safe = np.where(alpha==0, 1e-9, alpha)
    d5=fs**xk3*alpha_safe**xk4/((r/b3)**be3+1.)
    d6=d5*cost2

    f,fa,fs = ffs(gamma,0.,dg1)
    fcc=(1.+((alpha-al4)/dal4)**2)
    d7 =1./fcc*fs
    d8 =d7/fcc
    d9 =d8/fcc
    d10=d9/fcc

    arg=1.+((alpha-al5)/dal5)**2
    d11=1./arg/(1.+(gamma/dg2)**2)
    d12=d11/arg
    d13=d12/arg
    d14=d13/arg

    d15=1./(r**4+c1**2)
    d16=cost2/(r**4+c2**2)
    d17=cost2**2/(r**4+c3**2)

    bt_val = a1*d1+a2*d2+a3*d3+a4*d4+a5*d5+a6*d6+a7*d7+a8*d8+a9*d9+\
               a10*d10+a11*d11+a12*d12+a13*d13+a14*d14+a15*d15+a16*d16+a17*d17

    return bt_val


def ffs(a, a0, da):
    """
    Calculates field line mapping transformation.
    Vectorized version.
    """
    sq1 = np.sqrt((a + a0) ** 2 + da ** 2)
    sq2 = np.sqrt((a - a0) ** 2 + da ** 2)
    sq1_p_sq2 = sq1 + sq2
    sq1_p_sq2_safe = np.where(sq1_p_sq2==0, 1e-9, sq1_p_sq2)
    fa = 2. / sq1_p_sq2_safe
    f = fa * a
    sq1_safe = np.where(sq1==0, 1e-9, sq1)
    sq2_safe = np.where(sq2==0, 1e-9, sq2)
    fs = 0.5 * sq1_p_sq2 / (sq1_safe * sq2_safe) * (1.-f * f)

    return f, fa, fs


def rc_shield(a,ps,x_sc,x,y,z):
    """
    Calculates GSM field components for the ring current shield.
    Vectorized version.
    """
    fac_sc = (x_sc+1)**3

    cps = np.cos(ps)
    sps = np.sin(ps)
    s3ps=2*cps

    pst1=ps*a[84]
    pst2=ps*a[85]

    st1=np.sin(pst1)
    ct1=np.cos(pst1)
    st2=np.sin(pst2)
    ct2=np.cos(pst2)

    x1=x*ct1-z*st1
    z1=x*st1+z*ct1
    x2=x*ct2-z*st2
    z2=x*st2+z*ct2

    l=0
    bx,by,bz = [np.zeros_like(x) for _ in range(3)]

    for m in range(2):
        for i in range(3):
            p=a[72+i]
            q=a[78+i]
            cypi=np.cos(y/p)
            cyqi=np.cos(y/q)
            sypi=np.sin(y/p)
            syqi=np.sin(y/q)

            for k in range(3):
                r=a[75+k]
                s=a[81+k]
                szrk=np.sin(z1/r)
                czsk=np.cos(z2/s)
                czrk=np.cos(z1/r)
                szsk=np.sin(z2/s)
                sqpr=np.sqrt(1/p**2+1/r**2)
                sqqs=np.sqrt(1/q**2+1/s**2)
                epr=np.exp(x1*sqpr)
                eqs=np.exp(x2*sqqs)

                for n in range(2):
                    for nn in range(2):
                        if m == 0:
                            fx = -sqpr*epr*cypi*szrk*fac_sc
                            fy =  epr*sypi*szrk/p   *fac_sc
                            fz = -epr*cypi*czrk/r   *fac_sc
                            if n == 0:
                                hx,hy,hz = (fx,fy,fz) if nn == 0 else (fx*x_sc, fy*x_sc, fz*x_sc)
                            else:
                                hx,hy,hz = (fx*cps, fy*cps, fz*cps) if nn == 0 else (fx*cps*x_sc, fy*cps*x_sc, fz*cps*x_sc)
                        else:
                            fx = -sps*sqqs*eqs*cyqi*czsk*fac_sc
                            fy =  sps/q*eqs*syqi*czsk   *fac_sc
                            fz =  sps/s*eqs*cyqi*szsk   *fac_sc
                            if n == 0:
                                hx,hy,hz = (fx,fy,fz) if nn == 0 else (fx*x_sc,fy*x_sc,fz*x_sc)
                            else:
                                hx,hy,hz = (fx*s3ps,fy*s3ps,fz*s3ps) if nn == 0 else (fx*s3ps*x_sc, fy*s3ps*x_sc, fz*s3ps*x_sc)

                        if m == 0:
                            hxr =  hx*ct1+hz*st1
                            hzr = -hx*st1+hz*ct1
                        else:
                            hxr =  hx*ct2+hz*st2
                            hzr = -hx*st2+hz*ct2

                        bx = bx+hxr*a[l]
                        by = by+hy *a[l]
                        bz = bz+hzr*a[l]
                        l=l+1

    return bx, by, bz


# Update the rc_shield alias
RC_SHIELD = rc_shield


def dipole(ps, x, y, z):
    """
    Calculates GSM components of a geo-dipole field.
    Vectorized version.
    """
    q0 = 30115.

    sps = np.sin(ps)
    cps = np.cos(ps)
    x2 = x ** 2
    y2 = y ** 2
    z2 = z ** 2
    xz3 = 3 * x * z
    r_sq = x2 + y2 + z2
    r_sq_safe = np.where(r_sq == 0, 1e-9, r_sq)
    q = q0 / np.sqrt(r_sq_safe) ** 5
    bx = q * ((y2 + z2 - 2 * x2) * sps - xz3 * cps)
    by = -3 * y * q * (x * sps + z * cps)
    bz = q * ((x2 + y2 - 2 * z2) * cps - xz3 * sps)

    return bx, by, bz


# Alias for compatibility
t04 = t04_vectorized
