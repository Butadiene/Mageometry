"""
Vectorized implementation of the T01 Birkeland current system.

This module provides vectorized versions of the Birkeland field-aligned
current calculations for regions 1 and 2.
"""

import numpy as np
from typing import Tuple, Union


def r_s_vectorized(a: np.ndarray, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Vectorized deformation function for radial coordinate.
    
    Parameters
    ----------
    a : ndarray
        Coefficient array (31 elements)
    r : ndarray
        Radial coordinate
    theta : ndarray
        Polar angle
        
    Returns
    -------
    rs : ndarray
        Deformed radial coordinate
    """
    # Ensure arrays
    r = np.atleast_1d(r)
    theta = np.atleast_1d(theta)
    
    # Calculate terms
    term1 = r + a[1] / r
    term2 = a[2] * r / np.sqrt(r**2 + a[10]**2)
    term3 = a[3] * r / (r**2 + a[11]**2)
    
    # Theta-dependent terms
    cos_theta = np.cos(theta)
    term4 = a[4] + a[5] / r
    term5 = a[6] * r / np.sqrt(r**2 + a[12]**2)
    term6 = a[7] * r / (r**2 + a[13]**2)
    
    # 2*theta terms
    cos_2theta = np.cos(2 * theta)
    term7 = a[8] * r / np.sqrt(r**2 + a[14]**2)
    term8 = a[9] * r / (r**2 + a[15]**2)**2
    
    # Combine all terms
    rs = (term1 + term2 + term3 + 
          (term4 + term5 + term6) * cos_theta +
          (term7 + term8) * cos_2theta)
    
    return rs


def theta_s_vectorized(a: np.ndarray, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Vectorized deformation function for polar angle.
    
    Parameters
    ----------
    a : ndarray
        Coefficient array (31 elements)  
    r : ndarray
        Radial coordinate
    theta : ndarray
        Polar angle
        
    Returns
    -------
    thetas : ndarray
        Deformed polar angle
    """
    # Ensure arrays
    r = np.atleast_1d(r)
    theta = np.atleast_1d(theta)
    
    # sin(theta) terms
    sin_theta = np.sin(theta)
    term1 = a[16] + a[17] / r + a[18] / r**2
    term2 = a[19] * r / np.sqrt(r**2 + a[26]**2)
    
    # sin(2*theta) terms
    sin_2theta = np.sin(2 * theta)
    term3 = a[20]
    term4 = a[21] * r / np.sqrt(r**2 + a[27]**2)
    term5 = a[22] * r / (r**2 + a[28]**2)
    
    # sin(3*theta) terms
    sin_3theta = np.sin(3 * theta)
    term6 = a[23] + a[24] / r
    term7 = a[25] * r / (r**2 + a[29]**2)
    
    # Combine all terms
    thetas = (theta + 
              (term1 + term2) * sin_theta +
              (term3 + term4 + term5) * sin_2theta +
              (term6 + term7) * sin_3theta)
    
    return thetas


def fialcos_vectorized(r: np.ndarray, theta: np.ndarray, phi: np.ndarray, 
                      n: int, theta0: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized conical model of Birkeland current field.
    
    Based on the scalar fialcos function. Calculates the field-aligned current
    system in spherical coordinates with radial currents only (br=0).
    
    Parameters
    ----------
    r : ndarray
        Radial coordinate
    theta : ndarray
        Polar angle
    phi : ndarray
        Azimuthal angle
    n : int
        Number of modes to compute (n <= 10)
    theta0 : float
        Angular half-width of the cone
    dt : float
        Angular half-width of the current layer
        
    Returns
    -------
    btheta, bphi : ndarray
        Field components in spherical coordinates (scaled by 800)
    """
    # Ensure arrays
    r = np.atleast_1d(r)
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)
    
    # Handle scalar input
    scalar_input = r.size == 1
    
    # Initialize output arrays
    shape = r.shape
    btheta = np.zeros(shape)
    bphi = np.zeros(shape)
    
    # Calculate basic quantities
    sinte = np.sin(theta)
    ro = r * sinte
    coste = np.cos(theta)
    sinfi = np.sin(phi)
    cosfi = np.cos(phi)
    
    # tan(theta/2) and cot(theta/2) with safe division
    tg = np.divide(sinte, 1 + coste, out=np.zeros_like(sinte), where=(1 + coste) != 0)
    ctg = np.divide(sinte, 1 - coste, out=np.zeros_like(sinte), where=(1 - coste) != 0)
    
    # Current sheet boundaries
    tetanp = theta0 + dt
    tetanm = theta0 - dt
    
    # Pre-calculate boundary tangents
    tgp = np.tan(tetanp * 0.5)
    tgm = np.tan(tetanm * 0.5)
    tgm2 = tgm * tgm
    tgp2 = tgp * tgp
    
    # Initialize mode arrays
    btn = np.zeros((n, *shape))
    bpn = np.zeros((n, *shape))
    
    # Initialize recursion variables
    cosm1 = np.ones(shape)
    sinm1 = np.zeros(shape)
    tm = np.ones(shape)
    
    # These need to be tracked per element!
    tgm2m = np.ones(shape)  # tgm^(2m)
    tgp2m = np.ones(shape)  # tgp^(2m)
    
    # Determine which branch each element is in
    branch1 = theta < tetanm
    branch2 = (theta >= tetanm) & (theta < tetanp)
    branch3 = theta >= tetanp
    
    # Loop over modes
    for m in range(1, n + 1):
        # Update tm
        tm = tm * tg
        
        # Calculate cos(m*phi) and sin(m*phi) using recursion
        ccos = cosm1 * cosfi - sinm1 * sinfi
        ssin = sinm1 * cosfi + cosm1 * sinfi
        cosm1 = ccos
        sinm1 = ssin
        
        # Update recursion variables based on branch
        # Branch 2 and 3 need tgm2m updated
        tgm2m = np.where(branch2 | branch3, tgm2m * tgm2, tgm2m)
        # Only branch 3 needs tgp2m updated
        tgp2m = np.where(branch3, tgp2m * tgp2, tgp2m)
        
        # Initialize t and dtt for this mode
        t = np.zeros(shape)
        dtt = np.zeros(shape)
        
        # Branch 1: theta < tetanm
        if np.any(branch1):
            t[branch1] = tm[branch1]
            dtt[branch1] = 0.5 * m * tm[branch1] * (tg[branch1] + ctg[branch1])
        
        # Branch 2: tetanm <= theta < tetanp
        if np.any(branch2):
            fc = 1 / (tgp - tgm)
            fc1 = 1 / (2 * m + 1)
            tgm2m1 = tgm2m * tgm
            tg21 = 1 + tg * tg
            
            t[branch2] = fc * (tm[branch2] * (tgp - tg[branch2]) + 
                               fc1 * (tm[branch2] * tg[branch2] - tgm2m1[branch2] / tm[branch2]))
            dtt[branch2] = 0.5 * m * fc * tg21[branch2] * (
                tm[branch2] / tg[branch2] * (tgp - tg[branch2]) - 
                fc1 * (tm[branch2] - tgm2m1[branch2] / (tm[branch2] * tg[branch2]))
            )
        
        # Branch 3: theta >= tetanp
        if np.any(branch3):
            fc = 1 / (tgp - tgm)
            fc1 = 1 / (2 * m + 1)
            
            t[branch3] = fc * fc1 * (tgp2m[branch3] * tgp - tgm2m[branch3] * tgm) / tm[branch3]
            dtt[branch3] = -t[branch3] * m * 0.5 * (tg[branch3] + ctg[branch3])
        
        # Calculate field components for this mode
        # Avoid division by zero
        btn[m-1] = np.divide(m * t * ccos, ro, out=np.zeros_like(ro), where=ro != 0)
        bpn[m-1] = np.divide(-dtt * ssin, r, out=np.zeros_like(r), where=r != 0)
    
    # Extract the n-th mode and scale by 800
    btheta = btn[n-1] * 800.0
    bphi = bpn[n-1] * 800.0
    
    # Handle scalar output
    if scalar_input:
        return btheta.item(), bphi.item()
    else:
        return btheta, bphi


def birk_tot_vectorized(iopb: int, ps: float, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                       xkappa1: Union[float, np.ndarray], xkappa2: Union[float, np.ndarray]) -> Tuple[
                       np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                       np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized calculation of total Birkeland current field.
    
    Parameters
    ----------
    iopb : int
        Birkeland field mode flag:
        0 - all components
        1 - region 1, modes 1 & 2
        2 - region 2, modes 1 & 2
    ps : float
        Dipole tilt angle in radians
    x, y, z : ndarray
        GSM coordinates in Re
    xkappa1, xkappa2 : float or ndarray
        Scaling parameters for regions 1 and 2
        
    Returns
    -------
    bx11, by11, bz11, bx12, by12, bz12, bx21, by21, bz21, bx22, by22, bz22 : ndarray
        Field components for region 1 mode 1, region 1 mode 2, etc.
    """
    # Initialize all components
    shape = np.broadcast_shapes(x.shape, y.shape, z.shape)
    bx11 = np.zeros(shape)
    by11 = np.zeros(shape)
    bz11 = np.zeros(shape)
    bx12 = np.zeros(shape)
    by12 = np.zeros(shape)
    bz12 = np.zeros(shape)
    bx21 = np.zeros(shape)
    by21 = np.zeros(shape)
    bz21 = np.zeros(shape)
    bx22 = np.zeros(shape)
    by22 = np.zeros(shape)
    bz22 = np.zeros(shape)
    
    # Shielding coefficient arrays
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
        19.97575641,2.050148531,4.992657093,2.300564232,.2256245602,-.05841594319
    ])
    
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
        7.802304187,1.844169801,4.867254550,2.941393119,.1379899178,.06607020029
    ])
    
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
        15.91258637,1.804233877,6.578149088,2.515223491,.1930034238,-.02261109942
    ])
    
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
        6.996159733,1.971629939,4.436299219,2.904964304,.1486276863,.06859991529
    ])
    
    # Calculate fields based on mode
    if iopb == 0 or iopb == 1:
        # Region 1
        x_sc = xkappa1 - 1.1
        
        # Mode 1
        fx11, fy11, fz11 = birk_1n2_vectorized(1, 1, ps, x, y, z, xkappa1)
        hx11, hy11, hz11 = birk_shl_vectorized(sh11, ps, x_sc, x, y, z)
        bx11 = fx11 + hx11
        by11 = fy11 + hy11
        bz11 = fz11 + hz11
        
        # Mode 2
        fx12, fy12, fz12 = birk_1n2_vectorized(1, 2, ps, x, y, z, xkappa1)
        hx12, hy12, hz12 = birk_shl_vectorized(sh12, ps, x_sc, x, y, z)
        bx12 = fx12 + hx12
        by12 = fy12 + hy12
        bz12 = fz12 + hz12
    
    if iopb == 0 or iopb == 2:
        # Region 2
        x_sc = xkappa2 - 1.0
        
        # Mode 1
        fx21, fy21, fz21 = birk_1n2_vectorized(2, 1, ps, x, y, z, xkappa2)
        hx21, hy21, hz21 = birk_shl_vectorized(sh21, ps, x_sc, x, y, z)
        bx21 = fx21 + hx21
        by21 = fy21 + hy21
        bz21 = fz21 + hz21
        
        # Mode 2
        fx22, fy22, fz22 = birk_1n2_vectorized(2, 2, ps, x, y, z, xkappa2)
        hx22, hy22, hz22 = birk_shl_vectorized(sh22, ps, x_sc, x, y, z)
        bx22 = fx22 + hx22
        by22 = fy22 + hy22
        bz22 = fz22 + hz22
    
    return bx11, by11, bz11, bx12, by12, bz12, bx21, by21, bz21, bx22, by22, bz22


def birk_1n2_vectorized(numb: int, mode: int, ps: float, 
                       x: np.ndarray, y: np.ndarray, z: np.ndarray,
                       xkappa: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized calculation of region 1/2 Birkeland currents.
    
    Parameters
    ----------
    numb : int
        Region number (1 or 2)
    mode : int
        Harmonic mode (1 or 2)
    ps : float
        Dipole tilt angle in radians
    x, y, z : ndarray
        GSM coordinates in Re
    xkappa : float or ndarray
        Scaling factor
        
    Returns
    -------
    bx, by, bz : ndarray
        Field components in GSM system in nT
    """
    # Parameters
    beta = 0.9
    rh = 10.0
    eps = 3.0
    b = 0.5
    rho_0 = 7.0
    
    # Region-specific parameters
    if numb == 1:
        dphi = 0.055
        dtheta = 0.06
    else:  # numb == 2
        dphi = 0.030
        dtheta = 0.09
    
    # Coefficient arrays
    a11 = np.array([
        .1618068350, -.1797957553, 2.999642482, -.9322708978, -.6811059760,
        .2099057262, -8.358815746, -14.86033550, .3838362986, -16.30945494,
        4.537022847, 2.685836007, 27.97833029, 6.330871059, 1.876532361,
        18.95619213, .9651528100, .4217195118, -.08957770020, -1.823555887,
        .7457045438, -.5785916524, -1.010200918, .01112389357, .09572927448,
        -.3599292276, 8.713700514, .9763932955, 3.834602998, 2.492118385, .7113544659
    ])
    
    a12 = np.array([
        .7058026940, -.2845938535, 5.715471266, -2.472820880, -.7738802408,
        .3478293930, -11.37653694, -38.64768867, .6932927651, -212.4017288,
        4.944204937, 3.071270411, 33.05882281, 7.387533799, 2.366769108,
        79.22572682, .6154290178, .5592050551, -.1796585105, -1.654932210,
        .7309108776, -.4926292779, -1.130266095, -.009613974555, .1484586169,
        -.2215347198, 7.883592948, .02768251655, 2.950280953, 1.212634762, .5567714182
    ])
    
    a21 = np.array([
        .1278764024, -.2320034273, 1.805623266, -32.37241440, -.9931490648,
        .3175085630, -2.492465814, -16.21600096, .2695393416, -6.752691265,
        3.971794901, 14.54477563, 41.10158386, 7.912889730, 1.258297372,
        9.583547721, 1.014141963, .5104134759, -.1790430468, -1.756358428,
        .7561986717, -.6775248254, -.04014016420, .01446794851, .1200521731,
        -.2203584559, 4.508963850, .8221623576, 1.779933730, 1.102649543, .8867880020
    ])
    
    a22 = np.array([
        .4036015198, -.3302974212, 2.827730930, -45.44405830, -1.611103927,
        .4927112073, -.003258457559, -49.59014949, .3796217108, -233.7884098,
        4.312666980, 18.05051709, 28.95320323, 11.09948019, .7471649558,
        67.10246193, .5667096597, .6468519751, -.1560665317, -1.460805289,
        .7719653528, -.6658988668, .2515179349E-05, .02426021891, .1195003324,
        -.2625739255, 4.377172556, .2421190547, 2.503482679, 1.071587299, .7247997430
    ])
    
    # Select appropriate coefficients
    if numb == 1:
        if mode == 1:
            a = a11
        else:
            a = a12
    else:
        if mode == 1:
            a = a21
        else:
            a = a22
    
    # Scaled coordinates
    xsc = x * xkappa
    ysc = y * xkappa
    zsc = z * xkappa
    
    # Cylindrical coordinates
    rho = np.sqrt(xsc**2 + zsc**2)
    rsc = np.sqrt(xsc**2 + ysc**2 + zsc**2)
    rho2 = rho_0**2
    
    # Handle singularity at x=z=0
    phi = np.arctan2(-zsc, xsc)
    
    sphic = np.sin(phi)
    cphic = np.cos(phi)
    
    # Calculate bracket term
    rho_safe = np.where(rho < 1e-10, 1e-10, rho)
    brack = dphi + b * rho2 / (rho2 + 1) * (rho**2 - 1) / (rho2 + rho**2)
    
    # Deformation factors
    r1rh = (rsc - 1) / rh
    psias = beta * ps / (1 + r1rh**eps)**(1/eps)
    
    # Deformed angle
    phis = phi - brack * np.sin(phi) - psias
    dphisphi = 1 - brack * np.cos(phi)
    # Safe computation of derivatives
    rh_safe = np.where(rh < 1e-10, 1e-10, rh)
    rsc_safe = np.where(rsc < 1e-10, 1e-10, rsc)
    dphisrho = (-2 * b * rho2 * rho / (rho2 + rho**2)**2 * np.sin(phi) +
                beta * ps * r1rh**(eps-1) * rho / (rh_safe * rsc_safe * (1 + r1rh**eps)**(1/eps+1)))
    dphisdy = beta * ps * r1rh**(eps-1) * ysc / (rh_safe * rsc_safe * (1 + r1rh**eps)**(1/eps+1))
    
    sphics = np.sin(phis)
    cphics = np.cos(phis)
    
    # Deformed coordinates
    xs = rho * cphics
    zs = -rho * sphics
    
    # Calculate field from two cones
    # Mode number m is always mode for birk_1n2
    # theta0 is a[30] from coefficient array
    bxs, byas, bzs = twocones_vectorized(a, xs, ysc, zs, mode, a[30], dtheta)
    
    # Transform back
    brhoas = bxs * cphics - bzs * sphics
    bphias = -bxs * sphics - bzs * cphics
    
    brho_s = brhoas * dphisphi * xkappa
    bphi_s = (bphias - rho * (byas * dphisdy + brhoas * dphisrho)) * xkappa
    by_s = byas * dphisphi * xkappa
    
    # Convert to Cartesian
    bx = brho_s * cphic - bphi_s * sphic
    by = by_s
    bz = -brho_s * sphic - bphi_s * cphic
    
    return bx, by, bz


def twocones_vectorized(a: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                       m: int, theta0: float, dtheta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized two-cone field calculation.
    
    Adds fields from northern and southern cones with proper symmetry
    for region 1 Birkeland currents.
    
    Parameters
    ----------
    a : ndarray
        Coefficient array (31 elements)
    x, y, z : ndarray
        Coordinates in modified system
    m : int
        Mode number for fialcos
    theta0 : float
        Angular half-width of the cone
    dtheta : float
        Angular half-width of the current layer
        
    Returns
    -------
    bx, by, bz : ndarray
        Field components
    """
    # Northern cone
    bxn, byn, bzn = one_cone_vectorized(a, x, y, z, m, theta0, dtheta)
    
    # Southern cone (with symmetry)
    bxs, bys, bzs = one_cone_vectorized(a, x, -y, -z, m, theta0, dtheta)
    
    # Combine with proper symmetry
    bx = bxn - bxs
    by = byn + bys
    bz = bzn + bzs
    
    return bx, by, bz


def one_cone_vectorized(a: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                       m: int, theta0: float, dtheta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized single cone field calculation.
    
    Returns field components for a deformed conical current system, fitted to a 
    Biot-Savart field. Here only the northern cone is taken into account.
    
    Parameters
    ----------
    a : ndarray
        Coefficient array (31 elements)
    x, y, z : ndarray
        GSM coordinates in Re
    m : int
        Mode number for fialcos
    theta0 : float
        Angular half-width of the cone
    dtheta : float
        Angular half-width of the current layer
        
    Returns
    -------
    bx, by, bz : ndarray
        Field components in GSM system (nT)
    """
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    scalar_input = x.size == 1
    
    # Parameters for numerical differentiation
    dr = 1e-6
    dt = 1e-6
    
    # Convert to spherical coordinates
    rho2 = x**2 + y**2
    rho = np.sqrt(rho2)
    r = np.sqrt(rho2 + z**2)
    theta = np.arctan2(rho, z)
    phi = np.arctan2(y, x)
    
    # Apply coordinate deformation
    rs = r_s_vectorized(a, r, theta)
    thetas = theta_s_vectorized(a, r, theta)
    phis = phi  # No deformation in phi
    
    # Calculate field components at deformed position
    btast, bfast = fialcos_vectorized(rs, thetas, phis, m, theta0, dtheta)
    
    # Calculate derivatives for deformation tensor
    # drs/dr
    rs_plus_r = r_s_vectorized(a, r + dr, theta)
    rs_minus_r = r_s_vectorized(a, r - dr, theta)
    drsdr = (rs_plus_r - rs_minus_r) / (2 * dr)
    
    # drs/dtheta
    rs_plus_t = r_s_vectorized(a, r, theta + dt)
    rs_minus_t = r_s_vectorized(a, r, theta - dt)
    drsdt = (rs_plus_t - rs_minus_t) / (2 * dt)
    
    # dthetas/dr
    ts_plus_r = theta_s_vectorized(a, r + dr, theta)
    ts_minus_r = theta_s_vectorized(a, r - dr, theta)
    dtsdr = (ts_plus_r - ts_minus_r) / (2 * dr)
    
    # dthetas/dtheta
    ts_plus_t = theta_s_vectorized(a, r, theta + dt)
    ts_minus_t = theta_s_vectorized(a, r, theta - dt)
    dtsdt = (ts_plus_t - ts_minus_t) / (2 * dt)
    
    # Transform field components by deformation tensor
    stsst = np.divide(np.sin(thetas), np.sin(theta), 
                     out=np.zeros_like(theta), where=np.sin(theta) != 0)
    rsr = np.divide(rs, r, out=np.zeros_like(r), where=r != 0)
    
    # Note: br_ast is identically zero for radial currents
    br = -rsr / r * stsst * btast * drsdt
    btheta = rsr * stsst * btast * drsdr
    bphi = rsr * bfast * (drsdr * dtsdt - drsdt * dtsdr)
    
    # Convert to Cartesian coordinates
    s = np.divide(rho, r, out=np.zeros_like(r), where=r != 0)
    c = np.divide(z, r, out=np.zeros_like(r), where=r != 0)
    sf = np.divide(y, rho, out=np.zeros_like(rho), where=rho != 0)
    cf = np.divide(x, rho, out=np.zeros_like(rho), where=rho != 0)
    
    be = br * s + btheta * c
    
    # Apply amplitude scaling
    bx = a[0] * (be * cf - bphi * sf)
    by = a[0] * (be * sf + bphi * cf)
    bz = a[0] * (br * c - btheta * s)
    
    # Handle scalar output
    if scalar_input:
        return bx.item(), by.item(), bz.item()
    else:
        return bx, by, bz


def birk_shl_vectorized(a: np.ndarray, ps: float, x_sc: Union[float, np.ndarray],
                       x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized Birkeland current shielding field.
    
    Parameters
    ----------
    a : ndarray
        Coefficient array (86 elements)
    ps : float
        Dipole tilt angle in radians
    x_sc : float or ndarray
        Scaling factor
    x, y, z : ndarray
        GSM coordinates in Re
        
    Returns
    -------
    bx, by, bz : ndarray
        Shielding field components in nT
    """
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    x_sc = np.atleast_1d(x_sc)
    scalar_input = x.size == 1
    
    # Trigonometric functions of tilt
    cps = np.cos(ps)
    sps = np.sin(ps)
    s3ps = 2 * cps  # Approximation for small ps (matches scalar version)
    
    # Tilt rotation angles
    pst1 = ps * a[84]
    pst2 = ps * a[85]
    
    st1 = np.sin(pst1)
    ct1 = np.cos(pst1)
    st2 = np.sin(pst2)
    ct2 = np.cos(pst2)
    
    # Rotated coordinates
    x1 = x * ct1 - z * st1
    z1 = x * st1 + z * ct1
    x2 = x * ct2 - z * st2
    z2 = x * st2 + z * ct2
    
    # Initialize field components
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    
    # Coefficient index
    l = 0
    
    # Loop over symmetry modes
    for m in range(1, 3):  # m=1: perp. symmetry, m=2: parallel symmetry
        for i in range(1, 4):  # i = 1,2,3
            p = a[71 + i]
            q = a[77 + i]
            
            cypi = np.cos(y / p)
            cyqi = np.cos(y / q)
            sypi = np.sin(y / p)
            syqi = np.sin(y / q)
            
            for k in range(1, 4):  # k = 1,2,3
                r = a[74 + k]
                s = a[80 + k]
                
                szrk = np.sin(z1 / r)
                czsk = np.cos(z2 / s)
                czrk = np.cos(z1 / r)
                szsk = np.sin(z2 / s)
                
                sqpr = np.sqrt(1 / p**2 + 1 / r**2)
                sqqs = np.sqrt(1 / q**2 + 1 / s**2)
                
                epr = np.exp(x1 * sqpr)
                eqs = np.exp(x2 * sqqs)
                
                for n in range(1, 3):  # n = 1,2
                    for nn in range(1, 3):  # nn = 1,2
                        if m == 1:
                            # Perpendicular symmetry
                            fx = -sqpr * epr * cypi * szrk
                            fy = epr * sypi * szrk / p
                            fz = -epr * cypi * czrk / r
                            
                            if n == 1:
                                if nn == 1:
                                    hx, hy, hz = fx, fy, fz
                                else:
                                    hx, hy, hz = fx * x_sc, fy * x_sc, fz * x_sc
                            else:
                                if nn == 1:
                                    hx, hy, hz = fx * cps, fy * cps, fz * cps
                                else:
                                    hx, hy, hz = fx * cps * x_sc, fy * cps * x_sc, fz * cps * x_sc
                        else:
                            # Parallel symmetry
                            fx = -sps * sqqs * eqs * cyqi * czsk
                            fy = sps / q * eqs * syqi * czsk
                            fz = sps / s * eqs * cyqi * szsk
                            
                            if n == 1:
                                if nn == 1:
                                    hx, hy, hz = fx, fy, fz
                                else:
                                    hx, hy, hz = fx * x_sc, fy * x_sc, fz * x_sc
                            else:
                                if nn == 1:
                                    hx, hy, hz = fx * s3ps, fy * s3ps, fz * s3ps
                                else:
                                    hx, hy, hz = fx * s3ps * x_sc, fy * s3ps * x_sc, fz * s3ps * x_sc
                        
                        # Rotate back
                        if m == 1:
                            hxr = hx * ct1 + hz * st1
                            hzr = -hx * st1 + hz * ct1
                        else:
                            hxr = hx * ct2 + hz * st2
                            hzr = -hx * st2 + hz * ct2
                        
                        # Add contribution with coefficient
                        bx = bx + hxr * a[l]
                        by = by + hy * a[l]
                        bz = bz + hzr * a[l]
                        
                        l = l + 1
    
    # Handle scalar output
    if scalar_input:
        return bx.item(), by.item(), bz.item()
    else:
        return bx, by, bz