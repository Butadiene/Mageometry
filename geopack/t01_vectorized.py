"""
Vectorized implementation of the T01 magnetospheric field model.

This module provides a high-performance NumPy-based implementation of the
Tsyganenko 2001 (T01) magnetic field model, achieving significant speedup
over the scalar version while maintaining numerical accuracy.

Authors: N.A. Tsyganenko (original Fortran)
         Python vectorization following T01_VECTORIZATION_POLICY.md
"""

import numpy as np
import warnings
from dataclasses import dataclass
from typing import Union, Tuple, Optional
from scipy import special

# Import required components from geopack
from . import geopack
from .ring_current_vectorized import full_rc_vectorized
from .birkeland_vectorized import birk_tot_vectorized


@dataclass
class T01Parameters:
    """Container for T01 model parameters.
    
    All fields can be either scalars or arrays of the same shape as the input coordinates.
    This allows for different solar wind conditions at each point.
    """
    dxshift1: Union[float, np.ndarray]
    dxshift2: Union[float, np.ndarray]
    d: Union[float, np.ndarray]
    deltady: float  # Y-dependent thickness variation (constant)
    deltadx: float  # X-dependent thickness variation (constant)
    xkappa1: Union[float, np.ndarray]
    xkappa2: Union[float, np.ndarray]
    sc_sy: Union[float, np.ndarray]
    sc_pr: Union[float, np.ndarray]
    phi: Union[float, np.ndarray]
    g: Union[float, np.ndarray]
    rh0: Union[float, np.ndarray]
    xappa: Union[float, np.ndarray]  # Pressure scaling factor
    

def calculate_parameters(parmod: np.ndarray, ps: float, a: np.ndarray, 
                        n_points: Optional[int] = None) -> T01Parameters:
    """Calculate all T01 parameters from input.
    
    Parameters
    ----------
    parmod : array_like
        Model parameters, shape (6,) or (n_points, 6)
        [pdyn, dst, byimf, bzimf, g1, g2]
    ps : float
        Dipole tilt angle
    a : array_like
        Model coefficients array
    n_points : int, optional
        Number of points (used when parmod is 1D)
        
    Returns
    -------
    params : T01Parameters
        Container with all calculated parameters
    """
    # Handle both scalar and array inputs
    parmod = np.atleast_2d(parmod)
    if parmod.shape[0] == 1 and n_points is not None:
        # Broadcast scalar parameters to all points
        parmod = np.repeat(parmod, n_points, axis=0)
    
    # Extract parameters (now potentially arrays)
    pdyn = parmod[:, 0]
    dst = parmod[:, 1]
    byimf = parmod[:, 2]
    bzimf = parmod[:, 3]
    g1 = parmod[:, 4]
    g2 = parmod[:, 5]
    
    # Pressure scaling
    xappa = (pdyn / 2.0) ** a[38]
    
    # Initialize parameters container
    params = T01Parameters(
        dxshift1=a[25] + a[26] * g2,
        dxshift2=np.zeros_like(pdyn),
        d=np.full_like(pdyn, a[27]),
        deltady=a[28],  # Keep as scalar since it's constant
        deltadx=0.0,  # Not used in T01, set to 0
        xkappa1=a[34] + a[35] * g2,
        xkappa2=a[36] + a[37] * g2,
        sc_sy=None,  # Calculated below
        sc_pr=None,  # Calculated below
        phi=None,    # Calculated below
        g=np.full_like(pdyn, a[40]),
        rh0=np.full_like(pdyn, a[39]),
        xappa=xappa
    )
    
    # Ring current parameters (vectorized)
    params.phi = 1.5707963 * np.tanh(np.abs(dst) / a[33])
    znam = np.maximum(np.abs(dst), 20.0)
    params.sc_sy = a[29] * (20/znam)**a[30] * xappa
    params.sc_pr = a[31] * (20/znam)**a[32] * xappa
    
    # If single point, convert arrays back to scalars
    if parmod.shape[0] == 1:
        for field in params.__dataclass_fields__:
            value = getattr(params, field)
            if isinstance(value, np.ndarray) and value.size == 1:
                setattr(params, field, value.item())
    
    return params


def iterate_sigma_vectorized_masked(x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                                   sps: float, rh0: Union[float, np.ndarray], 
                                   rh2: float, max_iter: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized iterative sigma calculation with masked updates.
    
    Parameters
    ----------
    x, y, z : ndarray
        GSM coordinates in Re
    sps : float
        sin(ps) where ps is dipole tilt
    rh0 : float or ndarray
        Base hinge distance parameter
    rh2 : float
        Hinge distance z-dependence parameter
    max_iter : int
        Maximum iterations
        
    Returns
    -------
    xss, zss : ndarray
        Unwarped coordinates
    """
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    r = np.sqrt(x**2 + y**2 + z**2)
    xss = x.copy()
    zss = z.copy()
    
    # Track convergence for each point
    converged = np.zeros_like(x, dtype=bool)
    
    for i in range(max_iter):
        xsold = xss.copy()
        zsold = zss.copy()
        
        # Only update non-converged points
        active = ~converged
        if not np.any(active):
            break
            
        rh = rh0 + rh2 * (zss[active]/r[active])**2
        sinpsas = sps / np.power(1 + (r[active]/rh)**3, 1/3)
        sinpsas = np.clip(sinpsas, -1.0, 1.0)  # Ensure valid range
        cospsas = np.sqrt(np.maximum(1 - sinpsas**2, 0))
        
        zss[active] = x[active] * sinpsas + z[active] * cospsas
        xss[active] = x[active] * cospsas - z[active] * sinpsas
        
        # Check convergence
        dd = np.abs(xss - xsold) + np.abs(zss - zsold)
        converged = dd < 1e-6
    
    return xss, zss


def iterate_sigma_vectorized_full(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                                 sps: float, rh0: Union[float, np.ndarray],
                                 rh2: float, max_iter: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized iterative sigma calculation with full array operations.
    
    This version calculates updates for all points and uses np.where for
    conditional updates, which can be more efficient for certain array sizes.
    
    Parameters
    ----------
    x, y, z : ndarray
        GSM coordinates in Re
    sps : float
        sin(ps) where ps is dipole tilt
    rh0 : float or ndarray
        Base hinge distance parameter
    rh2 : float
        Hinge distance z-dependence parameter
    max_iter : int
        Maximum iterations
        
    Returns
    -------
    xss, zss : ndarray
        Unwarped coordinates
    """
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    r = np.sqrt(x**2 + y**2 + z**2)
    xss = x.copy()
    zss = z.copy()
    
    # Track convergence for each point
    converged = np.zeros_like(x, dtype=bool)
    
    for i in range(max_iter):
        xsold = xss.copy()
        zsold = zss.copy()
        
        # Calculate updates for ALL points (leverages NumPy's optimized loops)
        # Safe division for zsold/r
        r_safe = np.where(r < 1e-10, 1e-10, r)
        rh = rh0 + rh2 * (zsold/r_safe)**2
        # Safe division for r/rh
        rh_safe = np.where(rh < 1e-10, 1e-10, rh)
        sinpsas = sps / np.power(1 + (r/rh_safe)**3, 1/3)
        sinpsas = np.clip(sinpsas, -1.0, 1.0)  # Ensure valid range
        cospsas = np.sqrt(np.maximum(1 - sinpsas**2, 0))
        
        new_xss = x * cospsas - z * sinpsas
        new_zss = x * sinpsas + z * cospsas
        
        # Conditionally apply updates only to non-converged points
        xss = np.where(converged, xsold, new_xss)
        zss = np.where(converged, zsold, new_zss)
        
        # Update convergence mask
        dd = np.abs(xss - xsold) + np.abs(zss - zsold)
        converged |= dd < 1e-6
        
        if np.all(converged):
            break
    
    return xss, zss


def dipole_vectorized(ps: float, x: np.ndarray, y: np.ndarray, 
                     z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized dipole field calculation.
    
    Parameters
    ----------
    ps : float
        Dipole tilt angle in radians
    x, y, z : ndarray
        GSM coordinates in Re
        
    Returns
    -------
    bx, by, bz : ndarray
        Dipole field components in nT
    """
    sps = np.sin(ps)
    cps = np.cos(ps)
    
    p = x**2
    u = z**2
    v = 3 * z * x
    t = y**2
    
    # Safe division with epsilon to prevent overflow
    q = 30574.0 / np.power(p + t + u + 1e-15, 2.5)
    
    bx = q * ((t + u - 2 * p) * sps - v * cps)
    by = -3 * y * q * (x * sps + z * cps)
    bz = q * ((p + t - 2 * u) * cps - v * sps)
    
    return bx, by, bz


def t01_vectorized(parmod: np.ndarray, ps: float, x: Union[float, np.ndarray],
                  y: Union[float, np.ndarray], z: Union[float, np.ndarray],
                  debug: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized T01 magnetospheric field model.
    
    Parameters
    ----------
    parmod : array_like
        Model parameters [pdyn, dst, byimf, bzimf, g1, g2]
    ps : float
        Dipole tilt angle in radians
    x, y, z : float or array_like
        GSM coordinates in Re
        
    Returns
    -------
    bx, by, bz : float or ndarray
        Magnetic field components in nT
        Returns scalars if inputs were scalars, arrays otherwise
    """
    # Model coefficients (from original T01)
    a = np.array([
        1.00000, 2.47341, 0.40791, 0.30429, -0.10637, -0.89108, 3.29350,
        -0.05413, -0.00696, 1.07869, -0.02314, -0.66173, -0.68018, -0.03246,
        0.02681, 0.28062, 0.16535, -0.02939, 0.02639, -0.24891, -0.08063,
        0.08900, -0.02475, 0.05887, 0.57691, 0.65256, -0.03230, 2.24733,
        4.10546, 1.13665, 0.05506, 0.97669, 0.21164, 0.64594, 1.12556, 0.01389,
        1.02978, 0.02968, 0.15821, 9.00519, 28.17582, 1.35285, 0.42279
    ])
    
    # Track scalar inputs for proper return format
    scalar_input = np.isscalar(x) and np.isscalar(y) and np.isscalar(z)
    
    # Convert to arrays and broadcast
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    x, y, z = np.broadcast_arrays(x, y, z)
    
    # Check validity range
    invalid_mask = x < -15.0
    if np.any(invalid_mask):
        warnings.warn(
            f"T01 model used outside valid range (x < -15 Re) "
            f"for {np.sum(invalid_mask)} points. "
            f"Results set to NaN for these points."
        )
    
    # Special handling for points at or very near origin
    r = np.sqrt(x**2 + y**2 + z**2)
    origin_mask = r < 1e-5  # Points essentially at origin
    
    # Calculate parameters
    n_points = len(x)
    params = calculate_parameters(parmod, ps, a, n_points)
    
    # Extract some key parameters
    # Handle both 1D and 2D parmod arrays
    parmod = np.atleast_1d(parmod)
    if parmod.ndim == 1:
        pdyn = parmod[0]
        dst = parmod[1]
        byimf = parmod[2]
        bzimf = parmod[3]
        g1 = parmod[4]
        g2 = parmod[5]
    else:
        pdyn = parmod[:, 0]
        dst = parmod[:, 1]
        byimf = parmod[:, 2]
        bzimf = parmod[:, 3]
        g1 = parmod[:, 4]
        g2 = parmod[:, 5]
    
    dst_ast = dst * 0.8 - 13.0 * np.sqrt(pdyn)
    
    # Scale coordinates
    xx = x * params.xappa
    yy = y * params.xappa
    zz = z * params.xappa
    
    if debug:
        print(f"Before extall: xx.shape={xx.shape}, yy.shape={yy.shape}, zz.shape={zz.shape}")
    
    # Call main calculation
    # Pass both scaled and unscaled coordinates
    bx, by, bz = extall_vectorized(0, 0, 0, 0, a, 43, pdyn, dst_ast, 
                                  byimf, bzimf, g1, g2, ps, xx, yy, zz, params,
                                  x_unscaled=x, y_unscaled=y, z_unscaled=z)
    
    if debug:
        print(f"After extall: bx.shape={bx.shape}, by.shape={by.shape}, bz.shape={bz.shape}")
    
    # Set invalid points to NaN
    if np.any(invalid_mask):
        bx[invalid_mask] = np.nan
        by[invalid_mask] = np.nan
        bz[invalid_mask] = np.nan
    
    # For points at origin, set field to zero (external field only, dipole not included)
    if np.any(origin_mask):
        bx[origin_mask] = 0.0
        by[origin_mask] = 0.0
        bz[origin_mask] = 0.0
    
    # Return proper format
    if scalar_input:
        return bx.item(), by.item(), bz.item()
    else:
        return bx, by, bz


def shlcar3x3_vectorized_partial(x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                                ps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized 3x3x2 Cartesian harmonic shield with loop optimization.
    
    This implementation vectorizes over spatial points but keeps loops over
    harmonic indices for clarity and easier debugging.
    
    Parameters
    ----------
    x, y, z : ndarray
        GSM coordinates in Re
    ps : float
        Dipole tilt angle in radians
        
    Returns
    -------
    bx, by, bz : ndarray
        Shielding field components in nT
    """
    # Coefficients for shlcar3x3
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
        15.73319647,2.303504968,5.840511214,.8385953499E-01,.3477844929
    ])
    
    # Extract scale parameters
    p1, p2, p3 = a[36:39]
    r1, r2, r3 = a[39:42]
    q1, q2, q3 = a[42:45]
    s1, s2, s3 = a[45:48]
    t1, t2 = a[48:50]
    
    # Tilt-dependent calculations
    cps = np.cos(ps)
    sps = np.sin(ps)
    s2ps = 2 * cps  # Note: original comment says this was modified
    
    ct1 = np.cos(ps * t1)
    st1 = np.sin(ps * t1)
    ct2 = np.cos(ps * t2)
    st2 = np.sin(ps * t2)
    
    # Tilted coordinates
    x1 = x * ct1 - z * st1
    z1 = x * st1 + z * ct1
    x2 = x * ct2 - z * st2
    z2 = x * st2 + z * ct2
    
    # Initialize output arrays
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    
    # First set of harmonics (perpendicular symmetry)
    # Using loops for clarity - can be fully vectorized later
    p_vals = [p1, p2, p3]
    r_vals = [r1, r2, r3]
    
    hx_list = []
    hy_list = []
    hz_list = []
    
    # Calculate all 9 perpendicular harmonics
    for i, p in enumerate(p_vals):
        cyp = np.cos(y / p)
        syp = np.sin(y / p)
        
        for j, r in enumerate(r_vals):
            sqpr = np.sqrt(1/p**2 + 1/r**2)
            czr = np.cos(z1 / r)
            szr = np.sin(z1 / r)
            # Clip argument to prevent overflow/underflow
            arg = np.clip(sqpr * x1, -740.0, 88.0)
            expr = np.exp(arg)
            
            if j < 2:  # First two r values have simpler form
                fx = -sqpr * expr * cyp * szr
                hy = expr / p * syp * szr
                fz = -expr * cyp / r * czr
            else:  # r3 has special form
                fx = -expr * cyp * (sqpr * z1 * czr + szr / r * (x1 + 1/sqpr))
                hy = expr / p * syp * (z1 * czr + x1 / r * szr / sqpr)
                fz = -expr * cyp * (czr * (1 + x1 / r**2 / sqpr) - z1 / r * szr)
            
            hx = fx * ct1 + fz * st1
            hz = -fx * st1 + fz * ct1
            
            hx_list.append(hx)
            hy_list.append(hy)
            hz_list.append(hz)
    
    # Second set of harmonics (parallel symmetry)
    q_vals = [q1, q2, q3]
    s_vals = [s1, s2, s3]
    
    for i, q in enumerate(q_vals):
        cyq = np.cos(y / q)
        syq = np.sin(y / q)
        
        for j, s in enumerate(s_vals):
            sqqs = np.sqrt(1/q**2 + 1/s**2)
            czs = np.cos(z2 / s)
            szs = np.sin(z2 / s)
            # Clip argument to prevent overflow/underflow
            arg = np.clip(sqqs * x2, -740.0, 88.0)
            exqs = np.exp(arg)
            
            fx = -sqqs * exqs * cyq * czs * sps
            hy = exqs / q * syq * czs * sps
            fz = exqs * cyq / s * szs * sps
            
            hx = fx * ct2 + fz * st2
            hz = -fx * st2 + fz * ct2
            
            hx_list.append(hx)
            hy_list.append(hy)
            hz_list.append(hz)
    
    # Apply coefficients and sum contributions
    # Perpendicular terms
    for i in range(9):
        coeff_idx = i * 2
        a_perp = a[coeff_idx] + a[coeff_idx + 1] * cps
        bx += a_perp * hx_list[i]
        by += a_perp * hy_list[i]
        bz += a_perp * hz_list[i]
    
    # Parallel terms
    for i in range(9):
        coeff_idx = 18 + i * 2
        a_par = a[coeff_idx] + a[coeff_idx + 1] * s2ps
        bx += a_par * hx_list[9 + i]
        by += a_par * hy_list[9 + i]
        bz += a_par * hz_list[9 + i]
    
    return bx, by, bz


def extall_vectorized(iopgen: int, iopt: int, iopb: int, iopr: int,
                     a: np.ndarray, ntot: int, pdyn: Union[float, np.ndarray],
                     dst: Union[float, np.ndarray], byimf: Union[float, np.ndarray],
                     bzimf: Union[float, np.ndarray], vbimf1: Union[float, np.ndarray],
                     vbimf2: Union[float, np.ndarray], ps: float,
                     x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     params: T01Parameters, x_unscaled: np.ndarray = None, 
                     y_unscaled: np.ndarray = None, z_unscaled: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized external field calculation for T01.
    
    Parameters
    ----------
    iopgen : int
        General option flag:
        0 - calculate total field
        1 - dipole shielding only
        2 - tail field only
        3 - birkeland field only
        4 - ring current field only
        5 - interconnection field only
    iopt : int
        Tail field flag (0=both modes, 1=mode 1, 2=mode 2)
    iopb : int
        Birkeland field flag (0=all 4 terms, 1=region 1, 2=region 2)
    iopr : int
        Ring current flag (0=both src and prc, 1=src only, 2=prc only)
    a : ndarray
        Model coefficients
    ntot : int
        Total number of coefficients
    pdyn, dst, byimf, bzimf, vbimf1, vbimf2 : float or ndarray
        Solar wind and IMF parameters
    ps : float
        Dipole tilt angle
    x, y, z : ndarray
        Scaled GSM coordinates
    params : T01Parameters
        Pre-calculated parameters
        
    Returns
    -------
    bx, by, bz : ndarray
        Magnetic field components
    """
    # Constants
    a0_a = 34.586
    a0_s0 = 1.1960
    a0_x0 = 3.4397
    dsig = 0.003
    
    # Calculate some derived parameters
    sps = np.sin(ps)
    x0 = a0_x0 / params.xappa
    am = a0_a / params.xappa
    s0 = a0_s0
    
    # Note: x, y, z are already scaled coordinates passed from t01_vectorized
    # No need to scale again - this was causing double-scaling!
    xx = x
    yy = y
    zz = z
    
    # For sigma calculation, we need unscaled coordinates
    if x_unscaled is None:
        # If not provided, assume x, y, z are unscaled (for backward compatibility)
        x_unscaled = x / params.xappa
        y_unscaled = y / params.xappa
        z_unscaled = z / params.xappa
    
    # IMF clock angle
    if np.isscalar(byimf) and np.isscalar(bzimf):
        if byimf == 0 and bzimf == 0:
            theta = 0.0
        else:
            theta = np.arctan2(byimf, bzimf)
            if theta <= 0:
                theta += 2 * np.pi
    else:
        # Vectorized version
        theta = np.arctan2(byimf, bzimf)
        theta = np.where(theta <= 0, theta + 2 * np.pi, theta)
        # Handle zero IMF case
        zero_imf = (byimf == 0) & (bzimf == 0)
        theta = np.where(zero_imf, 0.0, theta)
    
    sthetah = np.sin(theta / 2.0) ** 2
    
    # IMF components outside magnetopause
    factimf = a[23] + a[24] * sthetah
    oimfx = np.zeros_like(x)
    # Ensure oimfy and oimfz have the same shape as x
    oimfy = np.full_like(x, byimf * factimf)
    oimfz = np.full_like(x, bzimf * factimf)
    
    # Calculate unwarped coordinates (sigma calculation)
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Use the iterative algorithm to find xss, zss
    # Choose the full array version as it showed better performance
    # IMPORTANT: Use unscaled coordinates for sigma calculation
    xss, zss = iterate_sigma_vectorized_full(x_unscaled, y_unscaled, z_unscaled, sps, params.rh0, -5.2)
    
    # Calculate sigma (magnetopause distance parameter)
    # Use unscaled y coordinate
    rho2 = y_unscaled**2 + zss**2
    asq = am**2
    xmxm = am + xss - x0
    xmxm = np.maximum(xmxm, 0)  # Cylinder boundary condition
    axx0 = xmxm**2
    aro = asq + rho2
    
    # Safe calculation of sigma
    discriminant = (aro + axx0)**2 - 4.0 * asq * axx0
    discriminant = np.maximum(discriminant, 0)
    sigma = np.sqrt((aro + axx0 + np.sqrt(discriminant)) / (2.0 * asq))
    
    # Define three regions
    mask_inside = sigma < (s0 - dsig)
    mask_layer = (sigma >= (s0 - dsig)) & (sigma < (s0 + dsig))
    mask_outside = sigma >= (s0 + dsig)
    
    # Initialize output arrays
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    
    # Process regions 1 and 2 (inside and boundary layer)
    mask_not_outside = mask_inside | mask_layer
    
    if np.any(mask_not_outside):
        # Calculate field components based on iopgen flag
        bxcf = np.zeros_like(x)
        bycf = np.zeros_like(y)
        bzcf = np.zeros_like(z)
        
        # Dipole shielding
        if (iopgen == 0) or (iopgen == 1):
            # Use scaled coordinates like the scalar version
            bxcf_temp, bycf_temp, bzcf_temp = shlcar3x3_vectorized_partial(
                xx[mask_not_outside], yy[mask_not_outside], zz[mask_not_outside], ps
            )
            # Apply xappa3 scaling like the scalar version
            xappa3 = params.xappa ** 3
            # Handle both scalar and array xappa3
            if np.isscalar(xappa3):
                bxcf[mask_not_outside] = bxcf_temp * xappa3
                bycf[mask_not_outside] = bycf_temp * xappa3
                bzcf[mask_not_outside] = bzcf_temp * xappa3
            else:
                # If xappa3 is an array, extract the values for mask_not_outside
                xappa3_local = xappa3[mask_not_outside] if xappa3.size > 1 else xappa3
                bxcf[mask_not_outside] = bxcf_temp * xappa3_local
                bycf[mask_not_outside] = bycf_temp * xappa3_local
                bzcf[mask_not_outside] = bzcf_temp * xappa3_local
        
        # Tail field
        bxt1 = np.zeros_like(x)
        byt1 = np.zeros_like(y)
        bzt1 = np.zeros_like(z)
        bxt2 = np.zeros_like(x)
        byt2 = np.zeros_like(y)
        bzt2 = np.zeros_like(z)
        
        if (iopgen == 0) or (iopgen == 2):
            # Get tail field for points inside magnetosphere
            if np.any(mask_not_outside):
                # Extract parameters for these points
                if np.isscalar(params.dxshift1):
                    dxshift1_local = params.dxshift1
                    dxshift2_local = params.dxshift2
                    d_local = params.d
                    g_local = params.g
                    rh0_local = params.rh0
                else:
                    dxshift1_local = params.dxshift1[mask_not_outside]
                    dxshift2_local = params.dxshift2[mask_not_outside]
                    d_local = params.d[mask_not_outside]
                    g_local = params.g[mask_not_outside]
                    rh0_local = params.rh0[mask_not_outside]
                
                # Calculate deformed tail field
                bxt1_temp, byt1_temp, bzt1_temp, bxt2_temp, byt2_temp, bzt2_temp = deformed_vectorized(
                    iopt, ps,
                    xx[mask_not_outside], yy[mask_not_outside], zz[mask_not_outside],
                    dxshift1_local, dxshift2_local, d_local, params.deltady, g_local, rh0_local
                )
                
                # Calculate tail amplitude factors
                dlp1 = (pdyn / 2.0) ** a[41]
                dlp2 = (pdyn / 2.0) ** a[42]
                # vbimf1 is actually g1 from parmod[4]
                tamp1 = a[1] + a[2] * dlp1 + a[3] * vbimf1 + a[4] * dst
                tamp2 = a[5] + a[6] * dlp2 + a[7] * vbimf1 + a[8] * dst
                
                # Handle array-valued amplitudes
                if not np.isscalar(tamp1) and np.size(tamp1) > 1:
                    tamp1 = tamp1[mask_not_outside]
                if not np.isscalar(tamp2) and np.size(tamp2) > 1:
                    tamp2 = tamp2[mask_not_outside]
                
                # Apply tail scaling with correct amplitudes
                # Note: deformed already returns fields at the correct scale, no xappa3 needed
                if iopt == 0:  # Both modes
                    bxt1[mask_not_outside] = bxt1_temp * tamp1 + bxt2_temp * tamp2
                    byt1[mask_not_outside] = byt1_temp * tamp1 + byt2_temp * tamp2
                    bzt1[mask_not_outside] = bzt1_temp * tamp1 + bzt2_temp * tamp2
                elif iopt == 1:  # Mode 1 only
                    bxt1[mask_not_outside] = bxt1_temp * tamp1
                    byt1[mask_not_outside] = byt1_temp * tamp1
                    bzt1[mask_not_outside] = bzt1_temp * tamp1
                elif iopt == 2:  # Mode 2 only
                    bxt1[mask_not_outside] = bxt2_temp * tamp2
                    byt1[mask_not_outside] = byt2_temp * tamp2
                    bzt1[mask_not_outside] = bzt2_temp * tamp2
        
        # Ring current field
        bxsrc = np.zeros_like(x)
        bysrc = np.zeros_like(y)
        bzsrc = np.zeros_like(z)
        bxprc = np.zeros_like(x)
        byprc = np.zeros_like(y)
        bzprc = np.zeros_like(z)
        
        if (iopgen == 0) or (iopgen == 4):
            # Use ring current parameters from params
            # Get ring current field for points inside magnetosphere
            if np.any(mask_not_outside):
                # Extract parameters if array-valued
                if np.isscalar(params.sc_sy):
                    sc_sy_local = params.sc_sy
                    sc_pr_local = params.sc_pr
                    phi_local = params.phi
                else:
                    sc_sy_local = params.sc_sy[mask_not_outside]
                    sc_pr_local = params.sc_pr[mask_not_outside]
                    phi_local = params.phi[mask_not_outside]
                
                # Calculate ring current fields (note: full_rc expects scaled coordinates!)
                bxsrc_temp, bysrc_temp, bzsrc_temp, bxprc_temp, byprc_temp, bzprc_temp = full_rc_vectorized(
                    iopr, ps,
                    xx[mask_not_outside], yy[mask_not_outside], zz[mask_not_outside],
                    sc_sy_local, sc_pr_local, phi_local
                )
                
                # Apply amplitude scaling only (no xappa3 since using scaled coordinates)
                # Correct ring current amplitude calculations
                a_src = a[9] + a[10] * dst + a[11] * np.sqrt(pdyn)
                a_prc = a[12] + a[13] * dst + a[14] * np.sqrt(pdyn)
                
                # Handle array-valued coefficients
                if not np.isscalar(a_src) and np.size(a_src) > 1:
                    a_src = a_src[mask_not_outside]
                if not np.isscalar(a_prc) and np.size(a_prc) > 1:
                    a_prc = a_prc[mask_not_outside]
                
                bxsrc[mask_not_outside] = bxsrc_temp * a_src
                bysrc[mask_not_outside] = bysrc_temp * a_src
                bzsrc[mask_not_outside] = bzsrc_temp * a_src
                bxprc[mask_not_outside] = bxprc_temp * a_prc
                byprc[mask_not_outside] = byprc_temp * a_prc
                bzprc[mask_not_outside] = bzprc_temp * a_prc
        
        # Birkeland current field
        bxr11 = np.zeros_like(x)
        byr11 = np.zeros_like(y)
        bzr11 = np.zeros_like(z)
        bxr12 = np.zeros_like(x)
        byr12 = np.zeros_like(y)
        bzr12 = np.zeros_like(z)
        bxr21 = np.zeros_like(x)
        byr21 = np.zeros_like(y)
        bzr21 = np.zeros_like(z)
        bxr22 = np.zeros_like(x)
        byr22 = np.zeros_like(y)
        bzr22 = np.zeros_like(z)
        
        if (iopgen == 0) or (iopgen == 3):
            # Use Birkeland current parameters from params
            # Get Birkeland field for points inside magnetosphere
            if np.any(mask_not_outside):
                # Extract parameters if array-valued
                if np.isscalar(params.xkappa1):
                    xkappa1_local = params.xkappa1
                    xkappa2_local = params.xkappa2
                else:
                    xkappa1_local = params.xkappa1[mask_not_outside]
                    xkappa2_local = params.xkappa2[mask_not_outside]
                
                # Calculate Birkeland fields (use scaled coordinates like scalar version)
                (bx11_temp, by11_temp, bz11_temp, bx12_temp, by12_temp, bz12_temp,
                 bx21_temp, by21_temp, bz21_temp, bx22_temp, by22_temp, bz22_temp) = birk_tot_vectorized(
                    iopb, ps,
                    xx[mask_not_outside], yy[mask_not_outside], zz[mask_not_outside],
                    xkappa1_local, xkappa2_local
                )
                
                # Apply amplitude scaling only (no xappa3 since using scaled coordinates)
                # vbimf2 is actually g2 from parmod[5]
                a_r11 = a[15] + a[16] * vbimf2
                a_r12 = a[17] + a[18] * vbimf2
                a_r21 = a[19] + a[20] * vbimf2
                a_r22 = a[21] + a[22] * vbimf2
                
                # Handle array-valued coefficients
                if not np.isscalar(a_r11) and np.size(a_r11) > 1:
                    a_r11 = a_r11[mask_not_outside]
                if not np.isscalar(a_r12) and np.size(a_r12) > 1:
                    a_r12 = a_r12[mask_not_outside]
                if not np.isscalar(a_r21) and np.size(a_r21) > 1:
                    a_r21 = a_r21[mask_not_outside]
                if not np.isscalar(a_r22) and np.size(a_r22) > 1:
                    a_r22 = a_r22[mask_not_outside]
                
                bxr11[mask_not_outside] = bx11_temp * a_r11
                byr11[mask_not_outside] = by11_temp * a_r11
                bzr11[mask_not_outside] = bz11_temp * a_r11
                bxr12[mask_not_outside] = bx12_temp * a_r12
                byr12[mask_not_outside] = by12_temp * a_r12
                bzr12[mask_not_outside] = bz12_temp * a_r12
                bxr21[mask_not_outside] = bx21_temp * a_r21
                byr21[mask_not_outside] = by21_temp * a_r21
                bzr21[mask_not_outside] = bz21_temp * a_r21
                bxr22[mask_not_outside] = bx22_temp * a_r22
                byr22[mask_not_outside] = by22_temp * a_r22
                bzr22[mask_not_outside] = bz22_temp * a_r22
        
        # Interconnection field (penetrated IMF)
        hximf = np.zeros_like(x)
        hyimf = np.zeros_like(y)
        hzimf = np.zeros_like(z)
        
        if (iopgen == 0) or (iopgen == 5):
            # Only transverse IMF components penetrate
            hximf = 0.0
            hyimf = byimf
            hzimf = bzimf
        
        # Total field (all components)
        # Note: bxt1, bxsrc, bxprc, and bxr* already have their amplitude coefficients applied
        # Dipole shielding amplitude includes xappa dependence
        
        # Initialize bbx, bby, bbz
        bbx = np.empty(0)
        bby = np.empty(0)
        bbz = np.empty(0)
        
        # Calculate bbx, bby, bbz only for points not outside
        if np.any(mask_not_outside):
            # Extract components for not-outside points
            bxcf_no = bxcf[mask_not_outside]
            bycf_no = bycf[mask_not_outside]
            bzcf_no = bzcf[mask_not_outside]
            bxt1_no = bxt1[mask_not_outside]
            byt1_no = byt1[mask_not_outside]
            bzt1_no = bzt1[mask_not_outside]
            bxsrc_no = bxsrc[mask_not_outside]
            bysrc_no = bysrc[mask_not_outside]
            bzsrc_no = bzsrc[mask_not_outside]
            bxprc_no = bxprc[mask_not_outside]
            byprc_no = byprc[mask_not_outside]
            bzprc_no = bzprc[mask_not_outside]
            bxr11_no = bxr11[mask_not_outside]
            byr11_no = byr11[mask_not_outside]
            bzr11_no = bzr11[mask_not_outside]
            bxr12_no = bxr12[mask_not_outside]
            byr12_no = byr12[mask_not_outside]
            bzr12_no = bzr12[mask_not_outside]
            bxr21_no = bxr21[mask_not_outside]
            byr21_no = byr21[mask_not_outside]
            bzr21_no = bzr21[mask_not_outside]
            bxr22_no = bxr22[mask_not_outside]
            byr22_no = byr22[mask_not_outside]
            bzr22_no = bzr22[mask_not_outside]
            
            # Get xappa and sthetah for not-outside points
            if np.isscalar(params.xappa):
                xappa = params.xappa
            else:
                xappa = params.xappa[mask_not_outside]
            
            if np.isscalar(sthetah):
                sthetah_no = sthetah
            else:
                sthetah_no = sthetah[mask_not_outside]
            
            # Calculate dipole shielding amplitude
            # Note: scalar version uses just a[0], not a[0] + a[9] * xappa
            a_s = a[0]
            
            # Get IMF components for not-outside points
            if np.isscalar(hximf):
                hximf_no = hximf
            else:
                hximf_no = hximf[mask_not_outside] if hximf.size > 1 else hximf
            
            if np.isscalar(hyimf):
                hyimf_no = hyimf
            else:
                hyimf_no = hyimf[mask_not_outside] if hyimf.size > 1 else hyimf
                
            if np.isscalar(hzimf):
                hzimf_no = hzimf  
            else:
                hzimf_no = hzimf[mask_not_outside] if hzimf.size > 1 else hzimf
            
            # Calculate total field for not-outside points
            # Note: The scalar version adds a[23]*hyimf + a[24]*hyimf*sthetah
            # which is NOT the same as factimf * hyimf
            bbx = (a_s * bxcf_no + bxt1_no + bxsrc_no + bxprc_no + 
                   bxr11_no + bxr12_no + bxr21_no + bxr22_no +
                   a[23] * hximf_no + a[24] * hximf_no * sthetah_no)
            bby = (a_s * bycf_no + byt1_no + bysrc_no + byprc_no + 
                   byr11_no + byr12_no + byr21_no + byr22_no +
                   a[23] * hyimf_no + a[24] * hyimf_no * sthetah_no)
            bbz = (a_s * bzcf_no + bzt1_no + bzsrc_no + bzprc_no + 
                   bzr11_no + bzr12_no + bzr21_no + bzr22_no +
                   a[23] * hzimf_no + a[24] * hzimf_no * sthetah_no)
        
        # Process inside magnetosphere (region 1)
        if np.any(mask_inside):
            # Create index mapping for bbx, bby, bbz arrays
            # mask_inside points that are also in mask_not_outside
            inside_in_not_outside = mask_inside[mask_not_outside]
            bx[mask_inside] = bbx[inside_in_not_outside]
            by[mask_inside] = bby[inside_in_not_outside]
            bz[mask_inside] = bbz[inside_in_not_outside]
        
        # Process boundary layer (region 2)
        if np.any(mask_layer):
            # Create index mapping for bbx, bby, bbz arrays
            # mask_layer points that are also in mask_not_outside
            layer_in_not_outside = mask_layer[mask_not_outside]
            
            # Interpolation factors
            fint = 0.5 * (1.0 - (sigma[mask_layer] - s0) / dsig)
            fext = 0.5 * (1.0 + (sigma[mask_layer] - s0) / dsig)
            
            # Get dipole field at these points (use unscaled coordinates)
            qx, qy, qz = dipole_vectorized(ps, x_unscaled[mask_layer], y_unscaled[mask_layer], z_unscaled[mask_layer])
            
            # Interpolate between internal and external fields
            # Correct formula: weighted average of internal model field (bbx) and external field (oimfx - qx)
            # This matches the scalar implementation algebra
            bx[mask_layer] = bbx[layer_in_not_outside] * fint + (oimfx[mask_layer] - qx) * fext
            by[mask_layer] = bby[layer_in_not_outside] * fint + (oimfy[mask_layer] - qy) * fext
            bz[mask_layer] = bbz[layer_in_not_outside] * fint + (oimfz[mask_layer] - qz) * fext
    
    # Process outside magnetosphere (region 3)
    if np.any(mask_outside):
        idx = mask_outside
        # Get dipole field (use unscaled coordinates)
        qx, qy, qz = dipole_vectorized(ps, x_unscaled[idx], y_unscaled[idx], z_unscaled[idx])
        
        # External field minus dipole
        bx[idx] = oimfx[idx] - qx
        by[idx] = oimfy[idx] - qy
        bz[idx] = oimfz[idx] - qz
    
    return bx, by, bz


def taildisk_vectorized(d0: Union[float, np.ndarray], deltadx: float, deltady: float,
                       x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized tail disk current field calculation.
    
    Computes the components of the tail current field, similar to Tsyganenko and Peredo (1994),
    but using spacewarping as described in Tsyganenko and Stern (1996).
    
    Parameters
    ----------
    d0 : float or ndarray
        Base thickness parameter
    deltadx : float
        X-dependent thickness variation
    deltady : float
        Y-dependent thickness variation
    x, y, z : ndarray
        GSM coordinates in Re
        
    Returns
    -------
    bx, by, bz : ndarray
        Field components in nT
    """
    # Model coefficients
    f = np.array([-71.09346626, -1014.308601, -1272.939359, -3224.935936, -44546.86232])
    b = np.array([10.90101242, 12.68393898, 13.51791954, 14.86775017, 15.12306404])
    c = np.array([0.7954069972, 0.6716601849, 1.174866319, 2.565249920, 10.01986790])
    
    # Calculate rho and derivatives
    rho = np.sqrt(x**2 + y**2)
    # Safe division for derivatives
    rho_safe = np.where(rho < 1e-10, 1e-10, rho)
    drhodx = x / rho_safe
    drhody = y / rho_safe
    # Handle rho=0 case
    drhodx = np.where(rho < 1e-10, 0.0, drhodx)
    drhody = np.where(rho < 1e-10, 0.0, drhody)
    
    # Calculate sheet thickness
    dex = np.exp(np.clip(x / 7.0, -740.0, 88.0))
    d = d0 + deltady * (y / 20.0)**2 + deltadx * dex
    dddy = deltady * y * 0.005
    dddx = deltadx / 7.0 * dex
    
    # Calculate dzeta and derivatives
    dzeta = np.sqrt(z**2 + d**2)
    ddzetadx = d * dddx / dzeta
    ddzetady = d * dddy / dzeta
    ddzetadz = z / dzeta
    
    # Initialize output arrays
    dbx = np.zeros_like(x)
    dby = np.zeros_like(y)
    dbz = np.zeros_like(z)
    
    # Vectorized loop over harmonics
    for i in range(5):
        bi = b[i]
        ci = c[i]
        
        # Calculate s1 and s2
        s1 = np.sqrt((rho + bi)**2 + (dzeta + ci)**2)
        s2 = np.sqrt((rho - bi)**2 + (dzeta + ci)**2)
        
        # Derivatives of s1 and s2
        ds1drho = (rho + bi) / s1
        ds2drho = (rho - bi) / s2
        ds1ddz = (dzeta + ci) / s1
        ds2ddz = (dzeta + ci) / s2
        
        ds1dx = ds1drho * drhodx + ds1ddz * ddzetadx
        ds1dy = ds1drho * drhody + ds1ddz * ddzetady
        ds1dz = ds1ddz * ddzetadz
        
        ds2dx = ds2drho * drhodx + ds2ddz * ddzetadx
        ds2dy = ds2drho * drhody + ds2ddz * ddzetady
        ds2dz = ds2ddz * ddzetadz
        
        # Calculate asas and derivatives
        s1ts2 = s1 * s2
        s1ps2 = s1 + s2
        s1ps2sq = s1ps2**2
        
        fac1 = np.sqrt(s1ps2sq - (2 * bi)**2)
        asas = fac1 / (s1ts2 * s1ps2sq)
        dasds1 = (1 / (fac1 * s2) - asas / s1ps2 * (s2 * s2 + s1 * (3 * s1 + 4 * s2))) / (s1 * s1ps2)
        dasds2 = (1 / (fac1 * s1) - asas / s1ps2 * (s1 * s1 + s2 * (3 * s2 + 4 * s1))) / (s2 * s1ps2)
        
        dasdx = dasds1 * ds1dx + dasds2 * ds2dx
        dasdy = dasds1 * ds1dy + dasds2 * ds2dy
        dasdz = dasds1 * ds1dz + dasds2 * ds2dz
        
        # Accumulate field components
        dbx = dbx - f[i] * x * dasdz
        dby = dby - f[i] * y * dasdz
        dbz = dbz + f[i] * (2 * asas + x * dasdx + y * dasdy)
    
    return dbx, dby, dbz


def shlcar5x5_vectorized(a: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                        dshift: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized 5x5 Cartesian harmonic shield calculation.
    
    Returns the shielding field represented by 5x5=25 "cartesian" harmonics.
    
    Parameters
    ----------
    a : ndarray
        Model coefficients (60 elements)
    x, y, z : ndarray
        GSM coordinates in Re
    dshift : float or ndarray
        Shift parameter
        
    Returns
    -------
    dhx, dhy, dhz : ndarray
        Shielding field components
    """
    # Initialize output arrays
    dhx = np.zeros_like(x)
    dhy = np.zeros_like(y)
    dhz = np.zeros_like(z)
    
    # Extract scale parameters
    p_scales = 1.0 / a[50:55]  # rp values
    r_scales = 1.0 / a[55:60]  # rr values
    
    # Vectorized implementation with loops
    l = 0
    for i in range(5):
        rp = p_scales[i]
        cypi = np.cos(y * rp)
        sypi = np.sin(y * rp)
        
        for k in range(5):
            rr = r_scales[k]
            szrk = np.sin(z * rr)
            czrk = np.cos(z * rr)
            sqpr = np.sqrt(rp**2 + rr**2)
            # Clip argument to prevent overflow/underflow
            arg = np.clip(x * sqpr, -740.0, 88.0)
            epr = np.exp(arg)
            
            # Field components for this harmonic
            dbx = -sqpr * epr * cypi * szrk
            dby = rp * epr * sypi * szrk
            dbz = -rr * epr * cypi * czrk
            
            # Apply coefficient
            coef = a[l] + a[l + 1] * dshift
            l += 2
            
            # Accumulate
            dhx = dhx + coef * dbx
            dhy = dhy + coef * dby
            dhz = dhz + coef * dbz
    
    return dhx, dhy, dhz


def unwarped_vectorized(iopt: int, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                       dxshift1: Union[float, np.ndarray], dxshift2: Union[float, np.ndarray],
                       d: Union[float, np.ndarray], deltady: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                              np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized calculation of unwarped tail field modes.
    
    Calculates GSM components of the shielded field of two tail modes with unit amplitudes,
    without any warping or bending.
    
    Parameters
    ----------
    iopt : int
        Tail field mode flag:
        0 - both modes added
        1 - mode 1 only
        2 - mode 2 only
    x, y, z : ndarray
        GSM coordinates in Re
    dxshift1, dxshift2 : float or ndarray
        X-shift parameters for modes 1 and 2
    d : float or ndarray
        Thickness parameter
    deltady : float
        Y-dependent thickness variation
        
    Returns
    -------
    bx1, by1, bz1, bx2, by2, bz2 : ndarray
        Field components for modes 1 and 2
    """
    # Mode 1 coefficients
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
        6.677425469,1.398283308,2.839005878,31.24817706,24.53577264
    ])
    
    # Mode 2 coefficients
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
        23.69472951,17.24977936,13.64902647,68.40989058,11.67828167
    ])
    
    # Constants
    deltadx1, alpha1, xshift1 = 1.0, 1.1, 6.0
    deltadx2, alpha2, xshift2 = 0.0, 0.25, 4.0
    xm1, xm2 = -12.0, -12.0
    
    # Initialize output arrays
    bx1 = np.zeros_like(x)
    by1 = np.zeros_like(y)
    bz1 = np.zeros_like(z)
    bx2 = np.zeros_like(x)
    by2 = np.zeros_like(y)
    bz2 = np.zeros_like(z)
    
    # Mode 1
    if iopt < 2:  # iopt = 0 or 1
        xsc1 = (x - xshift1 - dxshift1) * alpha1 - xm1 * (alpha1 - 1)
        ysc1 = y * alpha1
        zsc1 = z * alpha1
        d0sc1 = d * alpha1
        
        fx1, fy1, fz1 = taildisk_vectorized(d0sc1, deltadx1, deltady, xsc1, ysc1, zsc1)
        hx1, hy1, hz1 = shlcar5x5_vectorized(a1, x, y, z, dxshift1)
        
        bx1 = fx1 + hx1
        by1 = fy1 + hy1
        bz1 = fz1 + hz1
    
    # Mode 2
    if iopt != 1:  # iopt = 0 or 2
        xsc2 = (x - xshift2 - dxshift2) * alpha2 - xm2 * (alpha2 - 1)
        ysc2 = y * alpha2
        zsc2 = z * alpha2
        d0sc2 = d * alpha2
        
        fx2, fy2, fz2 = taildisk_vectorized(d0sc2, deltadx2, deltady, xsc2, ysc2, zsc2)
        hx2, hy2, hz2 = shlcar5x5_vectorized(a2, x, y, z, dxshift2)
        
        bx2 = fx2 + hx2
        by2 = fy2 + hy2
        bz2 = fz2 + hz2
    
    return bx1, by1, bz1, bx2, by2, bz2


def warped_vectorized(iopt: int, ps: float, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     dxshift1: Union[float, np.ndarray], dxshift2: Union[float, np.ndarray],
                     d: Union[float, np.ndarray], deltady: float,
                     g: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                           np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized calculation of warped tail field.
    
    Calculates GSM components of the warped field for two tail unit modes.
    The warping deformation is imposed on the unwarped field.
    
    Parameters
    ----------
    iopt : int
        Tail field mode flag (0=both, 1=mode 1, 2=mode 2)
    ps : float
        Dipole tilt angle in radians
    x, y, z : ndarray
        GSM coordinates in Re
    dxshift1, dxshift2 : float or ndarray
        X-shift parameters
    d : float or ndarray
        Thickness parameter
    deltady : float
        Y-dependent thickness variation
    g : float or ndarray
        Warping parameter
        
    Returns
    -------
    bx1, by1, bz1, bx2, by2, bz2 : ndarray
        Warped field components for modes 1 and 2
    """
    # Constants
    dgdx = 0.0
    xl = 20.0
    dxldx = 0.0
    
    sps = np.sin(ps)
    rho2 = y**2 + z**2
    rho = np.sqrt(rho2)
    
    # Handle y=0, z=0 case
    mask_zero = (y == 0) & (z == 0)
    if np.any(mask_zero):
        phi = np.zeros_like(y)
        cphi = np.ones_like(y)
        sphi = np.zeros_like(y)
        
        # For non-zero points
        mask_nonzero = ~mask_zero
        if np.any(mask_nonzero):
            phi[mask_nonzero] = np.arctan2(z[mask_nonzero], y[mask_nonzero])
            rho_safe = np.where(rho < 1e-10, 1e-10, rho)
            cphi[mask_nonzero] = y[mask_nonzero] / rho_safe[mask_nonzero]
            sphi[mask_nonzero] = z[mask_nonzero] / rho_safe[mask_nonzero]
    else:
        phi = np.arctan2(z, y)
        cphi = y / rho
        sphi = z / rho
    
    # Warping function (matching scalar version)
    rr4l4 = rho / (rho2**2 + xl**4)
    f = phi + g * rho2 * rr4l4 * cphi * sps
    dfdphi = 1 - g * rho2 * rr4l4 * sphi * sps
    dfdrho = g * rr4l4**2 * (3 * xl**4 - rho2**2) * cphi * sps
    dfdx = rr4l4 * cphi * sps * (dgdx * rho2 - g * rho * rr4l4 * 4 * xl**3 * dxldx)
    
    cf = np.cos(f)
    sf = np.sin(f)
    
    # Get unwarped field
    bx_as1, by_as1, bz_as1, bx_as2, by_as2, bz_as2 = unwarped_vectorized(
        iopt, x, y, z, dxshift1, dxshift2, d, deltady
    )
    
    # Transform mode 1
    brho_as = by_as1 * cf + bz_as1 * sf
    bphi_as = -by_as1 * sf + bz_as1 * cf
    brho_s = brho_as * dfdphi
    bphi_s = bphi_as - rho * (bx_as1 * dfdx + brho_as * dfdrho)
    bx1 = bx_as1 * dfdphi
    by1 = brho_s * cphi - bphi_s * sphi
    bz1 = brho_s * sphi + bphi_s * cphi
    
    # Transform mode 2
    brho_as = by_as2 * cf + bz_as2 * sf
    bphi_as = -by_as2 * sf + bz_as2 * cf
    brho_s = brho_as * dfdphi
    bphi_s = bphi_as - rho * (bx_as2 * dfdx + brho_as * dfdrho)
    bx2 = bx_as2 * dfdphi
    by2 = brho_s * cphi - bphi_s * sphi
    bz2 = brho_s * sphi + bphi_s * cphi
    
    return bx1, by1, bz1, bx2, by2, bz2


def deformed_vectorized(iopt: int, ps: float, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                       dxshift1: Union[float, np.ndarray], dxshift2: Union[float, np.ndarray],
                       d: Union[float, np.ndarray], deltady: float,
                       g: Union[float, np.ndarray], rh0: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                                              np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized calculation of deformed tail field.
    
    Calculates GSM components of two unit-amplitude tail field modes, taking into account
    both effects of dipole tilt: warping in y-z (done by warped) and bending
    in x-z (done by this function).
    
    Parameters
    ----------
    iopt : int
        Tail field mode flag (0=both modes, 1=mode 1 only, 2=mode 2 only)
    ps : float
        Dipole tilt angle in radians
    x, y, z : ndarray
        GSM coordinates in Re
    dxshift1, dxshift2 : float or ndarray
        X-shift parameters
    d : float or ndarray
        Thickness parameter
    deltady : float
        Y-dependent thickness variation
    g : float or ndarray
        Warping parameter
    rh0 : float or ndarray
        Hinge distance parameter
        
    Returns
    -------
    bx1, by1, bz1, bx2, by2, bz2 : ndarray
        Deformed field components for modes 1 and 2
    """
    # Constants
    rh2 = -5.2
    ieps = 3
    
    # Calculate deformation parameters
    sps = np.sin(ps)
    r2 = x**2 + y**2 + z**2
    r = np.sqrt(r2)
    
    # Safe division for zr
    r_safe = np.where(r < 1e-10, 1e-10, r)
    zr = z / r_safe
    
    # Hinge distance
    rh = rh0 + rh2 * zr**2
    # Safe computation of derivatives
    drhdr = np.where(r < 1e-10, 0.0, -zr / r * 2 * rh2 * zr)
    drhdz = np.where(r < 1e-10, 0.0, 2 * rh2 * zr / r)
    
    # Deformation function
    rrh = r / rh
    f = 1 / (1 + rrh**ieps)**(1 / ieps)
    dfdr = -rrh**(ieps - 1) * f**(ieps + 1) / rh
    dfdrh = -rrh * dfdr
    
    # Deformed tilt angles
    spsas = sps * f
    spsas = np.clip(spsas, -1.0, 1.0)  # Ensure valid range
    cpsas = np.sqrt(1 - spsas**2)
    
    # Deformed coordinates
    xas = x * cpsas - z * spsas
    zas = x * spsas + z * cpsas
    
    # Derivatives for coordinate transformation
    # Safe division by r
    facps = np.where(r < 1e-10, 0.0, sps / cpsas * (dfdr + dfdrh * drhdr) / r)
    psasx = facps * x
    psasy = facps * y
    psasz = facps * z + sps / cpsas * dfdrh * drhdz
    
    dxasdx = cpsas - zas * psasx
    dxasdy = -zas * psasy
    dxasdz = -spsas - zas * psasz
    dzasdx = spsas + xas * psasx
    dzasdy = xas * psasy
    dzasdz = cpsas + xas * psasz
    
    # Jacobian terms
    fac1 = dxasdz * dzasdy - dxasdy * dzasdz
    fac2 = dxasdx * dzasdz - dxasdz * dzasdx
    fac3 = dzasdx * dxasdy - dxasdx * dzasdy
    
    # Get warped field in deformed coordinates
    bxas1, byas1, bzas1, bxas2, byas2, bzas2 = warped_vectorized(
        iopt, ps, xas, y, zas, dxshift1, dxshift2, d, deltady, g
    )
    
    # Transform back to original coordinates (mode 1)
    bx1 = bxas1 * dzasdz - bzas1 * dxasdz + byas1 * fac1
    by1 = byas1 * fac2
    bz1 = bzas1 * dxasdx - bxas1 * dzasdx + byas1 * fac3
    
    # Transform back to original coordinates (mode 2)
    bx2 = bxas2 * dzasdz - bzas2 * dxasdz + byas2 * fac1
    by2 = byas2 * fac2
    bz2 = bzas2 * dxasdx - bxas2 * dzasdx + byas2 * fac3
    
    return bx1, by1, bz1, bx2, by2, bz2