"""
Vectorized implementation of the T01 ring current components.

This module provides vectorized versions of the ring current calculations
including symmetric (SRC) and partial (PRC) ring current fields.
"""

import numpy as np
from typing import Tuple, Union


def rc_symm_vectorized(x: Union[float, np.ndarray], y: Union[float, np.ndarray], z: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Vectorized calculation of symmetric ring current field.
    
    Calculates the field components from a model ring current due to its symmetric part.
    
    Parameters
    ----------
    x, y, z : ndarray
        GSM coordinates in Re
        
    Returns
    -------
    bx, by, bz : ndarray
        Field components in GSM system in nT
    """
    # Constants
    ds = 1e-2
    dc = 0.99994999875
    d = 1e-4
    drd = 5e3    # Check if inputs are scalars
    scalar_input = np.isscalar(x)
    

    
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    # Calculate coordinates
    rho2 = x**2 + y**2
    r2 = rho2 + z**2
    r = np.sqrt(r2)
    
    # Safe division
    r_safe = np.where(r < 1e-10, 1e-10, r)
    sint = np.sqrt(rho2) / r_safe
    cost = z / r_safe
    
    # Initialize output arrays
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    
    # Handle two cases: close to z-axis and general case
    mask_axis = sint < ds
    mask_general = ~mask_axis
    
    # Case 1: Close to z-axis (linear approximation)
    if np.any(mask_axis):
        idx = mask_axis
        rp = r[idx] + d
        rm = r[idx] - d
        
        # Calculate ap values
        a = ap_vectorized(r[idx], ds, dc) / ds
        dardr = (rp * ap_vectorized(rp, ds, dc) - rm * ap_vectorized(rm, ds, dc)) * drd
        
        fxy = z[idx] * (2 * a - dardr) / (r[idx] * r2[idx])
        bx[idx] = fxy * x[idx]
        by[idx] = fxy * y[idx]
        bz[idx] = (2 * a * cost[idx]**2 + dardr * sint[idx]**2) / r[idx]
    
    # Case 2: General case
    if np.any(mask_general):
        idx = mask_general
        theta = np.arctan2(sint[idx], cost[idx])
        tp = theta + d
        tm = theta - d
        
        sintp = np.sin(tp)
        sintm = np.sin(tm)
        costp = np.cos(tp)
        costm = np.cos(tm)
        
        rp = r[idx] + d
        rm = r[idx] - d
        
        # Calculate field components
        ap_tp = ap_vectorized(r[idx], sintp, costp)
        ap_tm = ap_vectorized(r[idx], sintm, costm)
        ap_rp = ap_vectorized(rp, sint[idx], cost[idx])
        ap_rm = ap_vectorized(rm, sint[idx], cost[idx])
        
        br = (sintp * ap_tp - sintm * ap_tm) / (r[idx] * sint[idx]) * drd
        bt = (rm * ap_rm - rp * ap_rp) / r[idx] * drd
        
        fxy = (br + bt * cost[idx] / sint[idx]) / r[idx]
        bx[idx] = fxy * x[idx]
        by[idx] = fxy * y[idx]
        bz[idx] = br * cost[idx] - bt * sint[idx]
    
    # Return scalars if inputs were scalars
    if scalar_input:
        return bx.item(), by.item(), bz.item()
    return bx, by, bz


def ap_vectorized(r: np.ndarray, sint: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Vectorized azimuthal vector potential for symmetric ring current.
    
    Parameters
    ----------
    r : ndarray
        Radial distance
    sint, cost : ndarray
        Sin and cos of colatitude
        
    Returns
    -------
    ap : ndarray
        Azimuthal component of vector potential
    """
    # Model parameters
    a1, a2 = -456.5289941, 375.9055332
    rrc1, dd1 = 4.274684950, 2.439528329
    rrc2, dd2 = 3.367557287, 3.146382545
    p1, r1, dr1, dla1 = -0.2291904607, 3.746064740, 1.508802177, 0.5873525737
    p2, r2, dr2, dla2 = 0.1556236119, 4.993638842, 3.324180497, 0.4368407663
    p3, r3, dr3 = 0.1855957207, 2.969226745, 2.243367377
    
    # Ensure arrays
    r = np.atleast_1d(r)
    sint = np.atleast_1d(sint)
    cost = np.atleast_1d(cost)
    
    # Handle proximity to axis
    sint1 = np.copy(sint)
    cost1 = np.copy(cost)
    prox = sint < 1e-2
    
    if np.any(prox):
        sint1[prox] = 1e-2
        cost1[prox] = 0.99994999875
    
    # Coordinate transformation
    # Safe divisions for alpha and gamma
    r_safe = np.where(r < 1e-10, 1e-10, r)
    alpha = sint1**2 / r_safe
    gamma = cost1 / r_safe**2
    
    # Calculate exponential arguments
    arg1 = -((r - r1) / dr1)**2 - (cost1 / dla1)**2
    arg2 = -((r - r2) / dr2)**2 - (cost1 / dla2)**2
    arg3 = -((r - r3) / dr3)**2
    
    # Safe exponentials
    dexp1 = np.where(arg1 < -740, 0.0, np.exp(arg1))
    dexp2 = np.where(arg2 < -740, 0.0, np.exp(arg2))
    dexp3 = np.where(arg3 < -740, 0.0, np.exp(arg3))
    
    # Deformed coordinates
    alpha_s = alpha * (1 + p1 * dexp1 + p2 * dexp2 + p3 * dexp3)
    gamma_s = gamma
    gammas2 = gamma_s**2
    
    # Inverse transformation
    alsqh = alpha_s**2 / 2
    f = 64/27 * gammas2 + alsqh**2
    q = (np.sqrt(f) + alsqh)**(1/3)
    c = q - 4 * gammas2**(1/3) / (3 * q)
    c = np.maximum(c, 0)
    
    g = np.sqrt(c**2 + 4 * gammas2**(1/3))
    rs = 4 / ((np.sqrt(2 * g - c) + np.sqrt(c)) * (g + c))
    costs = gamma_s * rs**2
    sints = np.sqrt(np.maximum(1 - costs**2, 0))
    rhos = rs * sints
    zs = rs * costs
    
    # Calculate elliptic integrals for two current loops
    ap = np.zeros_like(r)
    
    # First loop
    p = (rrc1 + rhos)**2 + zs**2 + dd1**2
    xk2 = 4 * rrc1 * rhos / p
    xk = np.sqrt(xk2)
    xkrho12 = xk * np.sqrt(rhos)
    
    # Elliptic integrals using polynomial approximations
    xk2s = 1 - xk2
    dl = np.log(1 / xk2s)
    
    elk = (1.38629436112 + xk2s * (0.09666344259 + xk2s * (0.03590092383 + 
           xk2s * (0.03742563713 + xk2s * 0.01451196212))) +
           dl * (0.5 + xk2s * (0.12498593597 + xk2s * (0.06880248576 + 
           xk2s * (0.03328355346 + xk2s * 0.00441787012)))))
    
    ele = (1 + xk2s * (0.44325141463 + xk2s * (0.0626060122 + 
           xk2s * (0.04757383546 + xk2s * 0.01736506451))) +
           dl * xk2s * (0.2499836831 + xk2s * (0.09200180037 + 
           xk2s * (0.04069697526 + xk2s * 0.00526449639))))
    
    aphi1 = ((1 - xk2 * 0.5) * elk - ele) / xkrho12
    
    # Second loop
    p = (rrc2 + rhos)**2 + zs**2 + dd2**2
    xk2 = 4 * rrc2 * rhos / p
    xk = np.sqrt(xk2)
    xkrho12 = xk * np.sqrt(rhos)
    
    xk2s = 1 - xk2
    dl = np.log(1 / xk2s)
    
    elk = (1.38629436112 + xk2s * (0.09666344259 + xk2s * (0.03590092383 + 
           xk2s * (0.03742563713 + xk2s * 0.01451196212))) +
           dl * (0.5 + xk2s * (0.12498593597 + xk2s * (0.06880248576 + 
           xk2s * (0.03328355346 + xk2s * 0.00441787012)))))
    
    ele = (1 + xk2s * (0.44325141463 + xk2s * (0.0626060122 + 
           xk2s * (0.04757383546 + xk2s * 0.01736506451))) +
           dl * xk2s * (0.2499836831 + xk2s * (0.09200180037 + 
           xk2s * (0.04069697526 + xk2s * 0.00526449639))))
    
    aphi2 = ((1 - xk2 * 0.5) * elk - ele) / xkrho12
    
    # Total potential
    ap = a1 * aphi1 + a2 * aphi2
    
    # Linear interpolation for proximity cases
    if np.any(prox):
        ap[prox] = ap[prox] * sint[prox] / sint1[prox]
    
    # Return scalar if input was scalar
    if np.isscalar(r):
        return ap.item()
    return ap


def prc_symm_vectorized(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized calculation of symmetric part of partial ring current.
    
    Parameters
    ----------
    x, y, z : ndarray
        GSM coordinates in Re
        
    Returns
    -------
    bx, by, bz : ndarray
        Field components in GSM system in nT
    """
    # Constants
    ds = 1e-2
    dc = 0.99994999875
    d = 1e-4
    drd = 5e3    # Check if inputs are scalars
    scalar_input = np.isscalar(x)
    

    
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    # Calculate coordinates
    rho2 = x**2 + y**2
    r2 = rho2 + z**2
    r = np.sqrt(r2)
    
    # Safe division
    r_safe = np.where(r < 1e-10, 1e-10, r)
    sint = np.sqrt(rho2) / r_safe
    cost = z / r_safe
    
    # Initialize output arrays
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    
    # Handle two cases: close to z-axis and general case
    mask_axis = sint < ds
    mask_general = ~mask_axis
    
    # Case 1: Close to z-axis
    if np.any(mask_axis):
        idx = mask_axis
        rp = r[idx] + d
        rm = r[idx] - d
        
        # Calculate apprc values
        a = apprc_vectorized(r[idx], ds, dc) / ds
        dardr = (rp * apprc_vectorized(rp, ds, dc) - rm * apprc_vectorized(rm, ds, dc)) * drd
        
        fxy = z[idx] * (2 * a - dardr) / (r[idx] * r2[idx])
        bx[idx] = fxy * x[idx]
        by[idx] = fxy * y[idx]
        bz[idx] = (2 * a * cost[idx]**2 + dardr * sint[idx]**2) / r[idx]
    
    # Case 2: General case
    if np.any(mask_general):
        idx = mask_general
        theta = np.arctan2(sint[idx], cost[idx])
        tp = theta + d
        tm = theta - d
        
        sintp = np.sin(tp)
        sintm = np.sin(tm)
        costp = np.cos(tp)
        costm = np.cos(tm)
        
        rp = r[idx] + d
        rm = r[idx] - d
        
        # Calculate field components
        ap_tp = apprc_vectorized(r[idx], sintp, costp)
        ap_tm = apprc_vectorized(r[idx], sintm, costm)
        ap_rp = apprc_vectorized(rp, sint[idx], cost[idx])
        ap_rm = apprc_vectorized(rm, sint[idx], cost[idx])
        
        br = (sintp * ap_tp - sintm * ap_tm) / (r[idx] * sint[idx]) * drd
        bt = (rm * ap_rm - rp * ap_rp) / r[idx] * drd
        
        fxy = (br + bt * cost[idx] / sint[idx]) / r[idx]
        bx[idx] = fxy * x[idx]
        by[idx] = fxy * y[idx]
        bz[idx] = br * cost[idx] - bt * sint[idx]
    
    # Return scalars if inputs were scalars
    if scalar_input:
        return bx.item(), by.item(), bz.item()
    return bx, by, bz


def apprc_vectorized(r: np.ndarray, sint: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Vectorized azimuthal vector potential for partial ring current symmetric part.
    
    Parameters
    ----------
    r : ndarray
        Radial distance
    sint, cost : ndarray
        Sin and cos of colatitude
        
    Returns
    -------
    apprc : ndarray
        Azimuthal component of vector potential
    """
    # Model parameters (35 parameters)
    params = [
        -80.11202281, 12.58246758, 6.560486035, 1.930711037, 3.827208119,
        0.7789990504, 0.3058309043, 0.1817139853, 0.1257532909, 3.422509402,
        0.04742939676, -4.800458958, -0.02845643596, 0.2188114228, 2.545944574,
        0.00813272793, 0.35868244, 103.1601001, -0.00764731187, 0.1046487459,
        2.958863546, 0.01172314188, 0.4382872938, 0.01134908150, 14.51339943,
        0.2647095287, 0.07091230197, 0.01512963586, 6.861329631, 0.1677400816,
        0.04433648846, 0.05553741389, 0.7665599464, 0.7277854652
    ]
    
    a1, a2, rrc1, dd1, rrc2, dd2 = params[0:6]
    p1, alpha1, dal1, beta1, dg1 = params[6:11]
    p2, alpha2, dal2, beta2, dg2, beta3 = params[11:17]
    p3, alpha3, dal3, beta4, dg3, beta5 = params[17:23]
    q0, q1, alpha4, dal4, dg4 = params[23:28]
    q2, alpha5, dal5, dg5, beta6, beta7 = params[28:34]
    
    # Ensure arrays
    r = np.atleast_1d(r)
    sint = np.atleast_1d(sint)
    cost = np.atleast_1d(cost)
    
    # Handle proximity to axis
    sint1 = np.copy(sint)
    cost1 = np.copy(cost)
    prox = sint < 1e-2
    
    if np.any(prox):
        sint1[prox] = 1e-2
        cost1[prox] = 0.99994999875
    
    # Coordinate transformation
    # Safe divisions for alpha and gamma
    r_safe = np.where(r < 1e-10, 1e-10, r)
    alpha = sint1**2 / r_safe
    gamma = cost1 / r_safe**2
    
    # Calculate exponential arguments
    arg1 = -(gamma / dg1)**2
    arg2 = -((alpha - alpha4) / dal4)**2 - (gamma / dg4)**2
    
    # Safe exponentials
    dexp1 = np.where(arg1 < -740, 0.0, np.exp(arg1))
    dexp2 = np.where(arg2 < -740, 0.0, np.exp(arg2))
    
    # Deformed alpha
    term1 = p1 / (1 + ((alpha - alpha1) / dal1)**2)**beta1 * dexp1
    term2 = (p2 * (alpha - alpha2) / 
             (1 + ((alpha - alpha2) / dal2)**2)**beta2 / 
             (1 + (gamma / dg2)**2)**beta3)
    term3 = (p3 * (alpha - alpha3)**2 / 
             (1 + ((alpha - alpha3) / dal3)**2)**beta4 / 
             (1 + (gamma / dg3)**2)**beta5)
    alpha_s = alpha * (1 + term1 + term2 + term3)
    
    # Deformed gamma
    term1 = q1 * (alpha - alpha4) * dexp2
    term2 = (q2 * (alpha - alpha5) / 
             (1 + ((alpha - alpha5) / dal5)**2)**beta6 / 
             (1 + (gamma / dg5)**2)**beta7)
    gamma_s = gamma * (1 + q0 + term1 + term2)
    
    gammas2 = gamma_s**2
    
    # Inverse transformation
    alsqh = alpha_s**2 / 2
    f = 64/27 * gammas2 + alsqh**2
    q = (np.sqrt(f) + alsqh)**(1/3)
    c = q - 4 * gammas2**(1/3) / (3 * q)
    c = np.maximum(c, 0)
    
    g = np.sqrt(c**2 + 4 * gammas2**(1/3))
    rs = 4 / ((np.sqrt(2 * g - c) + np.sqrt(c)) * (g + c))
    costs = gamma_s * rs**2
    sints = np.sqrt(np.maximum(1 - costs**2, 0))
    rhos = rs * sints
    zs = rs * costs
    
    # Calculate elliptic integrals for two current loops
    apprc = np.zeros_like(r)
    
    # First loop
    p = (rrc1 + rhos)**2 + zs**2 + dd1**2
    xk2 = 4 * rrc1 * rhos / p
    xk = np.sqrt(xk2)
    xkrho12 = xk * np.sqrt(rhos)
    
    # Elliptic integrals
    xk2s = 1 - xk2
    dl = np.log(1 / xk2s)
    
    elk = (1.38629436112 + xk2s * (0.09666344259 + xk2s * (0.03590092383 + 
           xk2s * (0.03742563713 + xk2s * 0.01451196212))) +
           dl * (0.5 + xk2s * (0.12498593597 + xk2s * (0.06880248576 + 
           xk2s * (0.03328355346 + xk2s * 0.00441787012)))))
    
    ele = (1 + xk2s * (0.44325141463 + xk2s * (0.0626060122 + 
           xk2s * (0.04757383546 + xk2s * 0.01736506451))) +
           dl * xk2s * (0.2499836831 + xk2s * (0.09200180037 + 
           xk2s * (0.04069697526 + xk2s * 0.00526449639))))
    
    aphi1 = ((1 - xk2 * 0.5) * elk - ele) / xkrho12
    
    # Second loop
    p = (rrc2 + rhos)**2 + zs**2 + dd2**2
    xk2 = 4 * rrc2 * rhos / p
    xk = np.sqrt(xk2)
    xkrho12 = xk * np.sqrt(rhos)
    
    xk2s = 1 - xk2
    dl = np.log(1 / xk2s)
    
    elk = (1.38629436112 + xk2s * (0.09666344259 + xk2s * (0.03590092383 + 
           xk2s * (0.03742563713 + xk2s * 0.01451196212))) +
           dl * (0.5 + xk2s * (0.12498593597 + xk2s * (0.06880248576 + 
           xk2s * (0.03328355346 + xk2s * 0.00441787012)))))
    
    ele = (1 + xk2s * (0.44325141463 + xk2s * (0.0626060122 + 
           xk2s * (0.04757383546 + xk2s * 0.01736506451))) +
           dl * xk2s * (0.2499836831 + xk2s * (0.09200180037 + 
           xk2s * (0.04069697526 + xk2s * 0.00526449639))))
    
    aphi2 = ((1 - xk2 * 0.5) * elk - ele) / xkrho12
    
    # Total potential
    apprc = a1 * aphi1 + a2 * aphi2
    
    # Linear interpolation for proximity cases
    if np.any(prox):
        apprc[prox] = apprc[prox] * sint[prox] / sint1[prox]
    
    # Return scalar if input was scalar
    if np.isscalar(r):
        return apprc.item()
    return apprc


def prc_quad_vectorized(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized quadrupole partial ring current field.
    
    Parameters
    ----------
    x, y, z : ndarray
        GSM coordinates in Re
        
    Returns
    -------
    bx, by, bz : ndarray
        Field components in GSM system in nT
    """
    # Constants
    d = 1e-4
    dd = 2e-4
    ds = 1e-2
    dc = 0.99994999875
    
    # Check if inputs are scalars
    scalar_input = np.isscalar(x)
    
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    # Calculate coordinates
    rho2 = x**2 + y**2
    r = np.sqrt(rho2 + z**2)
    rho = np.sqrt(rho2)
    
    # Safe division
    r_safe = np.where(r < 1e-10, 1e-10, r)
    rho_safe = np.where(rho < 1e-10, 1e-10, rho)
    sint = rho / r_safe
    cost = z / r_safe
    
    # Initialize output
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    
    # Handle two cases
    mask_general = sint > ds
    mask_axis = ~mask_general
    
    # General case (sint > ds)
    if np.any(mask_general):
        idx = mask_general
        cphi = x[idx] / rho_safe[idx]
        sphi = y[idx] / rho_safe[idx]
        
        # Calculate br and bt at current position
        br = br_prc_q_vectorized(r[idx], sint[idx], cost[idx])
        bt = bt_prc_q_vectorized(r[idx], sint[idx], cost[idx])
        
        # Calculate derivatives
        rp = r[idx] + d
        rm = r[idx] - d
        dbrr = (br_prc_q_vectorized(rp, sint[idx], cost[idx]) - 
                br_prc_q_vectorized(rm, sint[idx], cost[idx])) / dd
        
        theta = np.arctan2(sint[idx], cost[idx])
        tp = theta + d
        tm = theta - d
        sintp = np.sin(tp)
        costp = np.cos(tp)
        sintm = np.sin(tm)
        costm = np.cos(tm)
        
        dbtt = (bt_prc_q_vectorized(r[idx], sintp, costp) - 
                bt_prc_q_vectorized(r[idx], sintm, costm)) / dd
        
        # Field components
        bx[idx] = sint[idx] * (br + (br + r[idx] * dbrr + dbtt) * sphi**2) + cost[idx] * bt
        by[idx] = -sint[idx] * sphi * cphi * (br + r[idx] * dbrr + dbtt)
        bz[idx] = (br * cost[idx] - bt * sint[idx]) * cphi
    
    # Near-axis case (sint <= ds)
    if np.any(mask_axis):
        idx = mask_axis
        st = ds
        ct = dc * np.sign(z[idx])
        ct = np.where(z[idx] == 0, dc, ct)
        
        theta = np.arctan2(st, ct)
        tp = theta + d
        tm = theta - d
        sintp = np.sin(tp)
        costp = np.cos(tp)
        sintm = np.sin(tm)
        costm = np.cos(tm)
        
        br = br_prc_q_vectorized(r[idx], st, ct)
        bt = bt_prc_q_vectorized(r[idx], st, ct)
        
        rp = r[idx] + d
        rm = r[idx] - d
        dbrr = (br_prc_q_vectorized(rp, st, ct) - 
                br_prc_q_vectorized(rm, st, ct)) / dd
        dbtt = (bt_prc_q_vectorized(r[idx], sintp, costp) - 
                bt_prc_q_vectorized(r[idx], sintm, costm)) / dd
        
        fcxy = r[idx] * dbrr + dbtt
        rst2 = (r[idx] * st)**2
        
        bx[idx] = (br * (x[idx]**2 + 2 * y[idx]**2) + fcxy * y[idx]**2) / rst2 + bt * cost[idx]
        by[idx] = -(br + fcxy) * x[idx] * y[idx] / rst2
        # Safe division by st
        st_safe = np.where(st < 1e-10, 1e-10, st)
        bz[idx] = (br * cost[idx] / st_safe - bt) * x[idx] / r[idx]
    
    # Return scalars if inputs were scalars
    if scalar_input:
        return bx.item(), by.item(), bz.item()
    return bx, by, bz


def br_prc_q_vectorized(r: np.ndarray, sint: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Vectorized radial component of quadrupole PRC.
    
    Full implementation with all 18 terms matching the scalar version.
    """
    # Ensure arrays
    r = np.atleast_1d(r)
    sint = np.atleast_1d(sint)
    cost = np.atleast_1d(cost)
    scalar_input = len(r) == 1
    
    # Coefficients from scalar version
    a = np.array([-21.2666329, 32.24527521, -6.062894078, 7.515660734, 233.7341288, -227.1195714,
                  8.483233889, 16.80642754, -24.63534184, 9.067120578, -1.052686913, -12.08384538,
                  18.61969572, -12.71686069, 47017.35679, -50646.71204, 7746.058231, 1.531069371])
    
    xk = np.array([2.318824273, 0.7955534018, 3.477425908, 1.922155110])
    al = np.array([0.1417519429, 0.1401142771, 0.1485233485, 0.1295221828, 0.1811846095, 0.2265212965])
    dal = np.array([0.006388013110, 0.02306094179, 0.02319676273, 0.01753008801, 0.04841237481, 0.1301957209])
    
    b = np.array([5.303934488, 3.462235072, 7.830223587])
    be = np.array([4.213397467, 2.568743010, 8.492933868])
    
    dg1 = 0.01125504083
    dg2 = 0.01981805097
    c1 = 6.557801891
    c2 = 6.348576071
    c3 = 5.744436687
    drm = 0.5654023158
    
    # Basic parameters
    sint2 = sint**2
    sc = sint * cost
    # Safe divisions
    r_safe = np.where(r < 1e-10, 1e-10, r)
    alpha = sint2 / r_safe
    gamma = cost / r_safe**2
    
    # Term 1
    f1, fa1, fs1 = ffs_vectorized(alpha, al[0], dal[0])
    r_over_b1 = np.minimum(r / b[0], 100.0)
    d1 = sc * f1**xk[0] / (r_over_b1**be[0] + 1.0)
    d2 = d1 * cost**2
    
    # Term 3 (uses fs from f2)
    f2, fa2, fs2 = ffs_vectorized(alpha, al[1], dal[1])
    r_over_b2 = np.minimum(r / b[1], 100.0)
    d3 = sc * fs2**xk[1] / (r_over_b2**be[1] + 1.0)
    d4 = d3 * cost**2
    
    # Term 5 (uses fs from f3, complex expression)
    f3, fa3, fs3 = ffs_vectorized(alpha, al[2], dal[2])
    r_over_b3 = np.minimum(r / b[2], 100.0)
    d5 = sc * (alpha**xk[2]) * (fs3**xk[3]) / (r_over_b3**be[2] + 1.0)
    d6 = d5 * cost**2
    
    # Terms 7-14 (using proper indexing)
    arga = ((alpha - al[3]) / dal[3])**2 + 1.0
    argg = 1.0 + (gamma / dg1)**2
    d7 = sc / (arga * argg)
    d8 = d7 / arga
    d9 = d8 / arga
    d10 = d9 / arga
    
    arga = ((alpha - al[4]) / dal[4])**2 + 1.0
    argg = 1.0 + (gamma / dg2)**2
    d11 = sc / (arga * argg)
    d12 = d11 / arga
    d13 = d12 / arga
    d14 = d13 / arga
    
    # Terms 15-17
    d15 = sc / (r**4 + c1**4)
    d16 = sc / (r**4 + c2**4) * cost**2
    d17 = sc / (r**4 + c3**4) * cost**4
    
    # Term 18 (uses fs from f6)
    f6, fa6, fs6 = ffs_vectorized(alpha, al[5], dal[5])
    d18 = sc * fs6 / (1.0 + ((r - 1.2) / drm)**2)
    
    
    # Calculate final result
    result = (a[0] * d1 + a[1] * d2 + a[2] * d3 + a[3] * d4 + 
              a[4] * d5 + a[5] * d6 + a[6] * d7 + a[7] * d8 + 
              a[8] * d9 + a[9] * d10 + a[10] * d11 + a[11] * d12 + 
              a[12] * d13 + a[13] * d14 + a[14] * d15 + a[15] * d16 + 
              a[16] * d17 + a[17] * d18)
    
    # Return scalar if input was scalar
    if scalar_input:
        return result[0]
    return result


def bt_prc_q_vectorized(r: np.ndarray, sint: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Vectorized theta component of quadrupole PRC.
    
    Full implementation with all 17 terms matching the scalar version.
    """
    # Ensure arrays
    r = np.atleast_1d(r)
    sint = np.atleast_1d(sint)
    cost = np.atleast_1d(cost)
    scalar_input = len(r) == 1
    
    # Coefficients from scalar version
    a = np.array([12.74640393, -7.516393516, -5.476233865, 3.212704645, -59.10926169, 46.62198189,
                  -0.01644280062, 0.1234229112, -0.08579198697, 0.01321366966, 0.8970494003,
                  9.136186247, -38.19301215, 21.73775846, -410.0783424, -69.90832690, -848.8543440])
    
    xk = np.array([1.243288286, 1.376743507, 0.3157139940, 1.056309517])
    al = np.array([0.2071721360, 0.1568504222, 0.1701395257, 0.1280772299, 0.1648265607])
    dal = np.array([0.05030555417, 0.02092910682, 0.1019870070, 0.02189060799, 0.04701592613])
    
    b1 = 7.471332374
    b3 = 6.293740981
    be1 = 3.180533613
    be2 = 1.985148197
    be3 = 5.671824276
    
    dg1 = 0.01040696080
    dg2 = 0.01526400086
    c1 = 12.88384229
    c2 = 3.361775101
    c3 = 23.44173897
    
    # Basic parameters
    sint2 = sint**2
    cost2 = cost**2
    sc = sint * cost
    # Safe divisions
    r_safe = np.where(r < 1e-10, 1e-10, r)
    alpha = sint2 / r_safe
    gamma = cost / r_safe**2
    
    # Term 1 (no sc factor)
    f1, fa1, fs1 = ffs_vectorized(alpha, al[0], dal[0])
    r_over_b1 = np.minimum(r / b1, 100.0)
    d1 = f1**xk[0] / (r_over_b1**be1 + 1.0)
    d2 = d1 * cost2
    
    # Term 3 (uses fa from al[1])
    f2, fa2, fs2 = ffs_vectorized(alpha, al[1], dal[1])
    d3 = fa2**xk[1] / r**be2
    d4 = d3 * cost2
    
    # Term 5 (uses fs from al[2])
    f3, fa3, fs3 = ffs_vectorized(alpha, al[2], dal[2])
    r_over_b3 = np.minimum(r / b3, 100.0)
    d5 = fs3**xk[2] * alpha**xk[3] / (r_over_b3**be3 + 1.0)
    d6 = d5 * cost2
    
    # Terms 7-10: Special formulation using ffs on gamma
    f_gamma, fa_gamma, fs_gamma = ffs_vectorized(gamma, 0.0, dg1)
    fcc = 1.0 + ((alpha - al[3]) / dal[3])**2
    d7 = fs_gamma / fcc
    d8 = d7 / fcc
    d9 = d8 / fcc
    d10 = d9 / fcc
    
    # Terms 11-14
    arg = 1.0 + ((alpha - al[4]) / dal[4])**2
    d11 = 1.0 / (arg * (1.0 + (gamma / dg2)**2))
    d12 = d11 / arg
    d13 = d12 / arg
    d14 = d13 / arg
    
    # Terms 15-17: Note these use r**4 in denominator, not r**2
    d15 = 1.0 / (r**4 + c1**2)
    d16 = cost2 / (r**4 + c2**2)
    d17 = cost2**2 / (r**4 + c3**2)
    
    # Calculate final result
    result = (a[0] * d1 + a[1] * d2 + a[2] * d3 + a[3] * d4 + 
              a[4] * d5 + a[5] * d6 + a[6] * d7 + a[7] * d8 + 
              a[8] * d9 + a[9] * d10 + a[10] * d11 + a[11] * d12 + 
              a[12] * d13 + a[13] * d14 + a[14] * d15 + a[15] * d16 + 
              a[16] * d17)
    
    # Return scalar if input was scalar
    if scalar_input:
        return result[0]
    return result


def ffs_vectorized(a: np.ndarray, a0: float, da: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized auxiliary function for field calculations.
    
    Matches the scalar ffs function from t01.py.
    
    Parameters
    ----------
    a : ndarray
        Input array (typically alpha parameter)
    a0 : float
        Center parameter
    da : float
        Width parameter
        
    Returns
    -------
    f, fa, fs : ndarray
        Field function components
    """
    sq1 = np.sqrt((a + a0)**2 + da**2)
    sq2 = np.sqrt((a - a0)**2 + da**2)
    fa = 2.0 / (sq1 + sq2)
    f = fa * a
    fs = 0.5 * (sq1 + sq2) / (sq1 * sq2) * (1.0 - f * f)
    
    return f, fa, fs


def src_prc_vectorized(iopr: int, sc_sy: Union[float, np.ndarray], sc_pr: Union[float, np.ndarray],
                      phi: Union[float, np.ndarray], ps: float, 
                      x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                             np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized calculation of symmetric and partial ring current fields.
    
    Parameters
    ----------
    iopr : int
        Ring current flag (0=both, 1=SRC only, 2=PRC only)
    sc_sy, sc_pr : float or ndarray
        Scale factors for symmetric and partial components
    phi : float or ndarray
        Rotation angle (radians) of partial ring current
    ps : float
        Dipole tilt angle in radians
    x, y, z : ndarray
        GSM coordinates in Re
        
    Returns
    -------
    bxsrc, bysrc, bzsrc, bxprc, byprc, bzprc : ndarray
        Field components for symmetric and partial ring currents
    """
    # Tilt rotation
    cps = np.cos(ps)
    sps = np.sin(ps)
    
    xt = x * cps - z * sps
    zt = z * cps + x * sps
    
    # Initialize outputs
    bxsrc = np.zeros_like(x)
    bysrc = np.zeros_like(y)
    bzsrc = np.zeros_like(z)
    bxprc = np.zeros_like(x)
    byprc = np.zeros_like(y)
    bzprc = np.zeros_like(z)
    
    # Symmetric ring current
    if iopr <= 1:
        xts = xt / sc_sy
        yts = y / sc_sy
        zts = zt / sc_sy
        
        bxs, bys, bzs = rc_symm_vectorized(xts, yts, zts)
        
        # Transform back to GSM
        bxsrc = bxs * cps + bzs * sps
        bysrc = bys
        bzsrc = bzs * cps - bxs * sps
    
    # Partial ring current
    if iopr == 0 or iopr == 2:
        xta = xt / sc_pr
        yta = y / sc_pr
        zta = zt / sc_pr
        
        # Symmetric part of PRC
        bxa_s, bya_s, bza_s = prc_symm_vectorized(xta, yta, zta)
        
        # Rotate coordinates for quadrupole
        cp = np.cos(phi)
        sp = np.sin(phi)
        xr = xta * cp - yta * sp
        yr = xta * sp + yta * cp
        
        # Quadrupole field
        bxa_qr, bya_qr, bza_q = prc_quad_vectorized(xr, yr, zta)
        
        # Transform quadrupole back
        bxa_q = bxa_qr * cp + bya_qr * sp
        bya_q = -bxa_qr * sp + bya_qr * cp
        
        # Total PRC field
        bxp = bxa_s + bxa_q
        byp = bya_s + bya_q
        bzp = bza_s + bza_q
        
        # Transform back to GSM
        bxprc = bxp * cps + bzp * sps
        byprc = byp
        bzprc = bzp * cps - bxp * sps
    
    return bxsrc, bysrc, bzsrc, bxprc, byprc, bzprc


def rc_shield_vectorized(a_arr: np.ndarray, ps: float, x_sc: Union[float, np.ndarray],
                        x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized ring current shielding field.
    
    Correct implementation based on scalar rc_shield from t01.py.
    Properly accounts for dipole tilt angle using coordinate rotations.
    
    Parameters
    ----------
    a_arr : ndarray
        Coefficient array (40 elements for c_pr, 30 for c_sy)
    ps : float
        Dipole tilt angle in radians
    x_sc : float or ndarray
        Scaling factor (sc_pr - 1.0 or sc_sy - 1.0)
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
    
    # Broadcast x_sc if needed
    if x_sc.size == 1 and x.size > 1:
        x_sc = np.full_like(x, x_sc.item())
    
    # Initialize output
    bx = np.zeros_like(x)
    by = np.zeros_like(y)
    bz = np.zeros_like(z)
    
    # Scale factor
    fac_sc = (x_sc + 1.0) ** 3
    
    # Tilt parameters
    cps = np.cos(ps)
    sps = np.sin(ps)
    s3ps = 2 * cps  # Approximation for small ps (matches scalar version)
    
    # Tilt rotation angles from indices 84, 85
    pst1 = ps * a_arr[84]
    pst2 = ps * a_arr[85]
    
    st1 = np.sin(pst1)
    ct1 = np.cos(pst1)
    st2 = np.sin(pst2)
    ct2 = np.cos(pst2)
    
    # Rotated coordinates
    x1 = x * ct1 - z * st1
    z1 = x * st1 + z * ct1
    x2 = x * ct2 - z * st2
    z2 = x * st2 + z * ct2
    
    # Coefficient index
    l = 0
    
    # Two symmetries: m=0 for perpendicular, m=1 for parallel
    for m in range(2):
        # 3x3 harmonics
        for i in range(3):
            # Get harmonic parameters from correct indices
            p = a_arr[72 + i]  # p1, p2, p3 at indices 72, 73, 74
            q = a_arr[78 + i]  # q1, q2, q3 at indices 78, 79, 80
            
            # Calculate both p-based and q-based trig functions
            cypi = np.cos(y / p)
            cyqi = np.cos(y / q)
            sypi = np.sin(y / p)
            syqi = np.sin(y / q)
            
            for k in range(3):
                r = a_arr[75 + k]  # r1, r2, r3
                s = a_arr[81 + k]  # s1, s2, s3
                
                # Calculate all field quantities needed for both symmetries
                szrk = np.sin(z1 / r)
                czsk = np.cos(z2 / s)
                czrk = np.cos(z1 / r)
                szsk = np.sin(z2 / s)
                sqpr = np.sqrt(1/p**2 + 1/r**2)
                sqqs = np.sqrt(1/q**2 + 1/s**2)
                # Clip arguments to prevent overflow/underflow
                arg_pr = np.clip(x1 * sqpr, -740.0, 88.0)
                arg_qs = np.clip(x2 * sqqs, -740.0, 88.0)
                epr = np.exp(arg_pr)
                eqs = np.exp(arg_qs)
                
                # Four terms for each harmonic
                for n in range(2):
                    for nn in range(2):
                        if l >= len(a_arr):
                            continue
                            
                        # Calculate base field components
                        if m == 0:  # Perpendicular
                            fx = -sqpr * epr * cypi * szrk * fac_sc
                            fy = epr * sypi * szrk / p * fac_sc
                            fz = -epr * cypi * czrk / r * fac_sc
                        else:  # Parallel
                            fx = -sps * sqqs * eqs * cyqi * czsk * fac_sc
                            fy = sps / q * eqs * syqi * czsk * fac_sc
                            fz = sps / s * eqs * cyqi * szsk * fac_sc
                            
                        # Apply modulation factors
                        if m == 0:  # Perpendicular
                            if n == 0:
                                if nn == 0:
                                    hx, hy, hz = fx, fy, fz
                                else:
                                    hx = fx * x_sc
                                    hy = fy * x_sc
                                    hz = fz * x_sc
                            else:
                                if nn == 0:
                                    hx = fx * cps
                                    hy = fy * cps
                                    hz = fz * cps
                                else:
                                    hx = fx * cps * x_sc
                                    hy = fy * cps * x_sc
                                    hz = fz * cps * x_sc
                            
                            # Rotate back from tilted coordinates
                            hxr = hx * ct1 + hz * st1
                            hzr = -hx * st1 + hz * ct1
                        else:  # Parallel
                            if n == 0:
                                if nn == 0:
                                    hx, hy, hz = fx, fy, fz
                                else:
                                    hx = fx * x_sc
                                    hy = fy * x_sc
                                    hz = fz * x_sc
                            else:
                                if nn == 0:
                                    hx = fx * s3ps
                                    hy = fy * s3ps
                                    hz = fz * s3ps
                                else:
                                    hx = fx * s3ps * x_sc
                                    hy = fy * s3ps * x_sc
                                    hz = fz * s3ps * x_sc
                            
                            # Rotate back from tilted coordinates
                            hxr = hx * ct2 + hz * st2
                            hzr = -hx * st2 + hz * ct2
                        
                        # Accumulate with coefficient
                        bx += hxr * a_arr[l]
                        by += hy * a_arr[l]
                        bz += hzr * a_arr[l]
                        l += 1
    
    return bx, by, bz


def full_rc_vectorized(iopr: int, ps: float, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                      sc_sy: Union[float, np.ndarray], sc_pr: Union[float, np.ndarray],
                      phi: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                              np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized full ring current calculation including shielding.
    
    Parameters
    ----------
    iopr : int
        Ring current flag (0=both, 1=SRC only, 2=PRC only)
    ps : float
        Dipole tilt angle in radians
    x, y, z : ndarray
        GSM coordinates in Re
    sc_sy, sc_pr : float or ndarray
        Scale factors
    phi : float or ndarray
        PRC rotation angle
        
    Returns
    -------
    bxsrc, bysrc, bzsrc, bxprc, byprc, bzprc : ndarray
        Total ring current field components including shielding
    """
    # Shielding coefficients for SRC - full 86 element array
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
        0.3498952546, 0.6595919814
    ])
    
    # Shielding coefficients for PRC - full 86 element array
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
        15.69384715, 7.123215241, 2.300635346, 21.90881131, -.01775839370, .3996346710
    ])
    
    # Get unshielded ring current fields
    hxsrc, hysrc, hzsrc, hxprc, hyprc, hzprc = src_prc_vectorized(
        iopr, sc_sy, sc_pr, phi, ps, x, y, z
    )
    
    # Add shielding
    bxsrc = hxsrc
    bysrc = hysrc
    bzsrc = hzsrc
    bxprc = hxprc
    byprc = hyprc
    bzprc = hzprc
    
    # SRC shielding
    if iopr == 0 or iopr == 1:
        x_sc = sc_sy - 1.0
        fsx, fsy, fsz = rc_shield_vectorized(c_sy, ps, x_sc, x, y, z)
        bxsrc += fsx
        bysrc += fsy
        bzsrc += fsz
    
    # PRC shielding
    if iopr == 0 or iopr == 2:
        x_sc = sc_pr - 1.0
        fpx, fpy, fpz = rc_shield_vectorized(c_pr, ps, x_sc, x, y, z)
        bxprc += fpx
        byprc += fpy
        bzprc += fpz
    
    return bxsrc, bysrc, bzsrc, bxprc, byprc, bzprc