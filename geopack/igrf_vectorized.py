"""
Vectorized implementation of IGRF (International Geomagnetic Reference Field) functions.

This module provides vectorized versions of the IGRF field calculations that can process
multiple points simultaneously while maintaining exact numerical compatibility with the
scalar implementations.

Author: Claude (Anthropic)
Date: 2025-01-07
"""

import numpy as np
from . import geopack as gp


def igrf_geo_vectorized(r, theta, phi):
    """
    Vectorized IGRF calculation in spherical geographic coordinates.
    
    This is the core IGRF calculation that computes the geomagnetic field using
    spherical harmonic expansion. Fully vectorized to handle arrays of coordinates.
    
    Parameters
    ----------
    r : float or array_like
        Radial distance in Earth radii (Re)
    theta : float or array_like
        Colatitude in radians (0 to pi)
    phi : float or array_like
        Longitude in radians (0 to 2*pi)
    
    Returns
    -------
    br : float or ndarray
        Radial component of magnetic field (nT)
    btheta : float or ndarray
        Theta component of magnetic field (nT)
    bphi : float or ndarray
        Phi component of magnetic field (nT)
    """
    # Handle scalar inputs
    scalar_input = np.isscalar(r) and np.isscalar(theta) and np.isscalar(phi)
    r = np.atleast_1d(r)
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)
    
    # Broadcast inputs to same shape
    r, theta, phi = np.broadcast_arrays(r, theta, phi)
    shape = r.shape
    n_points = r.size
    
    # Flatten for processing
    r = r.ravel()
    theta = theta.ravel()
    phi = phi.ravel()
    
    # Initialize output arrays
    br = np.zeros(n_points)
    bt = np.zeros(n_points)
    bf = np.zeros(n_points)
    
    # Get IGRF coefficients from global state
    g = gp.g  # Gauss coefficients g
    h = gp.h  # Gauss coefficients h
    rec = gp.rec  # Recursion coefficients
    
    # Compute cos/sin values
    ct = np.cos(theta)
    st = np.sin(theta)
    
    # Handle pole singularity
    minst = 1e-5
    smlst = np.abs(st) < minst
    
    # Determine maximum order based on radius
    # In the scalar version: nm = 3 + 30/(r+2), capped at 13
    irp3 = (r + 2).astype(np.int64)
    nm = np.minimum(3 + 30 // irp3, 13)
    k = nm + 1
    k_max = np.max(k)
    
    # Create coefficient arrays a[n] = (1/r)^(n+2), b[n] = (n+1)*(1/r)^(n+2)
    ar = 1.0 / r  # 1/r for each point
    a = np.zeros((n_points, k_max))
    b = np.zeros((n_points, k_max))
    
    # Initialize first element
    a[:, 0] = ar * ar  # (1/r)^2
    b[:, 0] = a[:, 0]  # 1*(1/r)^2
    
    # Fill rest of arrays
    for n in range(1, k_max):
        a[:, n] = a[:, n-1] * ar
        b[:, n] = a[:, n] * (n + 1)
    
    # Initialize recursion variables
    d = np.zeros(n_points)
    p = np.ones(n_points)
    
    # m = 0 case
    m = 0
    smf = np.zeros(n_points)
    cmf = np.ones(n_points)
    p1 = p.copy()
    d1 = d.copy()
    p2 = np.zeros(n_points)
    d2 = np.zeros(n_points)
    
    l0 = 0
    mn = l0
    
    for n in range(m, k_max):
        # Only process points where n < k[i]
        mask = n < k
        if not np.any(mask):
            continue
            
        w = g[mn] * cmf + h[mn] * smf
        br[mask] += b[mask, n] * w[mask] * p1[mask]
        bt[mask] -= a[mask, n] * w[mask] * d1[mask]
        
        xk = rec[mn]
        # Recursion relations
        d0 = ct * d1 - st * p1 - xk * d2
        p0 = ct * p1 - xk * p2
        
        d2 = d1.copy()
        p2 = p1.copy()
        d1 = d0
        p1 = p0
        
        mn += n + 1
    
    # Update for next m
    d = st * d + ct * p
    p = st * p
    
    # m >= 1 cases
    l0 = 0
    for m in range(1, k_max):
        smf = np.sin(m * phi)
        cmf = np.cos(m * phi)
        p1 = p.copy()
        d1 = d.copy()
        p2 = np.zeros(n_points)
        d2 = np.zeros(n_points)
        tbf = np.zeros(n_points)
        
        l0 += m + 1
        mn = l0
        
        for n in range(m, k_max):
            # Only process points where n < k[i]
            mask = n < k
            if not np.any(mask):
                continue
                
            w = g[mn] * cmf + h[mn] * smf
            br[mask] += b[mask, n] * w[mask] * p1[mask]
            bt[mask] -= a[mask, n] * w[mask] * d1[mask]
            
            # For bphi component
            tp = p1.copy()
            tp[smlst] = d1[smlst]  # Use d1 at poles
            
            tbf[mask] += a[mask, n] * (g[mn] * smf[mask] - h[mn] * cmf[mask]) * tp[mask]
            
            xk = rec[mn]
            # Recursion relations
            d0 = ct * d1 - st * p1 - xk * d2
            p0 = ct * p1 - xk * p2
            
            d2 = d1.copy()
            p2 = p1.copy()
            d1 = d0
            p1 = p0
            
            mn += n + 1
        
        # Add phi component, handling poles
        bf_contribution = np.zeros_like(tbf)
        # For non-pole points, divide by sin(theta)
        non_pole = ~smlst
        bf_contribution[non_pole] = tbf[non_pole] * m / st[non_pole]
        # At poles, just use tbf
        bf_contribution[smlst] = tbf[smlst]
        
        # Only add for points where m < k
        mask = m < k
        bf[mask] += bf_contribution[mask]
        
        # Update for next m
        d = st * d + ct * p
        p = st * p
    
    # Reshape output
    br = br.reshape(shape)
    bt = bt.reshape(shape)
    bf = bf.reshape(shape)
    
    # Return scalar if input was scalar
    if scalar_input:
        return br.item(), bt.item(), bf.item()
    else:
        return br, bt, bf


def igrf_gsm_vectorized(xgsm, ygsm, zgsm):
    """
    Vectorized calculation of IGRF magnetic field in GSM coordinates.
    
    Parameters
    ----------
    xgsm, ygsm, zgsm : float or array_like
        Position in GSM coordinates (Earth radii)
    
    Returns
    -------
    bxgsm, bygsm, bzgsm : float or ndarray
        Magnetic field components in GSM coordinates (nT)
    """
    # Handle scalar inputs
    scalar_input = np.isscalar(xgsm)
    xgsm = np.atleast_1d(xgsm)
    ygsm = np.atleast_1d(ygsm)
    zgsm = np.atleast_1d(zgsm)
    
    # Transform GSM to GEO coordinates
    from .coordinates_vectorized import geogsm_vectorized
    xgeo, ygeo, zgeo = geogsm_vectorized(xgsm, ygsm, zgsm, -1)
    
    # Convert to spherical coordinates
    from .coordinates_vectorized_complex import sphcar_vectorized
    r, theta, phi = sphcar_vectorized(xgeo, ygeo, zgeo, -1)
    
    # Calculate field in spherical coordinates
    br, btheta, bphi = igrf_geo_vectorized(r, theta, phi)
    
    # Convert to Cartesian components
    from .coordinates_vectorized_complex import bspcar_vectorized
    bxgeo, bygeo, bzgeo = bspcar_vectorized(theta, phi, br, btheta, bphi)
    
    # Transform back to GSM
    bxgsm, bygsm, bzgsm = geogsm_vectorized(bxgeo, bygeo, bzgeo, 1)
    
    # Return scalar if input was scalar
    if scalar_input:
        return bxgsm.item(), bygsm.item(), bzgsm.item()
    else:
        return bxgsm, bygsm, bzgsm


def igrf_gsw_vectorized(xgsw, ygsw, zgsw):
    """
    Vectorized calculation of IGRF magnetic field in GSW coordinates.
    
    Parameters
    ----------
    xgsw, ygsw, zgsw : float or array_like
        Position in GSW coordinates (Earth radii)
    
    Returns
    -------
    bxgsw, bygsw, bzgsw : float or ndarray
        Magnetic field components in GSW coordinates (nT)
    """
    # Handle scalar inputs
    scalar_input = np.isscalar(xgsw)
    xgsw = np.atleast_1d(xgsw)
    ygsw = np.atleast_1d(ygsw)
    zgsw = np.atleast_1d(zgsw)
    
    # Transform GSW to GSM
    from .coordinates_vectorized import gswgsm_vectorized
    xgsm, ygsm, zgsm = gswgsm_vectorized(xgsw, ygsw, zgsw, 1)
    
    # Calculate field in GSM
    bxgsm, bygsm, bzgsm = igrf_gsm_vectorized(xgsm, ygsm, zgsm)
    
    # Transform field back to GSW
    bxgsw, bygsw, bzgsw = gswgsm_vectorized(bxgsm, bygsm, bzgsm, -1)
    
    # Return scalar if input was scalar
    if scalar_input:
        return bxgsw.item(), bygsw.item(), bzgsw.item()
    else:
        return bxgsw, bygsw, bzgsw