"""
Vectorized coordinate transformation functions with conditional logic.

This module contains the more complex coordinate transformation functions
that involve conditional logic (sphcar, bcarsp, bspcar).
"""

import numpy as np


def sphcar_vectorized(r, theta, phi, j):
    """
    Vectorized transformation between spherical and Cartesian coordinates.
    
    Parameters:
    -----------
    r, theta, phi : float or array-like
        Input coordinates (spherical if j>0, interpreted as x,y,z if j<0)
    j : int
        Direction flag: j>0 for spherical→Cartesian, j<0 for Cartesian→spherical
        
    Returns:
    --------
    tuple of (float or ndarray)
        Transformed coordinates (x,y,z if j>0, r,theta,phi if j<0)
    """
    # Check if input is scalar
    scalar_input = np.isscalar(r)
    
    # Ensure arrays
    r = np.atleast_1d(r)
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)
    
    if j > 0:  # Spherical to Cartesian
        sq = r * np.sin(theta)
        x = sq * np.cos(phi)
        y = sq * np.sin(phi)
        z = r * np.cos(theta)
        out1, out2, out3 = x, y, z
    else:  # Cartesian to Spherical
        # Here r,theta,phi are actually x,y,z
        x = r
        y = theta
        z = phi
        
        sq = np.sqrt(x*x + y*y)
        r_out = np.sqrt(sq*sq + z*z)
        
        # Handle conditional logic for phi
        phi_out = np.where(sq != 0, np.arctan2(y, x), 0.0)
        
        # Handle conditional logic for theta
        theta_out = np.where(sq != 0, 
                           np.arctan2(sq, z),
                           np.where(z < 0, np.pi, 0.0))
        
        out1, out2, out3 = r_out, theta_out, phi_out
    
    # Return scalar if input was scalar
    if scalar_input:
        return out1.item(), out2.item(), out3.item()
    else:
        return out1, out2, out3


def bspcar_vectorized(theta, phi, br, btheta, bphi):
    """
    Vectorized transformation of magnetic field components from spherical to Cartesian.
    
    Parameters:
    -----------
    theta, phi : float or array-like
        Spherical coordinate angles
    br, btheta, bphi : float or array-like
        Magnetic field components in spherical coordinates
        
    Returns:
    --------
    tuple of (float or ndarray)
        Magnetic field components in Cartesian coordinates (bx, by, bz)
    """
    # Check if input is scalar
    scalar_input = np.isscalar(theta)
    
    # Ensure arrays
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)
    br = np.atleast_1d(br)
    btheta = np.atleast_1d(btheta)
    bphi = np.atleast_1d(bphi)
    
    # Calculate trigonometric values
    s = np.sin(theta)
    c = np.cos(theta)
    sf = np.sin(phi)
    cf = np.cos(phi)
    
    # Transform field components
    be = br * s + btheta * c
    bx = be * cf - bphi * sf
    by = be * sf + bphi * cf
    bz = br * c - btheta * s
    
    # Return scalar if input was scalar
    if scalar_input:
        return bx.item(), by.item(), bz.item()
    else:
        return bx, by, bz


def bcarsp_vectorized(x, y, z, bx, by, bz):
    """
    Vectorized transformation of magnetic field components from Cartesian to spherical.
    
    Parameters:
    -----------
    x, y, z : float or array-like
        Cartesian coordinates
    bx, by, bz : float or array-like
        Magnetic field components in Cartesian coordinates
        
    Returns:
    --------
    tuple of (float or ndarray)
        Magnetic field components in spherical coordinates (br, btheta, bphi)
    """
    # Check if input is scalar
    scalar_input = np.isscalar(x)
    
    # Ensure arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    bx = np.atleast_1d(bx)
    by = np.atleast_1d(by)
    bz = np.atleast_1d(bz)
    
    # Calculate distances
    rho2 = x*x + y*y
    r2 = rho2 + z*z
    r = np.sqrt(r2)
    rho = np.sqrt(rho2)
    
    # Handle the case where rho = 0 (on z-axis)
    # When rho = 0, we're on the z-axis, so:
    # - br = ±bz (depending on sign of z)
    # - btheta = ∓bx (sign opposite to z)
    # - bphi = by
    
    # Calculate cos(phi) and sin(phi), handling rho = 0 case
    cphi = np.where(rho != 0, x / rho, 1.0)
    sphi = np.where(rho != 0, y / rho, 0.0)
    
    # Calculate ct and st for all points
    # Use regular division to get nan when r=0, matching scalar behavior
    ct = z / r
    st = rho / r
    
    # Calculate field components
    # Let division by zero produce nan to match scalar behavior
    br = (x * bx + y * by + z * bz) / r
    btheta = (bx * cphi + by * sphi) * ct - bz * st
    bphi = by * cphi - bx * sphi
    
    # Return scalar if input was scalar
    if scalar_input:
        return br.item(), btheta.item(), bphi.item()
    else:
        return br, btheta, bphi