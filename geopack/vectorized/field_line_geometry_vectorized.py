"""
Vectorized magnetic field line geometry analysis.

This module provides functions to calculate geometric properties of magnetic field lines
including the Frenet-Serret frame (tangent, normal, binormal vectors), curvature, and torsion.

Author: geopack-vectorize
"""

import numpy as np


def field_line_tangent_vectorized(model_func, parmod, ps, x, y, z):
    """
    Calculate unit tangent vectors along magnetic field lines.
    
    The tangent vector T is the normalized magnetic field vector: T = B/|B|
    
    Parameters
    ----------
    model_func : callable
        Magnetic field model function (e.g., t89_vectorized, t96_vectorized)
    parmod : array_like
        Model parameters specific to the chosen model
    ps : float
        Dipole tilt angle in radians
    x, y, z : float or array_like
        Position coordinates in GSM system (Re)
        
    Returns
    -------
    tx, ty, tz : float or ndarray
        Components of unit tangent vector
        Returns scalars for scalar input, arrays for array input
    """
    # Ensure arrays
    scalar_input = np.isscalar(x)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    # Get magnetic field
    bx, by, bz = model_func(parmod, ps, x, y, z)
    
    # Calculate magnitude
    b_mag = np.sqrt(bx**2 + by**2 + bz**2)
    
    # Handle zero field regions
    mask_nonzero = b_mag > 1e-10
    
    # Initialize output
    tx = np.zeros_like(x)
    ty = np.zeros_like(y)
    tz = np.zeros_like(z)
    
    # Normalize where field is non-zero
    tx = np.where(mask_nonzero, bx / b_mag, 0.0)
    ty = np.where(mask_nonzero, by / b_mag, 0.0)
    tz = np.where(mask_nonzero, bz / b_mag, 0.0)
    
    if scalar_input:
        return tx.item(), ty.item(), tz.item()
    else:
        return tx, ty, tz


def field_line_curvature_vectorized(model_func, parmod, ps, x, y, z, delta=0.01):
    """
    Calculate field line curvature using finite differences.
    
    Curvature κ = |dT/ds| where s is arc length parameter.
    
    Parameters
    ----------
    model_func : callable
        Magnetic field model function
    parmod : array_like
        Model parameters
    ps : float
        Dipole tilt angle in radians
    x, y, z : float or array_like
        Position coordinates in GSM system (Re)
    delta : float, optional
        Step size for finite differences (Re), default 0.01
        
    Returns
    -------
    curvature : float or ndarray
        Field line curvature (1/Re)
    """
    scalar_input = np.isscalar(x)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    # Get tangent at current point
    tx0, ty0, tz0 = field_line_tangent_vectorized(model_func, parmod, ps, x, y, z)
    
    # Step forward along field line
    x_plus = x + delta * tx0
    y_plus = y + delta * ty0
    z_plus = z + delta * tz0
    
    # Step backward along field line
    x_minus = x - delta * tx0
    y_minus = y - delta * ty0
    z_minus = z - delta * tz0
    
    # Get tangents at stepped positions
    tx_plus, ty_plus, tz_plus = field_line_tangent_vectorized(
        model_func, parmod, ps, x_plus, y_plus, z_plus
    )
    tx_minus, ty_minus, tz_minus = field_line_tangent_vectorized(
        model_func, parmod, ps, x_minus, y_minus, z_minus
    )
    
    # Central difference for dT/ds
    dtx_ds = (tx_plus - tx_minus) / (2 * delta)
    dty_ds = (ty_plus - ty_minus) / (2 * delta)
    dtz_ds = (tz_plus - tz_minus) / (2 * delta)
    
    # Curvature is magnitude of dT/ds
    curvature = np.sqrt(dtx_ds**2 + dty_ds**2 + dtz_ds**2)
    
    if scalar_input:
        return curvature.item()
    else:
        return curvature


def field_line_normal_vectorized(model_func, parmod, ps, x, y, z, delta=0.01):
    """
    Calculate normal vectors of magnetic field lines.
    
    Normal vector N = (dT/ds)/|dT/ds|
    
    Parameters
    ----------
    model_func : callable
        Magnetic field model function
    parmod : array_like
        Model parameters
    ps : float
        Dipole tilt angle in radians
    x, y, z : float or array_like
        Position coordinates in GSM system (Re)
    delta : float, optional
        Step size for finite differences (Re), default 0.01
        
    Returns
    -------
    nx, ny, nz : float or ndarray
        Components of unit normal vector
    """
    scalar_input = np.isscalar(x)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    # Get tangent at current point
    tx0, ty0, tz0 = field_line_tangent_vectorized(model_func, parmod, ps, x, y, z)
    
    # Step forward and backward along field line
    x_plus = x + delta * tx0
    y_plus = y + delta * ty0
    z_plus = z + delta * tz0
    
    x_minus = x - delta * tx0
    y_minus = y - delta * ty0
    z_minus = z - delta * tz0
    
    # Get tangents at stepped positions
    tx_plus, ty_plus, tz_plus = field_line_tangent_vectorized(
        model_func, parmod, ps, x_plus, y_plus, z_plus
    )
    tx_minus, ty_minus, tz_minus = field_line_tangent_vectorized(
        model_func, parmod, ps, x_minus, y_minus, z_minus
    )
    
    # Central difference for dT/ds
    dtx_ds = (tx_plus - tx_minus) / (2 * delta)
    dty_ds = (ty_plus - ty_minus) / (2 * delta)
    dtz_ds = (tz_plus - tz_minus) / (2 * delta)
    
    # Magnitude of dT/ds
    dt_mag = np.sqrt(dtx_ds**2 + dty_ds**2 + dtz_ds**2)
    
    # Handle regions with no curvature (straight field lines)
    mask_curved = dt_mag > 1e-10
    
    # Initialize output
    nx = np.zeros_like(x)
    ny = np.zeros_like(y)
    nz = np.zeros_like(z)
    
    # Normalize where curvature exists
    nx = np.where(mask_curved, dtx_ds / dt_mag, 0.0)
    ny = np.where(mask_curved, dty_ds / dt_mag, 0.0)
    nz = np.where(mask_curved, dtz_ds / dt_mag, 0.0)
    
    if scalar_input:
        return nx.item(), ny.item(), nz.item()
    else:
        return nx, ny, nz


def field_line_binormal_vectorized(model_func, parmod, ps, x, y, z, delta=0.01):
    """
    Calculate binormal vectors of magnetic field lines.
    
    Binormal vector B = T × N
    
    Parameters
    ----------
    model_func : callable
        Magnetic field model function
    parmod : array_like
        Model parameters
    ps : float
        Dipole tilt angle in radians
    x, y, z : float or array_like
        Position coordinates in GSM system (Re)
    delta : float, optional
        Step size for finite differences (Re), default 0.01
        
    Returns
    -------
    bx, by, bz : float or ndarray
        Components of unit binormal vector
    """
    scalar_input = np.isscalar(x)
    
    # Get tangent and normal vectors
    tx, ty, tz = field_line_tangent_vectorized(model_func, parmod, ps, x, y, z)
    nx, ny, nz = field_line_normal_vectorized(model_func, parmod, ps, x, y, z, delta)
    
    # Cross product T × N
    bx = ty * nz - tz * ny
    by = tz * nx - tx * nz
    bz = tx * ny - ty * nx
    
    if scalar_input:
        return bx, by, bz
    else:
        return bx, by, bz


def field_line_torsion_vectorized(model_func, parmod, ps, x, y, z, delta=0.01):
    """
    Calculate field line torsion using finite differences.
    
    Torsion τ = -N · (dB/ds) measures the rate of rotation of the osculating plane.
    
    Parameters
    ----------
    model_func : callable
        Magnetic field model function
    parmod : array_like
        Model parameters
    ps : float
        Dipole tilt angle in radians
    x, y, z : float or array_like
        Position coordinates in GSM system (Re)
    delta : float, optional
        Step size for finite differences (Re), default 0.01
        
    Returns
    -------
    torsion : float or ndarray
        Field line torsion (1/Re)
    """
    scalar_input = np.isscalar(x)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    # Get tangent vector at current point
    tx0, ty0, tz0 = field_line_tangent_vectorized(model_func, parmod, ps, x, y, z)
    
    # Get normal and binormal at current point
    nx0, ny0, nz0 = field_line_normal_vectorized(model_func, parmod, ps, x, y, z, delta)
    bx0, by0, bz0 = field_line_binormal_vectorized(model_func, parmod, ps, x, y, z, delta)
    
    # Step forward and backward along field line
    x_plus = x + delta * tx0
    y_plus = y + delta * ty0
    z_plus = z + delta * tz0
    
    x_minus = x - delta * tx0
    y_minus = y - delta * ty0
    z_minus = z - delta * tz0
    
    # Get binormal at stepped positions
    bx_plus, by_plus, bz_plus = field_line_binormal_vectorized(
        model_func, parmod, ps, x_plus, y_plus, z_plus, delta
    )
    bx_minus, by_minus, bz_minus = field_line_binormal_vectorized(
        model_func, parmod, ps, x_minus, y_minus, z_minus, delta
    )
    
    # Central difference for dB/ds
    dbx_ds = (bx_plus - bx_minus) / (2 * delta)
    dby_ds = (by_plus - by_minus) / (2 * delta)
    dbz_ds = (bz_plus - bz_minus) / (2 * delta)
    
    # Torsion = -N · (dB/ds)
    torsion = -(nx0 * dbx_ds + ny0 * dby_ds + nz0 * dbz_ds)
    
    if scalar_input:
        return torsion.item()
    else:
        return torsion


def field_line_frenet_frame_vectorized(model_func, parmod, ps, x, y, z, delta=0.01):
    """
    Calculate complete Frenet-Serret frame and curvature for field lines.
    
    Parameters
    ----------
    model_func : callable
        Magnetic field model function
    parmod : array_like
        Model parameters
    ps : float
        Dipole tilt angle in radians
    x, y, z : float or array_like
        Position coordinates in GSM system (Re)
    delta : float, optional
        Step size for finite differences (Re), default 0.01
        
    Returns
    -------
    tx, ty, tz : float or ndarray
        Components of unit tangent vector
    nx, ny, nz : float or ndarray
        Components of unit normal vector
    bx, by, bz : float or ndarray
        Components of unit binormal vector
    curvature : float or ndarray
        Field line curvature (1/Re)
    """
    # Get all components
    tx, ty, tz = field_line_tangent_vectorized(model_func, parmod, ps, x, y, z)
    nx, ny, nz = field_line_normal_vectorized(model_func, parmod, ps, x, y, z, delta)
    bx, by, bz = field_line_binormal_vectorized(model_func, parmod, ps, x, y, z, delta)
    curvature = field_line_curvature_vectorized(model_func, parmod, ps, x, y, z, delta)
    
    return tx, ty, tz, nx, ny, nz, bx, by, bz, curvature


def field_line_geometry_complete_vectorized(model_func, parmod, ps, x, y, z, delta=0.01):
    """
    Calculate complete field line geometry including Frenet frame, curvature, and torsion.
    
    Parameters
    ----------
    model_func : callable
        Magnetic field model function (e.g., t89_vectorized, t96_vectorized)
    parmod : array_like
        Model parameters specific to the chosen model
    ps : float
        Dipole tilt angle in radians
    x, y, z : float or array_like
        Position coordinates in GSM system (Re)
    delta : float, optional
        Step size for finite differences (Re), default 0.01
        
    Returns
    -------
    tx, ty, tz : float or ndarray
        Components of unit tangent vector
    nx, ny, nz : float or ndarray
        Components of unit normal vector
    bx, by, bz : float or ndarray
        Components of unit binormal vector
    curvature : float or ndarray
        Field line curvature (1/Re)
    torsion : float or ndarray
        Field line torsion (1/Re)
    """
    # Get Frenet frame and curvature
    tx, ty, tz, nx, ny, nz, bx, by, bz, curvature = field_line_frenet_frame_vectorized(
        model_func, parmod, ps, x, y, z, delta
    )
    
    # Get torsion
    torsion = field_line_torsion_vectorized(model_func, parmod, ps, x, y, z, delta)
    
    return tx, ty, tz, nx, ny, nz, bx, by, bz, curvature, torsion


# Note: The directional derivative functions have been moved to field_line_directional_derivatives_new.py
# which implements the correct 9 formulas with proper antisymmetry relations






