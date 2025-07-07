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


def field_line_tangent_normal_derivative_vectorized(model_func, parmod, ps, x, y, z, delta=0.01):
    """
    Calculate the normal derivative of the tangent vector: ∂T/∂n = (N · ∇)T
    
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
    dT_dn_x, dT_dn_y, dT_dn_z : float or ndarray
        Cartesian components of ∂T/∂n
    dT_dn_tangent : float or ndarray
        Tangent component: (∂T/∂n)·T (should be zero)
    dT_dn_normal : float or ndarray
        Normal component: (∂T/∂n)·N  
    dT_dn_binormal : float or ndarray
        Binormal component: (∂T/∂n)·B
    """
    scalar_input = np.isscalar(x)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    # Get Frenet frame at current point
    tx0, ty0, tz0, nx0, ny0, nz0, bx0, by0, bz0, _ = field_line_frenet_frame_vectorized(
        model_func, parmod, ps, x, y, z, delta
    )
    
    # Step in normal direction
    x_plus = x + delta * nx0
    y_plus = y + delta * ny0
    z_plus = z + delta * nz0
    
    x_minus = x - delta * nx0
    y_minus = y - delta * ny0
    z_minus = z - delta * nz0
    
    # Get tangent vectors at stepped positions
    tx_plus, ty_plus, tz_plus = field_line_tangent_vectorized(
        model_func, parmod, ps, x_plus, y_plus, z_plus
    )
    tx_minus, ty_minus, tz_minus = field_line_tangent_vectorized(
        model_func, parmod, ps, x_minus, y_minus, z_minus
    )
    
    # Central difference for ∂T/∂n
    dT_dn_x = (tx_plus - tx_minus) / (2 * delta)
    dT_dn_y = (ty_plus - ty_minus) / (2 * delta)
    dT_dn_z = (tz_plus - tz_minus) / (2 * delta)
    
    # Project onto Frenet frame
    dT_dn_tangent = dT_dn_x * tx0 + dT_dn_y * ty0 + dT_dn_z * tz0
    dT_dn_normal = dT_dn_x * nx0 + dT_dn_y * ny0 + dT_dn_z * nz0
    dT_dn_binormal = dT_dn_x * bx0 + dT_dn_y * by0 + dT_dn_z * bz0
    
    if scalar_input:
        return (dT_dn_x.item(), dT_dn_y.item(), dT_dn_z.item(),
                dT_dn_tangent.item(), dT_dn_normal.item(), dT_dn_binormal.item())
    else:
        return dT_dn_x, dT_dn_y, dT_dn_z, dT_dn_tangent, dT_dn_normal, dT_dn_binormal


def field_line_normal_normal_derivative_vectorized(model_func, parmod, ps, x, y, z, delta=0.01):
    """
    Calculate the normal derivative of the normal vector: ∂N/∂n = (N · ∇)N
    
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
    dN_dn_x, dN_dn_y, dN_dn_z : float or ndarray
        Cartesian components of ∂N/∂n
    dN_dn_tangent : float or ndarray
        Tangent component: (∂N/∂n)·T = -κ (negative curvature)
    dN_dn_normal : float or ndarray
        Normal component: (∂N/∂n)·N (should be zero)
    dN_dn_binormal : float or ndarray
        Binormal component: (∂N/∂n)·B
    """
    scalar_input = np.isscalar(x)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    # Get Frenet frame at current point
    tx0, ty0, tz0, nx0, ny0, nz0, bx0, by0, bz0, _ = field_line_frenet_frame_vectorized(
        model_func, parmod, ps, x, y, z, delta
    )
    
    # Step in normal direction
    x_plus = x + delta * nx0
    y_plus = y + delta * ny0
    z_plus = z + delta * nz0
    
    x_minus = x - delta * nx0
    y_minus = y - delta * ny0
    z_minus = z - delta * nz0
    
    # Get normal vectors at stepped positions
    nx_plus, ny_plus, nz_plus = field_line_normal_vectorized(
        model_func, parmod, ps, x_plus, y_plus, z_plus, delta
    )
    nx_minus, ny_minus, nz_minus = field_line_normal_vectorized(
        model_func, parmod, ps, x_minus, y_minus, z_minus, delta
    )
    
    # Central difference for ∂N/∂n
    dN_dn_x = (nx_plus - nx_minus) / (2 * delta)
    dN_dn_y = (ny_plus - ny_minus) / (2 * delta)
    dN_dn_z = (nz_plus - nz_minus) / (2 * delta)
    
    # Project onto Frenet frame
    dN_dn_tangent = dN_dn_x * tx0 + dN_dn_y * ty0 + dN_dn_z * tz0
    dN_dn_normal = dN_dn_x * nx0 + dN_dn_y * ny0 + dN_dn_z * nz0
    dN_dn_binormal = dN_dn_x * bx0 + dN_dn_y * by0 + dN_dn_z * bz0
    
    if scalar_input:
        return (dN_dn_x.item(), dN_dn_y.item(), dN_dn_z.item(),
                dN_dn_tangent.item(), dN_dn_normal.item(), dN_dn_binormal.item())
    else:
        return dN_dn_x, dN_dn_y, dN_dn_z, dN_dn_tangent, dN_dn_normal, dN_dn_binormal


def field_line_tangent_binormal_derivative_vectorized(model_func, parmod, ps, x, y, z, delta=0.01):
    """
    Calculate the binormal derivative of the tangent vector: ∂T/∂b = (B · ∇)T
    
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
    dT_db_x, dT_db_y, dT_db_z : float or ndarray
        Cartesian components of ∂T/∂b
    dT_db_tangent : float or ndarray
        Tangent component: (∂T/∂b)·T (should be zero)
    dT_db_normal : float or ndarray
        Normal component: (∂T/∂b)·N
    dT_db_binormal : float or ndarray
        Binormal component: (∂T/∂b)·B
    """
    scalar_input = np.isscalar(x)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    # Get Frenet frame at current point
    tx0, ty0, tz0, nx0, ny0, nz0, bx0, by0, bz0, _ = field_line_frenet_frame_vectorized(
        model_func, parmod, ps, x, y, z, delta
    )
    
    # Step in binormal direction
    x_plus = x + delta * bx0
    y_plus = y + delta * by0
    z_plus = z + delta * bz0
    
    x_minus = x - delta * bx0
    y_minus = y - delta * by0
    z_minus = z - delta * bz0
    
    # Get tangent vectors at stepped positions
    tx_plus, ty_plus, tz_plus = field_line_tangent_vectorized(
        model_func, parmod, ps, x_plus, y_plus, z_plus
    )
    tx_minus, ty_minus, tz_minus = field_line_tangent_vectorized(
        model_func, parmod, ps, x_minus, y_minus, z_minus
    )
    
    # Central difference for ∂T/∂b
    dT_db_x = (tx_plus - tx_minus) / (2 * delta)
    dT_db_y = (ty_plus - ty_minus) / (2 * delta)
    dT_db_z = (tz_plus - tz_minus) / (2 * delta)
    
    # Project onto Frenet frame
    dT_db_tangent = dT_db_x * tx0 + dT_db_y * ty0 + dT_db_z * tz0
    dT_db_normal = dT_db_x * nx0 + dT_db_y * ny0 + dT_db_z * nz0
    dT_db_binormal = dT_db_x * bx0 + dT_db_y * by0 + dT_db_z * bz0
    
    if scalar_input:
        return (dT_db_x.item(), dT_db_y.item(), dT_db_z.item(),
                dT_db_tangent.item(), dT_db_normal.item(), dT_db_binormal.item())
    else:
        return dT_db_x, dT_db_y, dT_db_z, dT_db_tangent, dT_db_normal, dT_db_binormal


def field_line_normal_binormal_derivative_vectorized(model_func, parmod, ps, x, y, z, delta=0.01):
    """
    Calculate the binormal derivative of the normal vector: ∂N/∂b = (B · ∇)N
    
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
    dN_db_x, dN_db_y, dN_db_z : float or ndarray
        Cartesian components of ∂N/∂b
    dN_db_tangent : float or ndarray
        Tangent component: (∂N/∂b)·T = -τ (negative torsion)
    dN_db_normal : float or ndarray
        Normal component: (∂N/∂b)·N (should be zero from orthogonality)
    dN_db_binormal : float or ndarray
        Binormal component: (∂N/∂b)·B (should be zero)
    """
    scalar_input = np.isscalar(x)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    # Get Frenet frame at current point
    tx0, ty0, tz0, nx0, ny0, nz0, bx0, by0, bz0, _ = field_line_frenet_frame_vectorized(
        model_func, parmod, ps, x, y, z, delta
    )
    
    # Step in binormal direction
    x_plus = x + delta * bx0
    y_plus = y + delta * by0
    z_plus = z + delta * bz0
    
    x_minus = x - delta * bx0
    y_minus = y - delta * by0
    z_minus = z - delta * bz0
    
    # Get normal vectors at stepped positions
    nx_plus, ny_plus, nz_plus = field_line_normal_vectorized(
        model_func, parmod, ps, x_plus, y_plus, z_plus, delta
    )
    nx_minus, ny_minus, nz_minus = field_line_normal_vectorized(
        model_func, parmod, ps, x_minus, y_minus, z_minus, delta
    )
    
    # Central difference for ∂N/∂b
    dN_db_x = (nx_plus - nx_minus) / (2 * delta)
    dN_db_y = (ny_plus - ny_minus) / (2 * delta)
    dN_db_z = (nz_plus - nz_minus) / (2 * delta)
    
    # Project onto Frenet frame
    dN_db_tangent = dN_db_x * tx0 + dN_db_y * ty0 + dN_db_z * tz0
    dN_db_normal = dN_db_x * nx0 + dN_db_y * ny0 + dN_db_z * nz0
    dN_db_binormal = dN_db_x * bx0 + dN_db_y * by0 + dN_db_z * bz0
    
    if scalar_input:
        return (dN_db_x.item(), dN_db_y.item(), dN_db_z.item(),
                dN_db_tangent.item(), dN_db_normal.item(), dN_db_binormal.item())
    else:
        return dN_db_x, dN_db_y, dN_db_z, dN_db_tangent, dN_db_normal, dN_db_binormal