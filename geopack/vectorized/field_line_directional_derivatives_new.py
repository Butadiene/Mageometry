"""
Vectorized field line directional derivatives implementation.

This module implements the 9 directional derivative formulas for the Frenet-Serret frame,
where T, n, and b are orthonormal unit vectors (|T| = |n| = |b| = 1).

The 9 formulas are:
- (∂T/∂T)·n = κ (curvature)
- (∂T/∂T)·b = 0
- (∂n/∂T)·b = τ (torsion)
- And 6 other related formulas with antisymmetry relations

Important: Since T, n, and b are unit vectors, their derivatives are perpendicular
to themselves: (∂T/∂T)·T = 0, (∂n/∂n)·n = 0, (∂b/∂b)·b = 0.
"""

import numpy as np
from .field_line_geometry_vectorized import (
    field_line_tangent_vectorized,
    field_line_normal_vectorized,
    field_line_binormal_vectorized,
    field_line_frenet_frame_vectorized
)


def field_line_directional_derivatives_vectorized(model_func, parmod, ps, x, y, z, delta=0.01):
    """
    Calculate all 9 directional derivative formulas for field line geometry.
    
    The Frenet-Serret frame consists of orthonormal unit vectors:
    - T: unit tangent vector (along field line)
    - n: unit normal vector (principal normal)
    - b: unit binormal vector (b = T × n)
    
    Since these are unit vectors, their derivatives are perpendicular to themselves.
    
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
    derivatives : dict
        Dictionary containing all 9 directional derivative values:
        - 'dT_dT_n': (∂T/∂T)·n = κ (curvature)
        - 'dT_dT_b': (∂T/∂T)·b = 0
        - 'dn_dT_b': (∂n/∂T)·b = τ (torsion)
        - 'dT_dn_n': (∂T/∂n)·n
        - 'dT_dn_b': (∂T/∂n)·b
        - 'dn_dn_b': (∂n/∂n)·b
        - 'dn_db_b': (∂n/∂b)·b
        - 'dn_db_T': (∂n/∂b)·T
        - 'db_db_T': (∂b/∂b)·T
        
        Note: Self-components (∂T/∂T)·T, (∂n/∂n)·n, (∂b/∂b)·b are always zero
        for unit vectors and are not included.
    """
    scalar_input = np.isscalar(x)
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    
    # Get Frenet frame at current point
    tx0, ty0, tz0, nx0, ny0, nz0, bx0, by0, bz0, _ = field_line_frenet_frame_vectorized(
        model_func, parmod, ps, x, y, z, delta
    )
    
    # Initialize results dictionary
    results = {}
    
    # === Tangential derivatives (∂/∂T) ===
    # Step in tangent direction
    x_t_plus = x + delta * tx0
    y_t_plus = y + delta * ty0
    z_t_plus = z + delta * tz0
    
    x_t_minus = x - delta * tx0
    y_t_minus = y - delta * ty0
    z_t_minus = z - delta * tz0
    
    # Get vectors at stepped positions
    tx_t_plus, ty_t_plus, tz_t_plus, nx_t_plus, ny_t_plus, nz_t_plus, bx_t_plus, by_t_plus, bz_t_plus, _ = \
        field_line_frenet_frame_vectorized(model_func, parmod, ps, x_t_plus, y_t_plus, z_t_plus, delta)
    
    tx_t_minus, ty_t_minus, tz_t_minus, nx_t_minus, ny_t_minus, nz_t_minus, bx_t_minus, by_t_minus, bz_t_minus, _ = \
        field_line_frenet_frame_vectorized(model_func, parmod, ps, x_t_minus, y_t_minus, z_t_minus, delta)
    
    # Central differences for ∂T/∂T, ∂n/∂T, ∂b/∂T
    dT_dT_x = (tx_t_plus - tx_t_minus) / (2 * delta)
    dT_dT_y = (ty_t_plus - ty_t_minus) / (2 * delta)
    dT_dT_z = (tz_t_plus - tz_t_minus) / (2 * delta)
    
    dn_dT_x = (nx_t_plus - nx_t_minus) / (2 * delta)
    dn_dT_y = (ny_t_plus - ny_t_minus) / (2 * delta)
    dn_dT_z = (nz_t_plus - nz_t_minus) / (2 * delta)
    
    db_dT_x = (bx_t_plus - bx_t_minus) / (2 * delta)
    db_dT_y = (by_t_plus - by_t_minus) / (2 * delta)
    db_dT_z = (bz_t_plus - bz_t_minus) / (2 * delta)
    
    # Key formulas
    results['dT_dT_n'] = dT_dT_x * nx0 + dT_dT_y * ny0 + dT_dT_z * nz0  # = κ
    results['dT_dT_b'] = dT_dT_x * bx0 + dT_dT_y * by0 + dT_dT_z * bz0  # = 0
    results['dn_dT_b'] = dn_dT_x * bx0 + dn_dT_y * by0 + dn_dT_z * bz0  # = τ
    
    # === Normal derivatives (∂/∂n) ===
    # Step in normal direction
    x_n_plus = x + delta * nx0
    y_n_plus = y + delta * ny0
    z_n_plus = z + delta * nz0
    
    x_n_minus = x - delta * nx0
    y_n_minus = y - delta * ny0
    z_n_minus = z - delta * nz0
    
    # Get vectors at stepped positions
    tx_n_plus, ty_n_plus, tz_n_plus, nx_n_plus, ny_n_plus, nz_n_plus, bx_n_plus, by_n_plus, bz_n_plus, _ = \
        field_line_frenet_frame_vectorized(model_func, parmod, ps, x_n_plus, y_n_plus, z_n_plus, delta)
    
    tx_n_minus, ty_n_minus, tz_n_minus, nx_n_minus, ny_n_minus, nz_n_minus, bx_n_minus, by_n_minus, bz_n_minus, _ = \
        field_line_frenet_frame_vectorized(model_func, parmod, ps, x_n_minus, y_n_minus, z_n_minus, delta)
    
    # Central differences
    dT_dn_x = (tx_n_plus - tx_n_minus) / (2 * delta)
    dT_dn_y = (ty_n_plus - ty_n_minus) / (2 * delta)
    dT_dn_z = (tz_n_plus - tz_n_minus) / (2 * delta)
    
    dn_dn_x = (nx_n_plus - nx_n_minus) / (2 * delta)
    dn_dn_y = (ny_n_plus - ny_n_minus) / (2 * delta)
    dn_dn_z = (nz_n_plus - nz_n_minus) / (2 * delta)
    
    db_dn_x = (bx_n_plus - bx_n_minus) / (2 * delta)
    db_dn_y = (by_n_plus - by_n_minus) / (2 * delta)
    db_dn_z = (bz_n_plus - bz_n_minus) / (2 * delta)
    
    # Key formulas
    results['dT_dn_n'] = dT_dn_x * nx0 + dT_dn_y * ny0 + dT_dn_z * nz0
    results['dT_dn_b'] = dT_dn_x * bx0 + dT_dn_y * by0 + dT_dn_z * bz0
    results['dn_dn_b'] = dn_dn_x * bx0 + dn_dn_y * by0 + dn_dn_z * bz0
    
    # === Binormal derivatives (∂/∂b) ===
    # Step in binormal direction
    x_b_plus = x + delta * bx0
    y_b_plus = y + delta * by0
    z_b_plus = z + delta * bz0
    
    x_b_minus = x - delta * bx0
    y_b_minus = y - delta * by0
    z_b_minus = z - delta * bz0
    
    # Get vectors at stepped positions
    tx_b_plus, ty_b_plus, tz_b_plus, nx_b_plus, ny_b_plus, nz_b_plus, bx_b_plus, by_b_plus, bz_b_plus, _ = \
        field_line_frenet_frame_vectorized(model_func, parmod, ps, x_b_plus, y_b_plus, z_b_plus, delta)
    
    tx_b_minus, ty_b_minus, tz_b_minus, nx_b_minus, ny_b_minus, nz_b_minus, bx_b_minus, by_b_minus, bz_b_minus, _ = \
        field_line_frenet_frame_vectorized(model_func, parmod, ps, x_b_minus, y_b_minus, z_b_minus, delta)
    
    # Central differences
    dT_db_x = (tx_b_plus - tx_b_minus) / (2 * delta)
    dT_db_y = (ty_b_plus - ty_b_minus) / (2 * delta)
    dT_db_z = (tz_b_plus - tz_b_minus) / (2 * delta)
    
    dn_db_x = (nx_b_plus - nx_b_minus) / (2 * delta)
    dn_db_y = (ny_b_plus - ny_b_minus) / (2 * delta)
    dn_db_z = (nz_b_plus - nz_b_minus) / (2 * delta)
    
    db_db_x = (bx_b_plus - bx_b_minus) / (2 * delta)
    db_db_y = (by_b_plus - by_b_minus) / (2 * delta)
    db_db_z = (bz_b_plus - bz_b_minus) / (2 * delta)
    
    # Key formulas
    results['dn_db_b'] = dn_db_x * bx0 + dn_db_y * by0 + dn_db_z * bz0
    results['dn_db_T'] = dn_db_x * tx0 + dn_db_y * ty0 + dn_db_z * tz0
    results['db_db_T'] = db_db_x * tx0 + db_db_y * ty0 + db_db_z * tz0
    
    # Also calculate the antisymmetric pairs for validation
    results['dn_dT_T'] = dn_dT_x * tx0 + dn_dT_y * ty0 + dn_dT_z * tz0  # = -κ
    results['db_dT_T'] = db_dT_x * tx0 + db_dT_y * ty0 + db_dT_z * tz0  # = 0
    results['db_dT_n'] = db_dT_x * nx0 + db_dT_y * ny0 + db_dT_z * nz0  # = -τ
    
    results['dn_dn_T'] = dn_dn_x * tx0 + dn_dn_y * ty0 + dn_dn_z * tz0  # = -(∂T/∂n)·n
    results['db_dn_T'] = db_dn_x * tx0 + db_dn_y * ty0 + db_dn_z * tz0  # = -(∂T/∂n)·b
    results['db_dn_n'] = db_dn_x * nx0 + db_dn_y * ny0 + db_dn_z * nz0  # = -(∂n/∂n)·b
    
    results['db_db_n'] = db_db_x * nx0 + db_db_y * ny0 + db_db_z * nz0  # = -(∂n/∂b)·b
    results['dT_db_n'] = dT_db_x * nx0 + dT_db_y * ny0 + dT_db_z * nz0  # = -(∂n/∂b)·T
    results['dT_db_b'] = dT_db_x * bx0 + dT_db_y * by0 + dT_db_z * bz0  # = -(∂b/∂b)·T
    
    if scalar_input:
        return {k: v.item() if hasattr(v, 'item') else v for k, v in results.items()}
    else:
        return results


def verify_antisymmetry_relations(derivatives):
    """
    Verify the antisymmetry relations between directional derivatives.
    
    Parameters
    ----------
    derivatives : dict
        Dictionary from field_line_directional_derivatives_vectorized
        
    Returns
    -------
    errors : dict
        Dictionary of antisymmetry relation errors
    """
    errors = {}
    
    # First set: Frenet-Serret formulas
    errors['κ_check'] = derivatives['dT_dT_n'] + derivatives['dn_dT_T']
    errors['τ_check'] = derivatives['dn_dT_b'] + derivatives['db_dT_n']
    errors['zero_check_1'] = derivatives['dT_dT_b'] - derivatives['db_dT_T']
    
    # Second set: Normal derivatives
    errors['dT_dn_n_check'] = derivatives['dT_dn_n'] + derivatives['dn_dn_T']
    errors['dT_dn_b_check'] = derivatives['dT_dn_b'] + derivatives['db_dn_T']
    errors['dn_dn_b_check'] = derivatives['dn_dn_b'] + derivatives['db_dn_n']
    
    # Third set: Binormal derivatives
    errors['dn_db_b_check'] = derivatives['dn_db_b'] + derivatives['db_db_n']
    errors['dn_db_T_check'] = derivatives['dn_db_T'] + derivatives['dT_db_n']
    errors['db_db_T_check'] = derivatives['db_db_T'] + derivatives['dT_db_b']
    
    return errors


def get_curvature_torsion_from_derivatives(derivatives):
    """
    Extract curvature and torsion from the directional derivatives.
    
    Parameters
    ----------
    derivatives : dict
        Dictionary from field_line_directional_derivatives_vectorized
        
    Returns
    -------
    curvature : float or ndarray
        Field line curvature κ = (∂T/∂T)·n
    torsion : float or ndarray
        Field line torsion τ = (∂n/∂T)·b
    """
    curvature = derivatives['dT_dT_n']
    torsion = derivatives['dn_dT_b']
    
    return curvature, torsion


def verify_unit_vectors(tx, ty, tz, nx, ny, nz, bx, by, bz, tol=1e-10):
    """
    Verify that T, n, and b are unit vectors and orthonormal.
    
    Parameters
    ----------
    tx, ty, tz : float or ndarray
        Components of tangent vector T
    nx, ny, nz : float or ndarray
        Components of normal vector n
    bx, by, bz : float or ndarray
        Components of binormal vector b
    tol : float, optional
        Tolerance for checks, default 1e-10
        
    Returns
    -------
    errors : dict
        Dictionary of errors for each check
    """
    errors = {}
    
    # Check unit length
    errors['|T| - 1'] = np.sqrt(tx**2 + ty**2 + tz**2) - 1.0
    errors['|n| - 1'] = np.sqrt(nx**2 + ny**2 + nz**2) - 1.0
    errors['|b| - 1'] = np.sqrt(bx**2 + by**2 + bz**2) - 1.0
    
    # Check orthogonality
    errors['T·n'] = tx*nx + ty*ny + tz*nz
    errors['T·b'] = tx*bx + ty*by + tz*bz
    errors['n·b'] = nx*bx + ny*by + nz*bz
    
    # Check b = T × n
    b_cross_x = ty*nz - tz*ny
    b_cross_y = tz*nx - tx*nz
    b_cross_z = tx*ny - ty*nx
    errors['b - T×n'] = np.sqrt((bx - b_cross_x)**2 + (by - b_cross_y)**2 + (bz - b_cross_z)**2)
    
    return errors