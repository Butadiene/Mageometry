"""
Directional derivatives of the Frenet-Serret frame along T, n, and b.

For the orthonormal frame (T, n, b) of a field line, this module computes
the nine independent projections of the directional derivatives
(∂v/∂u)·w with u, v ∈ {T, n, b}, w ⊥ v. Because the frame vectors are unit
vectors, (∂v/∂u)·v = 0 identically, and orthonormality implies the
antisymmetry (∂v/∂u)·w = -(∂w/∂u)·v, which `verify_antisymmetry_relations`
checks numerically.

The Frenet-Serret formulas appear as special cases of the tangential
derivatives: (∂T/∂T)·n = κ (curvature) and (∂n/∂T)·b = τ (torsion).

Derivatives are central finite differences of the frame at r ± δu. A point
is reported as NaN when the frame is undefined there or at any of the six
stencil points, or when the principal normal flips between the two sides
of a stencil (n(+δu)·n(-δu) below ``normal_flip_tol``), which happens
across inflection points where the difference quotient is meaningless.
"""

import numpy as np

from .field_line_geometry import (
    _as_arrays,
    _finish,
    _frame,
    DEFAULT_ORTHOGONALITY_TOL,
)

DEFAULT_NORMAL_FLIP_TOL = 0.9
"""Default ``normal_flip_tol``: smallest accepted ``n(+δu) . n(-δu)`` across a
finite-difference stencil (0.9 ≈ 26° of rotation)."""

# (key, differentiated vector index, projected-onto vector index) for each
# stencil direction; vector indices: 0 = T, 1 = n, 2 = b.
_PROJECTIONS = {
    'T': [('dT_dT_n', 0, 1), ('dT_dT_b', 0, 2), ('dn_dT_b', 1, 2),
          ('dn_dT_T', 1, 0), ('db_dT_T', 2, 0), ('db_dT_n', 2, 1)],
    'n': [('dT_dn_n', 0, 1), ('dT_dn_b', 0, 2), ('dn_dn_b', 1, 2),
          ('dn_dn_T', 1, 0), ('db_dn_T', 2, 0), ('db_dn_n', 2, 1)],
    'b': [('dn_db_b', 1, 2), ('dn_db_T', 1, 0), ('db_db_T', 2, 0),
          ('db_db_n', 2, 1), ('dT_db_n', 0, 1), ('dT_db_b', 0, 2)],
}


def field_line_directional_derivatives(field, x, y, z, delta=0.01,
                                       orthogonality_tol=DEFAULT_ORTHOGONALITY_TOL,
                                       normal_flip_tol=DEFAULT_NORMAL_FLIP_TOL):
    """
    All directional derivative projections of the Frenet-Serret frame.

    Parameters
    ----------
    field : callable
        Magnetic field function ``field(x, y, z) -> (bx, by, bz)`` accepting
        NumPy arrays (see `mageometry.geometry.field_line_geometry`).
    x, y, z : float or array_like
        Position coordinates (field's length unit).
    delta : float, optional
        Finite-difference step (field's length unit), used both for the
        frame itself and for the directional stencils. Default 0.01.
    orthogonality_tol : float, optional
        Frame validity threshold, see `field_line_normal`. Default 0.1.
    normal_flip_tol : float, optional
        Smallest accepted ``n(+δu) . n(-δu)`` across each stencil; points
        below it are NaN. Default 0.9.

    Returns
    -------
    derivatives : dict of float or ndarray
        The nine independent projections:

        - ``'dT_dT_n'``: (∂T/∂T)·n = κ (curvature)
        - ``'dT_dT_b'``: (∂T/∂T)·b = 0
        - ``'dn_dT_b'``: (∂n/∂T)·b = τ (torsion)
        - ``'dT_dn_n'``, ``'dT_dn_b'``, ``'dn_dn_b'``: derivatives along n
        - ``'dn_db_b'``, ``'dn_db_T'``, ``'db_db_T'``: derivatives along b

        plus their antisymmetric partners (``'dn_dT_T'`` = -κ, ``'db_dT_n'``
        = -τ, ``'db_dT_T'``, ``'dn_dn_T'``, ``'db_dn_T'``, ``'db_dn_n'``,
        ``'db_db_n'``, ``'dT_db_n'``, ``'dT_db_b'``) for validation with
        `verify_antisymmetry_relations`. Values are NaN where the frame is
        undefined or the stencil is invalid.
    """
    scalar_input, x, y, z = _as_arrays(x, y, z)

    frame0 = _frame(field, x, y, z, delta, orthogonality_tol)
    vec0 = (frame0[0:3], frame0[3:6], frame0[6:9])  # T, n, b at the base point

    invalid = ~np.isfinite(frame0[3])  # normal undefined at the base point
    results = {}
    inv2d = 0.5 / delta

    for direction, (ux, uy, uz) in zip(('T', 'n', 'b'), vec0):
        fp = _frame(field, x + delta * ux, y + delta * uy, z + delta * uz,
                    delta, orthogonality_tol)
        fm = _frame(field, x - delta * ux, y - delta * uy, z - delta * uz,
                    delta, orthogonality_tol)
        with np.errstate(invalid='ignore'):
            dot_n = fp[3] * fm[3] + fp[4] * fm[4] + fp[5] * fm[5]
            invalid |= ~(dot_n > normal_flip_tol)  # NaN -> invalid

        vp = (fp[0:3], fp[3:6], fp[6:9])
        vm = (fm[0:3], fm[3:6], fm[6:9])
        for key, iv, iw in _PROJECTIONS[direction]:
            dv = [(vp[iv][k] - vm[iv][k]) * inv2d for k in range(3)]
            w = vec0[iw]
            results[key] = dv[0] * w[0] + dv[1] * w[1] + dv[2] * w[2]

    for key, val in results.items():
        val = np.where(invalid, np.nan, val)
        results[key] = _finish(scalar_input, val)
    return results


def verify_antisymmetry_relations(derivatives):
    """
    Errors of the antisymmetry relations (∂v/∂u)·w + (∂w/∂u)·v = 0.

    Parameters
    ----------
    derivatives : dict
        Output of `field_line_directional_derivatives`.

    Returns
    -------
    errors : dict
        Each entry should be zero up to finite-difference error.
    """
    errors = {}

    # Tangential derivatives (Frenet-Serret formulas)
    errors['κ_check'] = derivatives['dT_dT_n'] + derivatives['dn_dT_T']
    errors['τ_check'] = derivatives['dn_dT_b'] + derivatives['db_dT_n']
    errors['zero_check_1'] = derivatives['dT_dT_b'] - derivatives['db_dT_T']

    # Normal derivatives
    errors['dT_dn_n_check'] = derivatives['dT_dn_n'] + derivatives['dn_dn_T']
    errors['dT_dn_b_check'] = derivatives['dT_dn_b'] + derivatives['db_dn_T']
    errors['dn_dn_b_check'] = derivatives['dn_dn_b'] + derivatives['db_dn_n']

    # Binormal derivatives
    errors['dn_db_b_check'] = derivatives['dn_db_b'] + derivatives['db_db_n']
    errors['dn_db_T_check'] = derivatives['dn_db_T'] + derivatives['dT_db_n']
    errors['db_db_T_check'] = derivatives['db_db_T'] + derivatives['dT_db_b']

    return errors


def get_curvature_torsion_from_derivatives(derivatives):
    """
    Curvature κ = (∂T/∂T)·n and torsion τ = (∂n/∂T)·b from the derivatives.

    Parameters
    ----------
    derivatives : dict
        Output of `field_line_directional_derivatives`.

    Returns
    -------
    curvature, torsion : float or ndarray
    """
    return derivatives['dT_dT_n'], derivatives['dn_dT_b']


def verify_unit_vectors(tx, ty, tz, nx, ny, nz, bx, by, bz):
    """
    Orthonormality errors of a frame: unit lengths, orthogonality, b = T × n.

    Parameters
    ----------
    tx, ty, tz, nx, ny, nz, bx, by, bz : float or ndarray
        Frame components, e.g. from `field_line_frenet_frame`.

    Returns
    -------
    errors : dict
        ``'|T| - 1'``, ``'|n| - 1'``, ``'|b| - 1'``, ``'T·n'``, ``'T·b'``,
        ``'n·b'``, ``'b - T×n'``; NaN where the frame is undefined.
    """
    errors = {}
    errors['|T| - 1'] = np.sqrt(tx**2 + ty**2 + tz**2) - 1.0
    errors['|n| - 1'] = np.sqrt(nx**2 + ny**2 + nz**2) - 1.0
    errors['|b| - 1'] = np.sqrt(bx**2 + by**2 + bz**2) - 1.0

    errors['T·n'] = tx * nx + ty * ny + tz * nz
    errors['T·b'] = tx * bx + ty * by + tz * bz
    errors['n·b'] = nx * bx + ny * by + nz * bz

    cx = ty * nz - tz * ny
    cy = tz * nx - tx * nz
    cz = tx * ny - ty * nx
    errors['b - T×n'] = np.sqrt((bx - cx)**2 + (by - cy)**2 + (bz - cz)**2)
    return errors
