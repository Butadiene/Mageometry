"""
|B| directional derivatives and the current density in the Frenet frame.

Writing B = B·T with T the unit tangent, the curl of B closes in the
Frenet-Serret frame (T, n, b) with no n-component of ∇×T:

    μ₀ J = B[(∂T/∂n)·b + (∂n/∂b)·T] T  +  (∂B/∂b) n  +  [Bκ − ∂B/∂n] b

The parallel current is pure field-line twist — the frame projections
``dT_dn_b + dn_db_T`` equal T·(∇×T) — while the curvature κ and the
transverse |B| gradients drive only the perpendicular components. The
frame's directional derivatives
(`~mageometry.geometry.field_line_directional_derivatives`) carry the
twist; this module adds the missing scalar ingredients — the |B| gradients
along the frame (`field_magnitude_derivatives`) — and assembles μ₀J
(`field_line_current_density`).

∇·B = 0 ties the tangential |B| gradient to the frame derivatives,

    ∂B/∂T = −B ∇·T = −B (dT_dn_n + dT_db_b),

which `verify_divergence_identity` evaluates as an end-to-end consistency
check of both modules — and, on fields that are not exactly solenoidal
(empirical models, interpolated data), as a Frenet-frame estimate of ∇·B.

Numerical method and NaN conventions follow
`~mageometry.geometry.field_line_geometry`: central differences with step
``delta`` along the local frame directions, NaN wherever the frame (or a
stencil frame, for the twist) is undefined. |B| differences themselves
need no frame at the stencil points, so ``dB_dT`` is defined wherever the
tangent is, and ``dB_dn``/``dB_db`` wherever the base-point frame is.
"""

import numpy as np

from .field_line_geometry import (
    _as_arrays,
    _finish,
    _frame,
    _DOC_FIELD,
    _DOC_DELTA,
    _DOC_TOL,
    DEFAULT_ORTHOGONALITY_TOL,
)
from .field_line_directional_derivatives import (
    field_line_directional_derivatives,
    DEFAULT_NORMAL_FLIP_TOL,
)


def _bmag(field, x, y, z):
    """|B| on arrays; NaN propagates from non-finite field values."""
    bx, by, bz = field(x, y, z)
    bx = np.asarray(bx, dtype=np.float64)
    by = np.asarray(by, dtype=np.float64)
    bz = np.asarray(bz, dtype=np.float64)
    return np.sqrt(bx * bx + by * by + bz * bz)


def field_magnitude_derivatives(field, x, y, z, delta=0.01,
                                orthogonality_tol=DEFAULT_ORTHOGONALITY_TOL):
    """
    Directional derivatives of |B| along the Frenet-Serret frame.

    Central finite differences of |B| along the local tangent, normal, and
    binormal: ∂B/∂u ≈ [B(r + δu) − B(r − δu)] / 2δ for u ∈ {T, n, b}.
    Unlike the frame's own directional derivatives, the differenced
    quantity is a scalar, so no frame is needed at the stencil points:
    ``dB_dT`` is defined wherever the tangent is (|B| finite and nonzero),
    ``dB_dn`` and ``dB_db`` wherever the base-point normal is.

    Parameters
    ----------
%s
%s
%s

    Returns
    -------
    derivatives : dict of float or ndarray
        - ``'B'``: |B| at the base point (field unit)
        - ``'dB_dT'``: ∂B/∂T, the mirror-force gradient along the line
        - ``'dB_dn'``: ∂B/∂n, toward the centre of curvature
        - ``'dB_db'``: ∂B/∂b

        Gradients are in field-unit/length-unit; NaN where the required
        frame vectors are undefined.

    Notes
    -----
    ∇·B = 0 makes ``dB_dT`` redundant with the frame derivatives:
    ∂B/∂T = −B(dT_dn_n + dT_db_b). It is provided both for direct use
    (bounce motion, mirror ratios) and so `verify_divergence_identity`
    can test the two computations against each other.
    """
    scalar_input, x, y, z = _as_arrays(x, y, z)

    tx, ty, tz, nx, ny, nz, bnx, bny, bnz = _frame(
        field, x, y, z, delta, orthogonality_tol)[:9]

    inv2d = 0.5 / delta
    results = {'B': _bmag(field, x, y, z)}
    for key, (ux, uy, uz) in (('dB_dT', (tx, ty, tz)),
                              ('dB_dn', (nx, ny, nz)),
                              ('dB_db', (bnx, bny, bnz))):
        bp = _bmag(field, x + delta * ux, y + delta * uy, z + delta * uz)
        bm = _bmag(field, x - delta * ux, y - delta * uy, z - delta * uz)
        results[key] = (bp - bm) * inv2d

    return {key: _finish(scalar_input, val) for key, val in results.items()}


def field_line_current_density(field, x, y, z, delta=0.01,
                               orthogonality_tol=DEFAULT_ORTHOGONALITY_TOL,
                               normal_flip_tol=DEFAULT_NORMAL_FLIP_TOL):
    """
    Current density μ₀J = ∇×B decomposed in the Frenet-Serret frame.

    Evaluates

        μ₀ J = B(dT_dn_b + dn_db_T) T + (∂B/∂b) n + (Bκ − ∂B/∂n) b

    from finite differences of the frame and of |B|. The parallel
    component is pure field-line twist, B·T·(∇×T): curvature and |B|
    gradients contribute only to the perpendicular components.

    Parameters
    ----------
%s
%s
%s
    normal_flip_tol : float, optional
        Smallest accepted ``n(+δu) . n(-δu)`` across the transverse
        stencils, as in
        `~mageometry.geometry.field_line_directional_derivatives`.
        Default `DEFAULT_NORMAL_FLIP_TOL` (0.9).

    Returns
    -------
    current : dict of float or ndarray
        - ``'mu0J_T'``, ``'mu0J_n'``, ``'mu0J_b'``: μ₀J projected on the
          frame (field-unit/length-unit)
        - ``'mu0J_x'``, ``'mu0J_y'``, ``'mu0J_z'``: the same vector in the
          field's Cartesian coordinates
        - ``'alpha'``: μ₀ j∥ / B = T·(∇×T), the twist per unit length
          (1/length-unit; the force-free α where J⊥ = 0)
        - ``'B'``, ``'curvature'``: |B| and κ at the base point, as
          by-products

        NaN where the frame or a transverse stencil frame is undefined.

    Notes
    -----
    For geopack fields (nT, Re), J [A/m²] = μ₀J [nT/Re] × 10⁻⁹ /
    (R_E μ₀) ≈ μ₀J × 0.125 nA/m² per nT/Re.

    The finite-difference error is set by ``delta`` exactly as for
    `field_line_directional_derivatives`; against a direct Cartesian
    finite-difference curl of a T96+IGRF field the three components agree
    to a median relative error ~1e-6 of |∇×B| at ``delta=0.002``.
    Validity should be screened with
    `~mageometry.geometry.field_line_frame_quality` in regions of weak or
    rapidly varying curvature.
    """
    scalar_input, x, y, z = _as_arrays(x, y, z)

    frame0 = _frame(field, x, y, z, delta, orthogonality_tol)
    tx, ty, tz, nx, ny, nz, bnx, bny, bnz, curvature = frame0[:10]
    invalid = ~np.isfinite(nx)

    B0 = _bmag(field, x, y, z)
    inv2d = 0.5 / delta

    # Transverse stencils: frame (for the twist projections) and |B| (for
    # the transverse gradients) at r ± δn and r ± δb.
    twist = {}
    grad = {}
    for direction, (ux, uy, uz) in (('n', (nx, ny, nz)), ('b', (bnx, bny, bnz))):
        xp, yp, zp = x + delta * ux, y + delta * uy, z + delta * uz
        xm, ym, zm = x - delta * ux, y - delta * uy, z - delta * uz
        fp = _frame(field, xp, yp, zp, delta, orthogonality_tol)
        fm = _frame(field, xm, ym, zm, delta, orthogonality_tol)
        with np.errstate(invalid='ignore'):
            dot_n = fp[3] * fm[3] + fp[4] * fm[4] + fp[5] * fm[5]
            invalid |= ~(dot_n > normal_flip_tol)  # NaN -> invalid
        if direction == 'n':
            # dT_dn_b = (∂T/∂n)·b
            twist['n'] = ((fp[0] - fm[0]) * bnx + (fp[1] - fm[1]) * bny
                          + (fp[2] - fm[2]) * bnz) * inv2d
        else:
            # dn_db_T = (∂n/∂b)·T
            twist['b'] = ((fp[3] - fm[3]) * tx + (fp[4] - fm[4]) * ty
                          + (fp[5] - fm[5]) * tz) * inv2d
        grad[direction] = (_bmag(field, xp, yp, zp)
                           - _bmag(field, xm, ym, zm)) * inv2d

    with np.errstate(invalid='ignore'):
        alpha = twist['n'] + twist['b']
        mu0J_T = B0 * alpha
        mu0J_n = grad['b']
        mu0J_b = B0 * curvature - grad['n']

    results = {
        'mu0J_T': mu0J_T, 'mu0J_n': mu0J_n, 'mu0J_b': mu0J_b,
        'mu0J_x': mu0J_T * tx + mu0J_n * nx + mu0J_b * bnx,
        'mu0J_y': mu0J_T * ty + mu0J_n * ny + mu0J_b * bny,
        'mu0J_z': mu0J_T * tz + mu0J_n * nz + mu0J_b * bnz,
        'alpha': alpha,
        'B': B0,
        'curvature': curvature,
    }
    keep = ('B', 'curvature')  # defined wherever their own inputs are
    for key, val in results.items():
        if key not in keep:
            val = np.where(invalid, np.nan, val)
        results[key] = _finish(scalar_input, val)
    return results


def verify_divergence_identity(field, x, y, z, delta=0.01,
                               orthogonality_tol=DEFAULT_ORTHOGONALITY_TOL,
                               normal_flip_tol=DEFAULT_NORMAL_FLIP_TOL):
    """
    ∇·B evaluated in the Frenet frame: ∂B/∂T + B(dT_dn_n + dT_db_b).

    For B = B·T, ∇·B = ∂B/∂T + B ∇·T with ∇·T = dT_dn_n + dT_db_b, so
    this assembles the divergence from two independent machineries:
    ``dB_dT`` from `field_magnitude_derivatives` (scalar |B| differences)
    and the frame divergence from `field_line_directional_derivatives`
    (frame vector differences). For a divergence-free field the result
    vanishes to finite-difference accuracy against a local scale such as
    ``B * curvature``, which makes it an end-to-end consistency check of
    both modules.

    A residual that does *not* shrink with ``delta`` is a property of the
    field, not of the differencing: empirical models are not exactly
    solenoidal everywhere (e.g. T96 reaches |∇·B| of order 1 nT/Re in
    parts of its Birkeland-current module), and interpolated simulation
    data inherits the interpolant's divergence. Compare with
    `~mageometry.io.GriddedField.divergence` for gridded data.

    Parameters
    ----------
%s
%s
%s

    Returns
    -------
    divergence : float or ndarray
        ∂B/∂T + B(dT_dn_n + dT_db_b) = ∇·B, in field-unit/length-unit.
        NaN where either ingredient is undefined.
    """
    mag = field_magnitude_derivatives(field, x, y, z, delta=delta,
                                      orthogonality_tol=orthogonality_tol)
    dd = field_line_directional_derivatives(
        field, x, y, z, delta=delta, orthogonality_tol=orthogonality_tol,
        normal_flip_tol=normal_flip_tol)
    return mag['dB_dT'] + mag['B'] * (dd['dT_dn_n'] + dd['dT_db_b'])


for _f in (field_magnitude_derivatives, field_line_current_density,
           verify_divergence_identity):
    _f.__doc__ %= (_DOC_FIELD, _DOC_DELTA, _DOC_TOL)
del _f
