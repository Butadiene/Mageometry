"""
Magnetic field line geometry: Frenet-Serret frame, curvature, torsion.

All functions take the magnetic field as a callable
``field(x, y, z) -> (bx, by, bz)`` accepting NumPy arrays, so any field
source can be analyzed: geopack models wrapped via
`mageometry.fields.geopack_field`, interpolated simulation output
(`GriddedField.field()`), or a custom callable. Units are the field's own:
positions in its length unit, curvature and torsion in 1/length-unit.

Numerical method
----------------
Derivatives along the field line are central finite differences with step
``delta`` along the local tangent: dT/ds ≈ [T(r + δT) - T(r - δT)] / 2δ.
The principal normal is the component of dT/ds perpendicular to T (the
parallel component of the finite-difference estimate is pure truncation
error and is projected out), so the returned frame is orthonormal by
construction.

Validity and NaN
----------------
Quantities that are undefined or numerically unreliable are returned as
NaN rather than a sentinel value, so they propagate through further
calculations and can be masked with ``np.isfinite``:

- the tangent is NaN where |B| is zero or non-finite (magnetic nulls,
  points outside an interpolated field's domain);
- the normal and binormal are NaN where the curvature is zero (straight
  lines) or where the finite-difference estimate is inconsistent, i.e.
  ``cos_theta = |T . dT/ds| / |dT/ds|`` exceeds ``orthogonality_tol``.
  ``cos_theta`` scales as δ²|κ'|/(3κ) — it grows where the curvature
  changes rapidly compared with ``delta`` — so exceeding the tolerance
  means ``delta`` is too large for the local curvature scale;
- curvature is returned wherever the tangents entering the difference are
  defined; torsion additionally needs valid frames at r ± δT.
"""

import numpy as np

DEFAULT_ORTHOGONALITY_TOL = 0.1
"""Default ``orthogonality_tol``: the largest accepted
``|T . dT/ds| / |dT/ds|``. Well-resolved fields give ~1e-6 at delta=0.01 and
~1e-3 at delta=0.25; values above 0.1 indicate finite differences that no
longer resolve the curvature."""

_DOC_FIELD = """    field : callable
        Magnetic field function ``field(x, y, z) -> (bx, by, bz)`` accepting
        NumPy arrays. Use `mageometry.fields.geopack_field` for the geopack
        (Tsyganenko/IGRF/dipole) models, `GriddedField.field()` for
        simulation data, or any custom callable.
    x, y, z : float or array_like
        Position coordinates (field's length unit)."""

_DOC_DELTA = """    delta : float, optional
        Finite-difference step along the field line (field's length unit).
        Default 0.01. Use about one grid cell for linearly interpolated
        data."""

_DOC_TOL = """    orthogonality_tol : float, optional
        Largest accepted ``|T . dT/ds| / |dT/ds|``; the normal and binormal
        are NaN where it is exceeded. Default `DEFAULT_ORTHOGONALITY_TOL`
        (0.1). Pass ``np.inf`` to disable the check."""


# ---------------------------------------------------------------------------
# Internal array-level core
# ---------------------------------------------------------------------------

def _as_arrays(x, y, z):
    """Broadcast inputs to 1D float64 arrays; report whether all were scalars."""
    scalar_input = np.isscalar(x) and np.isscalar(y) and np.isscalar(z)
    x, y, z = np.broadcast_arrays(np.atleast_1d(np.asarray(x, dtype=np.float64)),
                                  np.atleast_1d(np.asarray(y, dtype=np.float64)),
                                  np.atleast_1d(np.asarray(z, dtype=np.float64)))
    return scalar_input, x, y, z


def _finish(scalar_input, *arrays):
    """Return ``.item()`` values for scalar input, arrays otherwise."""
    if scalar_input:
        out = tuple(a.item() for a in arrays)
    else:
        out = tuple(arrays)
    return out[0] if len(out) == 1 else out


def _tangent(field, x, y, z):
    """Unit tangent B/|B| on arrays; NaN where |B| is zero or non-finite."""
    bx, by, bz = field(x, y, z)
    bx = np.asarray(bx, dtype=np.float64)
    by = np.asarray(by, dtype=np.float64)
    bz = np.asarray(bz, dtype=np.float64)
    b_mag = np.sqrt(bx * bx + by * by + bz * bz)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = np.where(b_mag > 0, 1.0 / b_mag, np.nan)
    return bx * inv, by * inv, bz * inv


def _frame(field, x, y, z, delta, orthogonality_tol):
    """
    Frenet-Serret frame on arrays (three field evaluations).

    Returns ``(tx, ty, tz, nx, ny, nz, bx, by, bz, curvature, cos_theta)``.
    """
    tx, ty, tz = _tangent(field, x, y, z)
    txp, typ, tzp = _tangent(field, x + delta * tx, y + delta * ty, z + delta * tz)
    txm, tym, tzm = _tangent(field, x - delta * tx, y - delta * ty, z - delta * tz)

    inv2d = 0.5 / delta
    dtx = (txp - txm) * inv2d
    dty = (typ - tym) * inv2d
    dtz = (tzp - tzm) * inv2d

    # Split dT/ds into components parallel and perpendicular to T.
    par = tx * dtx + ty * dty + tz * dtz
    px = dtx - par * tx
    py = dty - par * ty
    pz = dtz - par * tz
    curvature = np.sqrt(px * px + py * py + pz * pz)
    dt_mag = np.sqrt(dtx * dtx + dty * dty + dtz * dtz)

    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = np.where(dt_mag > 0, np.abs(par) / dt_mag, np.nan)
        valid = (curvature > 0) & (cos_theta <= orthogonality_tol)  # NaN -> False
        inv_k = np.where(valid, 1.0 / curvature, np.nan)

    nx, ny, nz = px * inv_k, py * inv_k, pz * inv_k
    bx = ty * nz - tz * ny
    by = tz * nx - tx * nz
    bz = tx * ny - ty * nx
    return tx, ty, tz, nx, ny, nz, bx, by, bz, curvature, cos_theta


def _torsion(field, x, y, z, delta, orthogonality_tol, frame0=None):
    """Torsion on arrays: τ = -N . dB/ds, with B differenced along T."""
    if frame0 is None:
        frame0 = _frame(field, x, y, z, delta, orthogonality_tol)
    tx, ty, tz, nx, ny, nz = frame0[:6]
    fp = _frame(field, x + delta * tx, y + delta * ty, z + delta * tz,
                delta, orthogonality_tol)
    fm = _frame(field, x - delta * tx, y - delta * ty, z - delta * tz,
                delta, orthogonality_tol)
    inv2d = 0.5 / delta
    dbx = (fp[6] - fm[6]) * inv2d
    dby = (fp[7] - fm[7]) * inv2d
    dbz = (fp[8] - fm[8]) * inv2d
    return -(nx * dbx + ny * dby + nz * dbz)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def field_line_tangent(field, x, y, z):
    """
    Unit tangent vector of the field line: T = B/|B|.

    Parameters
    ----------
%s

    Returns
    -------
    tx, ty, tz : float or ndarray
        Components of the unit tangent; NaN where |B| is zero or non-finite.
        Scalars for scalar input, arrays for array input.
    """
    scalar_input, x, y, z = _as_arrays(x, y, z)
    return _finish(scalar_input, *_tangent(field, x, y, z))


def field_line_curvature(field, x, y, z, delta=0.01):
    """
    Field line curvature κ = |dT/ds| by central finite differences.

    Parameters
    ----------
%s
%s

    Returns
    -------
    curvature : float or ndarray
        Curvature (1/length-unit); NaN where the tangent is undefined at the
        point or at r ± δT.
    """
    scalar_input, x, y, z = _as_arrays(x, y, z)
    frame = _frame(field, x, y, z, delta, np.inf)
    return _finish(scalar_input, frame[9])


def field_line_normal(field, x, y, z, delta=0.01,
                      orthogonality_tol=DEFAULT_ORTHOGONALITY_TOL):
    """
    Unit principal normal of the field line: N = (dT/ds)_⊥ / |(dT/ds)_⊥|.

    Parameters
    ----------
%s
%s
%s

    Returns
    -------
    nx, ny, nz : float or ndarray
        Components of the unit normal; NaN where undefined (zero curvature,
        undefined tangent) or where ``orthogonality_tol`` is exceeded.
    """
    scalar_input, x, y, z = _as_arrays(x, y, z)
    frame = _frame(field, x, y, z, delta, orthogonality_tol)
    return _finish(scalar_input, *frame[3:6])


def field_line_binormal(field, x, y, z, delta=0.01,
                        orthogonality_tol=DEFAULT_ORTHOGONALITY_TOL):
    """
    Unit binormal of the field line: B = T × N.

    Parameters
    ----------
%s
%s
%s

    Returns
    -------
    bx, by, bz : float or ndarray
        Components of the unit binormal; NaN wherever the normal is NaN.
    """
    scalar_input, x, y, z = _as_arrays(x, y, z)
    frame = _frame(field, x, y, z, delta, orthogonality_tol)
    return _finish(scalar_input, *frame[6:9])


def field_line_torsion(field, x, y, z, delta=0.01,
                       orthogonality_tol=DEFAULT_ORTHOGONALITY_TOL):
    """
    Field line torsion τ = -N · dB/ds by central finite differences.

    Parameters
    ----------
%s
%s
%s

    Returns
    -------
    torsion : float or ndarray
        Torsion (1/length-unit); NaN where the frame is undefined at the
        point or at r ± δT.
    """
    scalar_input, x, y, z = _as_arrays(x, y, z)
    return _finish(scalar_input, _torsion(field, x, y, z, delta, orthogonality_tol))


def field_line_frenet_frame(field, x, y, z, delta=0.01,
                            orthogonality_tol=DEFAULT_ORTHOGONALITY_TOL):
    """
    Frenet-Serret frame and curvature.

    Parameters
    ----------
%s
%s
%s

    Returns
    -------
    tx, ty, tz : float or ndarray
        Unit tangent components.
    nx, ny, nz : float or ndarray
        Unit normal components (NaN where undefined).
    bx, by, bz : float or ndarray
        Unit binormal components (NaN where undefined).
    curvature : float or ndarray
        Curvature (1/length-unit).
    """
    scalar_input, x, y, z = _as_arrays(x, y, z)
    frame = _frame(field, x, y, z, delta, orthogonality_tol)
    return _finish(scalar_input, *frame[:10])


def field_line_geometry_complete(field, x, y, z, delta=0.01,
                                 orthogonality_tol=DEFAULT_ORTHOGONALITY_TOL):
    """
    Frenet-Serret frame, curvature, and torsion.

    Parameters
    ----------
%s
%s
%s

    Returns
    -------
    tx, ty, tz, nx, ny, nz, bx, by, bz : float or ndarray
        Unit tangent, normal, and binormal components (normal and binormal
        NaN where undefined).
    curvature : float or ndarray
        Curvature (1/length-unit).
    torsion : float or ndarray
        Torsion (1/length-unit).
    """
    scalar_input, x, y, z = _as_arrays(x, y, z)
    frame = _frame(field, x, y, z, delta, orthogonality_tol)
    torsion = _torsion(field, x, y, z, delta, orthogonality_tol, frame0=frame)
    return _finish(scalar_input, *frame[:10], torsion)


def field_line_frame_quality(field, x, y, z, delta=0.01):
    """
    Finite-difference consistency diagnostic ``cos_theta``.

    ``cos_theta = |T . dT/ds| / |dT/ds|`` is zero for an exact derivative;
    it grows as δ²|κ'|/(3κ) where the curvature changes rapidly compared
    with ``delta``. Use it to choose ``delta`` and ``orthogonality_tol``.

    Parameters
    ----------
%s
%s

    Returns
    -------
    cos_theta : float or ndarray
        In [0, 1]; NaN where dT/ds is zero or undefined.
    """
    scalar_input, x, y, z = _as_arrays(x, y, z)
    frame = _frame(field, x, y, z, delta, np.inf)
    return _finish(scalar_input, frame[10])


field_line_tangent.__doc__ %= (_DOC_FIELD,)
field_line_curvature.__doc__ %= (_DOC_FIELD, _DOC_DELTA)
field_line_frame_quality.__doc__ %= (_DOC_FIELD, _DOC_DELTA)
for _f in (field_line_normal, field_line_binormal, field_line_torsion,
           field_line_frenet_frame, field_line_geometry_complete):
    _f.__doc__ %= (_DOC_FIELD, _DOC_DELTA, _DOC_TOL)
del _f
