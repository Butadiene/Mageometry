# mageometry/fields.py
"""
Magnetic field sources for Mageometry.

The geometry functions in `mageometry.geometry` accept any callable with
signature ``field(x, y, z) -> (bx, by, bz)`` (positions in GSM coordinates in
Re, field components in nT). This module provides adapters that wrap the
geopack field models into that form; interpolated fields from simulation
output files will be added here as well (planned `mageometry.io`).
"""

from .geopack import (
    dip,
    igrf_gsm_vectorized,
    t89_vectorized,
    t96_vectorized,
    t01_vectorized,
    t04_vectorized,
)

_EXTERNAL_MODELS = {
    't89': t89_vectorized,
    't96': t96_vectorized,
    't01': t01_vectorized,
    't04': t04_vectorized,
}

_INTERNAL_MODELS = {
    'dip': dip,
    'igrf': igrf_gsm_vectorized,
}


def geopack_field(external='t96', internal='dip', parmod=None, ps=None):
    """
    Build a ``field(x, y, z) -> (bx, by, bz)`` callable from geopack models.

    The returned callable evaluates the total magnetic field as the sum of an
    internal (Earth) field and an external (magnetospheric) Tsyganenko model,
    with the model parameters bound at construction time.

    `mageometry.geopack.recalc(ut)` must be called before building or
    evaluating the field: it sets the internal-model epoch and returns the
    dipole tilt angle `ps`.

    Parameters
    ----------
    external : str or None, optional
        External field model: 't89', 't96', 't01', 't04', or None for the
        internal field only. Default 't96'.
    internal : str or None, optional
        Internal field model: 'dip', 'igrf', or None for the external field
        only. Default 'dip'.
    parmod : array_like or int, optional
        Parameters of the external model: the Kp level (1-7) for 't89', or the
        10-element parameter array `[Pdyn, Dst, ByIMF, BzIMF, ...]` for
        't96'/'t01'/'t04'. Required when `external` is not None.
    ps : float, optional
        Dipole tilt angle in radians, as returned by
        `mageometry.geopack.recalc(ut)`. Required when `external` is not None.

    Returns
    -------
    field : callable
        ``field(x, y, z) -> (bx, by, bz)`` with x, y, z in GSM Re and the
        field components in nT (GSM). Accepts scalars or NumPy arrays.

    Examples
    --------
    >>> from mageometry import geopack, geopack_field, field_line_curvature
    >>> ps = geopack.recalc(100)
    >>> parmod = [2.0, -20.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0]
    >>> field = geopack_field('t96', 'dip', parmod, ps)
    >>> kappa = field_line_curvature(field, 5.0, 0.0, 0.0)
    """
    if external is None and internal is None:
        raise ValueError("At least one of `external` and `internal` must be given.")

    if external is not None:
        try:
            external_func = _EXTERNAL_MODELS[external]
        except KeyError:
            raise ValueError(
                f"Unknown external model {external!r}; "
                f"expected one of {sorted(_EXTERNAL_MODELS)} or None."
            ) from None
        if parmod is None or ps is None:
            raise ValueError(
                "`parmod` and `ps` are required when an external model is used."
            )
    else:
        external_func = None

    if internal is not None:
        try:
            internal_func = _INTERNAL_MODELS[internal]
        except KeyError:
            raise ValueError(
                f"Unknown internal model {internal!r}; "
                f"expected one of {sorted(_INTERNAL_MODELS)} or None."
            ) from None
    else:
        internal_func = None

    def field(x, y, z):
        if internal_func is None:
            return external_func(parmod, ps, x, y, z)
        bx, by, bz = internal_func(x, y, z)
        if external_func is not None:
            ex, ey, ez = external_func(parmod, ps, x, y, z)
            bx, by, bz = bx + ex, by + ey, bz + ez
        return bx, by, bz

    return field


__all__ = ["geopack_field"]
