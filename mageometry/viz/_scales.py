# mageometry/viz/_scales.py
"""
Backend-neutral colour-scale limits.

`resolve_scale` implements the library's colour conventions — symmetric about
zero for signed quantities, log10 for positive ones — as plain numbers, so
both the matplotlib layer (`viz`) and the PyVista layer (`viz3d`) share them.
"""

import numpy as np

__all__ = ["resolve_scale"]


def resolve_scale(values, quantity, log=None, vmin=None, vmax=None, percentile=98.0):
    """
    Colour-scale limits for ``values`` following the quantity's conventions:
    symmetric about zero for signed quantities, log10 for positive ones by
    default. Limits default to robust percentiles.

    Parameters
    ----------
    values : array_like
        The values to be coloured (NaN entries are ignored).
    quantity : Quantity
        Supplies the ``symmetric`` / ``positive`` / ``log`` conventions.
    log : bool, optional
        Override the quantity's log convention.
    vmin, vmax : float, optional
        Explicit limits (for symmetric quantities ``vmax`` sets the
        half-range).
    percentile : float, optional
        Robust percentile for the default limits. Default 98.

    Returns
    -------
    (vmin, vmax, log) : (float, float, bool)
        Scale limits and whether the scale is logarithmic. When ``log`` is
        True both limits are strictly positive.
    """
    finite = np.asarray(values)[np.isfinite(values)]
    if log is None:
        log = quantity.log
    if finite.size == 0:
        return 0.0, 1.0, False
    if quantity.symmetric:
        lim = np.percentile(np.abs(finite), percentile) if vmax is None else vmax
        lim = lim if lim > 0 else 1.0
        return float(-lim if vmin is None else vmin), float(lim), False
    if log:
        pos = finite[finite > 0]
        if pos.size:
            lo = np.percentile(pos, 100 - percentile) if vmin is None else vmin
            hi = np.percentile(pos, percentile) if vmax is None else vmax
            if hi <= lo:
                hi = lo * 10.0
            return float(lo), float(hi), True
    lo = (0.0 if quantity.positive else np.percentile(finite, 100 - percentile)) if vmin is None else vmin
    hi = np.percentile(finite, percentile) if vmax is None else vmax
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi), False
