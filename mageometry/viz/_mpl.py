# mageometry/viz/_mpl.py
"""Lazy matplotlib access and small shared helpers."""

import numpy as np


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "mageometry.viz requires the optional dependency matplotlib. "
            "Install it with: pip install matplotlib"
        ) from None
    return plt


def get_axes(ax=None, projection=None, figsize=None):
    """Return ``ax`` or a new axes (3D if ``projection='3d'``)."""
    plt = require_matplotlib()
    if ax is not None:
        return ax
    fig = plt.figure(figsize=figsize)
    return fig.add_subplot(111, projection=projection)


def is_3d(ax):
    return getattr(ax, 'name', '') == '3d'


def axis_label(name, unit=None):
    return f"{name} ({unit})" if unit else name


def label_axes(ax, names, unit=None):
    """
    Label the axes ``(h, v[, d])`` of ``ax``, keeping labels already set by an
    earlier plot unless a unit is given explicitly.
    """
    setters = [ax.set_xlabel, ax.set_ylabel]
    getters = [ax.get_xlabel, ax.get_ylabel]
    if len(names) == 3:
        setters.append(ax.set_zlabel)
        getters.append(ax.get_zlabel)
    for name, setter, getter in zip(names, setters, getters):
        if unit is not None or not getter():
            setter(axis_label(name, unit))


def color_norm(values, quantity, log=None, vmin=None, vmax=None, percentile=98.0):
    """
    Pick a colour normalisation for ``values`` following the quantity's
    conventions: symmetric about zero for signed quantities, log10 for
    positive ones by default. Limits default to robust percentiles.
    """
    import matplotlib.colors as mcolors

    finite = np.asarray(values)[np.isfinite(values)]
    if log is None:
        log = quantity.log
    if finite.size == 0:
        return mcolors.Normalize(0.0, 1.0)
    if quantity.symmetric:
        lim = np.percentile(np.abs(finite), percentile) if vmax is None else vmax
        lim = lim if lim > 0 else 1.0
        return mcolors.Normalize(-lim if vmin is None else vmin, lim)
    if log:
        pos = finite[finite > 0]
        if pos.size == 0:
            log = False
        else:
            lo = np.percentile(pos, 100 - percentile) if vmin is None else vmin
            hi = np.percentile(pos, percentile) if vmax is None else vmax
            if hi <= lo:
                hi = lo * 10.0
            return mcolors.LogNorm(lo, hi)
    lo = (0.0 if quantity.positive else np.percentile(finite, 100 - percentile)) if vmin is None else vmin
    hi = np.percentile(finite, percentile) if vmax is None else vmax
    if hi <= lo:
        hi = lo + 1.0
    return mcolors.Normalize(lo, hi)
