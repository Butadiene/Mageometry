# mageometry/viz/planes.py
"""Axis-aligned planes: sampling grids and projections."""

import numpy as np

_PLANES = {
    'xy': (0, 1, 2),
    'xz': (0, 2, 1),
    'yz': (1, 2, 0),
    'yx': (1, 0, 2),
    'zx': (2, 0, 1),
    'zy': (2, 1, 0),
}
_NAMES = ('x', 'y', 'z')


def plane_axes(plane):
    """
    Index triple ``(horizontal, vertical, normal)`` for a plane name.

    ``plane`` is two of ``'x'``, ``'y'``, ``'z'``: the first letter is the
    horizontal plot axis, the second the vertical, e.g. ``'xz'`` for the
    noon-midnight meridian with x horizontal.
    """
    try:
        return _PLANES[plane.lower()]
    except (KeyError, AttributeError):
        raise ValueError(f"plane must be one of {sorted(_PLANES)}, got {plane!r}.") from None


def plane_labels(plane):
    h, v, _ = plane_axes(plane)
    return _NAMES[h], _NAMES[v]


def plane_grid(plane, extent, n=100, offset=0.0):
    """
    Sample points on an axis-aligned plane.

    Parameters
    ----------
    plane : str
        ``'xy'``, ``'xz'``, ``'yz'`` (or reversed); see `plane_axes`.
    extent : sequence
        ``(hmin, hmax, vmin, vmax)`` in plot-axis order.
    n : int or (int, int), optional
        Number of samples along (horizontal, vertical). Default 100.
    offset : float, optional
        Coordinate along the plane normal. Default 0.

    Returns
    -------
    H, V : ndarray, shape (nv, nh)
        2D plot coordinates (``meshgrid`` with ``indexing='xy'``).
    x, y, z : ndarray, shape (nv, nh)
        3D coordinates of the samples, for passing to field/geometry
        functions (ravel them; results reshape to ``H.shape``).
    """
    h, v, nrm = plane_axes(plane)
    nh, nv = (n, n) if np.isscalar(n) else n
    hs = np.linspace(extent[0], extent[1], int(nh))
    vs = np.linspace(extent[2], extent[3], int(nv))
    H, V = np.meshgrid(hs, vs)
    coords = [None, None, None]
    coords[h] = H
    coords[v] = V
    coords[nrm] = np.full_like(H, float(offset))
    return H, V, coords[0], coords[1], coords[2]


def project(plane, x, y, z):
    """Return the in-plane ``(h, v)`` components of 3D coordinates or vectors."""
    h, v, _ = plane_axes(plane)
    comps = (x, y, z)
    return comps[h], comps[v]
