# mageometry/viz/frames.py
"""Frenet-Serret frame arrows."""

import numpy as np

from ..geometry import field_line_frenet_frame
from ._mpl import get_axes, is_3d, label_axes
from .planes import project, plane_labels

__all__ = ["plot_frenet_frame"]

_COLORS = {'T': 'tab:red', 'n': 'tab:green', 'b': 'tab:blue'}


def plot_frenet_frame(field, x, y, z, delta=0.01, ax=None, plane='xz', length=1.0,
                      vectors=('T', 'n', 'b'), unit=None, legend=True, **kwargs):
    """
    Draw the tangent / normal / binormal at one or more points.

    On a 3D axes the vectors are drawn in 3D; on a 2D axes they are
    projected onto ``plane``. Points where the frame is undefined (NaN
    normal) get only their tangent drawn.

    Parameters
    ----------
    field : callable
    x, y, z : float or array_like
        Points at which to draw the frame.
    delta : float, optional
        Finite-difference step.
    ax : matplotlib Axes, optional
        2D or 3D axes.
    plane : str, optional
        Projection plane for 2D axes. Default ``'xz'``.
    length : float, optional
        Arrow length in the field's length unit. Default 1.
    vectors : sequence of {'T', 'n', 'b'}, optional
        Which vectors to draw. Default all three.
    unit : str, optional
        Length unit for axis labels.
    legend : bool, optional
        Add a legend. Default True.
    **kwargs
        Passed to ``ax.quiver``.

    Returns
    -------
    dict
        ``{'T': quiver, 'n': quiver, 'b': quiver}`` for the drawn vectors.
    """
    ax = get_axes(ax)
    x, y, z = (np.atleast_1d(np.asarray(c, dtype=np.float64)).ravel() for c in (x, y, z))
    frame = field_line_frenet_frame(field, x, y, z, delta=delta)
    comps = {'T': frame[0:3], 'n': frame[3:6], 'b': frame[6:9]}
    out = {}
    for name in vectors:
        vx, vy, vz = (np.atleast_1d(c) for c in comps[name])
        ok = np.isfinite(vx)
        if not np.any(ok):
            continue
        kw = dict(color=_COLORS[name], label=name)
        if is_3d(ax):
            kw.update(length=length, arrow_length_ratio=0.25, linewidth=1.5)
            kw.update(kwargs)
            out[name] = ax.quiver(x[ok], y[ok], z[ok], vx[ok], vy[ok], vz[ok], **kw)
        else:
            h, v = project(plane, x, y, z)
            uh, uv = project(plane, vx, vy, vz)
            kw.update(angles='xy', scale_units='xy', scale=1.0, width=0.006)
            kw.update(kwargs)
            out[name] = ax.quiver(h[ok], v[ok], length * uh[ok], length * uv[ok], **kw)
    if is_3d(ax):
        label_axes(ax, ('x', 'y', 'z'), unit)
    else:
        label_axes(ax, plane_labels(plane), unit)
        ax.set_aspect('equal')
    if legend and out:
        ax.legend()
    return out
