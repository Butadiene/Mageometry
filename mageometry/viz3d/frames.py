# mageometry/viz3d/frames.py
"""Frenet-Serret frame arrows as 3D glyphs."""

import numpy as np

from ..geometry import field_line_frenet_frame
from ._pv import require_pyvista

__all__ = ["add_frenet_frame"]

# Same tab:red / tab:green / tab:blue as mageometry.viz.frames, as hex so no
# matplotlib colour lookup is needed.
_COLORS = {'T': '#d62728', 'n': '#2ca02c', 'b': '#1f77b4'}


def add_frenet_frame(plotter, field, x, y, z, delta=0.01, length=1.0,
                     vectors=('T', 'n', 'b'), legend=True, **kwargs):
    """
    Add tangent / normal / binormal arrows at one or more points.

    Points where the frame is undefined (NaN normal) get only their tangent
    drawn, matching `mageometry.viz.plot_frenet_frame`.

    Parameters
    ----------
    plotter : pyvista.Plotter
    field : callable
    x, y, z : float or array_like
        Points at which to draw the frame.
    delta : float, optional
        Finite-difference step.
    length : float, optional
        Arrow length in the field's length unit. Default 1.
    vectors : sequence of {'T', 'n', 'b'}, optional
        Which vectors to draw. Default all three.
    legend : bool, optional
        Add a legend for the drawn vectors. Default True.
    **kwargs
        Passed to ``plotter.add_mesh``.

    Returns
    -------
    dict
        ``{'T': actor, 'n': actor, 'b': actor}`` for the drawn vectors.
    """
    pv = require_pyvista()
    x, y, z = (np.atleast_1d(np.asarray(c, dtype=np.float64)).ravel()
               for c in (x, y, z))
    frame = field_line_frenet_frame(field, x, y, z, delta=delta)
    comps = {'T': frame[0:3], 'n': frame[3:6], 'b': frame[6:9]}
    out = {}
    entries = []
    for name in vectors:
        vx, vy, vz = (np.atleast_1d(c) for c in comps[name])
        ok = np.isfinite(vx) & np.isfinite(vy) & np.isfinite(vz)
        if not np.any(ok):
            continue
        cloud = pv.PolyData(np.column_stack([x[ok], y[ok], z[ok]]))
        cloud.point_data['vec'] = np.column_stack([vx[ok], vy[ok], vz[ok]])
        glyphs = cloud.glyph(orient='vec', scale=False, factor=length,
                             geom=pv.Arrow())
        out[name] = plotter.add_mesh(glyphs, color=_COLORS[name], **kwargs)
        entries.append([name, _COLORS[name]])
    if legend and entries:
        plotter.add_legend(entries, bcolor=None)
    return out
