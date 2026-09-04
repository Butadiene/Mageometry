# mageometry/viz3d/lines.py
"""Traced field lines as 3D polylines or tubes."""

import numpy as np

from ..viz._quantities import resolve_quantity
from ..viz._scales import resolve_scale
from ..viz.lines import _is_color
from ._pv import require_pyvista
from .mesh import trace_polydata

__all__ = ["add_field_lines"]


def add_field_lines(plotter, trace, color=None, field=None, delta=0.01,
                    cmap=None, log=None, vmin=None, vmax=None, tube_radius=None,
                    line_width=2.0, label=None, scalar_bar=True, **kwargs):
    """
    Add traced field lines to a PyVista plotter.

    Parameters
    ----------
    plotter : pyvista.Plotter
    trace : FieldLineTrace
        Result of `trace_field_lines`.
    color : None, str, array, or quantity, optional
        - None: plotter default colour;
        - a colour ('red', '#rrggbb', ...): all lines in that colour;
        - a quantity name or ``quantity(field, x, y, z)`` callable (needs
          ``field``): lines coloured by that value along the path;
        - an array of shape ``trace.x.shape``: colour by these values.
    field : callable, optional
        Required when ``color`` is a quantity.
    delta : float, optional
        Finite-difference step for quantity evaluation.
    cmap, log, vmin, vmax, label
        Colour scale options when colouring by a quantity (same conventions
        as `mageometry.viz`: log for positive quantities, symmetric diverging
        for signed ones).
    tube_radius : float, optional
        Render as tubes of this radius (field length units) instead of GL
        lines.
    line_width : float, optional
        Line width when not using tubes. Default 2.
    scalar_bar : bool, optional
        Show a scalar bar when colouring by a quantity. Default True.
    **kwargs
        Passed to ``plotter.add_mesh``.

    Returns
    -------
    pyvista.Actor
    """
    require_pyvista()
    by_value = color is not None and not _is_color(color)
    poly = trace_polydata(trace, color=color if by_value else None,
                          field=field, delta=delta, label=label)
    if poly.n_points == 0:
        raise ValueError("trace has no line with at least 2 points.")

    kw = dict(kwargs)
    if by_value:
        if isinstance(color, np.ndarray):
            q = resolve_quantity(lambda f, x, y, z: None, label or 'value')
        else:
            q = resolve_quantity(color, label)
        name = poly.point_data.active_scalars_name
        lo, hi, use_log = resolve_scale(np.asarray(poly.point_data[name]), q,
                                        log=log, vmin=vmin, vmax=vmax)
        kw.setdefault('scalar_bar_args', {'title': label or q.label})
        kw.update(scalars=name, cmap=cmap or q.cmap, clim=(lo, hi),
                  log_scale=use_log, nan_opacity=0.0, show_scalar_bar=scalar_bar)
    elif color is not None:
        kw['color'] = color

    mesh = poly.tube(radius=tube_radius) if tube_radius else poly
    if not tube_radius:
        kw.setdefault('line_width', line_width)
    return plotter.add_mesh(mesh, **kw)
