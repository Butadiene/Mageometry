# mageometry/viz3d/explore.py
"""One-call interactive viewer composing the viz3d primitives."""

import numpy as np

from ..tracing import trace_field_lines
from .lines import add_field_lines
from .slicer import slice_view

__all__ = ["explore"]


def explore(gridded_field, quantity='bmag', seeds=None, mode='ortho',
            line_color='curvature', field=None, delta=0.01, method='linear',
            trace_kwargs=None, tube_radius=None, plotter=None, show=True,
            **kwargs):
    """
    Interactive 3D view of a gridded field: slice widgets plus field lines.

    A convenience wrapper around `slice_view`, `trace_field_lines`, and
    `add_field_lines`; use those directly for full control.

    Parameters
    ----------
    gridded_field : GriddedField
    quantity : str, Quantity, or callable, optional
        Scalar shown on the slice planes. Default ``'bmag'``.
    seeds : (n, 3) array_like, optional
        Seed points; field lines are traced from them in both directions and
        added to the view.
    mode : {'ortho', 'plane'}, optional
        Slice widget mode (see `slice_view`). Default ``'ortho'``.
    line_color : quantity or colour, optional
        Colour of the traced lines (see `add_field_lines`). Default
        ``'curvature'``.
    field : callable, optional
        Field callable used for tracing and quantity evaluation. Default:
        ``gridded_field.field(method)``.
    delta : float, optional
        Finite-difference step for geometry quantities.
    method : str, optional
        Interpolation method for the default field callable.
    trace_kwargs : dict, optional
        Extra arguments for `trace_field_lines`. Defaults:
        ``direction='both'``, ``ds`` = the mean grid spacing, and
        ``bounds=gridded_field.bounds``.
    tube_radius : float, optional
        Render lines as tubes of this radius.
    plotter : pyvista.Plotter, optional
    show : bool, optional
        Open the render window (blocking). Default True.
    **kwargs
        Passed to `slice_view` (e.g. ``front_view=False`` to disable the
        face-on slice panels). Field lines are drawn into the main 3D
        subplot only.

    Returns
    -------
    pyvista.Plotter
    """
    gf = gridded_field
    if field is None:
        field = gf.field(method)
    plotter = slice_view(gf, quantity=quantity, mode=mode, field=field,
                         delta=delta, method=method, plotter=plotter,
                         show=False, **kwargs)
    if seeds is not None:
        seeds = np.atleast_2d(np.asarray(seeds, dtype=np.float64))
        if seeds.ndim != 2 or seeds.shape[1] != 3:
            raise ValueError("seeds must have shape (n, 3).")
        ds = float(np.mean([np.mean(np.diff(a)) for a in (gf.x, gf.y, gf.z)]))
        tk = dict(direction='both', ds=ds, bounds=gf.bounds)
        tk.update(trace_kwargs or {})
        trace = trace_field_lines(field, seeds[:, 0], seeds[:, 1], seeds[:, 2],
                                  **tk)
        add_field_lines(plotter, trace, color=line_color, field=field,
                        delta=delta, tube_radius=tube_radius)
    if show:
        plotter.show()
    return plotter
