# mageometry/viz/lines.py
"""Traced field lines: projections, 3D views, and profiles along the line."""

import numpy as np

from ._mpl import require_matplotlib, get_axes, is_3d, axis_label, label_axes, color_norm
from ._quantities import resolve_quantity
from .planes import plane_labels, project

__all__ = ["plot_field_lines", "plot_line_profiles"]


def plot_field_lines(trace, plane='xz', ax=None, color=None, field=None, delta=0.01,
                     cmap=None, log=None, vmin=None, vmax=None, colorbar=True,
                     label=None, unit=None, linewidth=1.2, **kwargs):
    """
    Draw traced field lines, projected onto a plane or in 3D.

    Parameters
    ----------
    trace : FieldLineTrace
        Result of `trace_field_lines`.
    plane : str, optional
        Projection plane (``'xz'`` default). Ignored when ``ax`` is a 3D axes.
    ax : matplotlib Axes, optional
        2D or 3D (``projection='3d'``) axes; new 2D axes if None.
    color : None, str, array, or quantity, optional
        - None: one colour per line (matplotlib cycle);
        - a matplotlib colour: all lines in that colour;
        - a quantity name or ``quantity(field, x, y, z)`` callable (needs
          ``field``): lines coloured by that value along the path;
        - an array of shape ``trace.x.shape``: colour by these values.
    field : callable, optional
        Required when ``color`` is a quantity.
    delta : float, optional
        Finite-difference step for quantity evaluation.
    cmap, log, vmin, vmax, colorbar, label
        Colour scale options when colouring by a quantity.
    unit : str, optional
        Length unit for axis labels.
    **kwargs
        Passed to the underlying ``plot``/``LineCollection``.

    Returns
    -------
    list of Line2D, or LineCollection / Line3DCollection when coloured by value
    """
    plt = require_matplotlib()
    ax = get_axes(ax)
    three_d = is_3d(ax)

    values = None
    q = None
    if color is not None and not _is_color(color):
        if isinstance(color, np.ndarray) and color.shape == trace.x.shape:
            values = color
            q = resolve_quantity(lambda f, x, y, z: None, label or 'value')
        else:
            if field is None:
                raise ValueError("field is required to colour lines by a quantity.")
            q = resolve_quantity(color, label)
            flat = q.evaluate(field, trace.x.ravel(), trace.y.ravel(), trace.z.ravel(), delta=delta)
            values = flat.reshape(trace.x.shape)

    if values is None:
        artists = []
        for i in range(trace.n_lines):
            x, y, z = trace.path(i)
            kw = dict(linewidth=linewidth, **kwargs)
            if color is not None:
                kw['color'] = color
            if three_d:
                artists += ax.plot(x, y, z, **kw)
            else:
                h, v = project(plane, x, y, z)
                artists += ax.plot(h, v, **kw)
        result = artists
    else:
        segments, seg_values = [], []
        for i in range(trace.n_lines):
            x, y, z = trace.path(i)
            vals = values[i, :x.size]
            pts = np.column_stack([x, y, z]) if three_d else np.column_stack(project(plane, x, y, z))
            segments.append(np.stack([pts[:-1], pts[1:]], axis=1))
            seg_values.append(0.5 * (vals[:-1] + vals[1:]))
        segments = np.concatenate(segments)
        seg_values = np.concatenate(seg_values)
        norm = color_norm(seg_values, q, log=log, vmin=vmin, vmax=vmax)
        if three_d:
            from mpl_toolkits.mplot3d.art3d import Line3DCollection as Coll
        else:
            from matplotlib.collections import LineCollection as Coll
        result = Coll(segments, cmap=cmap or q.cmap, norm=norm, linewidth=linewidth, **kwargs)
        result.set_array(seg_values)
        if three_d:
            ax.add_collection3d(result)
        else:
            ax.add_collection(result)
            ax.autoscale_view()
        if colorbar:
            plt.colorbar(result, ax=ax).set_label(label or q.label)

    if three_d:
        label_axes(ax, ('x', 'y', 'z'), unit)
        _equal_3d(ax, trace)
    else:
        label_axes(ax, plane_labels(plane), unit)
        ax.set_aspect('equal')
    return result


def plot_line_profiles(trace, field, quantities=('curvature', 'torsion'), delta=0.01,
                       axes=None, lines=None, log=None, unit=None, labels=None, **kwargs):
    """
    Profiles of quantities along traced field lines versus arc length.

    Parameters
    ----------
    trace : FieldLineTrace
    field : callable
        The field the lines were traced through.
    quantities : sequence of str or callable, optional
        One subplot per quantity. Default curvature and torsion.
    delta : float, optional
        Finite-difference step.
    axes : sequence of Axes, optional
        One per quantity; a new figure with stacked subplots if None.
    lines : sequence of int, optional
        Line indices to plot (default all).
    log : bool or sequence of bool, optional
        Log y-scale per quantity (default: the quantity's convention).
    unit : str, optional
        Length unit for the arc-length label.
    labels : sequence of str, optional
        Legend labels per line (default ``line i``).
    **kwargs
        Passed to ``ax.plot``.

    Returns
    -------
    list of Axes
    """
    plt = require_matplotlib()
    qs = [resolve_quantity(qn) for qn in quantities]
    if axes is None:
        fig, axes = plt.subplots(len(qs), 1, sharex=True, figsize=(8, 2.8 * len(qs)), squeeze=False)
        axes = axes[:, 0]
    axes = list(axes)
    if len(axes) != len(qs):
        raise ValueError("axes must have one entry per quantity.")
    if np.isscalar(log) or log is None:
        log = [log] * len(qs)
    lines = range(trace.n_lines) if lines is None else lines
    for ax, q, lg in zip(axes, qs, log):
        for k, i in enumerate(lines):
            x, y, z = trace.path(i)
            s = trace.arc_length(i)
            vals = q.evaluate(field, x, y, z, delta=delta)
            lab = labels[k] if labels is not None else f"line {i}"
            ax.plot(s, vals, label=lab, **kwargs)
        ax.set_ylabel(q.label)
        if q.log if lg is None else lg:
            ax.set_yscale('log')
        else:
            ax.axhline(0.0, color='k', lw=0.6, alpha=0.5)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize='small')
    if unit is not None or not axes[-1].get_xlabel():
        axes[-1].set_xlabel(axis_label('arc length s from seed', unit))
    return axes


def _is_color(c):
    if isinstance(c, np.ndarray):
        return False
    if callable(c):
        return False
    if isinstance(c, str):
        from ._quantities import QUANTITIES
        if c in QUANTITIES:
            return False
        import matplotlib.colors as mcolors
        return mcolors.is_color_like(c)
    try:
        import matplotlib.colors as mcolors
        return mcolors.is_color_like(c)
    except Exception:
        return False


def _equal_3d(ax, trace):
    xs, ys, zs = (np.asarray(a)[np.isfinite(a)] for a in (trace.x, trace.y, trace.z))
    if xs.size == 0:
        return
    centre = np.array([xs.mean(), ys.mean(), zs.mean()])
    half = 0.5 * max(np.ptp(xs), np.ptp(ys), np.ptp(zs), 1e-9)
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass
