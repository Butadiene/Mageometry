# mageometry/viz/maps.py
"""Maps of geometry quantities on axis-aligned planes."""

import numpy as np

from ._mpl import require_matplotlib, get_axes, label_axes, color_norm
from ._quantities import resolve_quantity
from .planes import plane_grid, plane_labels, project

__all__ = ["plot_geometry_map", "plot_field_direction"]


def plot_geometry_map(field, quantity='curvature', plane='xz', extent=(-15, 5, -8, 8),
                      n=120, offset=0.0, delta=0.01, ax=None, log=None,
                      vmin=None, vmax=None, cmap=None, colorbar=True, label=None,
                      arrows=False, arrow_step=6, mask=None, unit=None,
                      quantity_kwargs=None, **pcolormesh_kwargs):
    """
    Colour map of a scalar field-line quantity on a plane.

    Parameters
    ----------
    field : callable
        ``field(x, y, z) -> (bx, by, bz)``.
    quantity : str or callable, optional
        ``'curvature'`` (default), ``'torsion'``, ``'frame_quality'``,
        ``'bmag'``, ``'bx'``/``'by'``/``'bz'``, any directional-derivative
        key (``'dT_dn_n'``, ...), or ``quantity(field, x, y, z) -> values``.
    plane : str, optional
        ``'xy'``, ``'xz'`` (default), ``'yz'`` or reversed; first letter is
        the horizontal axis.
    extent : sequence, optional
        ``(hmin, hmax, vmin, vmax)`` in the field's length unit.
    n : int or (int, int), optional
        Samples along (horizontal, vertical). Default 120.
    offset : float, optional
        Position of the plane along its normal. Default 0.
    delta : float, optional
        Finite-difference step passed to the geometry functions.
    ax : matplotlib Axes, optional
        Axes to draw on; a new figure is created if None.
    log : bool, optional
        Log10 colour scale. Default: the quantity's convention (log for
        curvature, |B|, frame quality; linear otherwise).
    vmin, vmax : float, optional
        Colour limits; defaults are robust percentiles (symmetric about zero
        for signed quantities).
    cmap : str, optional
        Colormap; default per quantity.
    colorbar : bool, optional
        Add a colorbar. Default True.
    label : str, optional
        Colorbar label; default from the quantity.
    arrows : bool, optional
        Overlay in-plane field direction arrows. Default False.
    arrow_step : int, optional
        Subsampling stride for the arrows. Default 6.
    mask : callable, optional
        ``mask(x, y, z) -> bool array``; True where the map is blanked
        (e.g. inside a planet: ``lambda x, y, z: x**2+y**2+z**2 < 1``).
    unit : str, optional
        Length unit for the axis labels.
    quantity_kwargs : dict, optional
        Extra keywords for the geometry function (e.g. ``orthogonality_tol``).
    **pcolormesh_kwargs
        Passed to ``ax.pcolormesh``.

    Returns
    -------
    mappable
        The ``QuadMesh``; ``mappable.axes`` is the axes. NaN cells (undefined
        or masked) are left blank.
    """
    plt = require_matplotlib()
    q = resolve_quantity(quantity, label)
    H, V, x, y, z = plane_grid(plane, extent, n, offset)
    values = q.evaluate(field, x.ravel(), y.ravel(), z.ravel(), delta=delta,
                        **(quantity_kwargs or {})).reshape(H.shape)
    if mask is not None:
        values = np.where(np.asarray(mask(x, y, z), dtype=bool), np.nan, values)

    ax = get_axes(ax)
    norm = color_norm(values, q, log=log, vmin=vmin, vmax=vmax)
    mesh = ax.pcolormesh(H, V, values, norm=norm, cmap=cmap or q.cmap,
                         shading='nearest', **pcolormesh_kwargs)
    if arrows:
        plot_field_direction(field, plane=plane, extent=extent,
                             n=tuple(max(2, int(k) // arrow_step) for k in ((n, n) if np.isscalar(n) else n)),
                             offset=offset, ax=ax, mask=mask)
    label_axes(ax, plane_labels(plane), unit)
    ax.set_aspect('equal')
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    if colorbar:
        cb = plt.colorbar(mesh, ax=ax)
        cb.set_label(label or q.label)
    return mesh


def plot_field_direction(field, plane='xz', extent=(-15, 5, -8, 8), n=25, offset=0.0,
                         ax=None, length=None, mask=None, unit=None, **quiver_kwargs):
    """
    Unit arrows of the in-plane field direction on a plane.

    Arrows have a fixed length (``length``, default 1/40 of the plot width)
    and show direction only; where the in-plane component is negligible
    (field nearly normal to the plane) a dot is drawn instead.

    Returns
    -------
    matplotlib Quiver
    """
    H, V, x, y, z = plane_grid(plane, extent, n, offset)
    bx, by, bz = (np.asarray(c, dtype=np.float64) for c in field(x, y, z))
    bh, bv = project(plane, bx, by, bz)
    bmag = np.sqrt(bx * bx + by * by + bz * bz)
    inplane = np.hypot(bh, bv)
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = inplane / bmag
        uh, uv = bh / inplane, bv / inplane
    ok = np.isfinite(ratio) & (ratio >= 0.05)
    if mask is not None:
        ok &= ~np.asarray(mask(x, y, z), dtype=bool)
    dots = np.isfinite(ratio) & (ratio < 0.05)
    if mask is not None:
        dots &= ~np.asarray(mask(x, y, z), dtype=bool)

    ax = get_axes(ax)
    if length is None:
        length = (extent[1] - extent[0]) / 40.0
    kw = dict(angles='xy', scale_units='xy', scale=1.0, pivot='middle',
              color='k', alpha=0.7, width=0.003, headwidth=3, headlength=4)
    kw.update(quiver_kwargs)
    quiver = ax.quiver(H[ok], V[ok], length * uh[ok], length * uv[ok], **kw)
    if np.any(dots):
        ax.scatter(H[dots], V[dots], s=6, c=kw['color'], alpha=kw['alpha'], linewidths=0)
    label_axes(ax, plane_labels(plane), unit)
    ax.set_aspect('equal')
    return quiver
