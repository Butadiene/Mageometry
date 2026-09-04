# mageometry/viz3d/slicer.py
"""Interactive slice viewers for gridded volume data."""

import numpy as np

from ..viz._quantities import QUANTITIES, Quantity, resolve_quantity
from ..viz._scales import resolve_scale
from ._pv import get_plotter, require_pyvista
from .mesh import _array_name, to_rectilinear_grid

__all__ = ["slice_view"]

# Canonical face-on camera per dominant normal axis: (view direction from the
# plane towards the camera, view-up). Chosen so the panels match the 2D axis
# conventions of `mageometry.viz` (x slice: y right/z up; y slice: x right/
# z up; z slice: x right/y up).
_FACE_CAMERAS = {
    0: ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    1: ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    2: ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
}


def _face_camera(normal, origin, dist):
    """
    Camera placement looking at a slice plane face-on.

    Returns ``(position, up)`` for a camera at distance ``dist`` from
    ``origin`` along the plane normal. Normals (anti)parallel to a coordinate
    axis use the canonical side and up vector from ``_FACE_CAMERAS``
    regardless of the normal's sign; oblique normals view from the side the
    normal points to, with view-up as close to +z (or +y for near-z normals)
    as possible.
    """
    origin = np.asarray(origin, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    norm = np.linalg.norm(n)
    if not np.isfinite(norm) or norm == 0.0:
        direction, up = _FACE_CAMERAS[1]
        return origin + dist * np.asarray(direction), np.asarray(up)
    n = n / norm
    for axis in range(3):
        if abs(n[axis]) > 0.999:
            direction, up = _FACE_CAMERAS[axis]
            return origin + dist * np.asarray(direction), np.asarray(up)
    up = np.array([0.0, 0.0, 1.0]) if abs(n[2]) <= 0.9 else np.array([0.0, 1.0, 0.0])
    up = up - np.dot(up, n) * n
    up = up / np.linalg.norm(up)
    return origin + dist * n, up


def _widget_state(plotter):
    """
    The object carrying ``plane_widgets`` and ``plane_sliced_meshes``:
    ``plotter.widgets`` on newer PyVista, the plotter itself before that.
    """
    return plotter.widgets if hasattr(plotter, 'widgets') else plotter


def _add_face_panel(plotter, index, sliced_mesh, label, panel_kwargs,
                    widget, dist):
    """
    Add one face-on slice panel to subplot ``index``, aim its camera at the
    slice widget's plane, and keep it tracking the widget interactively.
    """
    plotter.subplot(index)
    plotter.add_mesh(sliced_mesh, **panel_kwargs)
    plotter.add_text(label, font_size=10)
    renderer = plotter.renderers[index]
    renderer.enable_parallel_projection()

    def _aim(*_args):
        _aim_face_camera(renderer, np.asarray(widget.GetNormal()),
                         np.asarray(widget.GetOrigin()), dist)

    _aim()
    # Fit the parallel scale to the slice, then aim again: reset_camera also
    # clamps the clipping range tightly around the (flat) slice, which would
    # clip it away as soon as the plane is dragged; _aim_face_camera restores
    # a generous range.
    renderer.reset_camera()
    _aim()
    # Re-aim only on interaction: the panel keeps its parallel scale (user
    # zoom) and recovers the face-on view after any manual rotation.
    widget.AddObserver('InteractionEvent', _aim)
    return renderer


def _aim_face_camera(renderer, normal, focal_point, dist):
    position, up = _face_camera(normal, focal_point, dist)
    camera = renderer.camera
    camera.focal_point = tuple(np.asarray(focal_point, dtype=np.float64))
    camera.position = tuple(position)
    camera.up = tuple(up)
    # The slice sits at distance `dist` from the camera, but during a drag
    # the rendered polygon lags the widget by up to the volume size (the
    # data refreshes on the configured interaction event). A wide fixed
    # clipping range keeps it visible throughout.
    camera.clipping_range = (0.01 * dist, 10.0 * dist)


def slice_view(data, quantity='bmag', mode='ortho', normal='y', field=None,
               delta=0.01, method='linear', cmap=None, log=None, vmin=None,
               vmax=None, percentile=98.0, nan_opacity=0.0, outline=True,
               front_view=None, plotter=None, show=True, **mesh_kwargs):
    """
    Interactively slice a scalar quantity through a gridded volume.

    Opens a 3D view with drag-able slice-plane widgets; the camera rotates
    with the left mouse button, zooms with the wheel, and pans with
    shift+drag (PyVista defaults). By default (``front_view``) the window
    also shows each slice face-on in a separate panel beside the 3D view;
    the panels update live as the planes are dragged, and each has its own
    camera (wheel-zoom and pan work per panel).

    Parameters
    ----------
    data : GriddedField or pyvista.DataSet
        The volume. A `GriddedField` is converted with `to_rectilinear_grid`
        (evaluating ``quantity`` on its nodes); a PyVista dataset is used
        as-is when ``quantity`` names one of its point-data arrays, otherwise
        the quantity is evaluated on its points (needs ``field``).
    quantity : str, Quantity, or callable, optional
        The scalar to show. Default ``'bmag'``. Geometry quantities on large
        grids are expensive — coarsen with ``data.subvolume(stride=...)``
        first.
    mode : {'ortho', 'plane'}, optional
        ``'ortho'`` (default): three orthogonal slice planes, each drag-able
        along its axis. ``'plane'``: one free plane — drag the arrow to
        translate it, grab the plane to rotate it.
    normal : str or tuple, optional
        Initial plane normal for ``mode='plane'``. Default ``'y'``.
    field : callable, optional
        Field callable for quantity evaluation. Default: built from the
        `GriddedField` with ``method``.
    delta : float, optional
        Finite-difference step for geometry quantities.
    method : str, optional
        Interpolation method for the default field callable. Prefer 'cubic'
        for smooth derivative quantities.
    cmap, log, vmin, vmax, percentile
        Colour scale options (same conventions as `mageometry.viz`: log for
        positive quantities, symmetric diverging for signed ones).
    nan_opacity : float, optional
        Opacity of NaN (undefined) values. Default 0 — NaN stays blank.
    outline : bool, optional
        Draw the volume outline box. Default True.
    front_view : bool, optional
        Show each slice face-on in a companion panel: three stacked panels
        (x/y/z, oriented like the 2D plots of `mageometry.viz`) for
        ``mode='ortho'``, one panel whose camera follows the widget normal
        for ``mode='plane'``. Panels use a parallel (orthographic)
        projection and update live while dragging; note the slice *data*
        refreshes when the drag ends (pass ``interaction_event='always'``
        to reslice continuously). Default: enabled when `slice_view`
        creates the plotter itself, off when ``plotter`` is given (the
        multi-view layout must be owned by this function, so
        ``front_view=True`` together with ``plotter`` raises ValueError).
    plotter : pyvista.Plotter, optional
        Draw into an existing plotter instead of creating one.
    show : bool, optional
        Open the render window (blocking). Default True; pass False to
        compose further and call ``plotter.show()`` yourself.
    **mesh_kwargs
        Passed to the underlying ``add_mesh_slice*`` call.

    Returns
    -------
    pyvista.Plotter
        With ``front_view`` the plotter is multi-view; the main 3D subplot
        is left active, so further ``add_*`` calls draw into it.
    """
    require_pyvista()
    from ..io import GriddedField

    if isinstance(data, GriddedField):
        mesh = to_rectilinear_grid(data, quantities=(quantity,), field=field,
                                   delta=delta, method=method)
        q = resolve_quantity(quantity)
        name = _array_name(quantity, q)
    else:
        mesh = data
        if isinstance(quantity, str) and quantity in mesh.point_data:
            name = quantity
            if quantity in QUANTITIES:
                q = QUANTITIES[quantity]
            else:
                q = Quantity(None, quantity)
        else:
            if field is None:
                raise ValueError(
                    f"quantity {quantity!r} is not a point-data array of the "
                    "dataset; pass `field` to evaluate it on the points."
                )
            q = resolve_quantity(quantity)
            name = _array_name(quantity, q)
            pts = np.asarray(mesh.points, dtype=np.float64)
            mesh.point_data[name] = q.evaluate(field, pts[:, 0], pts[:, 1],
                                               pts[:, 2], delta=delta)

    if mode not in ('ortho', 'plane'):
        raise ValueError("mode must be 'ortho' or 'plane'.")
    if front_view is None:
        front_view = plotter is None
    elif front_view and plotter is not None:
        raise ValueError(
            "front_view requires slice_view to create its own multi-view "
            "plotter; pass plotter=None or front_view=False."
        )

    lo, hi, use_log = resolve_scale(np.asarray(mesh.point_data[name]), q,
                                    log=log, vmin=vmin, vmax=vmax,
                                    percentile=percentile)
    if front_view:
        plotter = get_plotter(None, shape='1|3' if mode == 'ortho' else '1|1')
        plotter.subplot(0)
    else:
        plotter = get_plotter(plotter)
    kw = dict(scalars=name, cmap=cmap or q.cmap, clim=(lo, hi),
              log_scale=use_log, nan_opacity=nan_opacity,
              scalar_bar_args={'title': q.label}, **mesh_kwargs)
    if mode == 'ortho':
        plotter.add_mesh_slice_orthogonal(mesh, **kw)
    else:
        plotter.add_mesh_slice(mesh, normal=normal, **kw)
    if outline:
        plotter.add_mesh(mesh.outline(), color='grey')
    plotter.show_axes()

    if front_view:
        # The meshes in `plane_sliced_meshes` are updated in place by the
        # widget callbacks, so adding them to another renderer gives a live
        # face-on view for free; only the free plane needs camera tracking.
        dist = 2.0 * mesh.length
        state = _widget_state(plotter)
        panel_kw = dict(scalars=name, cmap=cmap or q.cmap, clim=(lo, hi),
                        log_scale=use_log, nan_opacity=nan_opacity,
                        show_scalar_bar=False)
        if mode == 'ortho':
            for i, axis in enumerate('xyz'):
                _add_face_panel(plotter, 1 + i, state.plane_sliced_meshes[i],
                                f'{axis} slice', panel_kw,
                                state.plane_widgets[i], dist)
        else:
            _add_face_panel(plotter, 1, state.plane_sliced_meshes[-1],
                            'slice', panel_kw, state.plane_widgets[-1], dist)
            plotter.show_axes()   # orientation triad: the panel view rotates
        plotter.subplot(0)

    if show:
        plotter.show()
    return plotter
