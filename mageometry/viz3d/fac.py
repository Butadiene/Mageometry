"""A spatial overview of field-aligned current, with signed regions and arrows."""

import numpy as np

from ..geometry import field_aligned_current_density
from ..io import GriddedField
from ..tracing import trace_field_lines
from ._pv import get_plotter, require_pyvista
from .mesh import to_rectilinear_grid, trace_polydata
from .slicer import _face_camera
from ._fac_slice import _FACSlice, _slice_settings

__all__ = ['fac_view']

_POSITIVE = '#c94343'
_NEGATIVE = '#2877ba'
_INK = '#263546'
_BACKGROUND = '#f5f7fa'


def _masked_field(field, mask):
    """Avoid evaluating a model inside an explicitly excluded region."""
    if mask is None:
        return field

    def evaluate(x, y, z):
        coords = np.broadcast_arrays(x, y, z)
        keep = ~np.broadcast_to(np.asarray(mask(*coords), dtype=bool), coords[0].shape)
        result = tuple(np.full(coords[0].shape, np.nan) for _ in range(3))
        if np.any(keep):
            components = field(*(c[keep] for c in coords))
            for out, component in zip(result, components):
                out[keep] = component
        return result

    return evaluate


def _preview_indices(shape, max_points):
    """Bound preview size while preserving domain endpoints on every axis."""
    counts = np.asarray(shape, dtype=int)
    if np.any(counts < 3):
        raise ValueError("FAC requires at least three grid nodes on every axis.")
    if max_points is not None:
        if not np.isscalar(max_points) or not np.isfinite(max_points) or max_points < 27:
            raise ValueError("max_points must be an integer >= 27 or None.")
        if int(max_points) != max_points:
            raise ValueError("max_points must be an integer >= 27 or None.")
        if np.prod(counts) > max_points:
            ratio = (max_points / np.prod(counts)) ** (1 / 3)
            counts = np.maximum(3, np.floor((counts - 1) * ratio).astype(int) + 1)
            while np.prod(counts) > max_points:
                axis = int(np.argmax(counts))
                counts[axis] -= 1
    return tuple(np.linspace(0, n - 1, k, dtype=int) for n, k in zip(shape, counts))


def _sample_fac(data, field=None, delta=None, max_points=120000, mask=None):
    """Sample a preview and compute FAC without constructing any VTK objects."""
    indices = _preview_indices(data.shape, max_points)
    axes = tuple(a[i] for a, i in zip((data.x, data.y, data.z), indices))
    b = data.b[np.ix_(*indices)].astype(float, copy=True)
    coords = np.meshgrid(*axes, indexing='ij')
    if mask is not None:
        b[np.broadcast_to(np.asarray(mask(*coords), dtype=bool), b.shape[:-1])] = np.nan

    if field is not None:
        if delta is None:
            delta = tuple(np.min(np.diff(a)) / 2 for a in axes)

        masked_field = _masked_field(field, mask)
        fac = field_aligned_current_density(masked_field, *coords, delta=delta)
        valid = np.all(np.isfinite(b), axis=-1)
        b = np.stack(np.broadcast_arrays(*masked_field(*coords), coords[0])[:3], axis=-1)
        b = np.where(valid[..., None], b, np.nan)
        fac = np.where(valid, fac, np.nan)
    else:
        if delta is not None:
            raise ValueError("delta requires field; grid FAC uses the preview's axis spacing.")
        # Nonuniform Cartesian central differences, with no extrapolated
        # boundary values. Require all six neighbouring samples to be valid.
        dx = np.gradient(b, axes[0], axis=0, edge_order=2)
        dy = np.gradient(b, axes[1], axis=1, edge_order=2)
        dz = np.gradient(b, axes[2], axis=2, edge_order=2)
        curl = np.stack((dy[..., 2] - dz[..., 1], dz[..., 0] - dx[..., 2],
                         dx[..., 1] - dy[..., 0]), axis=-1)
        valid = np.all(np.isfinite(b), axis=-1)
        stencil_valid = valid.copy()
        for axis in range(3):
            stencil_valid &= np.roll(valid, 1, axis) & np.roll(valid, -1, axis)
            boundary = [slice(None)] * 3
            boundary[axis] = [0, -1]
            stencil_valid[tuple(boundary)] = False
        magnitude = np.linalg.norm(b, axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            fac = np.sum(curl * (b / magnitude[..., None]), axis=-1)
        fac = np.where(stencil_valid & (magnitude > 0), fac, np.nan)

    preview = GriddedField(*axes, *np.moveaxis(b, -1, 0))
    return preview, fac


def _peak_projection(values, axis):
    """Signed value of the largest absolute sample along each sightline."""
    valid = np.isfinite(values)
    index = np.argmax(np.where(valid, np.abs(values), -np.inf), axis=axis)
    peak = np.take_along_axis(values, np.expand_dims(index, axis), axis=axis)
    return np.where(np.any(valid, axis=axis), np.squeeze(peak, axis=axis), np.nan)


def _region_seeds(points, values, threshold, count, separation):
    """Deterministic, spatially separated peaks, balanced across both signs."""
    if count == 0:
        return np.empty(0, dtype=int)
    candidates = []
    for sign in (1, -1):
        eligible = np.flatnonzero(np.isfinite(values) & (sign * values >= threshold)
                                 & (sign * values > 0))
        ranked = eligible[np.argsort(-np.abs(values[eligible]), kind='stable')]
        candidates.append(ranked)
    selected = []
    # Alternate signs so a weaker polarity is not lost among the strongest
    # samples of the other polarity. Refill from the remaining polarity.
    offsets = [0, 0]
    while len(selected) < count:
        added = False
        for sign in range(2):
            while offsets[sign] < len(candidates[sign]):
                index = candidates[sign][offsets[sign]]
                offsets[sign] += 1
                if not selected or np.all(np.linalg.norm(
                        points[selected] - points[index], axis=1) >= separation):
                    selected.append(index)
                    added = True
                    break
            if len(selected) == count:
                break
        if not added:
            break
    return np.asarray(selected, dtype=int)


def _valid_volume(mesh, values):
    """Remove whole cells touching undefined nodes before clipping regions."""
    finite = np.isfinite(values)
    cells = np.ones(tuple(n - 1 for n in values.shape), dtype=bool)
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                cells &= finite[i:i + cells.shape[0], j:j + cells.shape[1],
                                k:k + cells.shape[2]]
    return mesh.extract_cells(cells.ravel(order='F'))


def fac_view(gridded_field, field=None, delta=None, threshold=None,
             percentile=90.0, max_points=120000, mask=None, seeds=None,
             n_lines=10, trace_kwargs=None, current_scale=1.0,
             current_label='mu0 J parallel [field unit / length unit]',
             length_unit='grid unit', planet_radius=None,
             planet_center=(0.0, 0.0, 0.0), front_view=None,
             plotter=None, show=True, slice_normal=None, slice_origin=None,
             slice_only=False):
    """Locate field-aligned currents in a gridded magnetic field.

    Red regions carry positive current along B, blue regions negative
    current against B. Arrows show the direction of J_parallel; thin grey
    lines provide magnetic connectivity. A slider selects the absolute FAC
    threshold. Three companion maps show the signed peak of |FAC| along
    x/y/z, exposing structures hidden behind others in the 3D view.

    Parameters
    ----------
    gridded_field : GriddedField
        Grid and magnetic field, in any consistent coordinate system.
    field : callable, optional
        Analytic field for direct Cartesian finite differences and tracing.
        Default: compute curl directly on the preview grid with nonuniform
        central differences; boundary nodes and invalid stencils are NaN.
    delta : float or (3,) array_like, optional
        Step for an explicit field only. Default: half the preview spacing
        on each axis. Pass a resolved step for model-field diagnostics.
    threshold : float, optional
        Absolute displayed FAC cutoff after current_scale. Default: the
        percentile of finite |FAC|, a visibility setting, not significance.
    percentile : float, optional
        Default threshold percentile (90). Colour limits use the 98th
        percentile of |FAC| and remain fixed while adjusting the threshold.
    max_points : int or None, optional
        Maximum preview nodes (120000). Coarsening preserves the bounds;
        it changes numerical resolution. Use None to retain all nodes.
    mask : callable, optional
        ``mask(x, y, z)`` is True where samples should be excluded.
    seeds : (n, 3) array_like, optional
        Magnetic-line seeds. Default: up to n_lines separated FAC peaks.
        These context lines stay fixed when the threshold slider moves.
    n_lines : int, optional
        Number of automatic context lines; 0 disables them. Default 10.
    trace_kwargs : dict, optional
        Forwarded to trace_field_lines. Defaults: both directions, preview
        bounds, half the smallest spacing, and 600 steps per direction.
    current_scale : float, optional
        Positive multiplier for displayed FAC (1). For B in nT and positions
        in Re, use approximately 0.125 with current_label='J parallel [nA/m^2]'.
    current_label, length_unit : str, optional
        Display labels; native units by default. No units are inferred.
    planet_radius : float, optional
        Draw a reference sphere in coordinate units. Does not mask data.
    planet_center : (3,) array_like, optional
        Reference sphere centre, default (0, 0, 0).
    front_view : bool, optional
        Show peak maps (default when creating a plotter). They are
        projections, not slices or integrals; overlapping weaker structures
        of the opposite sign may be hidden. Cannot be True with plotter.
    plotter : pyvista.Plotter, optional
        Existing plotter for a single 3D view.
    show : bool, optional
        Open the desktop window; False returns an unshown plotter.
    slice_normal : {'x', 'y', 'z'}, (3,) array_like, or None, optional
        Initial normal of a draggable FAC cross-section in the main 3D
        view. None (default) starts with the slice hidden; press c to enable
        it. The slice uses the same colour scale as the peak maps and shows
        all FAC strengths, independently of the region threshold. Values
        are interpolated within valid preview cells, never across NaN cells.
    slice_origin : (3,) array_like, optional
        Initial point on the slice, in field coordinates. Default: grid
        centre. Drag the amber widget to move or rotate the plane.
    slice_only : bool, optional
        Start in a clean, face-on cross-section view (default False). F4
        toggles this mode and restores the overview camera and visibility.
        A dedicated position slider scans the cached slices; shift+drag
        pans and the wheel zooms. Defaults to an XZ slice if none is given.

    Returns
    -------
    pyvista.Plotter
        Main 3D subplot is active. Keys: x/y/z for axis views, r to reset,
        l to toggle magnetic lines, a to toggle current arrows, s for regions.
        c toggles the slice; F1/F2/F3 align it with the YZ/XZ/XY planes.
        F4 switches between the overview and the isolated, face-on slice.
    """
    pv = require_pyvista()
    owned_plotter = plotter is None
    slice_controller = None
    slice_normal, slice_origin = _slice_settings(slice_normal, slice_origin)
    if not np.isfinite(current_scale) or current_scale <= 0:
        raise ValueError("current_scale must be positive and finite.")
    if not np.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100.")
    if threshold is not None and (not np.isfinite(threshold) or threshold < 0):
        raise ValueError("threshold must be nonnegative and finite.")
    if not np.isfinite(n_lines) or n_lines < 0 or int(n_lines) != n_lines:
        raise ValueError("n_lines must be a nonnegative integer.")
    n_lines = int(n_lines)
    if planet_radius is not None and (not np.isfinite(planet_radius) or planet_radius <= 0):
        raise ValueError("planet_radius must be positive and finite.")
    if front_view is None:
        front_view = plotter is None
    if front_view and plotter is not None:
        raise ValueError("front_view requires fac_view to create its own plotter.")
    if seeds is not None:
        seeds = np.asarray(seeds, dtype=float)
        if seeds.ndim != 2 or seeds.shape[1] != 3 or not np.all(np.isfinite(seeds)):
            raise ValueError("seeds must be finite with shape (n, 3).")

    preview, values = _sample_fac(gridded_field, field, delta, max_points, mask)
    values *= current_scale
    mesh = to_rectilinear_grid(preview, quantities=())
    mesh.point_data['fac'] = values.ravel(order='F')
    volume = _valid_volume(mesh, values)
    magnitude = np.abs(values[np.isfinite(values)])
    peak = float(magnitude.max()) if magnitude.size else 0.0
    limit = float(np.percentile(magnitude, 98)) if magnitude.size else 1.0
    limit = limit if limit > 0 else peak or 1.0
    if threshold is None:
        threshold = float(np.percentile(magnitude, percentile)) if magnitude.size else 0.0
    # Zero means retain every nonzero current, not the current-free volume.
    floor = np.nextafter(0.0, 1.0)
    threshold = max(threshold, floor)

    if front_view:
        plotter = get_plotter(shape='1|3', splitting_position=0.7,
                              window_size=(1440, 960), border=False)
        plotter.subplot(0)
    else:
        plotter = get_plotter(plotter, window_size=(1200, 900))
    main_renderer = plotter.renderer
    main_location = np.atleast_1d(plotter.renderers.index_to_loc(plotter.renderers.active_index))

    def activate_main():
        plotter.subplot(*main_location)

    plotter.set_background(_BACKGROUND, all_renderers=False)
    plotter.add_text('FIELD-ALIGNED CURRENT', position=(0.035, 0.94),
                     viewport=True, font_size=19, color=_INK, name='fac-title')
    plotter.add_text('+ along B', position=(0.035, 0.885), viewport=True,
                     font_size=12, color=_POSITIVE)
    plotter.add_text('- against B', position=(0.24, 0.885), viewport=True,
                     font_size=12, color=_NEGATIVE)
    plotter.add_text('Grey: magnetic field lines', position=(0.47, 0.885),
                     viewport=True, font_size=10, color='#64748b')
    plotter.add_mesh(mesh.outline(), color='#c0cbd6', line_width=1,
                     name='fac-outline', pickable=False)
    plotter.show_bounds(bounds=mesh.bounds, color='#64748b', grid=False,
                        xtitle=f'x [{length_unit}]', ytitle=f'y [{length_unit}]',
                        ztitle=f'z [{length_unit}]', font_size=13, location='outer',
                        use_2d=True, use_3d_text=False, bold=False,
                        n_xlabels=3, n_ylabels=3, n_zlabels=3)
    plotter.add_axes(color=_INK, viewport=(0.78, 0.03, 0.96, 0.2))
    if planet_radius is not None:
        plotter.add_mesh(pv.Sphere(radius=planet_radius, center=planet_center),
                         color='#d7e0e8', smooth_shading=True, name='fac-planet')

    # Peak maps retain a common zero-centred colour scale. Threshold masking
    # affects only their displayed arrays, never the underlying FAC values.
    panels = []
    for axis in range(3) if front_view else ():
        plotter.subplot(axis + 1)
        plotter.set_background('#ffffff', all_renderers=False)
        axes = [preview.x, preview.y, preview.z]
        axes[axis] = np.array([mesh.center[axis]])
        panel = pv.RectilinearGrid(*axes)
        projected = _peak_projection(values, axis).ravel(order='F')
        panel.point_data['fac'] = projected.copy()
        actor = plotter.add_mesh(panel, scalars='fac', cmap='RdBu_r',
                                 clim=(-limit, limit), nan_opacity=0,
                                 lighting=False, show_scalar_bar=False,
                                 name='fac-projection')
        if axis == 2:
            plotter.add_scalar_bar(title=current_label, mapper=actor.mapper,
                                   color=_INK, title_font_size=10, label_font_size=9,
                                   vertical=False, width=0.85, height=0.1,
                                   position_x=0.08, position_y=0.02, fmt='%.2g')
        plane = ('YZ', 'XZ', 'XY')[axis]
        plotter.add_text(f'{plane} / peak along {"xyz"[axis]}', font_size=11,
                         color=_INK, position='upper_left')
        horizontal, vertical = ((1, 2), (0, 2), (0, 1))[axis]
        h, v = 'xyz'[horizontal], 'xyz'[vertical]
        plotter.add_text(f'horizontal: {h}  /  vertical: {v}',
                         position=(0.025, 0.83), viewport=True,
                         font_size=9, color='#64748b')
        h0, h1 = preview.bounds[horizontal]
        v0, v1 = preview.bounds[vertical]
        plotter.add_text(f'{h}: {h0:g} to {h1:g}  /  {v}: {v0:g} to {v1:g} [{length_unit}]',
                         position=(0.045, 0.145), viewport=True,
                         font_size=9, color='#64748b')
        plotter.add_mesh(panel.outline(), color='#d7e0e8', line_width=1, pickable=False)
        direction = np.eye(3)[axis]
        position, up = _face_camera(direction, panel.center, 2 * mesh.length)
        plotter.camera_position = [position, panel.center, up]
        plotter.enable_parallel_projection()
        plotter.reset_camera()
        plotter.camera.parallel_scale *= 1.4
        panels.append((panel, projected))
    if front_view:
        plotter.subplot(0)

    flat = values.ravel(order='F')
    spacing = min(float(np.min(np.diff(a))) for a in (preview.x, preview.y, preview.z))
    if seeds is None:
        selected = _region_seeds(mesh.points, flat, threshold, n_lines, 0.12 * mesh.length)
        seeds = mesh.points[selected]
    if len(seeds):
        line_field = preview.field() if field is None else field

        trace_field = _masked_field(line_field, mask)
        tk = dict(direction='both', ds=spacing / 2, bounds=preview.bounds, max_steps=600)
        tk.update(trace_kwargs or {})
        trace = trace_field_lines(trace_field, *np.asarray(seeds).T, **tk)
        lines = trace_polydata(trace)
        if lines.n_points:
            plotter.add_mesh(lines, color='#778999', line_width=1.4,
                             opacity=0.45, name='fac-lines', pickable=False)

    visibility = {'arrows': True, 'regions': True}

    def update(cutoff):
        cutoff = max(float(cutoff), floor)
        # Operate on the owned renderer even if the cursor is over a panel.
        activate_main()
        for sign, name, color in ((1, 'positive', _POSITIVE), (-1, 'negative', _NEGATIVE)):
            actor_name = f'fac-{name}'
            plotter.remove_actor(actor_name, reset_camera=False, render=False)
            if volume.n_cells and np.any(sign * flat >= cutoff):
                region = volume.clip_scalar(value=sign * cutoff, scalars='fac', invert=sign < 0)
                if region.n_cells:
                    actor = plotter.add_mesh(region.extract_surface(), color=color,
                                     opacity=0.5, smooth_shading=True, name=actor_name,
                                     reset_camera=False, render=False, show_scalar_bar=False)
                    actor.visibility = visibility['regions']
        plotter.remove_actor('fac-arrows', reset_camera=False, render=False)
        selected = _region_seeds(mesh.points, flat, cutoff, 32, 0.07 * mesh.length)
        if selected.size:
            arrows = pv.PolyData(mesh.points[selected])
            b = np.asarray(mesh.point_data['B'])[selected]
            arrows['direction'] = np.sign(flat[selected, None]) * b / np.linalg.norm(b, axis=1)[:, None]
            arrows['fac'] = flat[selected]
            glyphs = arrows.glyph(orient='direction', scale=False, factor=0.035 * mesh.length)
            actor = plotter.add_mesh(glyphs, scalars='fac', clim=(-limit, limit),
                                     cmap='RdBu_r', name='fac-arrows', show_scalar_bar=False,
                                     reset_camera=False, render=False)
            actor.visibility = visibility['arrows']
        for panel, projected in panels:
            panel.point_data['fac'] = np.where(np.abs(projected) >= cutoff, projected, np.nan)
        count = np.count_nonzero(np.isfinite(flat) & (np.abs(flat) >= cutoff))
        if not magnitude.size:
            status = 'FAC unavailable: no valid derivative stencils'
        elif count == 0:
            status = 'No FAC above threshold'
        else:
            status = f'{count:,} / {magnitude.size:,} valid nodes above threshold'
        plotter.add_text(status, position=(0.035, 0.22), viewport=True,
                         font_size=11, color=_INK, name='fac-status', render=False)
        plotter.add_text(f'|FAC| >= {cutoff:.3g}  /  {current_label}',
                         position=(0.035, 0.17), viewport=True, font_size=10,
                         color=_INK, name='fac-cutoff', render=False)
        if slice_controller is not None:
            slice_controller.focus.hide_overview()

    update(threshold)
    slider_max = max(peak * 1.01, threshold * 1.1) if peak or threshold > floor else 1.0
    threshold_widget = plotter.add_slider_widget(update, rng=(0, slider_max),
                              value=threshold, title='FAC strength threshold',
                              pointa=(0.06, 0.09), pointb=(0.64, 0.09), color=_INK,
                              title_height=0.017, fmt='%.2g', interaction_event='end')
    preview_shape = ' x '.join(map(str, preview.shape))
    plotter.add_text(f'Preview {preview_shape}  |  x/y/z: view  r: reset  l: lines  a: arrows  s: regions',
                     position=(0.035, 0.015), viewport=True, font_size=9, color='#64748b')
    main_renderer.view_isometric()
    main_renderer.enable_parallel_projection()
    main_renderer.reset_camera()
    main_renderer.camera.zoom(0.62)
    main_renderer.camera.SetWindowCenter(0, -0.1)
    initial_camera = main_renderer.camera_position

    def toggle(name):
        if slice_controller is not None and slice_controller.focus.active:
            return
        if name == 'regions':
            visibility[name] = not visibility[name]
            for polarity in ('positive', 'negative'):
                actor = main_renderer.actors.get(f'fac-{polarity}')
                if actor is not None:
                    actor.visibility = visibility[name]
            plotter.render()
            return
        actor = main_renderer.actors.get(f'fac-{name}')
        if name == 'arrows':
            visibility[name] = not visibility[name]
        if actor is not None:
            actor.visibility = not actor.visibility
        plotter.render()

    def reset():
        if slice_controller is not None and slice_controller.focus.active:
            slice_controller.focus.fit()
            plotter.render()
            return
        main_renderer.camera_position = initial_camera
        main_renderer.reset_camera()
        main_renderer.camera.zoom(0.62)
        plotter.render()

    def change_view(view):
        if slice_controller is None or not slice_controller.focus.active:
            view()

    for key, view in (('x', main_renderer.view_yz), ('y', main_renderer.view_xz),
                       ('z', main_renderer.view_xy), ('r', reset)):
        plotter.add_key_event(key, view if key == 'r' else lambda view=view: change_view(view))
    plotter.add_key_event('l', lambda: toggle('lines'))
    plotter.add_key_event('a', lambda: toggle('arrows'))
    plotter.add_key_event('s', lambda: toggle('regions'))
    slice_controller = _FACSlice(plotter, volume, mesh.bounds, limit, current_label, length_unit,
              slice_normal, slice_origin, scalar_bar=not front_view,
              activate_main=activate_main,
              panels=tuple(plotter.renderers)[1:] if front_view else (),
              overview_widgets=(threshold_widget, main_renderer.axes_widget),
              expand=owned_plotter)
    if slice_only:
        slice_controller.focus.enter()
    if show:
        plotter.show()
    return plotter
