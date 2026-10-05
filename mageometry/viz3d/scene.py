"""Render prepared session arrays without evaluating a magnetic field."""

import numpy as np

from ._pv import require_pyvista
from .fac import _peak_projection, _region_seeds, _valid_volume
from .slicer import _face_camera
from ._text_layout import _TextLayout


def display_key(result):
    return result['kind'] + ':' + result['component']


def _values_for_sign(values, value_sign):
    """Mask the unselected sign without changing prepared scientific arrays."""
    if value_sign == 'both':
        return values
    selected = values > 0 if value_sign == 'positive' else values < 0
    return np.where(selected & np.isfinite(values), values, np.nan)


def slice_axes(normal):
    """Return the horizontal/vertical unit vectors of the face-on slice."""
    direction, up = _face_camera(normal, np.zeros(3), 1.)
    return np.cross(up, direction), np.asarray(up)


def difference_step_label(result):
    geometry_step = result['resolved']['geometry_delta']
    component = result['component']
    if component in ('dalpha_ds', 'dalpha_ds_over_B', 'dfac_ds'):
        if result['analysis']['evaluation'] == 'grid':
            detail = '' if component == 'dfac_ds' else f'; alpha step {geometry_step:g}'
            return 'along-B gradient: preview axis spacing' + detail
        if component != 'dfac_ds':
            return f'alpha and along-B steps {geometry_step:g}'
        steps = result['resolved'].get('fac_delta', result['analysis']['delta'])
        steps = geometry_step if steps is None else steps
        return (f'along-B step {geometry_step:g}; FAC steps '
                + ', '.join(f'{step:g}' for step in np.atleast_1d(steps)))
    if result['component'] != 'fac':
        return f'geometry step {geometry_step:g}'
    if result['analysis']['evaluation'] == 'grid':
        return 'FAC steps: preview axis spacing'
    steps = result['resolved'].get('fac_delta', result['analysis']['delta'])
    if steps is None:
        steps = geometry_step
    return 'FAC steps ' + ', '.join(f'{step:g}' for step in np.atleast_1d(steps))


class GeometryScene:
    """A five-renderer scene embeddable in Qt or an off-screen Plotter.

    Parameters
    ----------
    plotter : pyvista.Plotter
        Plotter created with ``shape='1|4'``. Renderer 0 is 3D, 1 is the
        face-on slice, and 2 through 4 are signed peak projections.
    plane_changed : callable, optional
        Receives the normal and origin after dragging the 3D slice widget.
    """

    def __init__(self, plotter, plane_changed=None):
        self.p = plotter
        self.pv = require_pyvista()
        self.plane_changed = plane_changed
        self.result = None
        self.view = None
        self.widget = None
        self.last_normal = self.last_origin = None
        self.last_extent = None
        self.mesh = self.volume = None
        self.projections = []
        self.limit = 1.
        self.updating = False
        for index in range(5):
            self.p.subplot(index)
            self.p.set_background('#f5f7fa', all_renderers=False)
        self.text_layouts = [_TextLayout(self.p, renderer,
                            {'title': (.94, .22), 'validity': (.94, .06), 'empty': (.94, .06)})
                             for renderer in self.p.renderers]

    def camera_state(self):
        states = {}
        for index in range(5):
            camera = self.p.renderers[index].camera
            states[str(index)] = dict(position=list(camera.position),
                                      focal_point=list(camera.focal_point), up=list(camera.up),
                                      parallel_scale=float(camera.parallel_scale),
                                      parallel_projection=bool(camera.parallel_projection),
                                      clipping_range=list(camera.clipping_range))
        return states

    def restore_cameras(self, states):
        for index, state in states.items():
            camera = self.p.renderers[int(index)].camera
            for key, value in state.items():
                setattr(camera, key, value)

    def _add(self, data, name, **kwargs):
        if data.n_points:
            return self.p.add_mesh(data, name=name, reset_camera=False,
                                   render=False, show_scalar_bar=False, **kwargs)
        return None

    def clear(self):
        """Clear result actors and widgets while retaining scene illumination."""
        lights = [list(renderer.lights) for renderer in self.p.renderers]
        self.p.clear_plane_widgets()
        self.widget = None
        self.p.clear()
        # Plotter.clear() removes lights as well as actors. Preserve the
        # light kit (or caller-supplied lights); otherwise even a smooth
        # sphere is rendered as a flat disc after opening or updating data.
        for renderer, previous_lights in zip(self.p.renderers, lights):
            for light in previous_lights:
                renderer.add_light(light)
        self.result = self.view = None
        self.mesh = self.volume = None
        self.projections = []
        self.last_normal = self.last_origin = None
        self.last_extent = None

    def set_result(self, result, view, preserve_camera=True):
        # Build the potentially failing data objects before touching the scene.
        mesh = self.pv.RectilinearGrid(*result['axes'])
        mesh['value'] = result['values'].ravel(order='F')
        volume = _valid_volume(mesh, result['values'])
        projection_data = []
        for axis in range(3):
            axes = list(result['axes'])
            axes[axis] = np.array([mesh.center[axis]])
            projection_data.append((self.pv.RectilinearGrid(*axes),
                                    _peak_projection(result['values'], axis).ravel(order='F')))
        saved = self.camera_state() if preserve_camera and self.result is not None else view.get('cameras', {})
        self.updating = True
        try:
            self.clear()
            self.result, self.view = result, view
            self.mesh, self.volume, self.projections = mesh, volume, projection_data
            if view['origin'] is None:
                view['origin'] = list(mesh.center)
            self.set_layout(view['layout'], render=False)
            self.p.subplot(0)
            self._add(mesh.outline(), 'outline', color='#9caec0')
            radius = result['analysis']['planet_radius']
            if radius is not None:
                self._add(self.pv.Sphere(radius=radius), 'planet', color='#d7e0e8', smooth_shading=True)
            points, cells, offset = [], [], 0
            for path in result['paths']:
                if len(path) >= 2:
                    points.append(path)
                    cells.extend([len(path), *range(offset, offset + len(path))])
                    offset += len(path)
            if points:
                self._add(self.pv.PolyData(np.concatenate(points), lines=np.asarray(cells)),
                          'lines', color='#778999', line_width=1.4, opacity=.45, pickable=False)
            self.p.show_bounds(bounds=mesh.bounds, color='#596b7e', font_size=10,
                               xtitle='x', ytitle='y', ztitle='z', use_3d_text=False)
            for axis in range(3):
                self.p.subplot(axis + 2)
                panel, _ = self.projections[axis]
                self._add(panel.outline(), 'outline', color='#aab8c8')
                position, up = _face_camera(np.eye(3)[axis], panel.center, mesh.length * 2)
                self.p.camera_position = [position, panel.center, up]
                self.p.enable_parallel_projection()
                self.p.reset_camera()
            self.p.subplot(0)
            self.p.view_isometric()
            self.p.enable_parallel_projection()
            self.p.reset_camera()
            self.last_normal = self.last_origin = None
            self.update_display(render=False)
            renderer = self.p.renderers[0]
            existing_props = set(renderer.GetViewProps())
            self.widget = self.p.add_plane_widget(
                self._drag_plane, normal=view['normal'], origin=view['origin'], bounds=mesh.bounds,
                factor=1., color='#b77a18', outline_translation=False, outline_opacity=.08,
                test_callback=False,
                interaction_event='end')
            self.widget.GetPlaneProperty().SetOpacity(.08)
            for prop in set(renderer.GetViewProps()) - existing_props:
                prop.SetUseBounds(False)
            if saved:
                self.restore_cameras(saved)
            self.set_layout(view['layout'], render=False)
        finally:
            self.updating = False
        self.p.render()

    def _title(self, index, subtitle):
        r = self.result
        value_sign = self.view.get('value_sign', 'both')
        if value_sign != 'both':
            subtitle += f'; {value_sign} only'
        text = f"{r['case_label']}; {r['component']}; {r['contribution']}\n{subtitle}"
        if index in (0, 1):
            text += (f"\n{r['analysis']['evaluation']}; {difference_step_label(r)}"
                     if 'resolved' in r else '')
            if r['kind'] == 'attribution':
                text += '\nTotal-field frame'
        actor = self.p.add_text(text, name='title', font_size=8, color='#23344a',
                                position=(.03, .97), viewport=True, render=False)
        actor.GetTextProperty().SetVerticalJustificationToTop()

    def update_display(self, render=True):
        if self.result is None:
            return
        r, view, p = self.result, self.view, self.p
        key = display_key(r)
        self.limit = view['color_limits'].get(key, r['scale']['limit'])
        cutoff = max(view['thresholds'].get(key, r['scale']['threshold']), np.nextafter(0., 1.))
        interval = (view.get('value_intervals', {}).get(key)
                    if view.get('threshold_modes', {}).get(key) == 'interval' else None)
        p.subplot(0)
        value_sign = view.get('value_sign', 'both')
        values = _values_for_sign(r['values'], value_sign)
        flat = values.ravel(order='F')
        for sign, name, color in ((1, 'positive', '#c94343'), (-1, 'negative', '#2877ba')):
            p.remove_actor(name, reset_camera=False, render=False)
            if value_sign != 'both' and value_sign != name:
                continue
            if interval is None:
                low, high = (cutoff, np.inf) if sign > 0 else (-np.inf, -cutoff)
            else:
                low, high = interval
                low, high = (max(0., low), high) if sign > 0 else (low, min(0., high))
            finite = flat[np.isfinite(flat)]
            if (not self.volume.n_cells or low >= high or not finite.size
                    or not np.any(sign * finite > 0)
                    or finite.max() < low or finite.min() > high):
                continue
            region = self.volume
            if np.isfinite(low):
                region = region.clip_scalar(value=low, scalars='value', invert=False)
            if np.isfinite(high) and region.n_cells:
                region = region.clip_scalar(value=high, scalars='value', invert=True)
            if region.n_cells:
                self._add(region.extract_surface(), name, color=color,
                          opacity=.5, smooth_shading=True)
        p.remove_actor('arrows', reset_camera=False, render=False)
        basis = r['basis']
        if basis is not None:
            vectors = basis.reshape((-1, 3), order='F')
            eligible = np.where(np.all(np.isfinite(vectors), axis=-1), flat, np.nan)
            if interval is not None:
                eligible = np.where((eligible >= interval[0]) & (eligible <= interval[1]), eligible, np.nan)
            selected = _region_seeds(self.mesh.points, eligible, cutoff if interval is None else 0.,
                                     32, .07 * self.mesh.length)
            if len(selected):
                arrows = self.pv.PolyData(self.mesh.points[selected])
                arrows['direction'] = np.sign(flat[selected, None]) * vectors[selected]
                arrows['value'] = flat[selected]
                self._add(arrows.glyph(orient='direction', scale=False, factor=.035 * self.mesh.length),
                          'arrows', scalars='value', cmap='RdBu_r', clim=(-self.limit, self.limit))
        description = (f'threshold {cutoff:.4g}' if interval is None else
                       f'{interval[0]:.4g} <= value <= {interval[1]:.4g}')
        self._title(0, '3D overview; ' + description)
        count = int(np.count_nonzero(np.isfinite(r['values'])))
        context = 'grey: total-field lines' if r['analysis'].get('trace_enabled', True) else 'tracing disabled'
        p.add_text(f'{count:,} valid nodes; {context}', name='validity',
                   position=(.03, .15), viewport=True, font_size=8, color='#596b7e', render=False)
        for axis, (panel, projected) in enumerate(self.projections):
            p.subplot(axis + 2)
            if interval is not None:
                selected_values = np.where((values >= interval[0]) & (values <= interval[1]), values, np.nan)
                projected = _peak_projection(selected_values, axis).ravel(order='F')
            elif value_sign != 'both':
                projected = _peak_projection(values, axis).ravel(order='F')
            panel['value'] = projected if interval is not None else np.where(np.abs(projected) >= cutoff, projected, np.nan)
            self._scalar(panel, 'projection')
            self._title(axis + 2, f'{("YZ", "XZ", "XY")[axis]} signed peak along {"xyz"[axis]}')
        self.update_slice(render=False)
        self.update_visibility(render=False)
        p.subplot(0)
        if render:
            p.render()

    def update_visibility(self, render=True):
        """Show or hide existing 3D actors without rebuilding geometry."""
        if self.result is None:
            return
        actors = self.p.renderers[0].actors
        for name, key in (('positive', 'regions'), ('negative', 'regions'),
                          ('arrows', 'arrows'), ('lines', 'lines'), ('slice', 'plane')):
            if name in actors:
                actors[name].visibility = self.view[key]
        self._sync_plane_widget()
        if render:
            self.p.render()

    def update_colors(self, render=True):
        """Update mapper and legend ranges while retaining meshes and cameras."""
        if self.result is None:
            return
        key = display_key(self.result)
        self.limit = self.view['color_limits'].get(key, self.result['scale']['limit'])
        shared = (-self.limit, self.limit)
        slice_range = self.view.get('slice_color_ranges', {}).get(key)
        panels = (('arrows', 'slice'), ('slice',), ('projection',), ('projection',), ('projection',))
        for index, names in enumerate(panels):
            renderer = self.p.renderers[index]
            for name in names:
                actor = renderer.actors.get(name)
                if actor is None:
                    continue
                limits = slice_range if name == 'slice' and slice_range is not None else shared
                actor.mapper.scalar_range = limits
                actor.mapper.lookup_table.scalar_range = limits
            bar_name = self._bar_name(index)
            if bar_name in self.p.scalar_bars:
                manual_slice = index < 2 and slice_range is not None
                self.p.update_scalar_bar_range(slice_range if manual_slice else shared, name=bar_name)
                self.p.scalar_bars[bar_name].SetTitle(
                    self.result['label'] + (' / slice range' if manual_slice else ' / shared'))
        if render:
            self.p.render()

    def _bar_name(self, index):
        return f"{self.result['label']} / shared [{index}]"

    def _scalar(self, data, name, color_range=None):
        # Companion renderers use the same names. Plotter.remove_actor removes
        # matching actors in every renderer, including the shared 3D slice.
        self.p.renderer.remove_actor(name, reset_camera=False, render=False)
        bar_name = self._bar_name(self.p.renderers.active_index)
        if bar_name in self.p.scalar_bars:
            self.p.remove_scalar_bar(bar_name, render=False)
        actor = self._add(data, name, scalars='value', cmap='RdBu_r',
                          clim=(-self.limit, self.limit) if color_range is None else color_range,
                          nan_opacity=0, lighting=False)
        if actor is not None:
            # One independent bar per renderer; slice bounds can be overridden.
            bar = self.p.add_scalar_bar(title=bar_name, mapper=actor.mapper,
                                  color='#23344a', title_font_size=10, label_font_size=9,
                                  width=.8, height=.08, position_x=.1, position_y=.04,
                                  n_labels=3, render=False)
            bar.SetTitle(self.result['label'] + (' / shared' if color_range is None else ' / slice range'))
        return actor

    def update_slice(self, render=True):
        if self.result is None:
            return
        normal = np.asarray(self.view['normal'], dtype=float)
        normal /= np.linalg.norm(normal)
        origin = np.asarray(self.view['origin'], dtype=float)
        sliced = self.volume.slice(normal=normal, origin=origin) if self.volume.n_cells else self.pv.PolyData()
        extent = self.view.get('slice_extent')
        horizontal, vertical = slice_axes(normal)
        if extent is not None:
            for axis, low, high in ((horizontal, *extent[:2]), (vertical, *extent[2:])):
                if sliced.n_cells:
                    sliced = sliced.clip(normal=axis, origin=axis * low, invert=False)
                if sliced.n_cells:
                    sliced = sliced.clip(normal=axis, origin=axis * high, invert=True)
        if sliced.n_points:
            sliced['value'] = _values_for_sign(sliced['value'], self.view.get('value_sign', 'both'))
        color_range = self.view.get('slice_color_ranges', {}).get(display_key(self.result))
        self.p.subplot(0)
        actor = self._scalar(sliced, 'slice', color_range)
        if actor is not None:
            actor.visibility = self.view['plane']
        self.p.subplot(1)
        self._scalar(sliced, 'slice', color_range)
        self._title(1, f"Origin {np.round(origin, 3)}; all strengths")
        if not sliced.n_points or not np.any(np.isfinite(sliced['value'])):
            self.p.add_text('No matching data on this plane', name='empty', position=(.03, .15), viewport=True,
                            color='#596b7e', font_size=10, render=False)
        else:
            self.p.remove_actor('empty', reset_camera=False, render=False)
        camera = self.p.camera
        if (self.last_normal is None or not np.allclose(normal, self.last_normal)
                or (extent is None and self.last_extent is not None)):
            position, up = _face_camera(normal, origin, self.mesh.length * 2)
            self.p.camera_position = [position, origin, up]
            self.p.enable_parallel_projection()
            bounds = np.asarray(self.mesh.bounds).reshape(3, 2)
            corners = np.array(np.meshgrid(*bounds, indexing='ij')).reshape(3, -1).T
            camera.parallel_scale = float(np.ptp(corners @ np.asarray(up))) * .7
        elif self.last_origin is not None:
            shift = normal * np.dot(origin - self.last_origin, normal)
            camera.position = np.asarray(camera.position) + shift
            camera.focal_point = np.asarray(camera.focal_point) + shift
        if extent is not None and (self.last_normal is None or not np.allclose(normal, self.last_normal)
                                   or extent != self.last_extent):
            center = (normal * np.dot(origin, normal)
                      + horizontal * (extent[0] + extent[1]) / 2
                      + vertical * (extent[2] + extent[3]) / 2)
            position, up = _face_camera(normal, center, self.mesh.length * 2)
            self.p.camera_position = [position, center, up]
            self.p.enable_parallel_projection()
            viewport = self.p.renderer.GetViewport()
            width, height = self.p.window_size
            aspect = max(width * (viewport[2] - viewport[0]) / max(height * (viewport[3] - viewport[1]), 1), .01)
            camera.parallel_scale = 1.1 * max((extent[3] - extent[2]) / 2,
                                             (extent[1] - extent[0]) / (2 * aspect))
        camera.clipping_range = (.001 * self.mesh.length, 10 * self.mesh.length)
        self.last_normal, self.last_origin = normal.copy(), origin.copy()
        self.last_extent = list(extent) if extent is not None else None
        if self.widget is not None:
            self.widget.SetNormal(normal)
            self.widget.SetOrigin(origin)
        self.p.subplot(0)
        if render:
            self.p.render()

    def _drag_plane(self, normal, origin):
        if self.updating:
            return
        self.view['normal'], self.view['origin'] = list(normal), list(origin)
        self.update_slice()
        if self.plane_changed:
            self.plane_changed(normal, origin)

    def set_layout(self, layout, render=True):
        layouts = {
            'all': [(0, .25, .55, 1), (.55, .25, 1, 1),
                    (0, 0, 1/3, .25), (1/3, 0, 2/3, .25), (2/3, 0, 1, .25)],
            'three_d_slice': [(0, 0, .55, 1), (.55, 0, 1, 1), None, None, None],
            'three_d': [(0, 0, 1, 1), None, None, None, None],
            'slice': [None, (0, 0, 1, 1), None, None, None],
        }
        if layout not in layouts:
            raise ValueError('Unknown view layout.')
        viewports = layouts[layout]
        for index, renderer in enumerate(self.p.renderers):
            visible = viewports[index] is not None
            renderer.SetDraw(visible)
            renderer.SetInteractive(visible)
            # PyVista's mouse callbacks re-enable renderers based on their
            # viewport alone. Hidden full-window panels would then receive
            # drags intended for the visible camera. Collapse them outside
            # the drawable pixels, at the upper-right window boundary.
            viewport = viewports[index] if visible else (1, 1, 1, 1)
            renderer.SetViewport(*viewport)
        if self.view is not None:
            self.view['layout'] = layout
        self._sync_plane_widget()
        self.p.subplot(1 if layout == 'slice' else 0)
        if render:
            self.p.render()

    def _sync_plane_widget(self):
        if self.widget is not None and self.view is not None:
            enabled = self.view['layout'] != 'slice' and self.view['plane']
            if bool(self.widget.GetEnabled()) != enabled:
                if enabled:
                    # Disabling a VTK widget clears its renderer.
                    self.widget.SetCurrentRenderer(self.p.renderers[0])
                self.widget.SetEnabled(enabled)

    def reset_camera(self):
        if self.result is None:
            return
        index = 1 if self.view['layout'] == 'slice' else 0
        self.p.subplot(index)
        if index == 1:
            self.last_normal = None
            self.update_slice()
        else:
            self.p.view_isometric()
            self.p.reset_camera()
        self.p.render()

    def screenshot(self, path):
        self.p.render()
        self.p.screenshot(str(path))
