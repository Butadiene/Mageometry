"""Render prepared session arrays without evaluating a magnetic field."""

import weakref

import numpy as np

from ._pv import require_pyvista
from .fac import _peak_projection, _region_seeds, _valid_volume
from .slicer import _face_camera
from ._text_layout import _TextLayout
from ._current import _component_label, _component_color_range, _component_cmap


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


def _upright_camera(camera):
    """Adopt world Z for orbiting, including an exact top/bottom view."""
    if tuple(camera.up) == (0., 0., 1.):
        return
    offset = np.subtract(camera.position, camera.focal_point)
    distance = np.linalg.norm(offset)
    pole_margin = np.deg2rad(1.)
    if distance and np.linalg.norm(offset[:2]) < distance * np.sin(pole_margin):
        # Z cannot be both the viewing direction and view-up. Leave an axis
        # view by one degree, preserving its screen-up direction at the pole.
        horizontal = np.asarray(camera.up)[:2]
        length = np.linalg.norm(horizontal)
        horizontal = horizontal / length if length else np.array([0., 1.])
        sign = np.copysign(1., offset[2])
        offset[:2] = -sign * distance * np.sin(pole_margin) * horizontal
        offset[2] = sign * distance * np.cos(pole_margin)
        camera.position = np.asarray(camera.focal_point) + offset
    camera.up = (0., 0., 1.)


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
        self.bar_names = {}
        self.updating = False
        self.profile_pick = None
        self.profile_cursor = None
        for index in range(5):
            self.p.subplot(index)
            self.p.set_background('#f5f7fa', all_renderers=False)
        self.text_layouts = [_TextLayout(self.p, renderer,
                            {'title': (.94, .22), 'validity': (.94, .06), 'empty': (.94, .06)})
                             for renderer in self.p.renderers]
        self._configure_navigation()

    def _configure_navigation(self):
        from vtkmodules.vtkRenderingCore import VTKIS_ROTATE

        # Terrain keeps view-up fixed and limits elevation before the poles.
        # Handle Shift ourselves so left-button panning also releases VTK's
        # captured focus. PyVista's wheel helper assumes a rectangular subplot
        # grid, so select the renderer directly for our split layout.
        self.p.iren.enable_terrain_style(mouse_wheel_zooms=False, shift_pans=False)
        owner = weakref.ref(self)

        def left_press(style, event):
            scene = owner()
            if scene is not None:
                interactor = style.GetInteractor()
                position = interactor.GetEventPosition()
                if interactor.GetShiftKey() or any(
                        renderer.GetDraw() and renderer.IsInViewport(*position)
                        for renderer in scene.p.renderers[1:]):
                    style.StartPan()

        def mouse_move(style, event):
            scene = owner()
            if (scene is not None and style.GetState() == VTKIS_ROTATE
                    and style.GetCurrentRenderer() is scene.p.renderers[0]):
                _upright_camera(scene.p.renderers[0].camera)
            style.OnMouseMove()

        def left_release(style, event):
            # Terrain normally ends PAN only on middle-button release.
            style.OnMiddleButtonUp()

        def wheel_zoom(style, event):
            interactor = style.GetInteractor()
            style.FindPokedRenderer(*interactor.GetEventPosition())
            renderer = style.GetCurrentRenderer()
            if renderer is None:
                return
            camera = renderer.GetActiveCamera()
            # Match the existing trackball zoom sensitivity in both projections.
            factor = 1.1**(2 if event == 'MouseWheelForwardEvent' else -2)
            if camera.GetParallelProjection():
                camera.SetParallelScale(camera.GetParallelScale() / factor)
            else:
                camera.Dolly(factor)
            renderer.ResetCameraClippingRange()
            if interactor.GetLightFollowCamera():
                renderer.UpdateLightsGeometryToFollowCamera()
            interactor.Render()

        def right_press(style, event):
            scene = owner()
            if scene is not None and scene.profile_pick is not None:
                scene.pick_profile_line(*style.GetInteractor().GetEventPosition())
            else:
                style.OnRightButtonDown()

        # PyVista retains responsibility for capturing the starting renderer
        # and releasing it, including drags that cross panel boundaries.
        style = self.p.iren.style
        style.AddObserver('LeftButtonPressEvent', left_press, 1.)
        style.AddObserver('LeftButtonReleaseEvent', left_release, 1.)
        style.AddObserver('MouseMoveEvent', mouse_move)
        style.AddObserver('MouseWheelForwardEvent', wheel_zoom)
        style.AddObserver('MouseWheelBackwardEvent', wheel_zoom)
        style.AddObserver('RightButtonPressEvent', right_press)
        style.AddObserver('RightButtonReleaseEvent', lambda style, event: style.OnRightButtonUp())

    @property
    def eta_colors(self):
        return self.result['component'] == 'gamma' and self.view.get('gamma_eta', False)

    @property
    def color_component(self):
        return 'eta' if self.eta_colors else self.result['component']

    @property
    def color_range(self):
        return _component_color_range(self.color_component, self.limit)

    @property
    def color_key(self):
        return self.result['kind'] + ':eta' if self.eta_colors else display_key(self.result)

    @property
    def color_label(self):
        return _component_label('eta', None, '') if self.eta_colors else self.result['label']

    @property
    def default_color_limit(self):
        return 1. if self.eta_colors else self.result['scale']['limit']

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
        self.profile_cursor = None
        self.bar_names.clear()
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
        if result.get('eta_values') is not None:
            mesh['eta'] = result['eta_values'].ravel(order='F')
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
            self.update_traces(result['paths'], result.get('trace_status', 'ready'), render=False)
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

    def update_traces(self, paths, status='ready', render=True):
        """Replace context lines without rebuilding the field or moving cameras."""
        if self.result is None:
            return
        points, cells, line_ids, offset = [], [], [], 0
        for line_id, path in enumerate(paths):
            if len(path) >= 2:
                points.append(path)
                cells.extend([len(path), *range(offset, offset + len(path))])
                line_ids.append(line_id)
                offset += len(path)
        mesh = self.pv.PolyData(np.concatenate(points), lines=np.asarray(cells)) if points else None
        if mesh is not None:
            mesh.cell_data['line_id'] = np.asarray(line_ids)
            mesh.cell_data.active_scalars_name = None
        active = self.p.renderers.active_index
        try:
            self.p.subplot(0)
            self.p.renderer.remove_actor('lines', reset_camera=False, render=False)
            if mesh is not None:
                actor = self._add(mesh, 'lines', color='#778999', line_width=1.4,
                                  opacity=.45, pickable=self.profile_pick is not None)
                actor.visibility = self.view['lines']
            self.result['paths'] = paths
            self.set_trace_status(status, render=False)
        finally:
            self.p.subplot(active)
        if render:
            self.p.render()

    def set_profile_picking(self, callback=None):
        """Enable explicit right-click selection of context lines only."""
        self.profile_pick = callback
        actor = self.p.renderers[0].actors.get('lines')
        if actor is not None:
            actor.SetPickable(callback is not None)
        # The plane widget otherwise captures right-clicks before the style.
        # Keep its slice visible while temporarily suspending its handle.
        self._sync_plane_widget()
        self.p.render()

    def pick_profile_line(self, x, y):
        from vtkmodules.vtkRenderingCore import vtkCellPicker

        renderer = self.p.renderers[0]
        actor = renderer.actors.get('lines')
        if (self.profile_pick is None or actor is None or not actor.visibility
                or not renderer.GetDraw() or not renderer.IsInViewport(x, y)):
            return
        picker = vtkCellPicker()
        picker.SetTolerance(.004)
        picker.PickFromListOn()
        picker.AddPickList(actor)
        if picker.Pick(x, y, 0, renderer) and picker.GetCellId() >= 0:
            line_id = int(actor.mapper.dataset.cell_data['line_id'][picker.GetCellId()])
            self.profile_pick(line_id)

    def set_profile_line(self, record=None):
        """Highlight an analysis line without changing cameras or scene filters."""
        renderer = self.p.renderers[0]
        for name in ('profile-line', 'profile-seed', 'profile-cursor'):
            renderer.remove_actor(name, reset_camera=False, render=False)
        self.profile_cursor = None
        if record is not None:
            active = self.p.renderers.active_index
            try:
                self.p.subplot(0)
                if len(record['points']) > 1:
                    mesh = self.pv.lines_from_points(record['points'])
                    self._add(mesh, 'profile-line', color='#087f8c', line_width=4,
                              pickable=False).SetUseBounds(False)
                seed = self.pv.PolyData(np.array([record['seed']]))
                self._add(seed, 'profile-seed', color='#087f8c', point_size=12,
                          render_points_as_spheres=True, pickable=False).SetUseBounds(False)
                self.profile_cursor = self.pv.PolyData(np.array([record['seed']]))
                self._add(self.profile_cursor, 'profile-cursor', color='#ed9c28', point_size=10,
                          render_points_as_spheres=True, pickable=False).SetUseBounds(False)
            finally:
                self.p.subplot(active)
        self.p.render()

    def set_profile_cursor(self, point):
        if self.profile_cursor is not None:
            self.profile_cursor.points = np.asarray(point).reshape(1, 3)
            self.p.render()

    def set_trace_status(self, status, render=True):
        """Keep unfinished tracing visible in the scene, including PNG exports."""
        if self.result is None:
            return
        self.result['trace_status'] = status
        active = self.p.renderers.active_index
        try:
            self.p.subplot(0)
            messages = {'pending': 'field lines computing', 'cancelled': 'field lines cancelled',
                        'failed': 'field lines failed', 'disabled': 'tracing disabled'}
            context = messages.get(status, 'grey: total-field lines' if
                                   self.result['analysis'].get('trace_enabled', True) else 'tracing disabled')
            count = int(np.count_nonzero(np.isfinite(self.result['values'])))
            text = f'{count:,} valid nodes; {context}'
            actor = self.p.renderer.actors.get('validity')
            if actor is None:
                self.p.add_text(text, name='validity', position=(.03, .15), viewport=True,
                                font_size=8, color='#596b7e', render=False)
            else:
                actor.SetInput(text)
                self.text_layouts[0].state = None
        finally:
            self.p.subplot(active)
        if render:
            self.p.render()

    def _title(self, index, subtitle):
        r = self.result
        value_sign = self.view.get('value_sign', 'both')
        if value_sign != 'both':
            subtitle += f'; {value_sign} only'
        diagnostic = 'Gamma selection; eta colours' if self.eta_colors else r['component']
        text = f"{r['case_label']}; {diagnostic}; {r['contribution']}\n{subtitle}"
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
        self.limit = view['color_limits'].get(self.color_key, self.default_color_limit)
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
                colors = (dict(scalars='eta', cmap='RdBu_r', clim=(-self.limit, self.limit),
                               nan_opacity=0) if self.eta_colors else dict(color=color))
                self._add(region.extract_surface(), name, **colors,
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
                          'arrows', scalars='value', cmap=_component_cmap(self.color_component),
                          clim=self.color_range)
        description = (f'threshold {cutoff:.4g}' if interval is None else
                       f'{interval[0]:.4g} <= value <= {interval[1]:.4g}')
        if self.eta_colors:
            description += f" ({r['label']})"
        self._title(0, '3D overview; ' + description)
        self.set_trace_status(r.get('trace_status', 'ready'), render=False)
        for axis, (panel, projected) in enumerate(self.projections):
            p.subplot(axis + 2)
            if self.eta_colors:
                selected = (np.abs(values) >= cutoff if interval is None else
                            (values >= interval[0]) & (values <= interval[1]))
                valid = np.isfinite(values) & selected
                indices = np.argmax(np.where(valid, np.abs(values), -np.inf), axis=axis)
                colors = np.take_along_axis(r['eta_values'], np.expand_dims(indices, axis), axis=axis)
                # Colour the Gamma peak at its own location. Undefined eta
                # stays blank; never replace it with a different sightline sample.
                panel['value'] = np.where(np.any(valid, axis=axis), colors.squeeze(axis), np.nan).ravel(order='F')
            elif interval is not None:
                selected_values = np.where((values >= interval[0]) & (values <= interval[1]), values, np.nan)
                projected = _peak_projection(selected_values, axis).ravel(order='F')
            elif value_sign != 'both':
                projected = _peak_projection(values, axis).ravel(order='F')
            if not self.eta_colors:
                panel['value'] = projected if interval is not None else np.where(np.abs(projected) >= cutoff, projected, np.nan)
            self._scalar(panel, 'projection')
            peak = 'eta at Gamma peak' if self.eta_colors else 'signed peak'
            self._title(axis + 2, f'{("YZ", "XZ", "XY")[axis]} {peak} along {"xyz"[axis]}')
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
        key = self.color_key
        self.limit = self.view['color_limits'].get(key, self.default_color_limit)
        shared = self.color_range
        slice_range = self.view.get('slice_color_ranges', {}).get(key)
        main = ('positive', 'negative', 'slice') if self.eta_colors else ('arrows', 'slice')
        panels = (main, ('slice',), ('projection',), ('projection',), ('projection',))
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
                regions = index == 0 and self._region_bar_actor() is not None
                manual_slice = index < 2 and not regions and slice_range is not None
                self.p.update_scalar_bar_range(slice_range if manual_slice else shared, name=bar_name)
                self.p.scalar_bars[bar_name].SetTitle(
                    self.color_label + (' / regions' if regions else
                                        ' / slice range' if manual_slice else ' / shared'))
        if render:
            self.p.render()

    def _bar_name(self, index):
        return f"{self.color_label} / shared [{index}]"

    def _region_bar_actor(self):
        if self.eta_colors:
            actors = self.p.renderers[0].actors
            return actors.get('positive') or actors.get('negative')
        return None

    def _scalar(self, data, name, color_range=None):
        # Companion renderers use the same names. Plotter.remove_actor removes
        # matching actors in every renderer, including the shared 3D slice.
        self.p.renderer.remove_actor(name, reset_camera=False, render=False)
        index = self.p.renderers.active_index
        old_bar = self.bar_names.pop(index, None)
        if old_bar in self.p.scalar_bars:
            self.p.remove_scalar_bar(old_bar, render=False)
        actor = self._add(data, name, scalars='value', cmap=_component_cmap(self.color_component),
                          clim=self.color_range if color_range is None else color_range,
                          nan_opacity=0, lighting=False)
        region_actor = self._region_bar_actor() if index == 0 else None
        bar_actor = region_actor if region_actor is not None else actor
        if bar_actor is not None:
            # One independent bar per renderer; slice bounds can be overridden.
            # In combined mode, the 3D legend belongs to the regions. The
            # face-on slice legend reports any independent slice range.
            bar_name = self._bar_name(index)
            bar = self.p.add_scalar_bar(title=bar_name, mapper=bar_actor.mapper,
                                  color='#23344a', title_font_size=10, label_font_size=9,
                                  width=.8, height=.08, position_x=.1, position_y=.04,
                                  n_labels=3, render=False)
            bar.SetTitle(self.color_label + (' / regions' if region_actor is not None else
                                            ' / shared' if color_range is None else ' / slice range'))
            self.bar_names[index] = bar_name
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
            if self.eta_colors:
                sliced['value'] = np.where(np.isfinite(sliced['value']), sliced['eta'], np.nan)
        color_range = self.view.get('slice_color_ranges', {}).get(self.color_key)
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
            enabled = self.view['layout'] != 'slice' and self.view['plane'] and self.profile_pick is None
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
