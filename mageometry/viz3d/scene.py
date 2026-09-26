"""Render prepared session arrays without evaluating a magnetic field."""

import numpy as np

from ._pv import require_pyvista
from .fac import _peak_projection, _region_seeds, _valid_volume
from .slicer import _face_camera
from ._text_layout import _TextLayout


def display_key(result):
    return result['kind'] + ':' + result['component']


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
        self.result, self.view = result, view
        self.mesh, self.volume, self.projections = mesh, volume, projection_data
        if view['origin'] is None:
            view['origin'] = list(mesh.center)
        self.updating = True
        try:
            self.p.clear_plane_widgets()
            self.widget = None
            self.p.clear()
            self.set_layout(view['layout'], render=False)
            self.p.subplot(0)
            self._add(mesh.outline(), 'outline', color='#9caec0')
            radius = result['analysis']['planet_radius']
            if radius is not None:
                self._add(self.pv.Sphere(radius=radius), 'planet', color='#b3c5d5', smooth_shading=True)
            points, cells, offset = [], [], 0
            for path in result['paths']:
                if len(path) >= 2:
                    points.append(path)
                    cells.extend([len(path), *range(offset, offset + len(path))])
                    offset += len(path)
            if points:
                self._add(self.pv.PolyData(np.concatenate(points), lines=np.asarray(cells)),
                          'lines', color='#748698', line_width=1.5)
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
            self.widget = self.p.add_plane_widget(
                self._drag_plane, normal=view['normal'], origin=view['origin'], bounds=mesh.bounds,
                color='#c09631', outline_translation=False, test_callback=False,
                interaction_event='end')
            if saved:
                self.restore_cameras(saved)
            self.set_layout(view['layout'], render=False)
        finally:
            self.updating = False
        self.p.render()

    def _title(self, index, subtitle):
        r = self.result
        text = f"{r['case_label']}; {r['component']}; {r['contribution']}\n{subtitle}"
        if index in (0, 1):
            text += (f"\n{r['analysis']['evaluation']}; geometry step {r['resolved']['geometry_delta']:g}"
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
        p.subplot(0)
        flat = r['values'].ravel(order='F')
        for sign, name, color in ((1, 'positive', '#c94343'), (-1, 'negative', '#2877ba')):
            p.remove_actor(name, reset_camera=False, render=False)
            if self.volume.n_cells and np.any(sign * flat >= cutoff):
                region = self.volume.clip_scalar(value=sign * cutoff, scalars='value', invert=sign < 0)
                if region.n_cells:
                    actor = self._add(region.extract_surface(), name, color=color, opacity=.5)
                    actor.visibility = view['regions']
        p.remove_actor('arrows', reset_camera=False, render=False)
        basis = r['basis']
        if basis is not None:
            vectors = basis.reshape((-1, 3), order='F')
            eligible = np.where(np.all(np.isfinite(vectors), axis=-1), flat, np.nan)
            selected = _region_seeds(self.mesh.points, eligible, cutoff, 32, .07 * self.mesh.length)
            if len(selected):
                arrows = self.pv.PolyData(self.mesh.points[selected])
                arrows['direction'] = np.sign(flat[selected, None]) * vectors[selected]
                arrows['value'] = flat[selected]
                actor = self._add(arrows.glyph(orient='direction', scale=False, factor=.035 * self.mesh.length),
                                  'arrows', scalars='value', cmap='RdBu_r', clim=(-self.limit, self.limit))
                actor.visibility = view['arrows']
        if 'lines' in p.renderer.actors:
            p.renderer.actors['lines'].visibility = view['lines']
        self._title(0, f"3D overview; threshold {cutoff:.4g}")
        count = int(np.count_nonzero(np.isfinite(flat)))
        p.add_text(f'{count:,} valid nodes; grey: total-field lines', name='validity',
                   position=(.03, .15), viewport=True, font_size=8, color='#596b7e', render=False)
        for axis, (panel, projected) in enumerate(self.projections):
            p.subplot(axis + 2)
            panel['value'] = np.where(np.abs(projected) >= cutoff, projected, np.nan)
            self._scalar(panel, 'projection')
            self._title(axis + 2, f'{("YZ", "XZ", "XY")[axis]} signed peak along {"xyz"[axis]}')
        self.update_slice(render=False)
        p.subplot(0)
        if render:
            p.render()

    def _scalar(self, data, name):
        self.p.remove_actor(name, reset_camera=False, render=False)
        actor = self._add(data, name, scalars='value', cmap='RdBu_r',
                          clim=(-self.limit, self.limit), nan_opacity=0, lighting=False)
        if actor is not None:
            # One independent bar per renderer, with the same numerical range.
            bar_name = f"{self.result['label']} / shared [{self.p.renderers.active_index}]"
            if bar_name in self.p.scalar_bars:
                self.p.remove_scalar_bar(bar_name, render=False)
            bar = self.p.add_scalar_bar(title=bar_name, mapper=actor.mapper,
                                  color='#23344a', title_font_size=10, label_font_size=9,
                                  width=.8, height=.08, position_x=.1, position_y=.04,
                                  n_labels=3, render=False)
            bar.SetTitle(self.result['label'] + ' / shared')
        return actor

    def update_slice(self, render=True):
        if self.result is None:
            return
        normal = np.asarray(self.view['normal'], dtype=float)
        normal /= np.linalg.norm(normal)
        origin = np.asarray(self.view['origin'], dtype=float)
        sliced = self.volume.slice(normal=normal, origin=origin) if self.volume.n_cells else self.pv.PolyData()
        self.p.subplot(0)
        actor = self._scalar(sliced, 'slice')
        if actor is not None:
            actor.visibility = self.view['plane']
        self.p.subplot(1)
        self._scalar(sliced, 'slice')
        self._title(1, f"Origin {np.round(origin, 3)}; all finite values")
        if not sliced.n_points:
            self.p.add_text('No valid data on this plane', name='empty', position=(.03, .15), viewport=True,
                            color='#596b7e', font_size=10, render=False)
        else:
            self.p.remove_actor('empty', reset_camera=False, render=False)
        camera = self.p.camera
        if self.last_normal is None or not np.allclose(normal, self.last_normal):
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
        camera.clipping_range = (.001 * self.mesh.length, 10 * self.mesh.length)
        self.last_normal, self.last_origin = normal.copy(), origin.copy()
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
        viewports = [(0, .25, .55, 1), (.55, .25, 1, 1),
                     (0, 0, 1/3, .25), (1/3, 0, 2/3, .25), (2/3, 0, 1, .25)]
        for index, renderer in enumerate(self.p.renderers):
            visible = layout == 'all' or index == (0 if layout == 'three_d' else 1)
            renderer.SetDraw(visible)
            renderer.SetInteractive(visible)
            renderer.SetViewport(*(viewports[index] if layout == 'all' else (0, 0, 1, 1)))
        if self.widget is not None:
            self.widget.SetEnabled(layout != 'slice' and self.view['plane'])
        if self.view is not None:
            self.view['layout'] = layout
        self.p.subplot(1 if layout == 'slice' else 0)
        if render:
            self.p.render()

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
