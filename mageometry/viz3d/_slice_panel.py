"""Persistent face-on companion to the overview's movable slice."""

import numpy as np

from .slicer import _face_camera


class _SlicePanel:
    def __init__(self, owner, index):
        self.owner = owner
        self.plotter = owner.plotter
        self.index = index
        self.actor = None
        self.last_normal = None
        self.last_origin = None
        self.bar_title = None
        self.slider = None
        self.updating_slider = False
        self.refresh()

    def _move(self, offset):
        if self.updating_slider:
            return
        owner = self.owner
        origin = owner.origin + owner.normal * (offset - np.dot(owner.normal, owner.origin))
        owner.widget.SetOrigin(origin)
        owner._update(owner.normal, origin)
        if not owner.enabled and owner.actor is not None:
            owner.actor.visibility = False
        self.plotter.render()

    def refresh(self):
        owner, p = self.owner, self.plotter
        p.subplot(self.index)
        try:
            p.set_background('#eef2f6', all_renderers=False)
            source = owner.actor
            has_data = source is not None and source.visibility
            if self.bar_title is not None and (not has_data or
                    self.bar_title != f'Cross-section: {owner.label}'):
                p.remove_scalar_bar(self.bar_title, render=False)
                self.bar_title = None
            if has_data:
                if self.actor is None:
                    self.actor = p.add_mesh(
                        source.mapper.dataset, scalars=owner.scalar_name,
                        cmap='RdBu_r', clim=(-owner.limit, owner.limit),
                        lighting=False, show_scalar_bar=False, pickable=False,
                        name='fac-panel-slice', reset_camera=False, render=False)
                self.actor.mapper.dataset = source.mapper.dataset
                self.actor.mapper.array_name = owner.scalar_name
                self.actor.mapper.scalar_range = (-owner.limit, owner.limit)
                self.actor.mapper.Update()
                title = f'Cross-section: {owner.label}'
                if title != self.bar_title:
                    if self.bar_title is not None:
                        p.remove_scalar_bar(self.bar_title, render=False)
                    p.add_scalar_bar(title=title, mapper=self.actor.mapper,
                                     color='#263546', title_font_size=10,
                                     label_font_size=10, width=0.86, height=0.07,
                                     position_x=0.07, position_y=0.18, render=False)
                    self.bar_title = title
            if self.actor is not None:
                self.actor.visibility = has_data
            p.add_text(f'{owner.scalar_name} / CROSS-SECTION\n{owner.location()}',
                       position=(0.04, 0.89), viewport=True, font_size=12,
                       color='#263546', name='fac-panel-title', render=False)
            status = 'All strengths / fixed colour scale' if has_data else 'No valid data on this plane'
            p.add_text(status + '\nF1/F2/F3: YZ/XZ/XY  |  F4: enlarge',
                       position=(0.04, 0.025), viewport=True, font_size=9,
                       color='#64748b', name='fac-panel-status', render=False)
            normal, origin = owner.normal, owner.origin
            if self.last_normal is None or not np.allclose(normal, self.last_normal):
                length = owner.focus.length
                position, up = _face_camera(normal, origin, 2 * length)
                direction = np.asarray(position) - origin
                direction = np.sign(np.dot(direction, normal)) * normal
                up = np.asarray(up) - np.dot(up, direction) * direction
                up /= np.linalg.norm(up)
                p.camera_position = [origin + 2 * length * direction, origin, up]
                p.camera.parallel_projection = True
                p.renderer.reset_camera(bounds=owner.bounds)
                right = np.cross(up, direction)
                corners = owner.focus.corners
                width, height = p.window_size
                x0, y0, x1, y1 = p.renderer.GetViewport()
                aspect = width * (x1 - x0) / (height * (y1 - y0))
                p.camera.parallel_scale = max(np.ptp(corners @ up) / 0.54,
                                               np.ptp(corners @ right) / (aspect * 0.88)) / 2
                p.camera.SetWindowCenter(0, -0.10)
            else:
                shift = normal * np.dot(origin - self.last_origin, normal)
                p.camera.position = np.asarray(p.camera.position) + shift
                p.camera.focal_point = np.asarray(p.camera.focal_point) + shift
            self.last_normal, self.last_origin = normal.copy(), origin.copy()
            lo, hi = np.min(owner.focus.corners @ normal), np.max(owner.focus.corners @ normal)
            self.updating_slider = True
            try:
                if self.slider is None:
                    self.slider = p.add_slider_widget(
                        self._move, rng=(lo, hi), value=float(np.dot(normal, origin)),
                        title=f'Plane offset [{owner.unit}]', color='#263546',
                        pointa=(0.12, 0.12), pointb=(0.88, 0.12),
                        title_height=0.018, fmt='%.3g', interaction_event='always')
                    owner.focus.overview_widgets += (self.slider,)
                rep = self.slider.GetRepresentation()
                rep.SetMinimumValue(lo)
                rep.SetMaximumValue(hi)
                rep.SetValue(float(np.clip(np.dot(normal, origin), lo, hi)))
            finally:
                self.updating_slider = False
        finally:
            owner.activate_main()
