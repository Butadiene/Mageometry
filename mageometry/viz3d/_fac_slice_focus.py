"""Face-on, uncluttered inspection of a cached FAC slice."""

from itertools import product

import numpy as np

from .slicer import _face_camera


class _SliceFocus:
    """Temporarily isolate a slice, preserving the overview's view state."""

    def __init__(self, owner, panels=(), overview_widgets=(), expand=False):
        self.owner = owner
        self.plotter = owner.plotter
        self.renderer = self.plotter.renderer
        self.panels = panels
        self.overview_widgets = overview_widgets
        self.expand = expand
        self.active = False
        self.reserve_info = False
        self.slider = None
        self.props = set()
        self.layout_callbacks = []
        self.saved_visibility = {}
        self.widget_props = set()
        self.bar_title = f'{owner.scalar_name} cross-section: {owner.label}'
        self.bar_visible = False
        self.corners = np.array(list(product(*np.asarray(owner.bounds).reshape(3, 2))))
        self.length = np.linalg.norm(np.ptp(self.corners, axis=0))
        self.last_normal = None
        self.last_origin = None
        self._setting_slider = False

    def toggle(self):
        if self.active:
            self.leave()
        else:
            self.enter()
        self.plotter.render()

    def enter(self):
        self.owner.activate_main()
        self.was_enabled = self.owner.enabled
        self.saved_camera = self.renderer.camera.copy()
        # PyVista's copy omits VTK properties such as WindowCenter, which
        # would shift the restored scene into the header after an F4 round trip.
        self.saved_camera.DeepCopy(self.renderer.camera)
        self.saved_viewport = self.renderer.GetViewport()
        self.saved_background = self.renderer.background_color
        self.saved_style = self.plotter.iren.style
        self.saved_panels = [(p, p.GetDraw(), p.GetInteractive()) for p in self.panels]
        self.saved_widgets = [(w, w.GetEnabled(), w.GetCurrentRenderer())
                              for w in self.overview_widgets if w is not None]
        # Widget representations belong to the widgets, not to the scene's
        # actor visibility machinery. Off() removes them from the renderer,
        # so restoring actors by name cannot restore their visibility.
        self.widget_props = {w.GetRepresentation() for w, _, _ in self.saved_widgets
                             if hasattr(w, 'GetRepresentation')}
        self.saved_visibility = {}
        self.owner.enabled = True
        self.owner._ensure_widget()
        self.owner.widget.Off()
        self.active = True
        self.last_normal = None
        self.hide_overview()
        for panel, _, _ in self.saved_panels:
            panel.SetDraw(False)
            panel.SetInteractive(False)
        if self.expand:
            self.renderer.SetViewport(0, 0, 1, 1)
        for callback in self.layout_callbacks:
            callback()
        for widget, _, _ in self.saved_widgets:
            widget.Off()
        self.renderer.background_color = '#eef2f6'
        self.plotter.enable_image_style()
        self.focus_style = self.plotter.iren.style
        self.char_observer = self.focus_style.AddObserver('CharEvent', self._char_event)
        self.plotter.add_text(
            'F4: return to 3D  |  F1/F2/F3: YZ/XZ/XY  |  shift+drag: pan  |  wheel: zoom  |  r: fit',
            position=(0.035, 0.015), viewport=True, font_size=10,
            color='#64748b', name='fac-focus-help', render=False)
        self.refresh()

    def _char_event(self, style, event):
        # VTK's image style handles x/y/z itself after our KeyPressEvent.
        # Keep those overview shortcuts (and native 'r') from rotating or
        # resetting a face-on slice; retain native quit/other key handling.
        key = (self.plotter.iren.interactor.GetKeySym() or '').lower()
        if key not in ('x', 'y', 'z', 'r', 's', 'l', 'a'):
            style.OnChar()

    def hide_overview(self):
        """Also hide overview actors replaced by a threshold callback."""
        if not self.active:
            return
        for name, actor in self.renderer.actors.items():
            if (name == 'fac-slice' or name.startswith('fac-focus-')
                    or actor in self.props or actor in self.widget_props):
                continue
            self.saved_visibility.setdefault(name, actor.GetVisibility())
            actor.SetVisibility(False)

    def _move(self, offset):
        if not self.active or self._setting_slider:
            return
        normal, origin = self.owner.normal, self.owner.origin
        origin = origin + normal * (offset - np.dot(normal, origin))
        self.owner.widget.SetOrigin(origin)
        self.owner._update(normal, origin)

    def refresh(self):
        if not self.active:
            return
        self.owner.activate_main()
        if self.owner.case_label is None:
            self.plotter.add_text(f'{self.owner.scalar_name} / CROSS-SECTION',
                                 position=(0.035, 0.935), viewport=True,
                                 font_size=17, color='#263546',
                                 name='fac-focus-title', render=False)
        normal, origin = self.owner.normal, self.owner.origin
        changed = self.last_normal is None or not np.allclose(normal, self.last_normal)
        if changed:
            self.fit()
        else:
            # Follow only the normal displacement, preserving the user's
            # in-plane pan and zoom when scanning neighbouring sections.
            shift = normal * np.dot(origin - self.last_origin, normal)
            camera = self.renderer.camera
            if np.any(shift):
                camera.position = np.asarray(camera.position) + shift
                camera.focal_point = np.asarray(camera.focal_point) + shift
        self.last_normal, self.last_origin = normal.copy(), origin.copy()

        lo, hi = np.min(self.corners @ normal), np.max(self.corners @ normal)
        value = float(np.clip(np.dot(normal, origin), lo, hi))
        self._setting_slider = True
        try:
            if self.slider is None:
                before = set(self.renderer.GetViewProps())
                self.slider = self.plotter.add_slider_widget(
                    self._move, rng=(lo, hi), value=value, title=f'Plane offset [{self.owner.unit}]',
                    pointa=(0.18, 0.10), pointb=(0.82, 0.10), color='#263546',
                    title_height=0.018, fmt='%.3g', interaction_event='always')
                self.props.update(set(self.renderer.GetViewProps()) - before)
            rep = self.slider.GetRepresentation()
            rep.SetMinimumValue(lo)
            rep.SetMaximumValue(hi)
            rep.SetValue(value)
            axis = int(np.argmax(np.abs(normal)))
            title = f'{"xyz"[axis]} [{self.owner.unit}]' if normal[axis] > 0.999999 else f'Plane offset [{self.owner.unit}]'
            rep.SetTitleText(title)
            self.slider.SetCurrentRenderer(self.renderer)
            self.slider.On()
        finally:
            self._setting_slider = False

        actor = self.owner.actor
        has_data = actor is not None and actor.visibility
        if has_data and not self.bar_visible:
            bar = self.plotter.add_scalar_bar(
                title=self.bar_title, mapper=actor.mapper, color='#263546',
                title_font_size=13, label_font_size=12, vertical=False,
                width=0.60, height=0.06, position_x=0.20, position_y=0.17,
                fmt='%.3g', render=False)
            bar.SetVerticalTitleSeparation(6)
            self.props.add(bar)
            self.bar_visible = True
        self.plotter.add_text(self.owner.location(), position=(0.035, 0.875),
                             viewport=True, font_size=13, color='#263546',
                             name='fac-focus-location', render=False)
        key = self.owner.scalar_name
        message = f'All {key} strengths / fixed colour scale' if has_data else f'No valid {key} on this plane'
        if self.owner.case_label is not None and has_data:
            message = f'All {key} strengths / colour scale shared across datasets'
        self.plotter.add_text(message, position=(0.035, 0.835), viewport=True,
                             font_size=10, color='#64748b', name='fac-focus-status', render=False)
        self.hide_overview()

    def fit(self):
        """Fit the plane in the space between the title and controls."""
        self.owner.activate_main()
        normal, origin = self.owner.normal, self.owner.origin
        position, up = _face_camera(normal, origin, 2 * self.length)
        direction = (position - origin) / (2 * self.length)
        # _face_camera chooses a conventional side near coordinate axes;
        # retain that side but use the exact normal for oblique sections.
        direction = np.sign(np.dot(direction, normal)) * normal
        up = up - np.dot(up, direction) * direction
        up /= np.linalg.norm(up)
        right = np.cross(up, direction)
        actor = self.owner.actor
        if actor is not None and actor.visibility:
            actor.mapper.Update()
        points = actor.mapper.dataset.points if actor is not None and actor.visibility else self.corners
        if not len(points):
            points = self.corners
        horizontal, vertical = points @ right, points @ up
        h0, h1, v0, v1 = horizontal.min(), horizontal.max(), vertical.min(), vertical.max()
        center = right * (h0 + h1) / 2 + up * (v0 + v1) / 2 + normal * np.dot(normal, origin)
        camera = self.renderer.camera
        camera.focal_point = center
        camera.position = center + 2 * self.length * direction
        camera.up = up
        camera.parallel_projection = True
        camera.SetWindowCenter(0, 0.03 if self.reserve_info else -0.10)
        x0, y0, x1, y1 = self.renderer.GetViewport()
        width, height = self.plotter.window_size
        aspect = width * (x1 - x0) / (height * (y1 - y0))
        available_height = 0.39 if self.reserve_info else 0.54
        camera.parallel_scale = max((v1 - v0) / available_height, (h1 - h0) / (aspect * 0.88),
                                    self.length * 1e-6) / 2
        camera.clipping_range = (0.01 * self.length, 10 * self.length)
        names = ('x', 'y', 'z')
        hname = names[int(np.argmax(np.abs(right)))] if np.max(np.abs(right)) > 0.999999 else 'u'
        vname = names[int(np.argmax(np.abs(up)))] if np.max(np.abs(up)) > 0.999999 else 'v'
        self.plotter.add_text(
            f'Horizontal {hname}: {h0:.3g} to {h1:.3g}  /  vertical {vname}: {v0:.3g} to {v1:.3g} [{self.owner.unit}]',
            position=(0.035, 0.25), viewport=True, font_size=10, color='#64748b',
            name='fac-focus-coordinates', render=False)

    def leave(self, restore_slice=True):
        if not self.active:
            return
        self.owner.activate_main()
        self.active = False
        if self.slider is not None:
            self.slider.Off()
        if self.bar_visible:
            self.plotter.remove_scalar_bar(self.bar_title, render=False)
            self.bar_visible = False
        for name in list(self.renderer.actors):
            if name.startswith('fac-focus-'):
                self.plotter.remove_actor(name, reset_camera=False, render=False)
        for name, visible in self.saved_visibility.items():
            if name in self.renderer.actors:
                self.renderer.actors[name].SetVisibility(visible)
        for panel, draw, interactive in self.saved_panels:
            panel.SetDraw(draw)
            panel.SetInteractive(interactive)
        self.renderer.SetViewport(self.saved_viewport)
        for callback in self.layout_callbacks:
            callback()
        self.renderer.background_color = self.saved_background
        self.focus_style.RemoveObserver(self.char_observer)
        self.plotter.iren.style = self.saved_style
        for widget, enabled, renderer in self.saved_widgets:
            # Off() clears CurrentRenderer. Without restoring it, VTK picks
            # a renderer under the mouse and moves companion sliders there.
            widget.SetCurrentRenderer(renderer)
            widget.SetEnabled(enabled)
        self.renderer.camera.DeepCopy(self.saved_camera)
        if restore_slice and not self.was_enabled:
            self.owner.toggle()
        else:
            self.owner._ensure_widget()
