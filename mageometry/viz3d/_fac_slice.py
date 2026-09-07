"""Optional, draggable FAC cross-section in the overview's main renderer."""

from functools import partial

import numpy as np

from ._fac_slice_focus import _SliceFocus


def _slice_settings(normal, origin):
    """Validate user coordinates before creating any rendering resources."""
    if isinstance(normal, str):
        if normal.lower() not in ('x', 'y', 'z'):
            raise ValueError("slice_normal must be x, y, z, a nonzero vector, or None.")
        normal = np.eye(3)['xyz'.index(normal.lower())]
    elif normal is not None:
        normal = np.asarray(normal, dtype=float)
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            raise ValueError("slice_normal must be a finite nonzero 3-vector.")
        length = np.linalg.norm(normal)
        if not np.isfinite(length) or length == 0:
            raise ValueError("slice_normal must be a finite nonzero 3-vector.")
        normal = normal / length
    if origin is not None:
        origin = np.asarray(origin, dtype=float)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("slice_origin must be a finite 3-vector.")
    return normal, origin


class _FACSlice:
    """Own only this viewer's plane widget, slice actor, and slice legend."""

    def __init__(self, plotter, volume, bounds, limit, current_label,
                 length_unit, normal, origin, scalar_bar, activate_main,
                 panels=(), overview_widgets=(), expand=False):
        self.plotter = plotter
        self.volume = volume
        self.bounds = bounds
        self.limit = limit
        self.label = current_label
        self.unit = length_unit
        self.activate_main = activate_main
        self.with_scalar_bar = scalar_bar
        self.bar_title = f'Slice: {current_label}'
        self.bar_visible = False
        self.widget = None
        self.actor = None
        self.enabled = normal is not None
        self.normal = np.array([0., 1., 0.]) if normal is None else normal
        self.origin = np.mean(np.asarray(bounds).reshape(3, 2), axis=1) if origin is None else origin
        self.focus = None

        self._hint('Slice off')
        if self.enabled:
            self._ensure_widget()
        self.focus = _SliceFocus(self, panels, overview_widgets, expand)
        plotter.add_key_event('F4', self.focus.toggle)
        plotter.add_key_event('c', self.toggle)
        # Numeric '3' is VTK's stereo-rendering shortcut (CharEvent).
        for axis, key in enumerate(('F1', 'F2', 'F3')):
            plotter.add_key_event(key, partial(self.align, axis))

    def _hint(self, status):
        self.activate_main()
        self.plotter.add_text(
            f'{status}\nc: slice  |  F1/F2/F3: YZ/XZ/XY  |  F4: slice only  |  drag amber handle',
            position=(0.035, 0.79), viewport=True, font_size=9, color='#64748b',
            name='fac-slice-status', render=False)

    def _ensure_widget(self):
        if self.widget is None:
            self.activate_main()
            renderer = self.plotter.renderer
            existing_props = set(renderer.GetViewProps())
            # Use only valid cells: a slice must not interpolate through a
            # magnetic null, masked planet, or missing derivative stencil.
            self.widget = self.plotter.add_plane_widget(
                self._update, normal=self.normal, origin=self.origin,
                bounds=self.bounds, factor=1.0, color='#b77a18',
                outline_translation=False, outline_opacity=0.08,
                interaction_event='always', test_callback=False)
            self.widget.GetPlaneProperty().SetOpacity(0.08)
            # Widget arrows extend beyond the data box. Keep their bounds
            # out of camera resets and the overview's coordinate-axis labels.
            for prop in set(renderer.GetViewProps()) - existing_props:
                prop.SetUseBounds(False)
        if self.focus is not None and self.focus.active:
            self.widget.Off()
        else:
            self.widget.On()
        self._update(self.widget.GetNormal(), self.widget.GetOrigin())

    def _update(self, normal, origin):
        if not self.enabled:
            return
        self.activate_main()
        self.normal = np.asarray(normal, dtype=float)
        self.origin = np.asarray(origin, dtype=float)
        sliced = self.volume.slice(normal=normal, origin=origin) if self.volume.n_cells else None
        has_data = sliced is not None and sliced.n_cells > 0
        if has_data:
            if self.actor is None:
                self.actor = self.plotter.add_mesh(
                    sliced, scalars='fac', cmap='RdBu_r', clim=(-self.limit, self.limit),
                    lighting=False, nan_opacity=0, show_scalar_bar=False,
                    name='fac-slice', reset_camera=False, render=False, pickable=False)
            else:
                self.actor.mapper.dataset = sliced
            self.actor.visibility = True
            if self.with_scalar_bar and not self.bar_visible and not (self.focus and self.focus.active):
                self.plotter.add_scalar_bar(
                    title=self.bar_title, mapper=self.actor.mapper,
                    color='#263546', title_font_size=10, label_font_size=9,
                    vertical=False, width=0.56, height=0.04,
                    position_x=0.035, position_y=0.26, fmt='%.2g', render=False)
                self.bar_visible = True
        elif self.actor is not None:
            self.actor.visibility = False

        suffix = 'all FAC strengths' if has_data else 'no valid FAC on plane'
        self._hint(f'Slice {self.location()} / {suffix}')
        if self.focus is not None:
            self.focus.refresh()

    def location(self):
        """Human-readable slice coordinates for both viewing modes."""
        axis = int(np.argmax(np.abs(self.normal)))
        if abs(self.normal[axis]) > 0.999999:
            plane = ('YZ', 'XZ', 'XY')[axis]
            location = f'{plane}: {"xyz"[axis]} = {self.origin[axis]:.3g} [{self.unit}]'
        else:
            point = ', '.join(f'{v:.3g}' for v in self.origin)
            location = f'oblique at ({point}) [{self.unit}]'
        return location

    def toggle(self):
        if self.focus is not None and self.focus.active:
            self.focus.leave(restore_slice=False)
        self.enabled = not self.enabled
        if self.enabled:
            self._ensure_widget()
        else:
            if self.widget is not None:
                self.widget.Off()
            if self.actor is not None:
                self.actor.visibility = False
            if self.bar_visible:
                self.plotter.remove_scalar_bar(self.bar_title, render=False)
                self.bar_visible = False
            self._hint('Slice off')
        self.plotter.render()

    def align(self, axis):
        self.normal = np.eye(3)[axis]
        self.enabled = True
        if self.widget is not None:
            self.widget.SetNormal(self.normal)
        self._ensure_widget()
        self.plotter.render()
