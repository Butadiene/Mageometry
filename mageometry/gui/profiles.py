"""Qt/Matplotlib profile panel consuming prepared arrays only."""

import numpy as np
from PySide6 import QtCore as C, QtWidgets as W
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from ..geometry.line_profiles import BASIC, TRANSVERSE, CURRENT, ALONG_FIELD
from ..session.profiles import PROFILE_LABELS, STATUS_LABELS, default_profile_view, export_profile


class ProfilePanel(W.QWidget):
    requested = C.Signal(str, object)
    selected = C.Signal(object)
    cursor_moved = C.Signal(object)
    picking_changed = C.Signal(bool)
    error = C.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = self.data = None
        self.records = []
        self.state = default_profile_view()
        self.updating = False
        self.axes, self.cursor_lines = [], []
        self.setMinimumHeight(260)
        layout = W.QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        controls = W.QHBoxLayout()
        layout.addLayout(controls)
        controls.addWidget(W.QLabel('Field-line profile'))
        self.lines = W.QComboBox()
        self.lines.setMinimumWidth(200)
        self.lines.currentIndexChanged.connect(self._select)
        controls.addWidget(self.lines, 1)
        self.pick = W.QPushButton('Select line in 3D')
        self.pick.setCheckable(True)
        self.pick.setToolTip('When enabled, right-click a visible grey field line in the 3D view. '
                             'The plane handle is temporarily disabled; the slice stays visible.')
        self.pick.toggled.connect(lambda enabled: self.pick.setText('Right-click a line' if enabled else 'Select line in 3D'))
        self.pick.toggled.connect(self.picking_changed)
        controls.addWidget(self.pick)
        self.quantities = W.QToolButton()
        self.quantities.setText('Quantities…')
        self.quantities.setPopupMode(W.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.menu = W.QMenu(self.quantities)
        self.quantities.setMenu(self.menu)
        controls.addWidget(self.quantities)
        self.follow = W.QPushButton('Follow diagnostic')
        self.follow.clicked.connect(self._follow)
        controls.addWidget(self.follow)
        self.fit = W.QPushButton('Fit graphs')
        self.fit.clicked.connect(self._fit)
        controls.addWidget(self.fit)
        self.export = W.QPushButton('Export…')
        self.export.clicked.connect(self._export)
        self.export.setEnabled(False)
        controls.addWidget(self.export)
        self.figure = Figure(figsize=(9, 4), facecolor='white', constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(120)
        layout.addWidget(self.canvas, 1)
        footer = W.QHBoxLayout()
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        # Route file export through the provenance-aware action above.
        for action in self.toolbar.actions():
            if action.text() == 'Save':
                self.toolbar.removeAction(action)
        footer.addWidget(self.toolbar)
        self.readout = W.QLabel('Move over a graph to inspect the nearest evaluated point.')
        self.readout.setWordWrap(True)
        footer.addWidget(self.readout, 1)
        layout.addLayout(footer)
        self.info = W.QLabel('Prepare magnetic field lines, then select a line.')
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        self.canvas.mpl_connect('motion_notify_event', self._probe)
        self.canvas.mpl_connect('button_press_event', self._probe)

    def active_quantities(self):
        if self.result is None:
            return []
        allowed = TRANSVERSE + ('bmag',) if self.result['kind'] == 'attribution' else tuple(PROFILE_LABELS)
        names = [n for n in self.state['quantities'] if n in allowed]
        return names or list(dict.fromkeys((self.result['component'], 'bmag')))

    def set_context(self, result, state, records=None):
        self.updating = True
        try:
            self.result, self.state = result, state
            self.records = list(records if records is not None else (result or {}).get('trace_records', []))
            self.pick.setChecked(False)
            self.lines.clear()
            for record in self.records:
                seed = ', '.join(f'{x:g}' for x in record['seed'])
                self.lines.addItem(f"Seed {record['line_id'] + 1}: ({seed})", record['seed_id'])
            identity = state['seed_id']
            if identity is None and self.records:
                identity = self.records[0]['seed_id']
                state['seed_id'] = identity
            self.lines.setCurrentIndex(self.lines.findData(identity) if identity is not None else -1)
            self.pick.setEnabled(bool(self.records))
            self.quantities.setEnabled(result is not None)
            self._refresh_menu()
        finally:
            self.updating = False
        self.clear_data()
        if not self.records:
            status = (result or {}).get('trace_status', 'disabled')
            messages = {'pending': 'Waiting for magnetic field lines…',
                        'disabled': 'Enable Calculate field-line traces in Analysis and Apply.',
                        'failed': 'Tracing failed. Apply again to calculate field lines.',
                        'cancelled': 'Tracing was cancelled. Apply again to complete it.'}
            self.info.setText(messages.get(status, 'No field lines. Specify Trace seeds in Analysis and Apply.'))
        elif self.lines.currentIndex() < 0:
            self.info.setText('The previous seed is absent from this result. Select a line from the list.')
        self.request_current()

    def clear_data(self):
        self.data = None
        self.export.setEnabled(False)
        previous = self.updating
        self.updating = True
        try:
            # Clearing old axes emits limit callbacks. Those defaults must not
            # overwrite the new selection's automatic or saved plot ranges.
            self.figure.clear()
        finally:
            self.updating = previous
        self.axes, self.cursor_lines = [], []
        self.canvas.draw_idle()
        self.readout.setText('Move over a graph to inspect the nearest evaluated point.')
        self.selected.emit(None)

    def _refresh_menu(self):
        self.menu.clear()
        names = self.active_quantities()
        allowed = TRANSVERSE + ('bmag',) if self.result and self.result['kind'] == 'attribution' else tuple(PROFILE_LABELS)
        for title, keys in (('Field and geometry', BASIC), ('Transverse geometry', TRANSVERSE),
                            ('Currents and terms', ('fac',) + CURRENT), ('Along B', ALONG_FIELD)):
            eligible = [key for key in keys if key in allowed]
            if not eligible:
                continue
            menu = self.menu.addMenu(title)
            for key in eligible:
                label = ('Total |B|' if key == 'bmag' and self.result and self.result['kind'] == 'attribution'
                         else PROFILE_LABELS[key])
                action = menu.addAction(label)
                action.setCheckable(True)
                action.setChecked(key in names)
                action.setEnabled((key in names and len(names) > 1) or (key not in names and len(names) < 4))
                action.triggered.connect(lambda checked, name=key: self._quantity(name, checked))

    def _quantity(self, name, checked):
        names = self.active_quantities()
        if checked and name not in names and len(names) < 4:
            names.append(name)
        elif not checked and name in names and len(names) > 1:
            names.remove(name)
        self.state['quantities'] = names
        self._refresh_menu()
        self.request_current()

    def _follow(self):
        self.state['quantities'] = []
        self._refresh_menu()
        self.request_current()

    def _select(self, index):
        if self.updating:
            return
        self.state.update(seed_id=self.lines.currentData(), xlim=None, ylims={}, cursor_s=0.)
        self.request_current()

    def select_line_id(self, line_id):
        record = next((r for r in self.records if r['line_id'] == line_id), None)
        if record is not None:
            self.lines.setCurrentIndex(self.lines.findData(record['seed_id']))

    def request_current(self):
        identity = self.lines.currentData()
        if self.updating or not self.state['visible'] or self.result is None or identity is None:
            return
        record = next(r for r in self.records if r['seed_id'] == identity)
        self.clear_data()
        self.selected.emit(record)
        self.info.setText('Computing selected-line values…')
        self.requested.emit(identity, self.active_quantities())

    def set_data(self, data):
        if data['seed_id'] != self.lines.currentData() or list(data['values']) != self.active_quantities():
            return
        self.data = data
        self.export.setEnabled(True)
        self.info.setText(f"{data['evaluation']} | geometry step {data['geometry_delta']:g} | "
                          f"−B: {STATUS_LABELS[data['status_minus']]} | +B: {STATUS_LABELS[data['status_plus']]} | "
                          'Full line; independent of 3D filters')
        self.info.setToolTip('Pointwise diagnostics can differ from interpolated slice colours. '
                            'Grid mode uses the same preview interpolant as tracing. '
                            'Missing samples remain gaps; missing Frenet geometry does not mask invariant rates.\n'
                            f"Preview shape: {data['preview_shape']}; spacing: {data['grid_spacing']}; "
                            f"FAC step: {data['fac_delta']}")
        self._draw()

    def _draw(self):
        if self.data is None:
            return
        self.updating = True
        try:
            self.figure.clear()
            data, distance = self.data, self.data['s']
            names = list(data['values'])
            self.axes = list(self.figure.subplots(len(names), 1, sharex=True, squeeze=False)[:, 0])
            self.cursor_lines = []
            seed = ', '.join(f'{value:g}' for value in data['seed'])
            title = (f"{data['case_label']} | {data['contribution']} | seed ({seed})\n"
                     f"{data['evaluation']}; step {data['geometry_delta']:g} {data['analysis']['length_unit']}")
            self.figure.suptitle(title, fontsize=9)
            for index, (axis, name) in enumerate(zip(self.axes, names)):
                values = data['values'][name]
                axis.plot(distance, values, color=('#087f8c', '#b05b2a', '#5868ac', '#854d88')[index], lw=1.3)
                axis.axhline(0., color='#c7cdd2', lw=.7)
                axis.axvline(0., color='#a4adb7', ls=':', lw=.8)
                self.cursor_lines.append(axis.axvline(self.state['cursor_s'], color='#ed9c28', lw=1.))
                axis.set_ylabel(data['labels'][name].replace(' [', '\n['), fontsize=8)
                axis.tick_params(labelsize=8)
                axis.grid(alpha=.18)
                missing = np.count_nonzero(~np.isfinite(values))
                if missing:
                    text = 'Undefined throughout this line' if missing == len(values) else f'Missing {missing}/{len(values)}'
                    axis.text(.99, .9, text, ha='right', va='top', transform=axis.transAxes,
                              fontsize=8, color='#855a27')
                if name in self.state['ylims']:
                    axis.set_ylim(*self.state['ylims'][name])
                elif name == 'eta':
                    axis.set_ylim(-1., 1.)
                axis.profile_range_note = axis.text(.99, .06, '', ha='right', va='bottom',
                                                   transform=axis.transAxes, fontsize=8, color='#855a27')
                self._range_note(axis, name)
                axis.callbacks.connect('ylim_changed', lambda a, key=name: self._ylim(a, key))
            self.axes[-1].set_xlabel(f"s from seed [{data['analysis']['length_unit']}], increasing along B", fontsize=9)
            if self.state['xlim'] is not None:
                self.axes[-1].set_xlim(*self.state['xlim'])
            elif len(distance) > 1 and distance[-1] > distance[0]:
                self.axes[-1].set_xlim(distance[0], distance[-1])
            for axis in self.axes:
                axis.callbacks.connect('xlim_changed', self._xlim)
            self.toolbar.update()
            self.canvas.draw()
        finally:
            self.updating = False
        self.probe_at(self.state['cursor_s'])

    def _xlim(self, axis):
        if not self.updating:
            self.state['xlim'] = list(axis.get_xlim())

    def _ylim(self, axis, name):
        if not self.updating:
            self.state['ylims'][name] = list(axis.get_ylim())
            self._range_note(axis, name)

    def _range_note(self, axis, name):
        lo, hi = sorted(axis.get_ylim())
        values = self.data['values'][name]
        count = np.count_nonzero(np.isfinite(values) & ((values < lo) | (values > hi)))
        axis.profile_range_note.set_text(f'{count} samples outside y range' if count else '')

    def _fit(self):
        self.state.update(xlim=None, ylims={})
        self._draw()

    def _probe(self, event):
        if event.inaxes in self.axes and event.xdata is not None and not self.toolbar.mode:
            self.probe_at(event.xdata)

    def probe_at(self, distance):
        if self.data is None or not len(self.data['s']):
            return
        data = self.data
        index = int(np.argmin(np.abs(data['s'] - distance)))
        position = float(data['s'][index])
        self.state['cursor_s'] = position
        for artist in self.cursor_lines:
            artist.set_xdata([position, position])
        point = data['points'][index]
        values = '  '.join(f'{name}={array[index]:.5g}' if np.isfinite(array[index]) else f'{name}=undefined'
                          for name, array in data['values'].items())
        self.readout.setText(f"s={position:.5g}; xyz=({', '.join(f'{v:.5g}' for v in point)})\n{values}")
        self.cursor_moved.emit(point)
        self.canvas.draw_idle()

    def _export(self):
        if self.data is None:
            return
        path, _ = W.QFileDialog.getSaveFileName(self, 'Export field-line profile', 'field-line-profile.png',
                                               'PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;CSV (*.csv)')
        if path:
            try:
                export_profile(self.data, path, self.figure, self.state)
                self.info.setText('Saved profile and matching .profile.json metadata.')
            except Exception as exc:
                self.error.emit(str(exc))
