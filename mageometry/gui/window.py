"""Desktop workspace with independent draft, requested and displayed state."""

from copy import deepcopy
from pathlib import Path

import numpy as np
from PySide6 import QtCore as C, QtGui as G, QtWidgets as W
from pyvistaqt import QtInteractor

from ..session import empty_session, model_session, load_session, save_session, validate_session
from ..session.specs import new_group, new_id, default_analysis
from ..session.jobs import JobRunner
from ..viz3d._current import COMPONENTS, COMPONENT_LABELS, TRANSVERSE_COMPONENTS
from ..viz3d._contribution_data import CONTRIBUTIONS
from ..viz3d._source_info import _source_lines
from ..viz3d.scene import GeometryScene, display_key
from .forms import AnalysisForm, SourceDialog, numbers, line

DISPLAY_KEYS = ('normal', 'origin', 'layout', 'previous_layout', 'panels_hidden',
                'thresholds', 'color_limits', 'cameras', 'lines', 'arrows', 'regions', 'plane')


class MainWindow(W.QMainWindow):
    def __init__(self, session=None, parent=None, runner=None, auto_prepare=True):
        super().__init__(parent)
        self.setWindowTitle('Mageometry — magnetic geometry workspace')
        self.resize(1500, 940)
        self.session = validate_session(session or model_session())
        self.committed = {}
        self.displayed = None
        self.pending = None
        self.runner = runner
        self.busy = False
        self.syncing = False
        self.last_error = ''
        self._build()
        self.restore_window()
        self._populate()
        self.poll_timer = C.QTimer(self)
        self.poll_timer.timeout.connect(self.poll)
        self.poll_timer.start(60)
        if auto_prepare and self.group['cases']:
            C.QTimer.singleShot(0, self.apply)

    @property
    def group(self):
        return next(g for g in self.session['groups'] if g['id'] == self.session['active_group'])

    def _button(self, label, callback, layout):
        button = W.QPushButton(label)
        button.clicked.connect(lambda checked=False: self._guard(callback))
        layout.addWidget(button)
        return button

    def _guard(self, callback):
        try:
            callback()
        except Exception as exc:
            self.report_error(str(exc))

    def report_error(self, text):
        self.last_error = text
        self.status.setText('View unchanged. ' + text)

    def _build(self):
        toolbar = self.addToolBar('Session')
        toolbar.setObjectName('session-toolbar')
        toolbar.setMovable(False)
        for label, callback in [('New model', lambda: self.new_session(model_session())),
                                ('New By comparison', lambda: self.new_session(model_session((-5., -3., -1., 1., 3., 5.)))),
                                ('New file session', lambda: self.new_session(empty_session())),
                                ('Open session…', self.open), ('Save displayed session…', self.save),
                                ('Export PNG…', self.export)]:
            action = toolbar.addAction(label)
            action.triggered.connect(lambda checked=False, cb=callback: self._guard(cb))

        central = W.QWidget()
        self.setCentralWidget(central)
        layout = W.QVBoxLayout(central)
        selection = W.QHBoxLayout()
        layout.addLayout(selection)
        self.groups = W.QComboBox()
        self.cases = W.QComboBox()
        self.components = W.QComboBox()
        self.branches = W.QComboBox()
        for label, widget in [('Group', self.groups), ('Dataset', self.cases),
                              ('Diagnostic', self.components), ('Contribution', self.branches)]:
            selection.addWidget(W.QLabel(label))
            selection.addWidget(widget, 1)
        for key, label in COMPONENT_LABELS.items():
            self.components.addItem(label, key)
        for key, label in CONTRIBUTIONS.items():
            self.branches.addItem(label, key)
        self.groups.currentIndexChanged.connect(lambda: self._guard(self.select_group))
        self.cases.currentIndexChanged.connect(lambda: self.select('case', self.cases.currentData()))
        self.components.currentIndexChanged.connect(lambda: self.select('component', self.components.currentData()))
        self.branches.currentIndexChanged.connect(lambda: self.select('contribution', self.branches.currentData()))

        modes = W.QHBoxLayout()
        layout.addLayout(modes)
        self.mode_buttons = {}
        for label, key in [('All panels', 'all'), ('3D focus', 'three_d'), ('Slice focus', 'slice')]:
            button = self._button(label, lambda mode=key: self.set_layout(mode), modes)
            button.setCheckable(True)
            self.mode_buttons[key] = button
        self.panels_button = self._button('Hide side panels', self.toggle_panels, modes)
        self._button('Reset / fit', self.scene_reset, modes)
        for axis in 'xyz':
            self._button('View ' + axis, lambda a=axis: self.axis_view(a), modes)
        modes.addStretch()
        self.cancel_button = self._button('Cancel calculation', self.cancel, modes)
        self.cancel_button.setEnabled(False)

        self.header = W.QLabel('No prepared result. Set sources and analysis conditions, then Apply.')
        self.header.setWordWrap(True)
        self.header.setTextInteractionFlags(C.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.header)
        plane = W.QHBoxLayout()
        layout.addLayout(plane)
        self.normal = W.QComboBox()
        for label, value in [('YZ (normal x)', [1., 0., 0.]), ('XZ (normal y)', [0., 1., 0.]),
                              ('XY (normal z)', [0., 0., 1.]), ('Oblique…', None)]:
            self.normal.addItem(label, value)
        plane.addWidget(self.normal)
        plane.addWidget(W.QLabel('Origin x y z'))
        self.origin = W.QLineEdit()
        self.origin.setMaximumWidth(220)
        plane.addWidget(self.origin)
        self._button('Set plane', self.set_plane, plane)
        plane.addWidget(W.QLabel('Offset'))
        self.offset = W.QDoubleSpinBox()
        self.offset.setDecimals(5)
        self.offset.setRange(-1e10, 1e10)
        plane.addWidget(self.offset)
        self.slider = W.QSlider(C.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        plane.addWidget(self.slider, 1)
        self.normal.activated.connect(lambda: self._guard(self.align_plane))
        self.offset.valueChanged.connect(self.move_plane)
        self.slider.valueChanged.connect(self.move_slider)

        self.plotter = QtInteractor(central, shape='1|4', auto_update=False, border=False)
        layout.addWidget(self.plotter.interactor, 1)
        self.scene = GeometryScene(self.plotter, self.plane_dragged)
        self.status = W.QLabel('Ready')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self._case_dock()
        self._settings_dock()
        self._shortcuts()

    def _case_dock(self):
        self.case_dock = W.QDockWidget('Cases and sources', self)
        self.case_dock.setObjectName('case-dock')
        box = W.QWidget()
        layout = W.QVBoxLayout(box)
        self.case_list = W.QListWidget()
        self.case_list.currentRowChanged.connect(lambda row: self.cases.setCurrentIndex(row) if not self.syncing else None)
        layout.addWidget(self.case_list)
        for label, callback in [('Add model / files…', self.add_source), ('Edit selected source…', self.edit_source),
                                ('Rename case…', self.rename_case), ('Duplicate case', self.duplicate_case),
                                ('Remove case', self.remove_case), ('Use as reference', self.set_reference),
                                ('Add comparison group…', self.add_group), ('Move case to group…', self.move_case)]:
            self._button(label, callback, layout)
        self.source_info = W.QLabel()
        self.source_info.setWordWrap(True)
        layout.addWidget(self.source_info)
        self.background_info = W.QLabel('Background: none')
        self.background_info.setWordWrap(True)
        layout.addWidget(self.background_info)
        for label, callback in [('Assign background to case…', self.assign_background),
                                ('Assign dipole to model group', self.assign_dipoles),
                                ('Clear selected background', self.clear_background),
                                ('Reload sources / accept changed inputs', self.reload_sources)]:
            self._button(label, callback, layout)
        self.case_dock.setWidget(box)
        self.addDockWidget(C.Qt.DockWidgetArea.LeftDockWidgetArea, self.case_dock)

    def _settings_dock(self):
        self.settings_dock = W.QDockWidget('Analysis and display', self)
        self.settings_dock.setObjectName('settings-dock')
        tabs = W.QTabWidget()
        box = W.QWidget()
        layout = W.QVBoxLayout(box)
        scroll = W.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(C.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.analysis_form = AnalysisForm()
        scroll.setWidget(self.analysis_form)
        layout.addWidget(scroll)
        self.draft_hint = W.QLabel('Changes require Apply. Grid preview coarsening changes numerical evaluation.')
        self.draft_hint.setWordWrap(True)
        layout.addWidget(self.draft_hint)
        buttons = W.QHBoxLayout()
        layout.addLayout(buttons)
        self._button('Apply and recompute', self.apply, buttons)
        self._button('Discard edits', self.discard, buttons)
        tabs.addTab(box, 'Analysis')
        display = W.QWidget()
        form = W.QFormLayout(display)
        self.threshold = W.QDoubleSpinBox()
        self.threshold.setDecimals(9)
        self.threshold.setRange(0., 1e100)
        form.addRow('Absolute threshold', self.threshold)
        self.auto_limit = W.QCheckBox('Automatic shared colour range')
        self.auto_limit.setChecked(True)
        form.addRow(self.auto_limit)
        self.color_limit = W.QDoubleSpinBox()
        self.color_limit.setDecimals(9)
        self.color_limit.setRange(1e-9, 1e100)
        self.color_limit.setValue(1.)
        form.addRow('Symmetric colour limit', self.color_limit)
        self.layers = {}
        for name in ('lines', 'arrows', 'regions', 'plane'):
            check = W.QCheckBox('Show ' + name)
            check.setChecked(True)
            check.toggled.connect(lambda enabled, key=name: self.set_layer(key, enabled))
            form.addRow(check)
            self.layers[name] = check
        text = W.QLabel('Threshold affects regions, arrows and signed peak maps. Slices show all finite values. '
                        'Gradient attribution uses the total-field frame; residual gamma/eta are not scalar differences.')
        text.setWordWrap(True)
        form.addRow(text)
        tabs.addTab(display, 'Display')
        self.threshold.valueChanged.connect(self.set_threshold)
        self.auto_limit.toggled.connect(lambda value: self.set_color_limit())
        self.color_limit.valueChanged.connect(lambda value: self.set_color_limit())
        for widget in self.analysis_form.fields.values():
            signal = widget.currentIndexChanged if isinstance(widget, W.QComboBox) else widget.textEdited
            signal.connect(self.mark_draft)
        self.analysis_form.seeds.textChanged.connect(self.mark_draft)
        self.settings_dock.setWidget(tabs)
        self.addDockWidget(C.Qt.DockWidgetArea.RightDockWidgetArea, self.settings_dock)

    def _shortcuts(self):
        self.shortcuts = []
        actions = [('F4', self.toggle_slice), ('Escape', self.restore_layout),
                   ('F1', lambda: self.align_axis(0)), ('F2', lambda: self.align_axis(1)),
                   ('F3', lambda: self.align_axis(2)),
                   ('F5', lambda: self.cycle(self.components, -1)), ('F6', lambda: self.cycle(self.components, 1)),
                   ('F7', lambda: self.cycle(self.cases, -1)), ('F8', lambda: self.cycle(self.cases, 1)),
                   ('Alt+Left', lambda: self.cycle(self.branches, -1)),
                   ('Alt+Right', lambda: self.cycle(self.branches, 1)), ('r', self.scene_reset)]
        actions += [(key, lambda a=key: self.axis_view(a)) for key in 'xyz']
        actions += [(key, lambda name=name: self.layers[name].toggle())
                    for key, name in [('l', 'lines'), ('a', 'arrows'), ('s', 'regions'), ('c', 'plane')]]
        for key, callback in actions:
            shortcut = G.QShortcut(G.QKeySequence(key), self.plotter.interactor)
            shortcut.setContext(C.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda cb=callback: self._guard(cb))
            self.shortcuts.append(shortcut)

    def cycle(self, combo, direction):
        if combo.isEnabled() and combo.count():
            for step in range(1, combo.count() + 1):
                index = (combo.currentIndex() + direction * step) % combo.count()
                if combo.model().item(index).isEnabled():
                    combo.setCurrentIndex(index)
                    break

    def _populate(self, form=True):
        self.syncing = True
        try:
            g = self.group
            self.groups.clear()
            for group in self.session['groups']:
                self.groups.addItem(group['label'], group['id'])
            self.groups.setCurrentIndex(self.groups.findData(g['id']))
            self.cases.clear()
            self.case_list.clear()
            for case in g['cases']:
                label = case['label'] + (' [reference]' if case['id'] == g['reference'] else '')
                self.cases.addItem(label, case['id'])
                self.case_list.addItem(label)
            self.cases.setCurrentIndex(self.cases.findData(g['view']['case']))
            self.case_list.setCurrentRow(self.cases.currentIndex())
            self.components.setCurrentIndex(self.components.findData(g['view']['component']))
            for i in range(self.components.count()):
                self.components.model().item(i).setEnabled(g['analysis']['kind'] == 'field' or self.components.itemData(i) in TRANSVERSE_COMPONENTS)
            self.branches.setEnabled(g['analysis']['kind'] == 'attribution')
            self.branches.setCurrentIndex(self.branches.findData(g['view']['contribution']))
            if form:
                self.analysis_form.set_analysis(g['analysis'])
            self._source_labels()
        finally:
            self.syncing = False

    def _source_labels(self):
        case = self.selected_case()
        if case is None:
            self.source_info.setText('Add a model or simulation snapshot.')
            self.background_info.setText('Background: none')
            return
        source = case['source']
        if source['kind'] == 't96':
            p = source['parameters']
            details = (f"Epoch: {p['epoch']:g} Unix s\nPdyn: {p['pdyn']:g} nPa | Dst: {p['dst']:g} nT\n"
                       f"IMF By/Bz: {p['by']:g} / {p['bz']:g} nT")
        else:
            details = '\n'.join(f'{name}: {value}' for name, value in source.get('metadata', {}).items())
        self.source_info.setText(source.get('path', source['kind']) + '\n' + details)
        background = case.get('background')
        self.background_info.setText('Background: ' + (background.get('path', background['kind']) if background else 'none'))

    def selected_case(self):
        return next((case for case in self.group['cases'] if case['id'] == self.group['view']['case']), None)

    def mark_draft(self, *args):
        if not self.syncing:
            text = 'Unapplied changes. The displayed result retains its original conditions.'
            if self.analysis_form.fields['kind'].currentData() == 'attribution' and self.components.currentData() not in TRANSVERSE_COMPONENTS:
                text += ' Apply will select eta because attribution supports transverse diagnostics only.'
            self.draft_hint.setText(text)

    def _capture_view(self):
        if self.displayed is None:
            return
        self.scene.view['cameras'] = self.scene.camera_state()
        self.scene.view['panels_hidden'] = self.case_dock.isHidden() and self.settings_dock.isHidden()
        for group in self.session['groups']:
            if group['id'] == self.displayed['id']:
                for key in DISPLAY_KEYS:
                    group['view'][key] = deepcopy(self.scene.view[key])
                break

    def apply(self):
        self._guard(self._apply)

    def _apply(self):
        self._capture_view()
        g = self.group
        analysis = self.analysis_form.analysis()
        old_kind = g['analysis']['kind']
        history = g['view'].setdefault('last_components', {})
        history[old_kind] = g['view']['component']
        if analysis['kind'] == 'field' and old_kind != 'field':
            g['view']['component'] = history.get('field', 'alpha')
        g['analysis'] = analysis
        if analysis['kind'] == 'field':
            g['view']['contribution'] = 'total'
        elif g['view']['component'] not in TRANSVERSE_COMPONENTS:
            g['view']['component'] = 'eta'
        self.session = validate_session(self.session)
        self._populate(form=False)
        self.submit(self.group)

    def submit(self, group):
        if self.runner is None:
            self.runner = JobRunner()
        self.pending = deepcopy(group)
        self.runner.submit(group)
        self.busy = True
        self.cancel_button.setEnabled(True)
        self.status.setText('Preparing requested selection; the displayed result remains unchanged.')

    def select(self, key, value):
        if self.syncing or value is None:
            return
        self.group['view'][key] = value
        if key == 'component':
            self.group['view'].setdefault('last_components', {})[self.group['analysis']['kind']] = value
        self._populate(form=False)
        base = self.committed.get(self.group['id'])
        if base is None or value is None:
            self.status.setText('Press Apply to prepare this group.')
            return
        requested = deepcopy(base)
        for name in ('case', 'component', 'contribution'):
            requested['view'][name] = self.group['view'][name]
        if requested['view']['case'] not in [c['id'] for c in requested['cases']]:
            self.status.setText('Apply source changes before selecting the new case.')
            return
        self._guard(lambda: self.submit(requested))

    def poll(self):
        if self.runner is None:
            return
        for token, kind, value in self.runner.poll():
            if kind == 'progress':
                self.status.setText(value)
            elif kind == 'result' and self.pending is not None:
                self._guard(lambda: self.accept_result(value))
                self.busy = False
                self.cancel_button.setEnabled(False)
            elif kind == 'error':
                self.busy = False
                self.pending = None
                self.cancel_button.setEnabled(False)
                self.report_error(value[0])
                self.last_error = value[1]
                self.restore_selection()
        if self.busy and hasattr(self.runner, 'process') and not self.runner.process.is_alive():
            self.cancel()
            self.report_error('The numerical worker stopped. Apply again to restart it.')
            self.runner.close()
            self.runner = None

    def accept_result(self, result):
        candidate = deepcopy(self.pending)
        if candidate['id'] != self.group['id']:
            return
        self._capture_view()
        for key in DISPLAY_KEYS:
            candidate['view'][key] = deepcopy(self.group['view'][key])
        if self.displayed is not None and self.displayed['id'] == candidate['id']:
            unit_keys = ('current_scale', 'current_unit', 'length_unit')
            if any(self.displayed['analysis'][key] != candidate['analysis'][key] for key in unit_keys):
                candidate['view']['thresholds'] = {}
                candidate['view']['color_limits'] = {}
        candidate['view']['thresholds'].setdefault(display_key(result), result['scale']['threshold'])
        same_group = self.displayed is not None and self.displayed['id'] == candidate['id']
        self.scene.set_result(result, candidate['view'], preserve_camera=same_group)
        candidate['resolved'] = result['resolved']
        if candidate['analysis']['seeds'] is None:
            candidate['analysis']['seeds'] = result['resolved']['seeds']
            self.group['analysis']['seeds'] = result['resolved']['seeds']
            if not self.analysis_form.seeds.toPlainText().strip():
                self.analysis_form.seeds.setPlainText('\n'.join(line(seed) for seed in result['resolved']['seeds']) or 'none')
        self.committed[candidate['id']] = self.displayed = candidate
        self.group['resolved'] = deepcopy(result['resolved'])
        self.group['view'] = deepcopy(candidate['view'])
        self.pending = None
        self.last_error = ''
        self.header.setText(f"{result['case_label']} | {result['component']} | {CONTRIBUTIONS[result['contribution']]}\n"
                            f"{result['label']} | {result['analysis']['evaluation']} | geometry step {result['resolved']['geometry_delta']:g} | "
                            f"preview {tuple(result['resolved']['preview_shape'])} | "
                            f"{'total-field-frame attribution' if result['kind'] == 'attribution' else 'field geometry and currents'}")
        self.header.setToolTip('\n'.join(_source_lines(result['metadata'])))
        self.status.setText('Ready. Layout and slice changes reuse the prepared result.')
        try:
            pending_edits = self.analysis_form.analysis() != candidate['analysis']
        except ValueError:
            pending_edits = True
        self.draft_hint.setText('Unapplied edits remain; the result header describes the committed calculation.'
                               if pending_edits else 'Displayed settings committed. Further numerical edits require Apply.')
        self._populate(form=False)
        self.sync_display()

    def restore_selection(self):
        base = self.committed.get(self.group['id'])
        if base:
            for key in ('case', 'component', 'contribution'):
                self.group['view'][key] = base['view'][key]
            self._populate(form=False)

    def cancel(self, restore=True):
        if self.runner:
            self.runner.cancel()
        self.busy = False
        self.pending = None
        self.cancel_button.setEnabled(False)
        self.status.setText('Cancelled. The previous result and unapplied edits are retained.')
        if restore:
            self.restore_selection()

    def discard(self):
        self.cancel()
        base = self.committed.get(self.group['id'])
        if base:
            index = self.session['groups'].index(self.group)
            self.session['groups'][index] = deepcopy(base)
        self._populate()
        self.draft_hint.setText('Edits discarded.')

    def select_group(self):
        if self.syncing or self.groups.currentData() is None:
            return
        target = self.groups.currentData()
        self._capture_view()
        self.cancel()
        self.session['active_group'] = target
        self._populate()
        if self.group['cases']:
            self.apply()

    def new_session(self, session):
        self.cancel()
        self.session = validate_session(session)
        self.committed = {}
        self.displayed = None
        self.scene.result = None
        self.plotter.clear_plane_widgets()
        self.plotter.clear()
        self.header.setText('No prepared result in this session.')
        self.restore_window()
        self._populate()
        if self.group['cases']:
            self.apply()

    def _source_edited(self):
        self.cancel(restore=False)
        self.group.pop('resolved', None)
        self.group['revision'] = new_id()
        self._populate(form=False)
        self.mark_draft()

    def add_source(self):
        source = None if self.group['analysis']['evaluation'] == 'direct' else dict(kind='xdmf', path='', options={})
        dialog = SourceDialog(self, source)
        if dialog.exec() == W.QDialog.DialogCode.Accepted:
            empty = not self.group['cases']
            for source in dialog.sources:
                label = f"IMF By = {source['parameters']['by']:+g} nT" if source['kind'] == 't96' else Path(source['path']).name
                self.group['cases'].append(dict(id=new_id(), label=label, source=source, background=None))
            if empty:
                self.group['reference'] = self.group['view']['case'] = self.group['cases'][0]['id']
                self.group['analysis'] = default_analysis(dialog.sources[0]['kind'] == 't96')
                if dialog.sources[0]['kind'] != 't96':
                    self.group['analysis']['length_unit'] = dialog.sources[0].get('metadata', {}).get('length_unit', 'grid unit')
                self.group['view']['origin'] = [-6., 0., 0.] if dialog.sources[0]['kind'] == 't96' else None
                self.analysis_form.set_analysis(self.group['analysis'])
            self._source_edited()

    def edit_source(self):
        case = self.selected_case()
        if case is None:
            return
        dialog = SourceDialog(self, case['source'])
        if dialog.exec() == W.QDialog.DialogCode.Accepted:
            if len(dialog.sources) != 1:
                raise ValueError('Edit one case at a time; use Add model for a scan.')
            case['source'] = dialog.sources[0]
            background = case.get('background')
            if background and background.get('follow_case_epoch') and case['source']['kind'] == 't96':
                background['parameters']['epoch'] = case['source']['parameters']['epoch']
            self._source_edited()

    def rename_case(self):
        case = self.selected_case()
        if case:
            label, ok = W.QInputDialog.getText(self, 'Case label', 'Label', text=case['label'])
            if ok and label.strip():
                case['label'] = label.strip()
                self._source_edited()

    def duplicate_case(self):
        case = self.selected_case()
        if case:
            copy = deepcopy(case)
            copy.update(id=new_id(), label=case['label'] + ' copy')
            self.group['cases'].append(copy)
            self._source_edited()

    def remove_case(self):
        case = self.selected_case()
        if case:
            self.group['cases'].remove(case)
            next_case = self.group['cases'][0]['id'] if self.group['cases'] else None
            self.group['view']['case'] = next_case
            if self.group['reference'] == case['id']:
                self.group['reference'] = next_case
            self._source_edited()

    def set_reference(self):
        case = self.selected_case()
        if case:
            self.group['reference'] = case['id']
            self._source_edited()

    def add_group(self):
        label, ok = W.QInputDialog.getText(self, 'Comparison group', 'Group label')
        if ok and label.strip():
            self._capture_view()
            self.cancel()
            group = new_group(label.strip())
            group['view']['origin'] = None
            self.session['groups'].append(group)
            self.session['active_group'] = group['id']
            self._populate()

    def move_case(self):
        case = self.selected_case()
        choices = [g for g in self.session['groups'] if g['id'] != self.group['id']]
        if not case or not choices:
            raise ValueError('Create another comparison group before moving a case.')
        labels = [f"{i+1}: {g['label']}" for i, g in enumerate(choices)]
        label, ok = W.QInputDialog.getItem(self, 'Move case', 'Destination group', labels, editable=False)
        if ok:
            destination = choices[labels.index(label)]
            destination['cases'].append(deepcopy(case))
            if destination['reference'] is None:
                destination['reference'] = destination['view']['case'] = case['id']
            destination.pop('resolved', None)
            destination['revision'] = new_id()
            self.remove_case()

    def assign_background(self):
        case = self.selected_case()
        if case:
            initial = case.get('background')
            if initial is None:
                initial = dict(kind='dipole', parameters={'epoch': case['source']['parameters']['epoch']}) if case['source']['kind'] == 't96' else dict(kind='xdmf', path='', options={})
            dialog = SourceDialog(self, initial, background=True)
            if dialog.exec() == W.QDialog.DialogCode.Accepted:
                case['background'] = dialog.sources[0]
                self._source_edited()

    def assign_dipoles(self):
        if any(c['source']['kind'] != 't96' for c in self.group['cases']):
            raise ValueError('Dipole assignment requires model cases with explicit epochs.')
        for case in self.group['cases']:
            case['background'] = dict(kind='dipole', parameters={'epoch': case['source']['parameters']['epoch']},
                                      follow_case_epoch=True)
        self._source_edited()

    def clear_background(self):
        case = self.selected_case()
        if case:
            case['background'] = None
            self._source_edited()

    def reload_sources(self):
        self._source_edited()
        self.apply()

    def sync_display(self):
        if self.scene.result is None:
            return
        self.syncing = True
        try:
            view, result = self.scene.view, self.scene.result
            key = display_key(result)
            self.threshold.setValue(view['thresholds'].get(key, result['scale']['threshold']))
            self.auto_limit.setChecked(key not in view['color_limits'])
            self.color_limit.setValue(view['color_limits'].get(key, result['scale']['limit']))
            self.color_limit.setEnabled(not self.auto_limit.isChecked())
            for name, widget in self.layers.items():
                widget.setChecked(view[name])
            self.layers['arrows'].setEnabled(result['basis'] is not None)
            self.plane_dragged(view['normal'], view['origin'])
            self.case_dock.setVisible(not view['panels_hidden'])
            self.settings_dock.setVisible(not view['panels_hidden'])
            self.panels_button.setText('Show side panels' if view['panels_hidden'] else 'Hide side panels')
            for name, button in self.mode_buttons.items():
                button.setChecked(name == view['layout'])
        finally:
            self.syncing = False

    def plane_dragged(self, normal, origin):
        previous = self.syncing
        self.syncing = True
        try:
            self.origin.setText(line([round(float(x), 6) for x in origin]))
            axis = next((i for i in range(3) if np.allclose(normal, np.eye(3)[i])), 3)
            self.normal.setCurrentIndex(axis)
            if self.scene.mesh is not None:
                bounds = np.asarray(self.scene.mesh.bounds).reshape(3, 2)
                corners = np.array(np.meshgrid(*bounds, indexing='ij')).reshape(3, -1).T
                n = np.asarray(normal) / np.linalg.norm(normal)
                lo, hi = float((corners @ n).min()), float((corners @ n).max())
                self.plane_range = (lo, hi)
                value = float(np.dot(n, origin))
                self.offset.setRange(lo, hi)
                self.offset.setSingleStep((hi - lo) / 100)
                self.offset.setValue(value)
                self.slider.setValue(round(1000 * (value - lo) / (hi - lo)))
        finally:
            self.syncing = previous

    def set_plane(self):
        if self.scene.result is not None:
            self.scene.view['origin'] = numbers(self.origin.text(), 3)
            self.scene.update_slice()
            self.plane_dragged(self.scene.view['normal'], self.scene.view['origin'])

    def align_axis(self, axis):
        self.normal.setCurrentIndex(axis)
        self.align_plane()

    def align_plane(self):
        if self.scene.result is None:
            return
        normal = self.normal.currentData()
        if normal is None:
            text, ok = W.QInputDialog.getText(self, 'Oblique plane', 'Normal vector: nx ny nz', text=line(self.scene.view['normal']))
            if not ok:
                return
            normal = numbers(text, 3)
            if not np.all(np.isfinite(normal)) or np.linalg.norm(normal) == 0:
                raise ValueError('Use a finite nonzero normal.')
        self.scene.view['normal'] = (np.asarray(normal) / np.linalg.norm(normal)).tolist()
        self.scene.update_slice()
        self.plane_dragged(self.scene.view['normal'], self.scene.view['origin'])

    def move_plane(self, value):
        if self.syncing or self.scene.result is None:
            return
        normal = np.asarray(self.scene.view['normal'])
        origin = np.asarray(self.scene.view['origin'])
        self.scene.view['origin'] = (origin + normal * (value - np.dot(origin, normal))).tolist()
        self.scene.update_slice()
        self.plane_dragged(normal, self.scene.view['origin'])

    def move_slider(self, value):
        if not self.syncing and self.scene.result is not None:
            lo, hi = self.plane_range
            self.move_plane(lo + (hi - lo) * value / 1000)

    def set_threshold(self, value):
        if not self.syncing and self.scene.result is not None:
            self.scene.view['thresholds'][display_key(self.scene.result)] = value
            self.scene.update_display()

    def set_color_limit(self):
        if not self.syncing and self.scene.result is not None:
            key = display_key(self.scene.result)
            if self.auto_limit.isChecked():
                self.scene.view['color_limits'].pop(key, None)
            else:
                self.scene.view['color_limits'][key] = self.color_limit.value()
            self.color_limit.setEnabled(not self.auto_limit.isChecked())
            self.scene.update_display()

    def set_layer(self, key, value):
        if not self.syncing and self.scene.result is not None:
            self.scene.view[key] = value
            self.scene.update_display()
            self.scene.set_layout(self.scene.view['layout'])

    def set_layout(self, mode):
        if self.scene.result is None:
            return
        if mode != 'slice':
            self.scene.view['previous_layout'] = mode
        self.scene.set_layout(mode)
        self.sync_display()

    def toggle_slice(self):
        if self.scene.result is not None:
            view = self.scene.view
            self.set_layout(view.get('previous_layout', 'all') if view['layout'] == 'slice' else 'slice')

    def toggle_panels(self):
        if self.scene.result is not None:
            self.scene.view['panels_hidden'] = not self.scene.view['panels_hidden']
            self.sync_display()

    def restore_layout(self):
        if self.scene.result is not None:
            self.scene.view['panels_hidden'] = False
            self.set_layout('all')

    def scene_reset(self):
        self.scene.reset_camera()

    def axis_view(self, axis):
        if self.scene.result is not None and self.scene.view['layout'] != 'slice':
            self.plotter.subplot(0)
            getattr(self.plotter, {'x': 'view_yz', 'y': 'view_xz', 'z': 'view_xy'}[axis])()
            self.plotter.render()

    def open(self):
        path, _ = W.QFileDialog.getOpenFileName(self, 'Open session', '', 'Session (*.json)')
        if path:
            self.new_session(load_session(path))

    def saved_recipe(self):
        self._capture_view()
        recipe = deepcopy(self.session)
        recipe['groups'] = [deepcopy(self.committed.get(g['id'], g)) for g in recipe['groups']]
        recipe['window'] = dict(size=[self.width(), self.height()],
                                state=bytes(self.saveState().toBase64()).decode('ascii'))
        return validate_session(recipe)

    def restore_window(self):
        state = self.session.get('window', {})
        if state.get('state'):
            self.restoreState(C.QByteArray.fromBase64(state['state'].encode('ascii')))
        if state.get('size'):
            self.resize(*state['size'])

    def save(self):
        path, _ = W.QFileDialog.getSaveFileName(self, 'Save displayed session', 'mageometry-session.json', 'Session (*.json)')
        if path:
            save_session(self.saved_recipe(), path)
            self.status.setText('Saved the displayed recipes and views. Unapplied edits were not attached to displayed results.')

    def export(self):
        if self.scene.result is None:
            raise ValueError('Prepare a result before exporting.')
        path, _ = W.QFileDialog.getSaveFileName(self, 'Export displayed plots', 'geometry.png', 'PNG (*.png)')
        if path:
            self.scene.screenshot(path)
            recipe = self.saved_recipe()
            recipe['active_group'] = self.displayed['id']
            save_session(recipe, str(Path(path).with_suffix('.session.json')))
            self.status.setText('Saved PNG and the matching displayed session recipe.')

    def closeEvent(self, event):
        self.poll_timer.stop()
        if self.runner:
            self.runner.close()
        self.plotter.close()
        event.accept()
