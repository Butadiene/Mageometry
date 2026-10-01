"""Qt forms for scientific source and analysis settings."""

from copy import deepcopy
from pathlib import Path

from PySide6 import QtWidgets as W

from ..session.specs import MODEL_DEFAULTS, MODEL_KINDS, MODEL_PARAMETERS, model_source, validate_source


def numbers(text, count=None, integer=False):
    values = [int(word) if integer else float(word) for word in text.replace(',', ' ').split()]
    if count is not None and len(values) != count:
        raise ValueError(f'Expected {count} numbers, separated by spaces or commas.')
    return values


def line(values):
    if values is None:
        return ''
    if isinstance(values, (list, tuple)):
        return ' '.join(str(value) for value in values)
    return str(values)


class SourceDialog(W.QDialog):
    def __init__(self, parent=None, source=None, background=False):
        super().__init__(parent)
        self.setWindowTitle('Background source' if background else 'Magnetic source')
        self.resize(570, 660)
        self.sources = []
        self.source = deepcopy(source or model_source())
        self.background = background
        layout = W.QVBoxLayout(self)
        self.form = W.QFormLayout()
        self.form.setRowWrapPolicy(W.QFormLayout.RowWrapPolicy.WrapAllRows)
        layout.addLayout(self.form)
        self.kind = W.QComboBox()
        self.kind.addItems(['t96', 't89', 't01', 't04', 'xdmf', 'hdf5', 'vtk']
                           + (['dipole'] if background else []))
        self.form.addRow('Source type', self.kind)
        self.fields = {}
        labels = {'path': 'File', 'epoch': 'Epoch [Unix s]', 'pdyn': 'Pdyn [nPa]',
                  'dst': 'Dst [nT]', 'by': 'IMF By [nT] (list creates cases)', 'bz': 'IMF Bz [nT]',
                  'iopt': 'T89 activity bin iopt [1-7]', 'g1': 'G1 (T01)', 'g2': 'G2 (T01)',
                  'w': 'W1 W2 W3 W4 W5 W6 (T04)',
                  'bounds': 'Bounds: xmin xmax ymin ymax zmin zmax', 'shape': 'Grid nodes: nx ny nz',
                  'stride': 'Reader stride', 'arrays': 'Array name(s)', 'origin': 'HDF5 origin: x y z',
                  'spacing': 'HDF5 spacing: dx dy dz', 'h5_file': 'XDMF heavy-file override',
                  'coordinate_system': 'Coordinate system (declaration)',
                  'length_unit': 'Length unit (declaration)', 'field_unit': 'Field unit (declaration)'}
        for key, label in labels.items():
            widget = W.QLineEdit()
            self.fields[key] = widget
            self.form.addRow(label, widget)
        self.order = W.QCheckBox('HDF5 arrays stored as (nz, ny, nx)')
        self.form.addRow(self.order)
        browse = W.QPushButton('Browse file…')
        browse.clicked.connect(self.browse)
        layout.addWidget(browse)
        self.error = W.QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.StandardButton.Ok | W.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept_source)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        p = dict(MODEL_DEFAULTS, **self.source.get('parameters', {}))
        options = self.source.get('options', {})
        meta = self.source.get('metadata', {})
        values = dict(p, w=[p[f'w{i}'] for i in range(1, 7)], path=self.source.get('path', ''),
                      bounds=[v for interval in self.source.get('bounds', model_source()['bounds']) for v in interval],
                      shape=self.source.get('shape', [65, 49, 49]), stride=options.get('stride', 1),
                      arrays=options.get('name', options.get('datasets', options.get('components', ['BX', 'BY', 'BZ']))),
                      origin=options.get('origin'), spacing=options.get('spacing'),
                      h5_file=options.get('h5_file', ''),
                      coordinate_system=meta.get('coordinate_system', ''),
                      length_unit=meta.get('length_unit', ''), field_unit=meta.get('field_unit', ''))
        for key, widget in self.fields.items():
            widget.setText(line(values.get(key, '')))
        self.order.setChecked(options.get('zyx_order', True))
        self.fields['iopt'].setToolTip('Activity bin, not a numeric Kp value: 1 = Kp 0/0+, '
                                      '2 = 1-/1/1+, ..., 6 = 5-/5/5+, 7 = Kp >= 6-.')
        for key in ('g1', 'g2', 'w'):
            self.fields[key].setToolTip('Supply the model driving indices for the intended conditions. '
                                        'The zero defaults are demonstration inputs; the app does not '
                                        'derive them from solar-wind history.')
        self.kind.setCurrentText(self.source['kind'])
        self.kind.currentTextChanged.connect(self.show_fields)
        self.show_fields()

    def show_fields(self):
        kind = self.kind.currentText()
        model = kind in MODEL_KINDS or kind == 'dipole'
        visible = {'epoch'} if kind == 'dipole' else set()
        if kind in MODEL_KINDS:
            visible = set(MODEL_PARAMETERS[kind]) | {'bounds', 'shape'}
            if kind == 't04':
                visible.add('w')
        if not model:
            visible = {'path', 'stride', 'arrays', 'coordinate_system', 'length_unit', 'field_unit'}
            visible.update({'h5_file'} if kind == 'xdmf' else {'origin', 'spacing'} if kind == 'hdf5' else set())
        for key, widget in self.fields.items():
            widget.setVisible(key in visible)
            self.form.labelForField(widget).setVisible(key in visible)
        self.order.setVisible(kind == 'hdf5')
        if kind == 'vtk' and len(self.fields['arrays'].text().split()) != 1:
            self.fields['arrays'].setText('B')
        elif kind in ('xdmf', 'hdf5') and len(self.fields['arrays'].text().split()) != 3:
            self.fields['arrays'].setText('BX BY BZ')

    def browse(self):
        path, _ = W.QFileDialog.getOpenFileName(self, 'Magnetic data', self.fields['path'].text(),
                                                'Magnetic data (*.xmf *.xdmf *.vti *.vtr *.h5 *.hdf5);;All files (*)')
        if path:
            self.fields['path'].setText(path)
            ext = Path(path).suffix.lower()
            self.kind.setCurrentText('xdmf' if ext in ('.xmf', '.xdmf') else 'vtk' if ext in ('.vti', '.vtr') else 'hdf5')

    def accept_source(self):
        try:
            kind = self.kind.currentText()
            text = lambda key: self.fields[key].text().strip()
            if kind in MODEL_KINDS or kind == 'dipole':
                parameters = {'epoch': float(text('epoch'))}
                if kind in MODEL_KINDS:
                    keys = set(MODEL_PARAMETERS[kind]) - {'epoch', 'by', 'iopt'}
                    parameters.update({key: float(text(key)) for key in keys if not key.startswith('w')})
                    if kind == 't89':
                        parameters['iopt'] = int(text('iopt'))
                    if kind == 't04':
                        parameters.update(zip((f'w{i}' for i in range(1, 7)), numbers(text('w'), 6)))
                    values = [None] if kind == 't89' else numbers(text('by'))
                    if not values or len(set(values)) != len(values):
                        raise ValueError('Supply distinct IMF By values.')
                    if self.background and len(values) != 1:
                        raise ValueError('A background source takes one IMF By value.')
                    bounds = numbers(text('bounds'), 6)
                    self.sources = [dict(kind=kind, parameters=dict(parameters, **({} if by is None else {'by': by})),
                                         bounds=[bounds[i:i+2] for i in (0, 2, 4)],
                                         shape=numbers(text('shape'), 3, True)) for by in values]
                else:
                    self.sources = [dict(kind=kind, parameters=parameters)]
            else:
                options = dict(stride=int(text('stride')))
                names = text('arrays').split()
                if kind == 'vtk':
                    if len(names) != 1:
                        raise ValueError('VTK requires one vector array name.')
                    options['name'] = names[0]
                else:
                    if len(names) != 3:
                        raise ValueError('Supply three scalar component array names.')
                    options['components' if kind == 'xdmf' else 'datasets'] = names
                if kind == 'xdmf' and text('h5_file'):
                    options['h5_file'] = str(Path(text('h5_file')).resolve())
                if kind == 'hdf5':
                    options.update(origin=numbers(text('origin'), 3), spacing=numbers(text('spacing'), 3),
                                   zyx_order=self.order.isChecked())
                metadata = {key: text(key) for key in ('coordinate_system', 'length_unit', 'field_unit') if text(key)}
                if not text('path'):
                    raise ValueError('Select a source file.')
                self.sources = [dict(kind=kind, path=str(Path(text('path')).resolve()), options=options, metadata=metadata)]
            for source in self.sources:
                validate_source(source)
            self.accept()
        except (ValueError, TypeError) as exc:
            self.error.setText(str(exc))


class AnalysisForm(W.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.form = W.QFormLayout(self)
        self.form.setRowWrapPolicy(W.QFormLayout.RowWrapPolicy.WrapAllRows)
        self.form.setFieldGrowthPolicy(W.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.fields = {}
        for key, choices in (('kind', [('Field geometry and currents', 'field'), ('Gradient attribution', 'attribution')]),
                             ('evaluation', [('Direct model', 'direct'), ('Grid interpolation', 'grid')])):
            combo = W.QComboBox()
            for label, value in choices:
                combo.addItem(label, value)
            self.fields[key] = combo
            self.form.addRow(key.capitalize(), combo)
        self.trace_enabled = W.QCheckBox('Calculate field-line traces')
        self.trace_enabled.setChecked(True)
        self.trace_enabled.setToolTip('Uncheck and Apply to skip both automatic seed selection and tracing. '
                                      'Existing seed and integration settings are retained for re-enabling.')
        self.form.addRow(self.trace_enabled)
        labels = {'geometry_delta': 'Geometry step (shared; blank: auto)',
                  'delta': 'FAC steps (blank: Geometry step)',
                  'max_points': 'Analysis/display preview nodes', 'cache_size': 'Retained case previews',
                  'mask_radius': 'Inner exclusion radius', 'planet_radius': 'Reference sphere radius (blank: none)',
                  'current_scale': 'Current display multiplier', 'current_unit': 'Current display unit (blank: native)',
                  'length_unit': 'Coordinate length unit', 'percentile': 'Default threshold percentile',
                  'n_lines': 'Automatic seed count', 'ds': 'Trace step (blank: auto)',
                  'max_steps': 'Trace maximum steps', 'r0': 'Trace inner radius (blank: none)',
                  'rlim': 'Trace outer radius (blank: none)', 'err': 'Trace error tolerance',
                  'bounds': 'Trace bounds: xmin xmax ymin ymax zmin zmax (blank: grid)'}
        for key, label in labels.items():
            if key == 'delta':
                self.fac_override = W.QCheckBox('Override direct FAC steps')
                self.form.addRow(self.fac_override)
            widget = W.QLineEdit()
            self.fields[key] = widget
            self.form.addRow(label, widget)
        tips = {
            'current_scale': ('Display value = native value times this positive multiplier. '
                              'T89/T96/T01/T04 use nT and Re (Earth radii). The default 0.125 '
                              'converts mu0 J from nT/Re to J in nA/m^2; for example, '
                              '2 nT/Re becomes about 0.25 nA/m^2. File default 1 keeps native '
                              'values. The same multiplier applies to dFAC/ds; alpha derivatives '
                              'are unscaled.'),
            'delta': ('Optional curl(B) steps for FAC and the FAC values used in dFAC/ds. '
                      'Otherwise FAC follows Geometry step. Enter one value or three x/y/z '
                      'steps in coordinate units (Re for the model). '
                      'Grid evaluation uses preview axis spacing instead. '
                      'This is separate from the field-line trace step.'),
            'geometry_delta': ('Shared spatial difference step for geometry diagnostics and '
                               'direct FAC unless FAC steps are overridden. Blank uses 0.002 '
                               'in direct mode, or the smallest preview axis spacing in grid '
                               'mode. Also sets the along-B displacement for direct alpha/FAC '
                               'derivatives; grid gradients use preview axis spacing. '
                               'In coordinate length units (Re for the model).'),
            'ds': 'Integration step for tracing field lines, in coordinate length units.',
        }
        for key, tip in tips.items():
            self.fields[key].setToolTip(tip)
            self.form.labelForField(self.fields[key]).setToolTip(tip)
        direction = W.QComboBox()
        for label, value in [('Both', 'both'), ('Along B', 1), ('Against B', -1)]:
            direction.addItem(label, value)
        self.fields['direction'] = direction
        self.form.addRow('Trace direction', direction)
        self.seeds = W.QPlainTextEdit()
        self.seeds.setMaximumHeight(110)
        self.seeds.setPlaceholderText('One x y z seed per line; blank = automatic; none = no tracing')
        self.form.addRow('Trace seeds', self.seeds)
        for widget in self.fields.values():
            self.form.labelForField(widget).setWordWrap(True)
        self.fields['evaluation'].currentIndexChanged.connect(self._evaluation_changed)
        self.fac_override.toggled.connect(self._evaluation_changed)
        self.fac_override.setToolTip(tips['delta'])
        self.trace_enabled.toggled.connect(self._trace_changed)
        self._evaluation_changed()

    def _trace_changed(self):
        enabled = self.trace_enabled.isChecked()
        for key in ('n_lines', 'ds', 'max_steps', 'r0', 'rlim', 'err', 'bounds', 'direction'):
            self.fields[key].setEnabled(enabled)
        self.seeds.setEnabled(enabled)

    def _evaluation_changed(self):
        direct = self.fields['evaluation'].currentData() == 'direct'
        self.fac_override.setEnabled(direct)
        show_fac = direct and self.fac_override.isChecked()
        self.fields['delta'].setVisible(show_fac)
        self.form.labelForField(self.fields['delta']).setVisible(show_fac)
        self.fields['delta'].setEnabled(show_fac)
        self.fields['geometry_delta'].setPlaceholderText('0.002' if direct else 'Minimum preview spacing')

    def set_analysis(self, analysis):
        self.base = deepcopy(analysis)
        for key, widget in self.fields.items():
            value = analysis.get(key, analysis.get('trace', {}).get(key))
            if key == 'direction' and value is None:
                value = 'both'
            if isinstance(widget, W.QComboBox):
                widget.setCurrentIndex(widget.findData(value))
            else:
                if key == 'bounds' and value:
                    value = [v for interval in value for v in interval]
                widget.setText(line(value))
        seeds = analysis['seeds']
        self.seeds.setPlainText('' if seeds is None else '\n'.join(line(seed) for seed in seeds) if seeds else 'none')
        self.fac_override.setChecked(analysis['delta'] is not None)
        self.trace_enabled.setChecked(analysis.get('trace_enabled', True))
        self._trace_changed()
        self._evaluation_changed()

    def analysis(self):
        result = deepcopy(self.base)
        result['trace_enabled'] = self.trace_enabled.isChecked()
        for key in ('kind', 'evaluation'):
            result[key] = self.fields[key].currentData()
        text = lambda key: self.fields[key].text().strip()
        for key in ('max_points', 'cache_size', 'n_lines'):
            result[key] = int(text(key))
        for key in ('mask_radius', 'current_scale', 'percentile'):
            result[key] = float(text(key))
        for key in ('geometry_delta', 'planet_radius'):
            result[key] = float(text(key)) if text(key) else None
        for key in ('length_unit', 'current_unit'):
            result[key] = text(key) or None
        result['length_unit'] = result['length_unit'] or 'grid unit'
        override = result['evaluation'] == 'direct' and self.fac_override.isChecked()
        steps = numbers(text('delta')) if override else []
        result['delta'] = steps[0] if len(steps) == 1 else steps or None
        result['trace'] = {'direction': self.fields['direction'].currentData()}
        for key in ('ds', 'r0', 'rlim', 'err', 'max_steps'):
            if text(key):
                result['trace'][key] = int(text(key)) if key == 'max_steps' else float(text(key))
        if text('bounds'):
            values = numbers(text('bounds'), 6)
            result['trace']['bounds'] = [values[i:i+2] for i in (0, 2, 4)]
        seeds = self.seeds.toPlainText().strip()
        result['seeds'] = None if not seeds else [] if seeds.lower() == 'none' else [numbers(row, 3) for row in seeds.splitlines() if row.strip()]
        return result
