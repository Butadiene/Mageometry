"""Qt forms for scientific source and analysis settings."""

from copy import deepcopy
from pathlib import Path

from PySide6 import QtWidgets as W

from ..session.specs import model_source, validate_source


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
        self.kind.addItems(['t96', 'xdmf', 'hdf5', 'vtk'] + (['dipole'] if background else []))
        self.form.addRow('Source type', self.kind)
        self.fields = {}
        labels = {'path': 'File', 'epoch': 'Epoch [Unix s]', 'pdyn': 'Pdyn [nPa]',
                  'dst': 'Dst [nT]', 'by': 'IMF By [nT] (list creates cases)', 'bz': 'IMF Bz [nT]',
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
        p = self.source.get('parameters', model_source()['parameters'])
        options = self.source.get('options', {})
        meta = self.source.get('metadata', {})
        values = dict(p, path=self.source.get('path', ''),
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
        self.kind.setCurrentText(self.source['kind'])
        self.kind.currentTextChanged.connect(self.show_fields)
        self.show_fields()

    def show_fields(self):
        kind = self.kind.currentText()
        model = kind in ('t96', 'dipole')
        visible = {'epoch'} if kind == 'dipole' else set()
        if kind == 't96':
            visible = {'epoch', 'pdyn', 'dst', 'by', 'bz', 'bounds', 'shape'}
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
            if kind in ('t96', 'dipole'):
                parameters = {'epoch': float(text('epoch'))}
                if kind == 't96':
                    parameters.update({key: float(text(key)) for key in ('pdyn', 'dst', 'bz')})
                    values = numbers(text('by'))
                    if not values or len(set(values)) != len(values):
                        raise ValueError('Supply distinct IMF By values.')
                    if self.background and len(values) != 1:
                        raise ValueError('A background source takes one IMF By value.')
                    bounds = numbers(text('bounds'), 6)
                    self.sources = [dict(kind=kind, parameters=dict(parameters, by=by),
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
        labels = {'delta': 'FAC step (direct only)', 'geometry_delta': 'Geometry step (blank: auto)',
                  'max_points': 'Analysis/display preview nodes', 'cache_size': 'Retained case previews',
                  'mask_radius': 'Inner exclusion radius', 'planet_radius': 'Reference sphere radius (blank: none)',
                  'current_scale': 'Current display multiplier', 'current_unit': 'Current display unit (blank: native)',
                  'length_unit': 'Coordinate length unit', 'percentile': 'Default threshold percentile',
                  'n_lines': 'Automatic seed count', 'ds': 'Trace step (blank: auto)',
                  'max_steps': 'Trace maximum steps', 'r0': 'Trace inner radius (blank: none)',
                  'rlim': 'Trace outer radius (blank: none)', 'err': 'Trace error tolerance',
                  'bounds': 'Trace bounds: xmin xmax ymin ymax zmin zmax (blank: grid)'}
        for key, label in labels.items():
            widget = W.QLineEdit()
            self.fields[key] = widget
            self.form.addRow(label, widget)
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

    def _evaluation_changed(self):
        self.fields['delta'].setEnabled(self.fields['evaluation'].currentData() == 'direct')

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
        self._evaluation_changed()

    def analysis(self):
        result = deepcopy(self.base)
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
        steps = numbers(text('delta')) if result['evaluation'] == 'direct' else []
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
