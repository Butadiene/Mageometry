"""Display controls for prepared scene data, independent of session/job state."""

import numpy as np
from PySide6 import QtCore as C, QtWidgets as W

from ..viz3d.scene import display_key, slice_axes
from .forms import line, numbers


class DisplayPanel(W.QWidget):
    error = C.Signal(str)

    def __init__(self, scene, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.syncing = False
        self._build()
        self.sync()

    def _guard(self, callback):
        try:
            callback()
        except Exception as exc:
            self.error.emit(str(exc))

    def _build(self):
        display_layout = W.QVBoxLayout(self)
        display_layout.setContentsMargins(6, 6, 6, 6)
        display_layout.setSpacing(12)

        def section(title):
            group = W.QGroupBox(title)
            form = W.QFormLayout(group)
            form.setContentsMargins(8, 8, 8, 8)
            form.setRowWrapPolicy(W.QFormLayout.RowWrapPolicy.WrapLongRows)
            form.setFieldGrowthPolicy(W.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            display_layout.addWidget(group)
            return form

        shared_form = section('Shared')
        self.gamma_eta = W.QCheckBox('Colour Gamma by eta')
        self.gamma_eta.setToolTip(
            'Available with Diagnostic = Gamma. Select regions by Gamma and colour them by eta '
            'from the same contribution. Peak maps show eta at the Gamma peak; slices show eta '
            'without the Gamma threshold. No field recalculation is needed.')
        shared_form.addRow(self.gamma_eta)
        self.quantity_hint = W.QLabel()
        self.quantity_hint.setWordWrap(True)
        shared_form.addRow(self.quantity_hint)
        self.value_sign = W.QComboBox()
        for label, value in [('All values', 'both'), ('Positive only (> 0)', 'positive'),
                              ('Negative only (< 0)', 'negative')]:
            self.value_sign.addItem(label, value)
        self.value_sign.setAccessibleName('Value sign')
        self.value_sign.setToolTip(
            'Filter regions, arrows, slices and peak maps by sign. '
            'Peak maps select the strongest value of the chosen sign along each sightline. '
            'Applied immediately without recalculating the field.')
        self.value_sign.setEnabled(False)
        shared_form.addRow('Value sign', self.value_sign)
        self.auto_limit = W.QCheckBox('Automatic shared colour range')
        self.auto_limit.setChecked(True)
        shared_form.addRow(self.auto_limit)
        self.color_limit = W.QDoubleSpinBox()
        self.color_limit.setDecimals(9)
        self.color_limit.setRange(1e-9, 1e100)
        self.color_limit.setValue(1.)
        shared_form.addRow('Colour upper limit', self.color_limit)
        shared_tip = ('Value sign applies to the selected diagnostic in all panels. Shared colours apply '
                      'to eta-coloured Gamma regions, arrows and peak maps, '
                      'and to slices when Slice uses shared colour range is checked. '
                      'Gamma, Gamma/|alpha| and |B| kappa use [0, upper limit]; '
                      'signed diagnostics use [-upper limit, upper limit].')
        self.auto_limit.setToolTip(shared_tip)
        self.color_limit.setToolTip(shared_tip)

        form = section('3D and peak maps')
        self.threshold_mode = W.QComboBox()
        self.threshold_mode.addItem('Absolute threshold', 'absolute')
        self.threshold_mode.addItem('Value interval [a, b]', 'interval')
        form.addRow('3D / peak filter', self.threshold_mode)
        self.value_interval = W.QLineEdit()
        self.value_interval.setAccessibleName('Value interval a b')
        self.value_interval.setToolTip('Signed inclusive bounds a b, with a < b. Scientific notation accepted. '
                                       'Filter 3D regions, arrows and peak maps; slice values remain unchanged.')
        form.addRow('Value interval: a b', self.value_interval)
        self.threshold = W.QDoubleSpinBox()
        self.threshold.setDecimals(9)
        self.threshold.setRange(0., 1e100)
        self.threshold.setKeyboardTracking(False)
        self.threshold.setEnabled(False)
        form.addRow('Absolute threshold', self.threshold)
        self.threshold_slider = W.QSlider(C.Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 1000)
        self.threshold_slider.setAccessibleName('Absolute threshold')
        self.threshold_slider.setEnabled(False)
        self.threshold_slider_max = 1.
        self.threshold_range = W.QLabel()
        threshold_controls = W.QVBoxLayout()
        threshold_controls.addWidget(self.threshold_slider)
        threshold_controls.addWidget(self.threshold_range)
        form.addRow(threshold_controls)
        self.auto_threshold_range = W.QCheckBox('Automatic threshold slider range')
        self.auto_threshold_range.setChecked(True)
        self.auto_threshold_range.setEnabled(False)
        form.addRow(self.auto_threshold_range)
        self.threshold_slider_upper = W.QLineEdit()
        self.threshold_slider_upper.setEnabled(False)
        self.threshold_slider_upper.setAccessibleName('Threshold slider upper bound')
        self.threshold_slider_upper.setToolTip(
            'Positive upper bound in display units; scientific notation is accepted. '
            'The lower bound is zero. Lowering the upper bound below the current '
            'threshold also lowers the threshold. Press Enter to apply.')
        form.addRow('Slider upper bound', self.threshold_slider_upper)
        threshold_tip = ('Applied immediately in the diagnostic display units. Automatic slider range uses '
                         'the shared peak across cases and contributions. A higher threshold expands the range.')
        self.threshold.setToolTip(threshold_tip)
        self.threshold_slider.setToolTip(threshold_tip)
        self.layers = {}
        for name in ('lines', 'arrows', 'regions'):
            check = W.QCheckBox('Show ' + name)
            check.setChecked(True)
            check.toggled.connect(lambda enabled, key=name: self.set_layer(key, enabled))
            form.addRow(check)
            self.layers[name] = check
        text = W.QLabel('Filters regions, arrows and peak maps. Slices retain all strengths of the selected sign.')
        text.setWordWrap(True)
        form.addRow(text)

        slice_form = section('Slice')
        text = W.QLabel('Colour and extent apply to both the 3D plane and the face-on slice.')
        text.setWordWrap(True)
        slice_form.addRow(text)
        self.auto_slice_color = W.QCheckBox('Slice uses shared colour range')
        self.auto_slice_color.setChecked(True)
        slice_form.addRow(self.auto_slice_color)
        self.slice_color_range = W.QLineEdit()
        self.slice_color_range.setAccessibleName('Slice colour min max')
        self.slice_color_range.setToolTip('Colour-bar minimum and maximum for both slice panels. '
                                          'Values outside the range keep the endpoint colours. '
                                          'Does not filter data or change 3D/peak colour limits.')
        slice_form.addRow('Slice colour: min max', self.slice_color_range)
        self.auto_slice_extent = W.QCheckBox('Full slice extent')
        self.auto_slice_extent.setChecked(True)
        slice_form.addRow(self.auto_slice_extent)
        self.slice_horizontal = W.QLineEdit()
        self.slice_vertical = W.QLineEdit()
        self.slice_horizontal.setAccessibleName('Slice horizontal min max')
        self.slice_vertical.setAccessibleName('Slice vertical min max')
        self.slice_horizontal_label = W.QLabel('Slice horizontal: min max')
        self.slice_vertical_label = W.QLabel('Slice vertical: min max')
        for label, widget in ((self.slice_horizontal_label, self.slice_horizontal),
                              (self.slice_vertical_label, self.slice_vertical)):
            label.setWordWrap(True)
            widget.setToolTip('Crop both slice panels to this coordinate interval. For oblique planes, '
                              'coordinates are projections onto the face-on horizontal/vertical axes '
                              'measured from the world origin. Scientific notation accepted.')
            slice_form.addRow(label, widget)
        self.layers['plane'] = W.QCheckBox('Show plane in 3D')
        self.layers['plane'].setChecked(True)
        self.layers['plane'].setToolTip('Show the slice and its handle in 3D. The face-on slice stays visible.')
        self.layers['plane'].toggled.connect(lambda enabled: self.set_layer('plane', enabled))
        slice_form.addRow(self.layers['plane'])
        display_layout.addStretch()
        for section_form in (shared_form, form, slice_form):
            for row in range(section_form.rowCount()):
                item = section_form.itemAt(row, W.QFormLayout.ItemRole.LabelRole)
                if item is not None and isinstance(item.widget(), W.QLabel):
                    item.widget().setWordWrap(True)
        self.value_sign.currentIndexChanged.connect(self.set_value_sign)
        self.gamma_eta.toggled.connect(self.set_gamma_eta)
        self.threshold_mode.currentIndexChanged.connect(lambda: self._guard(self.set_value_interval))
        self.value_interval.editingFinished.connect(lambda: self._guard(self.set_value_interval))
        self.auto_slice_color.toggled.connect(lambda: self._guard(self.set_slice_color_range))
        self.slice_color_range.editingFinished.connect(lambda: self._guard(self.set_slice_color_range))
        self.auto_slice_extent.toggled.connect(lambda: self._guard(self.set_slice_extent))
        self.slice_horizontal.editingFinished.connect(lambda: self._guard(self.set_slice_extent))
        self.slice_vertical.editingFinished.connect(lambda: self._guard(self.set_slice_extent))
        self.threshold.valueChanged.connect(self.set_threshold)
        self.threshold_slider.valueChanged.connect(self.move_threshold_slider)
        self.auto_threshold_range.toggled.connect(lambda value: self._guard(self.set_threshold_slider_limit))
        self.threshold_slider_upper.editingFinished.connect(lambda: self._guard(self.set_threshold_slider_limit))
        self.auto_limit.toggled.connect(lambda value: self.set_color_limit())
        self.color_limit.valueChanged.connect(lambda value: self.set_color_limit())

    def sync(self):
        ready = self.scene.result is not None
        self.setEnabled(ready)
        self.value_sign.setEnabled(ready)
        self.threshold.setEnabled(ready)
        self.threshold_slider.setEnabled(ready)
        self.auto_threshold_range.setEnabled(ready)
        for widget in (self.threshold_mode, self.value_interval, self.auto_slice_color,
                       self.slice_color_range, self.auto_slice_extent, self.slice_horizontal, self.slice_vertical):
            widget.setEnabled(ready)
        if self.scene.result is None:
            self.threshold_range.clear()
            self.threshold_slider_upper.clear()
            self.threshold_slider_upper.setEnabled(False)
            return
        self.syncing = True
        try:
            view, result = self.scene.view, self.scene.result
            self.gamma_eta.setEnabled(result['component'] == 'gamma')
            self.gamma_eta.setChecked(view.get('gamma_eta', False))
            self.quantity_hint.setText(
                f"Filter / sign: {result['label']}\nColours: {self.scene.color_label}")
            self.value_sign.setCurrentIndex(self.value_sign.findData(view['value_sign']))
            key = display_key(result)
            self.sync_threshold(view['thresholds'].get(key, result['scale']['threshold']), reset_range=True)
            self.threshold_mode.setCurrentIndex(self.threshold_mode.findData(view['threshold_modes'].get(key, 'absolute')))
            filter_limit = view['color_limits'].get(key, result['scale']['limit'])
            self.value_interval.setText(line(view['value_intervals'].get(key, [-filter_limit, filter_limit])))
            self._sync_filter_enabled()
            key = self.scene.color_key
            limit = view['color_limits'].get(key, self.scene.default_color_limit)
            self.auto_limit.setChecked(key not in view['color_limits'])
            self.color_limit.setValue(limit)
            self.color_limit.setEnabled(not self.auto_limit.isChecked())
            self.auto_slice_color.setChecked(key not in view['slice_color_ranges'])
            self.slice_color_range.setText(line(view['slice_color_ranges'].get(key, self.scene.color_range)))
            self.slice_color_range.setEnabled(not self.auto_slice_color.isChecked())
            self.auto_slice_extent.setChecked(view['slice_extent'] is None)
            self.sync_slice_extent()
            for name, widget in self.layers.items():
                widget.setChecked(view[name])
            self.layers['arrows'].setEnabled(result['basis'] is not None)
        finally:
            self.syncing = False

    def sync_slice_extent(self):
        if self.scene.mesh is None:
            return
        axes = slice_axes(self.scene.view['normal'])
        bounds = np.asarray(self.scene.mesh.bounds).reshape(3, 2)
        corners = np.array(np.meshgrid(*bounds, indexing='ij')).reshape(3, -1).T
        extent = self.scene.view['slice_extent']
        for index, (axis, widget, label) in enumerate(zip(axes,
                (self.slice_horizontal, self.slice_vertical),
                (self.slice_horizontal_label, self.slice_vertical_label))):
            name = next(('xyz'[i] for i in range(3) if np.allclose(axis, np.eye(3)[i])),
                        'u' if index == 0 else 'v')
            direction = 'horizontal' if index == 0 else 'vertical'
            unit = self.scene.result['analysis']['length_unit']
            label.setText(f'Slice {direction} {name} [{unit}]: min max')
            values = extent[2*index:2*index+2] if extent is not None else [float((corners @ axis).min()), float((corners @ axis).max())]
            widget.setText(line(values))
            widget.setEnabled(extent is not None)

    def sync_threshold(self, value, *, reset_range=False):
        previous = self.syncing
        self.syncing = True
        try:
            key = display_key(self.scene.result)
            limits = self.scene.view['threshold_slider_limits']
            if key in limits:
                self.threshold_slider_max = min(limits[key], self.threshold.maximum())
            elif reset_range:
                peak = self.scene.result['scale']['peak']
                self.threshold_slider_max = min(1.01 * peak, self.threshold.maximum()) if peak > 0 else 1.
            self.threshold_slider_max = max(self.threshold_slider_max, value)
            upper = self.threshold_slider_max
            if key in limits:
                limits[key] = upper
            self.auto_threshold_range.setChecked(key not in limits)
            self.threshold_slider_upper.setEnabled(key in limits)
            self.threshold_slider_upper.setText(f'{upper:.16g}')
            # Keep small diagnostics visible without quantizing manual input to slider ticks.
            exponent = np.log10(upper) - 3
            if value > 0:
                exponent = min(exponent, np.log10(value))
            self.threshold.setDecimals(min(323, max(9, int(np.ceil(-exponent)) + 2)))
            self.threshold.setSingleStep(upper / self.threshold_slider.maximum())
            self.threshold.setValue(value)
            self.threshold_slider.setValue(round(self.threshold_slider.maximum() * value / upper))
            self.threshold_range.setText(f'Slider range: 0 – {upper:.6g}')
        finally:
            self.syncing = previous

    def move_threshold_slider(self, position):
        if not self.syncing and self.scene.result is not None:
            self.set_threshold(self.threshold_slider_max * position / self.threshold_slider.maximum())

    def set_threshold_slider_limit(self):
        if self.syncing or self.scene.result is None:
            return
        key = display_key(self.scene.result)
        limits = self.scene.view['threshold_slider_limits']
        value = self.scene.view['thresholds'][key]
        if self.auto_threshold_range.isChecked():
            limits.pop(key, None)
            self.sync_threshold(value, reset_range=True)
        else:
            try:
                upper = float(self.threshold_slider_upper.text())
            except ValueError:
                upper = np.nan
            if not np.isfinite(upper) or not 0 < upper <= self.threshold.maximum():
                raise ValueError('Slider upper bound must be a positive finite number no greater than 1e100.')
            limits[key] = upper
            if value > upper:
                self.set_threshold(upper)
            else:
                self.sync_threshold(value)

    def set_value_sign(self):
        if not self.syncing and self.scene.result is not None:
            self.scene.view['value_sign'] = self.value_sign.currentData()
            self.scene.update_display()

    def set_gamma_eta(self, enabled):
        if not self.syncing and self.scene.result is not None:
            self.scene.view['gamma_eta'] = enabled
            self.scene.update_display()
            self.sync()

    def set_threshold(self, value):
        if not self.syncing and self.scene.result is not None:
            self.sync_threshold(value)
            self.scene.view['thresholds'][display_key(self.scene.result)] = self.threshold.value()
            self.scene.update_display()

    def set_color_limit(self):
        if not self.syncing and self.scene.result is not None:
            key = self.scene.color_key
            if self.auto_limit.isChecked():
                self.scene.view['color_limits'].pop(key, None)
            else:
                self.scene.view['color_limits'][key] = self.color_limit.value()
            self.color_limit.setEnabled(not self.auto_limit.isChecked())
            self.scene.update_colors()
            if self.auto_slice_color.isChecked():
                self.slice_color_range.setText(line(self.scene.color_range))

    @staticmethod
    def _range(text, label):
        values = numbers(text, 2)
        if not np.all(np.isfinite(values)) or values[0] >= values[1]:
            raise ValueError(label + ' requires finite min < max.')
        return values

    def _sync_filter_enabled(self):
        absolute = self.threshold_mode.currentData() == 'absolute'
        for widget in (self.threshold, self.threshold_slider, self.auto_threshold_range):
            widget.setEnabled(absolute)
        self.threshold_slider_upper.setEnabled(absolute and not self.auto_threshold_range.isChecked())
        self.value_interval.setEnabled(not absolute)

    def set_value_interval(self):
        if self.syncing or self.scene.result is None:
            return
        key = display_key(self.scene.result)
        mode = self.threshold_mode.currentData()
        self._sync_filter_enabled()
        if mode == 'interval':
            values = self._range(self.value_interval.text(), 'Value interval')
            self.scene.view['value_intervals'][key] = values
        self.scene.view['threshold_modes'][key] = mode
        self.scene.update_display()

    def set_slice_color_range(self):
        if self.syncing or self.scene.result is None:
            return
        key = self.scene.color_key
        self.slice_color_range.setEnabled(not self.auto_slice_color.isChecked())
        if self.auto_slice_color.isChecked():
            self.scene.view['slice_color_ranges'].pop(key, None)
            self.slice_color_range.setText(line([-self.scene.limit, self.scene.limit]))
        else:
            values = self._range(self.slice_color_range.text(), 'Slice colour range')
            self.scene.view['slice_color_ranges'][key] = values
        self.scene.update_colors()

    def set_slice_extent(self):
        if self.syncing or self.scene.result is None:
            return
        for widget in (self.slice_horizontal, self.slice_vertical):
            widget.setEnabled(not self.auto_slice_extent.isChecked())
        extent = None if self.auto_slice_extent.isChecked() else (
            self._range(self.slice_horizontal.text(), 'Slice horizontal range')
            + self._range(self.slice_vertical.text(), 'Slice vertical range'))
        self.scene.view['slice_extent'] = extent
        self.scene.last_normal = None
        self.sync_slice_extent()
        self.scene.update_slice()

    def set_layer(self, key, value):
        if not self.syncing and self.scene.result is not None:
            self.scene.view[key] = value
            self.scene.update_visibility()
