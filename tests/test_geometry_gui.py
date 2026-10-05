"""Qt controller state with a real off-screen VTK scene and deterministic jobs."""

from copy import deepcopy
import importlib.util
import os
import unittest
from unittest.mock import patch

import numpy as np

from mageometry.session import model_session
from test_session_scene import packet, gamma_packet

HAVE_GUI = all(importlib.util.find_spec(module) for module in ('PySide6', 'pyvistaqt', 'pyvista'))


class FakeRunner:
    def __init__(self):
        self.requests = []
        self.events = []
        self.token = 0

    def submit(self, group):
        self.token += 1
        self.requests.append(deepcopy(group))
        return self.token

    def poll(self):
        events, self.events = self.events, []
        return events

    def cancel(self):
        self.token += 1
        self.events = []

    def close(self):
        pass


@unittest.skipUnless(HAVE_GUI, 'Optional Qt GUI dependencies unavailable')
class TestGeometryGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        import pyvista as pv
        from PySide6.QtWidgets import QWidget
        from mageometry.gui.window import MainWindow

        def plotter(parent, **kwargs):
            result = pv.Plotter(shape=kwargs['shape'], off_screen=True, border=False)
            object.__setattr__(result, 'interactor', QWidget(parent))
            return result

        self.runner = FakeRunner()
        self.session = model_session((-1., 1.))
        group = self.session['groups'][0]
        group['view']['origin'] = [0., 0., 0.]
        with patch('mageometry.gui.window.QtInteractor', side_effect=plotter):
            self.window = MainWindow(self.session, runner=self.runner, auto_prepare=False)
        self.result = packet()
        self.result.update(case=group['view']['case'], case_label='First case',
                           analysis=deepcopy(group['analysis']),
                           resolved={'inputs': {}, 'seeds': [], 'geometry_delta': .002,
                                     'preview_shape': [7, 7, 7]})
        self.window.pending = deepcopy(group)
        self.window.accept_result(self.result)

    def tearDown(self):
        self.window.close()

    def test_new_by_comparison_uses_the_cli_comparison_preset(self):
        from PySide6.QtWidgets import QToolBar
        from mageometry.gui.cli import session_from_args
        toolbar = self.window.findChild(QToolBar, 'session-toolbar')
        action = next(action for action in toolbar.actions() if action.text() == 'New By comparison')
        action.trigger()
        self.assertFalse(self.window.last_error)
        group = self.runner.requests[-1]
        expected, _ = session_from_args(['--by', '-10', '-5', '0', '5', '10'])
        expected = expected['groups'][0]
        # The form records the integrator's implicit default explicitly.
        expected['analysis']['trace'].setdefault('direction', 'both')
        self.assertEqual([case['source'] for case in group['cases']],
                         [case['source'] for case in expected['cases']])
        self.assertEqual(group['analysis'], expected['analysis'])
        # Compare recipe defaults; the UI additionally records diagnostic history.
        self.assertEqual({k: group['view'][k] for k in expected['view'] if k != 'case'},
                         {k: v for k, v in expected['view'].items() if k != 'case'})
        self.assertEqual(group['reference'], group['cases'][0]['id'])
        self.assertEqual(group['view']['case'], group['reference'])

    def test_gamma_eta_display_uses_separate_units_without_jobs_and_survives_pending_result(self):
        window, panel = self.window, self.window.display_panel
        self.assertFalse(panel.gamma_eta.isEnabled())
        result = dict(self.result, **{key: value for key, value in gamma_packet().items()
                                     if key in ('component', 'values', 'eta_values', 'label')})
        window.pending = deepcopy(window.group)
        window.pending['view']['component'] = 'gamma'
        window.accept_result(result)
        cameras = window.scene.camera_state()
        panel.set_threshold(2.5)
        np.testing.assert_allclose(np.fromstring(panel.slice_color_range.text(), sep=' '), [0., 5.])
        panel.gamma_eta.setChecked(True)
        np.testing.assert_allclose(np.fromstring(panel.slice_color_range.text(), sep=' '), [-1., 1.])
        self.assertTrue(panel.gamma_eta.isEnabled())
        self.assertIn('Filter / sign: Gamma [1 / grid unit]', panel.quantity_hint.text())
        self.assertIn('Colours: eta [dimensionless]', panel.quantity_hint.text())
        panel.auto_limit.setChecked(False)
        panel.color_limit.setValue(.7)
        self.assertEqual(window.scene.view['thresholds']['field:gamma'], 2.5)
        self.assertEqual(window.scene.view['color_limits']['field:eta'], .7)
        self.assertNotIn('field:gamma', window.scene.view['color_limits'])
        self.assertEqual(window.scene.camera_state(), cameras)
        self.assertIs(window.scene.result, result)
        self.assertFalse(self.runner.requests)
        recipe = window.saved_recipe()
        self.assertTrue(recipe['groups'][0]['view']['gamma_eta'])
        window.select('case', window.group['cases'][1]['id'])
        panel.gamma_eta.setChecked(False)
        updated = deepcopy(result)
        updated['case'] = window.group['cases'][1]['id']
        window.accept_result(updated)
        self.assertFalse(window.scene.eta_colors)
        self.assertFalse(panel.gamma_eta.isChecked())
        self.assertEqual(panel.threshold.value(), 2.5)
        self.assertEqual(panel.color_limit.value(), result['scale']['limit'])
        panel.gamma_eta.setChecked(True)
        self.assertEqual(panel.color_limit.value(), .7)
        self.assertEqual(len(self.runner.requests), 1)

    def test_gamma_ratio_is_selectable_and_keeps_its_dimensionless_label(self):
        window = self.window
        component = 'gamma_over_abs_alpha'
        index = window.components.findData(component)
        self.assertGreaterEqual(index, 0)
        self.assertIn('Gamma/|alpha|', window.components.itemText(index))
        window.components.setCurrentIndex(index)
        self.assertEqual(self.runner.requests[-1]['view']['component'], component)
        result = deepcopy(self.result)
        result.update(component=component, label='Gamma/|alpha| [dimensionless]',
                      values=np.abs(result['values']))
        window.accept_result(result)
        self.assertIn('Gamma/|alpha| [dimensionless]', window.header.text())
        self.assertFalse(window.display_panel.layers['arrows'].isEnabled())
        self.assertFalse(window.display_panel.gamma_eta.isEnabled())
        panel = window.display_panel
        np.testing.assert_allclose(np.fromstring(panel.slice_color_range.text(), sep=' '), [0., 5.])
        panel.auto_limit.setChecked(False)
        panel.color_limit.setValue(3.)
        np.testing.assert_allclose(np.fromstring(panel.slice_color_range.text(), sep=' '), [0., 3.])
        self.assertEqual(window.plotter.renderers[1].actors['slice'].mapper.scalar_range, (0., 3.))
        panel.auto_limit.setChecked(True)
        np.testing.assert_allclose(np.fromstring(panel.slice_color_range.text(), sep=' '), [0., 5.])
        self.assertEqual(window.saved_recipe()['groups'][0]['view']['component'], component)
        window.analysis_form.fields['kind'].setCurrentIndex(
            window.analysis_form.fields['kind'].findData('attribution'))
        window.assign_dipoles()
        window.apply()
        self.assertFalse(window.last_error)
        self.assertEqual(self.runner.requests[-1]['view']['component'], component)
        self.assertTrue(window.components.model().item(index).isEnabled())

    def test_layouts_do_not_submit_jobs_or_apply_draft(self):
        self.assertEqual(self.window.scene.view['layout'], 'three_d_slice')
        self.assertTrue(self.window.mode_buttons['three_d_slice'].isChecked())
        self.window.analysis_form.fields['geometry_delta'].setText('.001')
        cameras = self.window.scene.camera_state()
        self.window.mode_buttons['three_d_slice'].click()
        self.window.toggle_slice()
        self.assertEqual(self.window.scene.view['layout'], 'slice')
        self.window.toggle_slice()
        self.assertEqual(self.window.scene.view['layout'], 'three_d_slice')
        self.assertTrue(self.window.mode_buttons['three_d_slice'].isChecked())
        self.assertEqual(self.window.saved_recipe()['groups'][0]['view']['layout'], 'three_d_slice')
        self.window.toggle_panels()
        self.assertTrue(self.window.scene.view['panels_hidden'])
        self.window.restore_layout()
        self.assertFalse(self.window.scene.view['panels_hidden'])
        self.assertEqual(self.window.scene.camera_state(), cameras)
        self.assertFalse(self.runner.requests)
        self.assertEqual(self.window.analysis_form.fields['geometry_delta'].text(), '.001')
        self.assertEqual(self.window.saved_recipe()['groups'][0]['analysis']['geometry_delta'], .002)

    def test_case_request_retains_header_until_result_or_error(self):
        previous_header = self.window.header.text()
        next_case = self.window.group['cases'][1]['id']
        self.window.select('case', next_case)
        self.assertEqual(self.runner.requests[-1]['view']['case'], next_case)
        self.assertEqual(self.window.header.text(), previous_header)
        self.runner.events = [(self.runner.token, 'error', ('Example failure', 'traceback'))]
        self.window.poll()
        self.assertEqual(self.window.header.text(), previous_header)
        self.assertEqual(self.window.cases.currentData(), self.result['case'])
        self.assertIs(self.window.scene.result, self.result)

    def test_result_mode_distinguishes_model_grids_and_file_data(self):
        from PySide6.QtWidgets import QDialog
        window = self.window
        self.assertEqual(window.result_mode.text(), 'Displayed: Model\nDirect model')
        evaluation = window.analysis_form.fields['evaluation']
        evaluation.setCurrentIndex(evaluation.findData('grid'))
        window.apply()
        self.assertEqual(window.result_mode.text(), 'Displayed: Model\nDirect model')
        result = deepcopy(self.result)
        result['analysis'] = deepcopy(window.pending['analysis'])
        window.accept_result(result)
        self.assertEqual(window.result_mode.text(), 'Displayed: Model\nGrid interpolation')
        with patch('mageometry.gui.window.SourceDialog') as dialog:
            dialog.return_value.exec.return_value = QDialog.DialogCode.Accepted
            dialog.return_value.sources = [dict(kind='xdmf', path='/tmp/ea01sw000.xmf', options={'stride': 4})]
            window.add_source()
        self.assertEqual(window.result_mode.text(), 'Displayed: Model\nGrid interpolation')
        window.apply()
        self.assertEqual(window.result_mode.text(), 'Displayed: Model\nGrid interpolation')
        result = deepcopy(self.result)
        result.update(case=window.pending['view']['case'], case_label='ea01sw000.xmf',
                      analysis=deepcopy(window.pending['analysis']))
        window.accept_result(result)
        self.assertEqual(window.result_mode.text(), 'Displayed: File data\nGrid interpolation')
        self.assertEqual(window.scene.result['case'], window.group['view']['case'])
        window.toggle_panels()
        self.assertFalse(window.result_mode.isHidden())

    def test_result_mode_ignores_unapplied_failed_and_cancelled_changes(self):
        from mageometry.session import empty_session
        window = self.window
        original = window.result_mode.text()
        evaluation = window.analysis_form.fields['evaluation']
        evaluation.setCurrentIndex(evaluation.findData('grid'))
        self.assertEqual(window.result_mode.text(), original)
        window.apply()
        self.runner.events = [(self.runner.token, 'error', ('Example failure', 'traceback'))]
        window.poll()
        self.assertEqual(window.result_mode.text(), original)
        window.apply()
        window.cancel()
        self.assertEqual(window.result_mode.text(), original)
        window.new_session(empty_session())
        self.assertEqual(window.result_mode.text(), 'Displayed: None\nAwaiting calculation')

    def publish_pending_traces(self):
        self.window.analysis_form.fields['geometry_delta'].setText('.001')
        self.window.apply()
        result = deepcopy(self.result)
        result.update(trace_status='pending', paths=[])
        result['analysis']['geometry_delta'] = .001
        result['resolved']['geometry_delta'] = .001
        self.runner.events = [(self.runner.token, 'result', result)]
        self.window.poll()
        return result

    def test_field_is_usable_while_tracing_and_completion_preserves_user_edits(self):
        window = self.window
        result = self.publish_pending_traces()
        self.assertIs(window.scene.result, result)
        self.assertTrue(window.busy)
        self.assertTrue(window.cancel_button.isEnabled())
        self.assertTrue(window.display_panel.isEnabled())
        self.assertIn('computing field lines', window.status.text())
        self.assertEqual(window.saved_recipe()['groups'][0]['analysis']['geometry_delta'], .001)
        window.analysis_form.fields['geometry_delta'].setText('.0005')
        window.set_layout('slice')
        window.display_panel.layers['lines'].setChecked(False)
        window.display_panel.threshold.setValue(1.5)
        window.scene.view['origin'] = [.2, 0., 0.]
        window.scene.update_slice()
        cameras = window.scene.camera_state()
        actors = [dict(renderer.actors) for renderer in window.plotter.renderers]
        paths = [np.array([[-1., 0., 0.], [1., 0., 0.]])]
        self.runner.events = [(self.runner.token, 'traces', paths)]
        with patch.object(window.scene, 'set_result', side_effect=AssertionError('Rebuilt field')), \
                patch.object(window.scene, 'update_display', side_effect=AssertionError('Rebuilt regions')):
            window.poll()
        self.assertFalse(window.busy)
        self.assertFalse(window.cancel_button.isEnabled())
        self.assertEqual(result['trace_status'], 'ready')
        self.assertIs(result['paths'], paths)
        self.assertFalse(window.plotter.renderers[0].actors['lines'].visibility)
        self.assertEqual(window.scene.camera_state(), cameras)
        for index, renderer in enumerate(window.plotter.renderers):
            for name in ('positive', 'negative', 'slice', 'projection'):
                if name in actors[index]:
                    self.assertIs(renderer.actors[name], actors[index][name])
        self.assertEqual(window.analysis_form.fields['geometry_delta'].text(), '.0005')
        saved = window.saved_recipe()['groups'][0]
        self.assertEqual(saved['analysis']['geometry_delta'], .001)
        self.assertEqual(saved['view']['layout'], 'slice')
        self.assertEqual(saved['view']['origin'], [.2, 0., 0.])
        self.assertEqual(saved['view']['thresholds']['field:alpha'], 1.5)

    def test_cancel_after_field_publication_retains_field_and_ignores_late_traces(self):
        result = self.publish_pending_traces()
        token = self.runner.token
        self.window.cancel()
        self.assertIs(self.window.scene.result, result)
        self.assertEqual(result['trace_status'], 'cancelled')
        self.assertEqual(self.window.saved_recipe()['groups'][0]['analysis']['geometry_delta'], .001)
        self.assertTrue(self.window.saved_recipe()['groups'][0]['analysis']['trace_enabled'])
        self.runner.events = [(token, 'traces', [np.zeros((2, 3))])]
        self.window.poll()
        self.assertEqual(result['paths'], [])
        self.assertFalse(self.window.busy)

    def test_new_request_discards_late_trace_completion_from_previous_field(self):
        result = self.publish_pending_traces()
        token = self.runner.token
        self.window.select('case', self.window.group['cases'][1]['id'])
        self.runner.events = [(token, 'traces', [np.zeros((2, 3))]),
                              (token, 'error', ('Stale error', 'stale traceback'))]
        self.window.poll()
        self.assertEqual(result['trace_status'], 'cancelled')
        self.assertEqual(result['paths'], [])
        self.assertTrue(self.window.busy)
        self.assertIsNotNone(self.window.pending)
        self.assertEqual(self.window.last_error, '')

    def test_trace_failure_retains_committed_field_and_marks_incomplete_lines(self):
        result = self.publish_pending_traces()
        self.runner.events = [(self.runner.token, 'error', ('Trace failed', 'traceback'))]
        self.window.poll()
        self.assertIs(self.window.scene.result, result)
        self.assertEqual(result['trace_status'], 'failed')
        self.assertFalse(self.window.busy)
        self.assertIn('Trace failed', self.window.status.text())
        self.assertIn('failed', self.window.plotter.renderers[0].actors['validity'].GetInput())
        self.assertEqual(self.window.saved_recipe()['groups'][0]['analysis']['geometry_delta'], .001)

    def test_threshold_controls_update_display_without_applying_analysis(self):
        window = self.window
        window.analysis_form.fields['geometry_delta'].setText('.001')
        cameras = window.scene.camera_state()
        original_slice = window.plotter.renderers[1].actors['slice'].mapper.dataset['value'].copy()
        self.assertEqual(window.display_panel.threshold.value(), 2.)
        with patch.object(window.scene, 'update_display', wraps=window.scene.update_display) as update:
            window.display_panel.threshold.setValue(1.234567891)
            self.assertEqual(window.scene.view['thresholds']['field:alpha'], 1.234567891)
            update.assert_called_once()
            self.assertAlmostEqual(window.display_panel.threshold_slider.value() / 1000 * window.display_panel.threshold_slider_max,
                                   window.display_panel.threshold.value(), delta=window.display_panel.threshold_slider_max / 2000)
            update.reset_mock()
            window.display_panel.threshold_slider.setValue(500)
            self.assertAlmostEqual(window.display_panel.threshold.value(), 4.04)
            self.assertEqual(window.scene.view['thresholds']['field:alpha'], window.display_panel.threshold.value())
            update.assert_called_once()
        self.assertEqual(cameras, window.scene.camera_state())
        np.testing.assert_allclose(window.plotter.renderers[1].actors['slice'].mapper.dataset['value'],
                                   original_slice, equal_nan=True)
        self.assertFalse(self.runner.requests)
        self.assertEqual(window.analysis_form.fields['geometry_delta'].text(), '.001')
        saved = window.saved_recipe()['groups'][0]
        self.assertEqual(saved['analysis']['geometry_delta'], .002)
        self.assertEqual(saved['view']['thresholds']['field:alpha'], window.display_panel.threshold.value())

    def test_value_sign_updates_display_without_applying_analysis(self):
        window = self.window
        window.analysis_form.fields['geometry_delta'].setText('.001')
        cameras = window.scene.camera_state()
        self.assertEqual(window.display_panel.value_sign.currentData(), 'both')
        for mode, excluded in (('positive', 'negative'), ('negative', 'positive')):
            with self.subTest(mode=mode):
                with patch.object(window.scene, 'update_display', wraps=window.scene.update_display) as update:
                    window.display_panel.value_sign.setCurrentIndex(window.display_panel.value_sign.findData(mode))
                    update.assert_called_once()
                self.assertEqual(window.scene.view['value_sign'], mode)
                self.assertIn(mode, window.plotter.renderers[0].actors)
                self.assertNotIn(excluded, window.plotter.renderers[0].actors)
                self.assertEqual(window.scene.camera_state(), cameras)
                self.assertFalse(self.runner.requests)
                saved = window.saved_recipe()['groups'][0]
                self.assertEqual(saved['view']['value_sign'], mode)
                self.assertEqual(saved['analysis']['geometry_delta'], .002)
        self.assertEqual(window.analysis_form.fields['geometry_delta'].text(), '.001')

    def test_colour_and_visibility_controls_retain_scene_geometry(self):
        window = self.window
        panel = window.display_panel
        actors = [dict(renderer.actors) for renderer in window.plotter.renderers]
        cameras = window.scene.camera_state()
        with patch.object(window.scene, 'update_display', side_effect=AssertionError('Rebuilt regions')), \
                patch.object(window.scene, 'update_slice', side_effect=AssertionError('Rebuilt slice')):
            for key, checkbox in panel.layers.items():
                checkbox.setChecked(False)
                self.assertFalse(window.scene.view[key])
                checkbox.setChecked(True)
            panel.auto_limit.setChecked(False)
            panel.color_limit.setValue(3.)
            panel.set_color_limit()
            panel.slice_color_range.setText('-.2 .8')
            panel.auto_slice_color.setChecked(False)
            np.testing.assert_allclose(window.plotter.renderers[1].actors['slice'].mapper.scalar_range, [-.2, .8])
            panel.auto_slice_color.setChecked(True)
            np.testing.assert_allclose(window.plotter.renderers[1].actors['slice'].mapper.scalar_range, [-3., 3.])
        self.assertEqual(window.last_error, '')
        self.assertFalse(self.runner.requests)
        self.assertEqual(window.scene.camera_state(), cameras)
        for index, renderer in enumerate(window.plotter.renderers):
            self.assertEqual(dict(renderer.actors), actors[index])
        saved = window.saved_recipe()['groups'][0]['view']
        self.assertEqual(saved['color_limits']['field:alpha'], 3.)
        self.assertNotIn('field:alpha', saved['slice_color_ranges'])
        self.assertTrue(all(saved[key] for key in panel.layers))

    def test_interval_and_slice_ranges_are_display_only_and_survive_pending_result(self):
        window = self.window
        window.analysis_form.fields['geometry_delta'].setText('.001')
        window.display_panel.value_interval.setText('-.4 1.2')
        window.display_panel.threshold_mode.setCurrentIndex(window.display_panel.threshold_mode.findData('interval'))
        self.assertFalse(window.display_panel.threshold.isEnabled())
        self.assertTrue(window.display_panel.value_interval.isEnabled())
        window.display_panel.slice_color_range.setText('-.2 .8')
        window.display_panel.auto_slice_color.setChecked(False)
        window.display_panel.slice_horizontal.setText('-1 1')
        window.display_panel.slice_vertical.setText('-.5 .5')
        window.display_panel.auto_slice_extent.setChecked(False)
        self.assertFalse(self.runner.requests)
        saved = window.saved_recipe()['groups'][0]
        self.assertEqual(saved['analysis']['geometry_delta'], .002)
        self.assertEqual(saved['view']['threshold_modes']['field:alpha'], 'interval')
        self.assertEqual(saved['view']['value_intervals']['field:alpha'], [-.4, 1.2])
        self.assertEqual(saved['view']['slice_color_ranges']['field:alpha'], [-.2, .8])
        self.assertEqual(saved['view']['slice_extent'], [-1., 1., -.5, .5])
        window.select('component', 'eta')
        window.display_panel.value_interval.setText('-.3 .9')
        window.display_panel.set_value_interval()
        updated = deepcopy(self.result)
        updated.update(component='eta', scale={'limit': 1., 'peak': 1., 'threshold': .25})
        self.runner.events = [(self.runner.token, 'result', updated)]
        window.poll()
        self.assertEqual(window.display_panel.threshold_mode.currentData(), 'absolute')
        self.assertTrue(window.display_panel.auto_slice_color.isChecked())
        self.assertFalse(window.display_panel.auto_slice_extent.isChecked())
        self.assertEqual(window.scene.view['value_intervals']['field:alpha'], [-.3, .9])
        window.select('component', 'alpha')
        self.runner.events = [(self.runner.token, 'result', self.result)]
        window.poll()
        self.assertEqual(window.display_panel.threshold_mode.currentData(), 'interval')
        self.assertEqual(window.display_panel.value_interval.text(), '-0.3 0.9')
        self.assertEqual(window.display_panel.slice_color_range.text(), '-0.2 0.8')
        saved = window.saved_recipe()
        window.new_session(saved)
        self.runner.events = [(self.runner.token, 'result', self.result)]
        window.poll()
        self.assertEqual(window.scene.view['slice_extent'], [-1., 1., -.5, .5])
        self.assertEqual(window.display_panel.threshold_mode.currentData(), 'interval')
        self.assertFalse(window.display_panel.auto_slice_color.isChecked())

    def test_invalid_ranges_leave_applied_display_state_and_axes_labels_follow_plane(self):
        window = self.window
        window.display_panel.value_interval.setText('1 1')
        window.display_panel.threshold_mode.setCurrentIndex(window.display_panel.threshold_mode.findData('interval'))
        self.assertNotIn('field:alpha', window.scene.view['threshold_modes'])
        self.assertTrue(window.display_panel.value_interval.isEnabled())
        window.display_panel.value_interval.setText('-.5 .5')
        window.display_panel.set_value_interval()
        window.display_panel.auto_slice_color.setChecked(False)
        window.display_panel.auto_slice_extent.setChecked(False)
        previous = deepcopy(window.scene.view)
        for widget, action in ((window.display_panel.value_interval, window.display_panel.set_value_interval),
                               (window.display_panel.slice_color_range, window.display_panel.set_slice_color_range),
                               (window.display_panel.slice_horizontal, window.display_panel.set_slice_extent)):
            original = widget.text()
            for text in ('1 1', '3 2', 'nan 2', '0 inf', '1'):
                widget.setText(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    action()
                self.assertEqual(window.scene.view, previous)
            widget.setText(original)
        window.align_axis(2)
        self.assertIn('horizontal x', window.display_panel.slice_horizontal_label.text())
        self.assertIn('vertical y', window.display_panel.slice_vertical_label.text())
        window.scene._drag_plane([1., 1., 1.], [0., 0., 0.])
        self.assertIn('horizontal u', window.display_panel.slice_horizontal_label.text())
        self.assertIn('vertical v', window.display_panel.slice_vertical_label.text())

    def test_disabling_traces_requires_apply_and_retains_automatic_seed_choice(self):
        window = self.window
        window.analysis_form.seeds.clear()
        window.analysis_form.trace_enabled.setChecked(False)
        self.assertFalse(window.analysis_form.seeds.isEnabled())
        self.assertFalse(window.analysis_form.fields['ds'].isEnabled())
        self.assertFalse(self.runner.requests)
        self.assertTrue(window.saved_recipe()['groups'][0]['analysis']['trace_enabled'])
        window.apply()
        submitted = self.runner.requests[-1]
        self.assertFalse(submitted['analysis']['trace_enabled'])
        self.assertIsNone(submitted['analysis']['seeds'])
        updated = deepcopy(self.result)
        updated['analysis'] = deepcopy(submitted['analysis'])
        updated['resolved']['seeds'] = None
        self.runner.events = [(self.runner.token, 'result', updated)]
        window.poll()
        self.assertFalse(window.saved_recipe()['groups'][0]['analysis']['trace_enabled'])
        self.assertIsNone(window.saved_recipe()['groups'][0]['analysis']['seeds'])
        self.assertEqual(window.analysis_form.seeds.toPlainText(), '')
        window.analysis_form.trace_enabled.setChecked(True)
        window.apply()
        self.assertTrue(self.runner.requests[-1]['analysis']['trace_enabled'])
        self.assertIsNone(self.runner.requests[-1]['analysis']['seeds'])


    def test_value_sign_survives_pending_result_and_session_restore(self):
        window = self.window
        window.select('component', 'eta')
        # A display edit made during a calculation must take precedence on completion.
        window.display_panel.value_sign.setCurrentIndex(window.display_panel.value_sign.findData('negative'))
        updated = deepcopy(self.result)
        updated.update(component='eta', scale={'limit': 1., 'peak': 1., 'threshold': .25})
        self.runner.events = [(self.runner.token, 'result', updated)]
        window.poll()
        self.assertEqual(window.scene.view['value_sign'], 'negative')
        self.assertEqual(window.display_panel.value_sign.currentData(), 'negative')
        self.assertNotIn('positive', window.plotter.renderers[0].actors)
        saved = window.saved_recipe()
        window.new_session(saved)
        self.assertFalse(window.display_panel.value_sign.isEnabled())
        self.runner.events = [(self.runner.token, 'result', updated)]
        window.poll()
        self.assertTrue(window.display_panel.value_sign.isEnabled())
        self.assertEqual(window.display_panel.value_sign.currentData(), 'negative')
        self.assertEqual(window.scene.view['value_sign'], 'negative')

    def test_manual_threshold_expands_slider_and_keeps_range_stable(self):
        window = self.window
        window.display_panel.threshold.setValue(20.)
        self.assertEqual(window.display_panel.threshold_slider.value(), window.display_panel.threshold_slider.maximum())
        self.assertEqual(window.display_panel.threshold_slider_max, 20.)
        self.assertNotIn('positive', window.plotter.renderers[0].actors)
        self.assertIn('slice', window.plotter.renderers[1].actors)
        window.display_panel.threshold_slider.setValue(250)
        self.assertEqual(window.display_panel.threshold.value(), 5.)
        self.assertEqual(window.display_panel.threshold_slider_max, 20.)
        window.display_panel.threshold_slider.setValue(1000)
        self.assertEqual(window.display_panel.threshold.value(), 20.)
        window.display_panel.threshold_slider.setValue(0)
        self.assertEqual(window.scene.view['thresholds']['field:alpha'], 0.)

    def test_threshold_range_follows_result_and_restores_saved_value(self):
        window = self.window
        window.display_panel.threshold.setValue(12.)
        window.select('component', 'eta')
        # The controls continue to describe the displayed result while a job is pending.
        self.assertEqual(window.display_panel.threshold.value(), 12.)
        updated = deepcopy(self.result)
        updated.update(component='eta', scale={'limit': 1., 'peak': 1., 'threshold': .25})
        self.runner.events = [(self.runner.token, 'result', updated)]
        window.poll()
        self.assertEqual(window.display_panel.threshold.value(), .25)
        self.assertAlmostEqual(window.display_panel.threshold_slider_max, 1.01)
        window.select('component', 'alpha')
        self.runner.events = [(self.runner.token, 'result', self.result)]
        window.poll()
        self.assertEqual(window.display_panel.threshold.value(), 12.)
        self.assertEqual(window.display_panel.threshold_slider.value(), 1000)
        saved = window.saved_recipe()
        window.new_session(saved)
        self.assertFalse(window.display_panel.threshold.isEnabled())
        self.assertFalse(window.display_panel.threshold_slider.isEnabled())
        self.runner.events = [(self.runner.token, 'result', self.result)]
        window.poll()
        self.assertTrue(window.display_panel.threshold_slider.isEnabled())
        self.assertEqual(window.display_panel.threshold.value(), 12.)
        self.assertEqual(window.display_panel.threshold_slider_max, 12.)

    def test_threshold_slider_handles_zero_invalid_and_small_values(self):
        window = self.window
        for peak, threshold, invalid in ((0., 0., False), (0., 0., True), (8e-12, 2e-12, False)):
            updated = deepcopy(self.result)
            updated['values'] *= peak / 8
            if invalid:
                updated['values'][:] = np.nan
            updated['scale'].update(peak=peak, threshold=threshold)
            window.scene.view['thresholds']['field:alpha'] = threshold
            window.pending = deepcopy(window.group)
            window.accept_result(updated)
            window.display_panel.threshold_slider.setValue(500)
            self.assertGreater(window.display_panel.threshold.value(), 0.)
            self.assertAlmostEqual(window.display_panel.threshold.value() / window.display_panel.threshold_slider_max, .5)
            self.assertEqual(window.scene.view['thresholds']['field:alpha'], window.display_panel.threshold.value())

    def test_manual_slider_range_clamps_threshold_and_preserves_layout_state(self):
        window = self.window
        window.analysis_form.fields['geometry_delta'].setText('.001')
        cameras = window.scene.camera_state()
        window.display_panel.auto_threshold_range.setChecked(False)
        window.display_panel.threshold_slider_upper.setText('1e-2')
        with patch.object(window.scene, 'update_display', wraps=window.scene.update_display) as update:
            window.display_panel.threshold_slider_upper.editingFinished.emit()
            update.assert_called_once()
        self.assertEqual(window.display_panel.threshold.value(), .01)
        self.assertEqual(window.display_panel.threshold_slider_max, .01)
        window.display_panel.threshold_slider.setValue(250)
        self.assertEqual(window.display_panel.threshold.value(), .0025)
        window.set_layout('three_d_slice')
        self.assertEqual(window.display_panel.threshold_slider_max, .01)
        self.assertEqual(window.scene.camera_state(), cameras)
        self.assertFalse(self.runner.requests)
        self.assertEqual(window.analysis_form.fields['geometry_delta'].text(), '.001')
        # A larger explicit threshold extends the manual range without losing the input.
        window.display_panel.threshold.setValue(.05)
        self.assertEqual(window.scene.view['threshold_slider_limits']['field:alpha'], .05)
        window.display_panel.auto_threshold_range.setChecked(True)
        self.assertNotIn('field:alpha', window.scene.view['threshold_slider_limits'])
        self.assertAlmostEqual(window.display_panel.threshold_slider_max, 8.08)
        self.assertEqual(window.display_panel.threshold.value(), .05)
        self.assertFalse(window.display_panel.threshold_slider_upper.isEnabled())

    def test_manual_slider_range_survives_case_diagnostic_and_session_changes(self):
        window = self.window
        window.display_panel.auto_threshold_range.setChecked(False)
        window.display_panel.threshold_slider_upper.setText('20')
        window.display_panel.threshold_slider_upper.editingFinished.emit()
        window.select('case', window.group['cases'][1]['id'])
        updated = deepcopy(self.result)
        updated['case'] = window.group['cases'][1]['id']
        self.runner.events = [(self.runner.token, 'result', updated)]
        window.poll()
        self.assertEqual(window.display_panel.threshold_slider_max, 20.)
        window.select('component', 'eta')
        updated.update(component='eta', scale={'limit': 1., 'peak': 1., 'threshold': .25})
        self.runner.events = [(self.runner.token, 'result', updated)]
        window.poll()
        self.assertTrue(window.display_panel.auto_threshold_range.isChecked())
        self.assertAlmostEqual(window.display_panel.threshold_slider_max, 1.01)
        window.select('component', 'alpha')
        updated.update(component='alpha', scale=self.result['scale'])
        self.runner.events = [(self.runner.token, 'result', updated)]
        window.poll()
        self.assertEqual(window.display_panel.threshold_slider_max, 20.)
        saved = window.saved_recipe()
        self.assertEqual(saved['groups'][0]['view']['threshold_slider_limits'], {'field:alpha': 20.})
        window.new_session(saved)
        self.assertFalse(window.display_panel.auto_threshold_range.isEnabled())
        self.assertFalse(window.display_panel.threshold_slider_upper.isEnabled())
        self.runner.events = [(self.runner.token, 'result', updated)]
        window.poll()
        self.assertFalse(window.display_panel.auto_threshold_range.isChecked())
        self.assertTrue(window.display_panel.threshold_slider_upper.isEnabled())
        self.assertEqual(window.display_panel.threshold_slider_max, 20.)
        self.assertEqual(window.display_panel.threshold.value(), 2.)

    def test_manual_slider_range_rejects_invalid_input_and_accepts_small_scales(self):
        window = self.window
        window.display_panel.auto_threshold_range.setChecked(False)
        previous = window.display_panel.threshold_slider_max
        for text in ('', 'bad', '0', '-1', 'nan', 'inf', '1e101'):
            with self.subTest(text=text):
                window.display_panel.threshold_slider_upper.setText(text)
                window.display_panel.threshold_slider_upper.editingFinished.emit()
                self.assertIn('Slider upper bound must', window.status.text())
                self.assertEqual(window.display_panel.threshold_slider_max, previous)
                self.assertEqual(window.display_panel.threshold.value(), 2.)
        window.display_panel.threshold_slider_upper.setText('1e-12')
        window.display_panel.threshold_slider_upper.editingFinished.emit()
        self.assertEqual(window.display_panel.threshold_slider_max, 1e-12)
        self.assertEqual(window.display_panel.threshold.value(), 1e-12)
        window.display_panel.threshold_slider.setValue(500)
        self.assertEqual(window.display_panel.threshold.value(), 5e-13)
        self.assertFalse(self.runner.requests)

    def test_changed_display_units_reset_manual_slider_range(self):
        window = self.window
        window.display_panel.auto_threshold_range.setChecked(False)
        window.display_panel.threshold_slider_upper.setText('20')
        window.display_panel.threshold_slider_upper.editingFinished.emit()
        window.analysis_form.fields['length_unit'].setText('km')
        window.apply()
        updated = deepcopy(self.result)
        updated['analysis']['length_unit'] = 'km'
        self.runner.events = [(self.runner.token, 'result', updated)]
        window.poll()
        self.assertEqual(window.scene.view['threshold_slider_limits'], {})
        self.assertTrue(window.display_panel.auto_threshold_range.isChecked())
        self.assertAlmostEqual(window.display_panel.threshold_slider_max, 8.08)

    def test_cancel_and_save_keep_committed_analysis(self):
        self.window.analysis_form.fields['geometry_delta'].setText('.001')
        self.window.apply()
        self.assertEqual(self.runner.requests[-1]['analysis']['geometry_delta'], .001)
        self.window.cancel()
        self.assertIsNone(self.window.pending)
        saved = self.window.saved_recipe()
        self.assertEqual(saved['groups'][0]['analysis']['geometry_delta'], .002)
        self.assertEqual(self.window.analysis_form.fields['geometry_delta'].text(), '.001')

    def test_late_completion_preserves_newer_form_edits_and_view(self):
        self.window.analysis_form.fields['geometry_delta'].setText('.001')
        self.window.apply()
        self.window.analysis_form.fields['geometry_delta'].setText('.0005')
        self.window.set_layout('three_d_slice')
        updated = deepcopy(self.result)
        updated['analysis']['geometry_delta'] = .001
        updated['resolved']['geometry_delta'] = .001
        self.runner.events = [(self.runner.token, 'result', updated)]
        self.window.poll()
        self.assertEqual(self.window.scene.view['layout'], 'three_d_slice')
        self.assertEqual(self.window.analysis_form.fields['geometry_delta'].text(), '.0005')
        self.assertEqual(self.window.saved_recipe()['groups'][0]['analysis']['geometry_delta'], .001)

    def test_removing_case_does_not_restore_deleted_selection(self):
        removed = self.window.group['view']['case']
        self.window.remove_case()
        self.assertNotEqual(self.window.group['view']['case'], removed)
        self.assertNotIn(removed, [case['id'] for case in self.window.group['cases']])
        self.assertEqual(self.window.group['view']['case'], self.window.group['reference'])

    def test_invalid_apply_does_not_submit_or_clear_scene(self):
        count = len(self.runner.requests)
        self.window.analysis_form.fields['max_points'].setText('2')
        self.window.apply()
        self.assertEqual(len(self.runner.requests), count)
        self.assertIs(self.window.scene.result, self.result)
        self.assertIn('max_points', self.window.status.text())

    def test_fac_override_is_optional_and_geometry_remains_the_base(self):
        form = self.window.analysis_form
        self.assertEqual(form.fields['geometry_delta'].text(), '0.002')
        self.assertFalse(form.fac_override.isChecked())
        self.assertTrue(form.fields['delta'].isHidden())
        form.fields['geometry_delta'].setText('.01')
        self.window.apply()
        self.assertEqual(self.runner.requests[-1]['analysis']['geometry_delta'], .01)
        self.assertIsNone(self.runner.requests[-1]['analysis']['delta'])
        self.window.cancel()
        form.fac_override.setChecked(True)
        self.assertFalse(form.fields['delta'].isHidden())
        form.fields['delta'].setText('.02 .03 .04')
        self.window.apply()
        self.assertEqual(self.runner.requests[-1]['analysis']['delta'], [.02, .03, .04])
        self.assertEqual(self.runner.requests[-1]['analysis']['geometry_delta'], .01)
        self.window.cancel()
        form.fac_override.setChecked(False)
        self.assertIsNone(form.analysis()['delta'])
        form.fac_override.setChecked(True)
        form.fields['evaluation'].setCurrentIndex(form.fields['evaluation'].findData('grid'))
        self.assertFalse(form.fac_override.isEnabled())
        self.assertTrue(form.fields['delta'].isHidden())
        self.assertIsNone(form.analysis()['delta'])
        form.fields['evaluation'].setCurrentIndex(form.fields['evaluation'].findData('direct'))
        self.assertEqual(form.analysis()['delta'], [.02, .03, .04])

    def test_restored_fac_override_is_visible_and_can_be_discarded(self):
        form = self.window.analysis_form
        analysis = deepcopy(self.session['groups'][0]['analysis'])
        analysis.update(geometry_delta=.02, delta=.03)
        form.set_analysis(analysis)
        self.assertTrue(form.fac_override.isChecked())
        self.assertFalse(form.fields['delta'].isHidden())
        self.assertEqual(form.analysis()['delta'], .03)
        self.window.discard()
        self.assertFalse(form.fac_override.isChecked())
        self.assertTrue(form.fields['delta'].isHidden())

    def test_fac_result_displays_its_effective_steps(self):
        for evaluation, steps, label in (
            ('direct', [.01, .02, .03], 'FAC steps 0.01, 0.02, 0.03'),
            ('direct', .002, 'FAC steps 0.002'),
            ('grid', None, 'FAC steps: preview axis spacing'),
        ):
            with self.subTest(evaluation=evaluation, steps=steps):
                result = deepcopy(self.result)
                result['component'] = 'fac'
                result['analysis'].update(evaluation=evaluation, delta=steps)
                result['resolved']['fac_delta'] = steps
                self.window.pending = deepcopy(self.window.group)
                self.window.pending['analysis'] = result['analysis']
                self.window.accept_result(result)
                self.assertIn(label, self.window.header.text())
                self.assertIn(label, self.window.plotter.renderers[0].actors['title'].GetInput())

    def test_model_source_edits_and_dipole_epochs_work_for_every_model(self):
        from mageometry.session.specs import model_source, model_case_label
        from PySide6.QtWidgets import QDialog
        for model in ('t89', 't01', 't04', 't96'):
            with self.subTest(model=model):
                self.window.assign_dipoles()
                source = model_source(model=model, parameters={'epoch': 1234.})
                with patch('mageometry.gui.window.SourceDialog') as dialog:
                    dialog.return_value.exec.return_value = QDialog.DialogCode.Accepted
                    dialog.return_value.sources = [source]
                    self.window.edit_source()
                selected = self.window.selected_case()
                self.assertEqual(selected['source'], source)
                self.assertEqual(selected['label'], model_case_label(source))
                self.assertEqual(selected['background']['parameters']['epoch'], 1234.)
                self.assertIn(model, self.window.source_info.text())
                self.assertIn('1234', self.window.source_info.text())
                if model == 't89':
                    self.assertIn('iopt', self.window.source_info.text())
                    self.assertNotIn('IMF', self.window.source_info.text())
                elif model == 't01':
                    self.assertIn('G1/G2', self.window.source_info.text())
                elif model == 't04':
                    self.assertIn('W1-W6', self.window.source_info.text())
                self.window.apply()
                self.assertEqual(self.runner.requests[-1]['cases'][0]['source']['kind'], model)

    def test_display_panel_requires_a_prepared_result(self):
        from mageometry.session import empty_session

        session = deepcopy(self.window.session)
        self.window.new_session(empty_session())
        self.assertFalse(self.window.display_panel.isEnabled())
        self.assertFalse(self.window.display_panel.auto_limit.isEnabled())
        self.assertFalse(self.window.display_panel.layers['plane'].isEnabled())
        self.window.new_session(session)
        self.assertFalse(self.window.display_panel.isEnabled())
        self.window.accept_result(self.result)
        self.assertTrue(self.window.display_panel.isEnabled())
        self.assertTrue(self.window.display_panel.auto_limit.isEnabled())
        self.assertTrue(self.window.display_panel.layers['plane'].isEnabled())

    def test_new_file_group_can_start_with_any_model_and_model_units(self):
        from mageometry.session import empty_session
        from mageometry.session.specs import model_source
        from PySide6.QtWidgets import QDialog
        for model in ('t89', 't01', 't04'):
            with self.subTest(model=model):
                self.window.new_session(empty_session())
                with patch('mageometry.gui.window.SourceDialog') as dialog:
                    dialog.return_value.exec.return_value = QDialog.DialogCode.Accepted
                    dialog.return_value.sources = [model_source(model=model)]
                    self.window.add_source()
                analysis = self.window.group['analysis']
                self.assertEqual(analysis['evaluation'], 'direct')
                self.assertEqual(analysis['current_scale'], .125)
                self.assertEqual(analysis['current_unit'], 'nA/m^2')
                self.assertIn('2 nT/Re', self.window.analysis_form.fields['current_scale'].toolTip())
                self.window.apply()
                self.assertEqual(self.runner.requests[-1]['cases'][0]['source']['kind'], model)

    def test_adding_files_to_model_group_uses_separate_grid_defaults(self):
        from PySide6.QtWidgets import QDialog
        window = self.window
        original = deepcopy(window.group)
        cameras = window.scene.camera_state()
        window.analysis_form.fields['max_points'].setText('4321')
        source = dict(kind='xdmf', path='/tmp/ea01sw000.xmf', options={'stride': 4},
                      metadata={'length_unit': 'km'})
        with patch('mageometry.gui.window.SourceDialog') as dialog:
            dialog.return_value.exec.return_value = QDialog.DialogCode.Accepted
            dialog.return_value.sources = [source]
            window.add_source()
        self.assertEqual(len(window.session['groups']), 2)
        previous = window.session['groups'][0]
        self.assertEqual(previous['cases'], original['cases'])
        self.assertEqual(previous['analysis']['evaluation'], 'direct')
        self.assertEqual(previous['analysis']['current_scale'], .125)
        self.assertEqual(previous['analysis']['max_points'], 4321)
        self.assertEqual(previous['view']['cameras'], cameras)
        self.assertIs(window.scene.result, self.result)
        self.assertFalse(self.runner.requests)
        group = window.group
        self.assertNotEqual(group['id'], original['id'])
        self.assertEqual(group['cases'][0]['source'], source)
        self.assertEqual(group['view']['case'], group['cases'][0]['id'])
        self.assertIsNone(group['view']['origin'])
        analysis = window.analysis_form.analysis()
        self.assertEqual(analysis['evaluation'], 'grid')
        self.assertEqual(analysis['current_scale'], 1.)
        self.assertIsNone(analysis['current_unit'])
        self.assertEqual(analysis['length_unit'], 'km')
        self.assertIsNone(analysis['geometry_delta'])
        self.assertIsNone(analysis['seeds'])
        self.assertEqual(analysis['mask_radius'], 0.)
        self.assertIsNone(analysis['planet_radius'])
        self.assertIn('separate', window.status.text())
        window.apply()
        self.assertFalse(window.last_error)
        self.assertEqual(len(self.runner.requests[-1]['cases']), 1)
        self.assertEqual(self.runner.requests[-1]['analysis']['evaluation'], 'grid')
        self.assertEqual(self.runner.requests[-1]['cases'][0]['source']['options']['stride'], 4)
        # Subsequent files remain in the file comparison group.
        with patch('mageometry.gui.window.SourceDialog') as dialog:
            dialog.return_value.exec.return_value = QDialog.DialogCode.Accepted
            dialog.return_value.sources = [dict(source, path='/tmp/ea01sw001.xmf')]
            window.add_source()
        self.assertEqual(len(window.session['groups']), 2)
        self.assertEqual(len(window.group['cases']), 2)

    def test_adding_models_to_file_group_retains_file_settings(self):
        from PySide6.QtWidgets import QDialog
        from mageometry.session import empty_session
        from mageometry.session.specs import model_source
        self.window.new_session(empty_session())
        with patch('mageometry.gui.window.SourceDialog') as dialog:
            dialog.return_value.exec.return_value = QDialog.DialogCode.Accepted
            dialog.return_value.sources = [dict(kind='xdmf', path='/tmp/file.xmf', options={'stride': 4})]
            self.window.add_source()
            file_group = deepcopy(self.window.group)
            dialog.return_value.sources = [model_source(by=by) for by in (-5., 5.)]
            self.window.add_source()
        self.assertEqual(len(self.window.session['groups']), 2)
        previous = self.window.session['groups'][0]
        self.assertEqual(previous['cases'], file_group['cases'])
        self.assertEqual(previous['analysis']['evaluation'], 'grid')
        self.assertEqual(previous['analysis']['current_scale'], 1.)
        self.assertEqual(self.window.group['analysis']['evaluation'], 'direct')
        self.assertEqual(self.window.group['analysis']['current_scale'], .125)
        self.assertEqual(len(self.window.group['cases']), 2)
        self.window.apply()
        self.assertFalse(self.window.last_error)
        self.assertEqual(len(self.runner.requests[-1]['cases']), 2)

    def test_along_field_diagnostics_select_and_show_derivative_steps(self):
        for component, evaluation, label in (
            ('dalpha_ds', 'direct', 'alpha and along-B steps 0.002'),
            ('dalpha_ds_over_B', 'direct', 'alpha and along-B steps 0.002'),
            ('dfac_ds', 'direct', 'along-B step 0.002; FAC steps 0.01, 0.02, 0.03'),
            ('dalpha_ds', 'grid', 'along-B gradient: preview axis spacing; alpha step 0.002'),
            ('dfac_ds', 'grid', 'along-B gradient: preview axis spacing'),
        ):
            with self.subTest(component=component, evaluation=evaluation):
                index = self.window.components.findData(component)
                self.assertGreaterEqual(index, 0)
                self.window.components.setCurrentIndex(index)
                self.assertEqual(self.runner.requests[-1]['view']['component'], component)
                result = deepcopy(self.result)
                result['component'] = component
                result['analysis']['evaluation'] = evaluation
                result['resolved']['fac_delta'] = [.01, .02, .03]
                self.window.pending = deepcopy(self.window.group)
                self.window.pending['analysis'] = result['analysis']
                self.window.accept_result(result)
                self.assertIn(label, self.window.header.text())
                self.assertIn(label, self.window.plotter.renderers[0].actors['title'].GetInput())
                self.assertFalse(self.window.display_panel.layers['arrows'].isEnabled())
                self.assertEqual(self.window.saved_recipe()['groups'][0]['view']['component'], component)

    def test_window_size_and_dock_state_are_saved_and_restored(self):
        self.window.resize(1650, 1000)
        self.window.toggle_panels()
        recipe = self.window.saved_recipe()
        self.assertEqual(recipe['window']['size'], [1650, 1000])
        self.assertTrue(recipe['window']['state'])
        self.window.resize(1500, 940)
        self.window.session = recipe
        self.window.restore_window()
        self.assertEqual([self.window.width(), self.window.height()], [1650, 1000])
        self.assertTrue(self.window.case_dock.isHidden())
        self.assertTrue(self.window.settings_dock.isHidden())

    def test_open_session_retains_lighting_before_and_after_preparation(self):
        recipe = self.window.saved_recipe()
        self.window.plotter.renderers[0].lights[0].intensity = .37
        lights = [list(renderer.lights) for renderer in self.window.plotter.renderers]
        self.window.new_session(recipe)
        self.assertIsNone(self.window.scene.result)
        self.assertIsNone(self.window.scene.widget)
        for renderer, expected in zip(self.window.plotter.renderers, lights):
            self.assertEqual(renderer.lights, expected)
        self.window.accept_result(deepcopy(self.result))
        self.assertIsNotNone(self.window.scene.result)
        for renderer, expected in zip(self.window.plotter.renderers, lights):
            self.assertEqual(renderer.lights, expected)
        self.assertAlmostEqual(self.window.plotter.renderers[0].lights[0].intensity, .37)


if __name__ == '__main__':
    unittest.main()
