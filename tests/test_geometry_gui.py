"""Qt controller state with a real off-screen VTK scene and deterministic jobs."""

from copy import deepcopy
import importlib.util
import os
import unittest
from unittest.mock import patch

import numpy as np

from mageometry.session import model_session
from test_session_scene import packet

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

    def test_layouts_do_not_submit_jobs_or_apply_draft(self):
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

    def test_threshold_controls_update_display_without_applying_analysis(self):
        window = self.window
        window.analysis_form.fields['geometry_delta'].setText('.001')
        cameras = window.scene.camera_state()
        original_slice = window.plotter.renderers[1].actors['slice'].mapper.dataset['value'].copy()
        self.assertEqual(window.threshold.value(), 2.)
        with patch.object(window.scene, 'update_display', wraps=window.scene.update_display) as update:
            window.threshold.setValue(1.234567891)
            self.assertEqual(window.scene.view['thresholds']['field:alpha'], 1.234567891)
            update.assert_called_once()
            self.assertAlmostEqual(window.threshold_slider.value() / 1000 * window.threshold_slider_max,
                                   window.threshold.value(), delta=window.threshold_slider_max / 2000)
            update.reset_mock()
            window.threshold_slider.setValue(500)
            self.assertAlmostEqual(window.threshold.value(), 4.04)
            self.assertEqual(window.scene.view['thresholds']['field:alpha'], window.threshold.value())
            update.assert_called_once()
        self.assertEqual(cameras, window.scene.camera_state())
        np.testing.assert_allclose(window.plotter.renderers[1].actors['slice'].mapper.dataset['value'],
                                   original_slice, equal_nan=True)
        self.assertFalse(self.runner.requests)
        self.assertEqual(window.analysis_form.fields['geometry_delta'].text(), '.001')
        saved = window.saved_recipe()['groups'][0]
        self.assertEqual(saved['analysis']['geometry_delta'], .002)
        self.assertEqual(saved['view']['thresholds']['field:alpha'], window.threshold.value())

    def test_manual_threshold_expands_slider_and_keeps_range_stable(self):
        window = self.window
        window.threshold.setValue(20.)
        self.assertEqual(window.threshold_slider.value(), window.threshold_slider.maximum())
        self.assertEqual(window.threshold_slider_max, 20.)
        self.assertNotIn('positive', window.plotter.renderers[0].actors)
        self.assertIn('slice', window.plotter.renderers[1].actors)
        window.threshold_slider.setValue(250)
        self.assertEqual(window.threshold.value(), 5.)
        self.assertEqual(window.threshold_slider_max, 20.)
        window.threshold_slider.setValue(1000)
        self.assertEqual(window.threshold.value(), 20.)
        window.threshold_slider.setValue(0)
        self.assertEqual(window.scene.view['thresholds']['field:alpha'], 0.)

    def test_threshold_range_follows_result_and_restores_saved_value(self):
        window = self.window
        window.threshold.setValue(12.)
        window.select('component', 'eta')
        # The controls continue to describe the displayed result while a job is pending.
        self.assertEqual(window.threshold.value(), 12.)
        updated = deepcopy(self.result)
        updated.update(component='eta', scale={'limit': 1., 'peak': 1., 'threshold': .25})
        self.runner.events = [(self.runner.token, 'result', updated)]
        window.poll()
        self.assertEqual(window.threshold.value(), .25)
        self.assertAlmostEqual(window.threshold_slider_max, 1.01)
        window.select('component', 'alpha')
        self.runner.events = [(self.runner.token, 'result', self.result)]
        window.poll()
        self.assertEqual(window.threshold.value(), 12.)
        self.assertEqual(window.threshold_slider.value(), 1000)
        saved = window.saved_recipe()
        window.new_session(saved)
        self.assertFalse(window.threshold.isEnabled())
        self.assertFalse(window.threshold_slider.isEnabled())
        self.runner.events = [(self.runner.token, 'result', self.result)]
        window.poll()
        self.assertTrue(window.threshold_slider.isEnabled())
        self.assertEqual(window.threshold.value(), 12.)
        self.assertEqual(window.threshold_slider_max, 12.)

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
            window.threshold_slider.setValue(500)
            self.assertGreater(window.threshold.value(), 0.)
            self.assertAlmostEqual(window.threshold.value() / window.threshold_slider_max, .5)
            self.assertEqual(window.scene.view['thresholds']['field:alpha'], window.threshold.value())

    def test_manual_slider_range_clamps_threshold_and_preserves_layout_state(self):
        window = self.window
        window.analysis_form.fields['geometry_delta'].setText('.001')
        cameras = window.scene.camera_state()
        window.auto_threshold_range.setChecked(False)
        window.threshold_slider_upper.setText('1e-2')
        with patch.object(window.scene, 'update_display', wraps=window.scene.update_display) as update:
            window.threshold_slider_upper.editingFinished.emit()
            update.assert_called_once()
        self.assertEqual(window.threshold.value(), .01)
        self.assertEqual(window.threshold_slider_max, .01)
        window.threshold_slider.setValue(250)
        self.assertEqual(window.threshold.value(), .0025)
        window.set_layout('three_d_slice')
        self.assertEqual(window.threshold_slider_max, .01)
        self.assertEqual(window.scene.camera_state(), cameras)
        self.assertFalse(self.runner.requests)
        self.assertEqual(window.analysis_form.fields['geometry_delta'].text(), '.001')
        # A larger explicit threshold extends the manual range without losing the input.
        window.threshold.setValue(.05)
        self.assertEqual(window.scene.view['threshold_slider_limits']['field:alpha'], .05)
        window.auto_threshold_range.setChecked(True)
        self.assertNotIn('field:alpha', window.scene.view['threshold_slider_limits'])
        self.assertAlmostEqual(window.threshold_slider_max, 8.08)
        self.assertEqual(window.threshold.value(), .05)
        self.assertFalse(window.threshold_slider_upper.isEnabled())

    def test_manual_slider_range_survives_case_diagnostic_and_session_changes(self):
        window = self.window
        window.auto_threshold_range.setChecked(False)
        window.threshold_slider_upper.setText('20')
        window.threshold_slider_upper.editingFinished.emit()
        window.select('case', window.group['cases'][1]['id'])
        updated = deepcopy(self.result)
        updated['case'] = window.group['cases'][1]['id']
        self.runner.events = [(self.runner.token, 'result', updated)]
        window.poll()
        self.assertEqual(window.threshold_slider_max, 20.)
        window.select('component', 'eta')
        updated.update(component='eta', scale={'limit': 1., 'peak': 1., 'threshold': .25})
        self.runner.events = [(self.runner.token, 'result', updated)]
        window.poll()
        self.assertTrue(window.auto_threshold_range.isChecked())
        self.assertAlmostEqual(window.threshold_slider_max, 1.01)
        window.select('component', 'alpha')
        updated.update(component='alpha', scale=self.result['scale'])
        self.runner.events = [(self.runner.token, 'result', updated)]
        window.poll()
        self.assertEqual(window.threshold_slider_max, 20.)
        saved = window.saved_recipe()
        self.assertEqual(saved['groups'][0]['view']['threshold_slider_limits'], {'field:alpha': 20.})
        window.new_session(saved)
        self.assertFalse(window.auto_threshold_range.isEnabled())
        self.assertFalse(window.threshold_slider_upper.isEnabled())
        self.runner.events = [(self.runner.token, 'result', updated)]
        window.poll()
        self.assertFalse(window.auto_threshold_range.isChecked())
        self.assertTrue(window.threshold_slider_upper.isEnabled())
        self.assertEqual(window.threshold_slider_max, 20.)
        self.assertEqual(window.threshold.value(), 2.)

    def test_manual_slider_range_rejects_invalid_input_and_accepts_small_scales(self):
        window = self.window
        window.auto_threshold_range.setChecked(False)
        previous = window.threshold_slider_max
        for text in ('', 'bad', '0', '-1', 'nan', 'inf', '1e101'):
            with self.subTest(text=text):
                window.threshold_slider_upper.setText(text)
                window.threshold_slider_upper.editingFinished.emit()
                self.assertIn('Slider upper bound must', window.status.text())
                self.assertEqual(window.threshold_slider_max, previous)
                self.assertEqual(window.threshold.value(), 2.)
        window.threshold_slider_upper.setText('1e-12')
        window.threshold_slider_upper.editingFinished.emit()
        self.assertEqual(window.threshold_slider_max, 1e-12)
        self.assertEqual(window.threshold.value(), 1e-12)
        window.threshold_slider.setValue(500)
        self.assertEqual(window.threshold.value(), 5e-13)
        self.assertFalse(self.runner.requests)

    def test_changed_display_units_reset_manual_slider_range(self):
        window = self.window
        window.auto_threshold_range.setChecked(False)
        window.threshold_slider_upper.setText('20')
        window.threshold_slider_upper.editingFinished.emit()
        window.analysis_form.fields['length_unit'].setText('km')
        window.apply()
        updated = deepcopy(self.result)
        updated['analysis']['length_unit'] = 'km'
        self.runner.events = [(self.runner.token, 'result', updated)]
        window.poll()
        self.assertEqual(window.scene.view['threshold_slider_limits'], {})
        self.assertTrue(window.auto_threshold_range.isChecked())
        self.assertAlmostEqual(window.threshold_slider_max, 8.08)

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
