"""Qt controller state with a real off-screen VTK scene and deterministic jobs."""

from copy import deepcopy
import importlib.util
import os
import unittest
from unittest.mock import patch

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
        self.window.set_layout('three_d')
        self.window.toggle_slice()
        self.assertEqual(self.window.scene.view['layout'], 'slice')
        self.window.toggle_slice()
        self.assertEqual(self.window.scene.view['layout'], 'three_d')
        self.window.toggle_panels()
        self.assertTrue(self.window.scene.view['panels_hidden'])
        self.window.restore_layout()
        self.assertFalse(self.window.scene.view['panels_hidden'])
        self.assertEqual(self.window.scene.camera_state(), cameras)
        self.assertFalse(self.runner.requests)
        self.assertEqual(self.window.analysis_form.fields['geometry_delta'].text(), '.001')
        self.assertIsNone(self.window.saved_recipe()['groups'][0]['analysis']['geometry_delta'])

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

    def test_cancel_and_save_keep_committed_analysis(self):
        self.window.analysis_form.fields['geometry_delta'].setText('.001')
        self.window.apply()
        self.assertEqual(self.runner.requests[-1]['analysis']['geometry_delta'], .001)
        self.window.cancel()
        self.assertIsNone(self.window.pending)
        saved = self.window.saved_recipe()
        self.assertIsNone(saved['groups'][0]['analysis']['geometry_delta'])
        self.assertEqual(self.window.analysis_form.fields['geometry_delta'].text(), '.001')

    def test_late_completion_preserves_newer_form_edits_and_view(self):
        self.window.analysis_form.fields['geometry_delta'].setText('.001')
        self.window.apply()
        self.window.analysis_form.fields['geometry_delta'].setText('.0005')
        self.window.set_layout('slice')
        updated = deepcopy(self.result)
        updated['analysis']['geometry_delta'] = .001
        updated['resolved']['geometry_delta'] = .001
        self.runner.events = [(self.runner.token, 'result', updated)]
        self.window.poll()
        self.assertEqual(self.window.scene.view['layout'], 'slice')
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


if __name__ == '__main__':
    unittest.main()
