"""Desktop CLI options become portable sessions before any rendering starts."""

import io
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from mageometry.gui.cli import export_screenshot, main, session_from_args
from mageometry.session import load_session, model_session, save_session


class TestGUICLI(unittest.TestCase):
    def test_no_trace_cli_disables_calculation_without_erasing_model_seeds(self):
        session, _ = session_from_args(['--no-trace'])
        analysis = session['groups'][0]['analysis']
        self.assertFalse(analysis['trace_enabled'])
        self.assertTrue(analysis['seeds'])

    def test_along_field_diagnostics_survive_cli_and_session_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gradient.json'
            for component in ('dalpha_ds', 'dalpha_ds_over_B', 'dfac_ds'):
                with self.subTest(component=component):
                    session, _ = session_from_args(['--component', component])
                    self.assertEqual(session['groups'][0]['view']['component'], component)
                    save_session(session, path)
                    self.assertEqual(load_session(path)['groups'][0]['view']['component'], component)

    def test_help_without_optional_rendering_dependencies(self):
        code = """
import importlib.abc
import runpy
import sys
class NoRenderer(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('pyvista', 'pyvistaqt', 'PySide6'):
            raise AssertionError('Unexpected rendering dependency: ' + fullname)
sys.meta_path.insert(0, NoRenderer())
sys.argv = ['mageometry.gui', '--help']
runpy.run_module('mageometry.gui', run_name='__main__')
"""
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ('--layout', 'three_d_slice', '--by', '--xmf', '--session'):
            self.assertIn(option, result.stdout)

    def test_saved_session_keeps_analysis_and_accepts_layout_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'saved.json'
            expected = model_session((-1., 1.))
            expected['groups'][0]['view']['component'] = 'eta'
            save_session(expected, path)
            session, screenshot = session_from_args(['--session', str(path),
                                                     '--layout', 'three_d_slice'],
                                                    default_by=(-10., -5., 0., 5., 10.),
                                                    default_model_parameters={'dst': -30., 'bz': -10.})
        self.assertIsNone(screenshot)
        group = session['groups'][0]
        self.assertEqual(group['analysis'], expected['groups'][0]['analysis'])
        self.assertEqual(group['cases'], expected['groups'][0]['cases'])
        self.assertEqual(group['view']['component'], 'eta')
        self.assertEqual(group['view']['layout'], 'three_d_slice')
        self.assertEqual(group['view']['previous_layout'], 'three_d_slice')

    def test_empty_session_and_grid_model(self):
        empty, _ = session_from_args(['--empty'], default_by=(-5., 5.))
        self.assertEqual(empty['groups'][0]['cases'], [])
        model, _ = session_from_args(['--evaluation', 'grid'])
        self.assertEqual(model['groups'][0]['analysis']['evaluation'], 'grid')
        self.assertIsNone(model['groups'][0]['analysis']['delta'])
        self.assertIsNone(model['groups'][0]['analysis']['geometry_delta'])

    def test_geometry_is_primary_and_fac_delta_is_an_optional_override(self):
        for args, geometry, fac in (
            ([], .002, None),
            (['--geometry-delta', '.01'], .01, None),
            (['--fac-delta', '.03'], .002, .03),
            (['--delta', '.03'], .002, .03),
            (['--geometry-delta', '.01', '--fac-delta', '.03'], .01, .03),
        ):
            with self.subTest(args=args):
                session, _ = session_from_args(args)
                analysis = session['groups'][0]['analysis']
                self.assertEqual(analysis['geometry_delta'], geometry)
                self.assertEqual(analysis['delta'], fac)

    def test_invalid_options_fail_before_launch(self):
        for args in (
            ['--by', '0', '0'], ['--by', 'nan'], ['--shape', '2', '7', '7'],
            ['--by', '-5', '5', '--initial-by', '0'], ['--initial-by', '0'],
            ['--evaluation', 'grid', '--delta', '.002'],
            ['--threshold', 'nan'], ['--max-points', '8'], ['--delta', '0'],
            ['--contribution', 'residual'], ['--background', 'dipole', '--component', 'fac'],
            ['--component', 'q'], ['--component', 'sigma'],
            ['--empty', '--by', '5'], ['--session', 'saved.json', '--vtk', 'field.vti'],
            ['--empty', '--screenshot', 'empty.png'],
            ['--layout', 'three_d_slice', '--slice-only'],
        ):
            with self.subTest(args=args), patch('mageometry.gui.app.run') as launch, \
                    patch.object(sys, 'stderr', new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as error:
                    main(args)
                self.assertEqual(error.exception.code, 2)
                launch.assert_not_called()

    def test_screenshot_uses_desktop_recipe_without_launching_qt(self):
        with patch('mageometry.gui.cli.export_screenshot') as export, \
                patch('mageometry.gui.app.run') as launch:
            main(['--by', '-5', '5', '--background', 'dipole', '--contribution', 'residual',
                  '--layout', 'three_d_slice', '--screenshot', 'comparison.png'])
        launch.assert_not_called()
        session, path = export.call_args.args
        self.assertEqual(path, 'comparison.png')
        group = session['groups'][0]
        self.assertEqual(len(group['cases']), 2)
        self.assertEqual(group['view']['layout'], 'three_d_slice')
        self.assertEqual(group['view']['component'], 'eta')
        self.assertEqual(group['view']['contribution'], 'residual')

    @unittest.skipUnless(importlib.util.find_spec('pyvista'), 'pyvista unavailable')
    def test_export_saves_resolved_seeds_and_layout_without_mutating_input(self):
        from test_session_scene import packet

        session = model_session()
        group = session['groups'][0]
        group['analysis']['seeds'] = None
        group['view'].update(layout='three_d_slice', origin=[0., 0., 0.])
        result = packet()
        result.update(case=group['view']['case'], resolved={
            'seeds': [[1., 0., 0.]], 'geometry_delta': .002,
            'preview_shape': [7, 7, 7], 'inputs': {}})
        with tempfile.TemporaryDirectory() as directory, \
                patch('mageometry.session.SessionEngine') as engine:
            engine.return_value.prepare.return_value = result
            path = Path(directory) / 'view.png'
            export_screenshot(session, path)
            self.assertGreater(path.stat().st_size, 0)
            saved = load_session(path.with_suffix('.session.json'))['groups'][0]
        self.assertEqual(saved['view']['layout'], 'three_d_slice')
        self.assertEqual(saved['analysis']['seeds'], [[1., 0., 0.]])
        self.assertEqual(len(saved['view']['cameras']), 5)
        self.assertIsNone(group['analysis']['seeds'])
        self.assertNotIn('resolved', group)


if __name__ == '__main__':
    unittest.main()
