"""The standalone CLI shares one dispatch path across all launchers."""

import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

from mageometry.viz3d import cli


class TestViewerCLI(unittest.TestCase):
    def test_no_trace_passes_empty_seeds_and_zero_automatic_count(self):
        with patch.object(cli, 'make_cases', return_value={'case': object()}), \
                patch.object(cli, 'make_fields'), \
                patch.object(cli.viz3d, 'compare_geometry') as render, patch('builtins.print'):
            cli.main(['--by', '-5', '5', '--no-trace'])
        self.assertEqual(render.call_args.kwargs['seeds'], [])
        self.assertEqual(render.call_args.kwargs['n_lines'], 0)

    def test_module_help_does_not_import_rendering_dependencies(self):
        code = """
import importlib.abc
import runpy
import sys
class NoRenderer(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('pyvista', 'pyvistaqt', 'PySide6'):
            raise AssertionError('Unexpected rendering dependency: ' + fullname)
sys.meta_path.insert(0, NoRenderer())
sys.argv = ['mageometry.viz3d', '--help']
runpy.run_module('mageometry.viz3d', run_name='__main__')
"""
        result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                                text=True, cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ('--xmf', '--by', '--background', '--screenshot'):
            self.assertIn(option, result.stdout)

    def test_common_launcher_supports_comparison_and_selected_case(self):
        with patch.object(cli, 'make_cases', return_value={'case': object()}) as cases, \
                patch.object(cli, 'make_fields') as fields, \
                patch.object(cli.viz3d, 'compare_geometry') as render, patch('builtins.print'):
            cli.main(['--by', '-5', '5', '--initial-by', '5', '--component', 'beta_g',
                  '--shape', '9', '7', '7', '--color-limit', '.1', '--cache-size', '1'])
        cases.assert_called_once_with([-5., 5.], (9, 7, 7))
        fields.assert_called_once_with([-5., 5.])
        options = render.call_args.kwargs
        self.assertEqual(options['initial_case'], 'IMF By = +5 nT')
        self.assertEqual(options['color_limits'], {'beta_g': .1})
        self.assertEqual(options['cache_size'], 1)
        self.assertEqual(options['slice_normal'], 'x')
        self.assertEqual(options['slice_origin'], (-6., 0., 0.))
        self.assertEqual(options['seeds'].shape, (6, 3))

    def test_single_model_supports_resolution_and_evaluation_options(self):
        for evaluation, delta in (('grid', None), ('direct', .001)):
            args = ['--shape', '9', '7', '7', '--evaluation', evaluation]
            expected = {'shape': (9, 7, 7)}
            if delta is not None:
                args += ['--delta', str(delta)]
                expected['delta'] = delta
            else:
                expected['evaluation'] = evaluation
            with self.subTest(evaluation=evaluation), \
                    patch.object(cli, 'model_snapshot', return_value=(object(), {})) as model, \
                    patch.object(cli.viz3d, 'geometry_view') as render, patch('builtins.print'):
                cli.main(args)
            model.assert_called_once_with(**expected)
            self.assertIsNone(render.call_args.kwargs['slice_normal'])

    def test_invalid_combinations_are_rejected_before_loading(self):
        runs = (
            ['--by', '-5', '5', '--vtk', 'field.vti'],
            ['--by', '0', '0'], ['--by', 'nan'], ['--shape', '2', '7', '7'],
            ['--by', '-5', '5', '--background', 'dipole'],
            ['--vtk', 'field.vti', '--evaluation', 'direct'],
            ['--vtk', 'field.vti', '--shape', '9', '7', '7'],
            ['--initial-by', '5'], ['--color-limit', '1'], ['--cache-size', '1'],
            ['--by', '-5', '5', '--initial-by', '0'],
            ['--by', '0', '--cache-size', '0'], ['--by', '0', '--color-limit', '-1'],
            ['--threshold', 'nan'], ['--max-points', '8'], ['--delta', '0'],
            ['--evaluation', 'grid', '--delta', '.002'],
            ['--origin', '0', '0', '0'],
        )
        for args in runs:
            with self.subTest(args=args), \
                    patch.object(cli, 'model_snapshot') as model, \
                    patch.object(cli, 'make_cases') as cases, \
                    patch.object(cli, 'load_vtk') as load, \
                    patch.object(sys, 'stderr', new_callable=io.StringIO), \
                    self.assertRaises(SystemExit) as error:
                cli.main(args)
            self.assertEqual(error.exception.code, 2)
            for unused in (model, cases, load):
                unused.assert_not_called()

    def test_screenshot_saves_and_closes_without_interactive_window(self):
        fake_pv = MagicMock()
        for args, loader, renderer, data in (
            ([], 'model_snapshot', 'geometry_view', (object(), {})),
            (['--vtk', 'field.vti'], 'load_vtk', 'geometry_view', object()),
            (['--by', '-5', '5', '--evaluation', 'grid'],
             'make_cases', 'compare_geometry', {'case': object()}),
        ):
            with self.subTest(args=args), patch.dict(sys.modules, pyvista=fake_pv), \
                    patch.object(cli, loader, return_value=data), \
                    patch.object(cli.viz3d, renderer) as render, patch('builtins.print'):
                cli.main([*args, '--screenshot', 'preview.png'])
            self.assertTrue(fake_pv.OFF_SCREEN)
            render.return_value.screenshot.assert_called_once_with('preview.png')
            render.return_value.close.assert_called_once_with()
            render.return_value.show.assert_not_called()

    def test_comparison_example_reuses_shared_model_helpers(self):
        from examples import compare_t96_by
        from mageometry.session import presets
        with patch.object(presets, 'make_cases') as cases, patch.object(presets, 'make_fields') as fields:
            self.assertIs(compare_t96_by.make_cases(shape=(9, 7, 7)), cases.return_value)
            self.assertIs(compare_t96_by.make_fields(), fields.return_value)
        cases.assert_called_once_with((-10., -5., 0., 5., 10.), (9, 7, 7), dst=-30., bz=-10.)
        fields.assert_called_once_with((-10., -5., 0., 5., 10.), dst=-30., bz=-10.)


if __name__ == '__main__':
    unittest.main()
