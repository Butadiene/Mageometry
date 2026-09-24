"""Simulation viewers require explicit input and preserve shared controls."""

import importlib
import io
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch


EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'


class TestGeometryViewerExamples(unittest.TestCase):

    def setUp(self):
        paths = patch.object(sys, 'path', [str(EXAMPLES)] + sys.path)
        paths.start()
        self.addCleanup(paths.stop)
        self.viewer = importlib.import_module('fac_viewer')

    def run_viewer(self, script, *args):
        with patch.object(sys, 'argv', [script, *args]), patch('builtins.print'):
            runpy.run_path(str(EXAMPLES / script), run_name='__main__')

    def test_simulation_viewer_requires_snapshot(self):
        with patch.object(self.viewer, 'load_xdmf') as xdmf, \
                patch.object(self.viewer, 'load_vtk') as vtk, \
                patch.object(self.viewer, 'load_hdf5') as hdf5, \
                patch.object(self.viewer, 'model_snapshot') as model, \
                patch.object(self.viewer.viz3d, 'geometry_view') as render, \
                patch.object(sys, 'stderr', new_callable=io.StringIO) as stderr:
            with self.assertRaises(SystemExit) as error:
                self.run_viewer('geometry_viewer_simulation.py')
        self.assertEqual(error.exception.code, 2)
        self.assertIn('specify a snapshot file with --xmf, --vtk, or --h5',
                      stderr.getvalue())
        for unused in (xdmf, vtk, hdf5, model, render):
            unused.assert_not_called()

    def test_simulation_viewer_loads_supplied_xmf(self):
        with patch.object(self.viewer, 'load_xdmf') as load, \
                patch.object(self.viewer.viz3d, 'geometry_view') as render, \
                patch.object(self.viewer, 'model_snapshot') as model:
            self.run_viewer('geometry_viewer_simulation.py', '--xmf', 'my run/step.xmf')
        load.assert_called_once_with('my run/step.xmf', h5_file=None, stride=1)
        model.assert_not_called()
        self.assertIs(render.call_args.args[0], load.return_value)
        self.assertEqual(render.call_args.kwargs['component'], 'alpha')
        render.return_value.show.assert_called_once_with()

    def test_simulation_viewer_accepts_loading_and_display_options(self):
        with patch.object(self.viewer, 'load_xdmf') as load, \
                patch.object(self.viewer.viz3d, 'geometry_view') as render:
            self.run_viewer('geometry_viewer_simulation.py', '--xmf', 'step.xmf',
                            '--h5', 'heavy.h5', '--stride', '4',
                            '--component', 'sigma', '--slice', 'x', '--slice-only')
        load.assert_called_once_with('step.xmf', h5_file='heavy.h5', stride=4)
        self.assertEqual(render.call_args.kwargs['component'], 'sigma')
        self.assertEqual(render.call_args.kwargs['slice_normal'], 'x')
        self.assertTrue(render.call_args.kwargs['slice_only'])

    def test_simulation_viewer_loads_supplied_vtk(self):
        with patch.object(self.viewer, 'load_xdmf') as xdmf, \
                patch.object(self.viewer, 'load_vtk') as vtk, \
                patch.object(self.viewer.viz3d, 'geometry_view') as render:
            self.run_viewer('geometry_viewer_simulation.py', '--vtk', 'step.vti')
        xdmf.assert_not_called()
        vtk.assert_called_once_with('step.vti', stride=1)
        self.assertIs(render.call_args.args[0], vtk.return_value)

    def test_simulation_viewer_loads_supplied_hdf5(self):
        with patch.object(self.viewer, 'load_xdmf') as xdmf, \
                patch.object(self.viewer, 'load_hdf5') as hdf5, \
                patch.object(self.viewer.viz3d, 'geometry_view') as render:
            self.run_viewer('geometry_viewer_simulation.py', '--h5', 'field.h5',
                            '--origin', '-2', '-3', '-4',
                            '--spacing', '0.5', '1', '2', '--stride', '2')
        xdmf.assert_not_called()
        hdf5.assert_called_once_with('field.h5', origin=(-2., -3., -4.),
                                     spacing=(0.5, 1., 2.), stride=2)
        self.assertIs(render.call_args.args[0], hdf5.return_value)

    def test_simulation_viewer_rejects_invalid_input_options(self):
        cases = (
            (['--xmf', 'step.xmf', '--vtk', 'step.vti'], 'not allowed'),
            (['--vtk', 'step.vti', '--h5', 'field.h5'],
             '--h5 cannot be combined with --vtk'),
            (['--h5', 'field.h5'], 'direct HDF5 needs --origin and --spacing'),
            (['--xmf', 'step.xmf', '--stride', '0'], '--stride must be positive'),
        )
        for args, message in cases:
            with self.subTest(args=args), \
                    patch.object(sys, 'stderr', new_callable=io.StringIO) as stderr:
                with self.assertRaises(SystemExit) as error:
                    self.run_viewer('geometry_viewer_simulation.py', *args)
                self.assertEqual(error.exception.code, 2)
                self.assertIn(message, stderr.getvalue())

    def test_original_viewers_keep_model_defaults(self):
        for script, component in (('geometry_viewer.py', 'alpha'), ('fac_viewer.py', 'fac')):
            grid = object()
            with self.subTest(script=script), \
                    patch.object(self.viewer, 'load_xdmf') as load, \
                    patch.object(self.viewer, 'model_snapshot', return_value=(grid, {})) as model, \
                    patch.object(self.viewer.viz3d, 'geometry_view') as render:
                if script == 'fac_viewer.py':
                    # Run the imported main so its model/loader mocks stay in scope.
                    with patch.object(sys, 'argv', [script]), patch('builtins.print'):
                        self.viewer.main()
                else:
                    self.run_viewer(script)
                load.assert_not_called()
                model.assert_called_once_with()
                self.assertIs(render.call_args.args[0], grid)
                self.assertEqual(render.call_args.kwargs['component'], component)


if __name__ == '__main__':
    unittest.main()
