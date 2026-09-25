"""Simulation viewers require explicit input and preserve shared controls."""

import importlib
import io
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch

import numpy as np

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
                            '--component', 'beta_g', '--slice', 'x', '--slice-only')
        load.assert_called_once_with('step.xmf', h5_file='heavy.h5', stride=4)
        self.assertEqual(render.call_args.kwargs['component'], 'beta_g')
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

    def test_model_metadata_matches_evaluated_parameters(self):
        field = lambda x, y, z: (np.ones_like(x), np.ones_like(y), np.ones_like(z))
        with patch.object(self.viewer.geopack, 'recalc', return_value=0.123) as recalc, \
                patch.object(self.viewer, 'geopack_field', return_value=field) as model:
            grid, options = self.viewer.model_snapshot()
        external, internal, parmod, ps = model.call_args_list[0].args
        self.assertEqual((external, internal), ('t96', 'dip'))
        self.assertIs(options['field'], field)
        self.assertEqual(grid.metadata['model'], 'T96 + dipole')
        self.assertEqual(grid.metadata['parameters'], {
            'Pdyn [nPa]': parmod[0], 'Dst [nT]': parmod[1],
            'IMF By [nT]': parmod[2], 'IMF Bz [nT]': parmod[3],
            'Dipole tilt [rad]': ps, 'Epoch [Unix s]': recalc.call_args.args[0]})
        self.assertIs(options['background_choices']['Dipole'], field)
        model.assert_called_with(None, 'dip', ps=ps)

    def test_simulation_gui_loader_preserves_input_stride(self):
        path = Path('background.vti')
        with patch.object(self.viewer, 'load_vtk'), \
                patch.object(self.viewer.viz3d, 'geometry_view') as render:
            self.run_viewer('geometry_viewer_simulation.py', '--vtk', 'run/total.vti', '--stride', '4')
        options = render.call_args.kwargs
        self.assertEqual(options['background_directory'], Path('run').resolve())
        with patch('mageometry.viz3d._background.load_vtk') as load:
            options['background_loader'](path)
        load.assert_called_once_with(path, stride=4)

    def test_model_background_uses_declared_dipole_tilt(self):
        from types import SimpleNamespace
        grid = SimpleNamespace(metadata={'parameters': {'Dipole tilt [rad]': .123}})
        with patch.object(self.viewer, 'model_snapshot', return_value=(grid, {})), \
                patch.object(self.viewer, 'geopack_field') as background, \
                patch.object(self.viewer.viz3d, 'transverse_contribution_view') as render:
            self.run_viewer('geometry_viewer.py', '--background', 'dipole',
                            '--contribution', 'residual')
        background.assert_called_once_with(None, 'dip', ps=.123)
        self.assertIs(render.call_args.kwargs['background'], background.return_value)
        self.assertEqual(render.call_args.kwargs['background_label'], 'Dipole')
        self.assertEqual(render.call_args.kwargs['component'], 'eta')
        self.assertEqual(render.call_args.kwargs['contribution'], 'residual')

    def test_background_snapshot_loading(self):
        for extension, loader in (('xmf', 'load_xdmf'), ('vtk', 'load_vtk')):
            total, background = object(), object()
            with self.subTest(extension=extension), \
                    patch.object(self.viewer, loader, side_effect=[total, background]) as load, \
                    patch.object(self.viewer.viz3d, 'transverse_contribution_view') as render:
                self.run_viewer('geometry_viewer_simulation.py',
                                '--' + extension, 'total.' + extension,
                                '--background-' + extension, 'background.' + extension,
                                '--stride', '3', '--component', 'gamma')
            self.assertEqual(load.call_args.kwargs, {'stride': 3})
            self.assertIs(render.call_args.args[0], total)
            self.assertIs(render.call_args.kwargs['background'], background)
            self.assertEqual(render.call_args.kwargs['component'], 'gamma')
            self.assertEqual(render.call_args.kwargs['contribution'], 'total')

    def test_invalid_background_options_rejected_before_loading(self):
        for args in (['--contribution', 'residual'],
                     ['--background', 'dipole', '--component', 'fac'],
                     ['--background', 'dipole', '--vtk', 'total.vti']):
            with self.subTest(args=args), \
                    patch.object(self.viewer, 'model_snapshot') as model, \
                    patch.object(self.viewer, 'load_vtk') as load, \
                    patch.object(sys, 'stderr', new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as error:
                    self.run_viewer('geometry_viewer.py', *args)
                self.assertEqual(error.exception.code, 2)
                model.assert_not_called()
                load.assert_not_called()

    def test_simulation_cli_accepts_eta(self):
        with patch.object(self.viewer, 'load_vtk'), \
                patch.object(self.viewer.viz3d, 'geometry_view') as render:
            self.run_viewer('geometry_viewer_simulation.py', '--vtk', 'step.vti', '--component', 'eta')
        self.assertEqual(render.call_args.kwargs['component'], 'eta')

    def test_simulation_cli_accepts_shear_names(self):
        for name in ('beta_g', 'delta_g'):
            with self.subTest(name=name), patch.object(self.viewer, 'load_vtk'), \
                    patch.object(self.viewer.viz3d, 'geometry_view') as render:
                self.run_viewer('geometry_viewer_simulation.py', '--vtk', 'step.vti', '--component', name)
            self.assertEqual(render.call_args.kwargs['component'], name)

    def test_simulation_cli_rejects_removed_names_before_loading(self):
        for name in ('sigma', 'q'):
            with self.subTest(name=name), patch.object(self.viewer, 'load_vtk') as load, \
                    patch.object(self.viewer.viz3d, 'geometry_view') as render, \
                    patch.object(sys, 'stderr', new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as error:
                    self.run_viewer('geometry_viewer_simulation.py', '--vtk', 'step.vti',
                                    '--component', name)
                self.assertEqual(error.exception.code, 2)
                load.assert_not_called()
                render.assert_not_called()


if __name__ == '__main__':
    unittest.main()
