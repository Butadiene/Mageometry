"""Example launchers build editable desktop recipes without eager source loading."""

import io
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'


class TestGeometryViewerExamples(unittest.TestCase):
    def run_viewer(self, script, *args):
        with patch.object(sys, 'argv', [script, *args]), \
                patch('mageometry.gui.app.run') as launch:
            runpy.run_path(str(EXAMPLES / script), run_name='__main__')
        launch.assert_called_once()
        return launch.call_args.args[0]['groups'][0]

    def test_model_launchers_open_desktop_with_their_presets(self):
        for script, component, values in (
            ('geometry_gui.py', 'alpha', [0.]),
            ('compare_t96_by.py', 'alpha', [-5., -3., -1., 1., 3., 5.]),
        ):
            with self.subTest(script=script):
                group = self.run_viewer(script, '--layout', 'three_d_slice')
                self.assertEqual(group['view']['component'], component)
                self.assertEqual(group['view']['layout'], 'three_d_slice')
                self.assertEqual([c['source']['parameters']['by'] for c in group['cases']], values)
                self.assertEqual(group['analysis']['geometry_delta'], .002)
                self.assertIsNone(group['analysis']['delta'])

    def test_standard_launcher_can_start_on_fac(self):
        group = self.run_viewer('geometry_gui.py', '--component', 'fac', '--layout', 'three_d_slice')
        self.assertEqual(group['view']['component'], 'fac')
        self.assertEqual(group['view']['layout'], 'three_d_slice')
        self.assertEqual(group['analysis']['kind'], 'field')
        self.assertEqual(len(group['cases']), 1)
        self.assertEqual(group['cases'][0]['source']['parameters']['by'], 0.)

    def test_comparison_overrides_and_backgrounds_reach_desktop(self):
        group = self.run_viewer('compare_t96_by.py', '--by', '-2', '2', '--initial-by', '2',
                                '--shape', '9', '7', '7', '--background', 'dipole',
                                '--contribution', 'residual', '--component', 'beta_g',
                                '--color-limit', '.1', '--threshold', '.01', '--cache-size', '1',
                                '--delta', '.001', '--geometry-delta', '.003')
        self.assertEqual(group['view']['case'], group['cases'][1]['id'])
        self.assertEqual(group['reference'], group['cases'][0]['id'])
        self.assertEqual(group['view']['contribution'], 'residual')
        self.assertEqual(group['view']['color_limits'], {'attribution:beta_g': .1})
        self.assertEqual(group['view']['thresholds'], {'attribution:beta_g': .01})
        self.assertEqual(group['analysis']['kind'], 'attribution')
        self.assertEqual(group['analysis']['cache_size'], 1)
        self.assertEqual(group['analysis']['delta'], .001)
        self.assertEqual(group['analysis']['geometry_delta'], .003)
        for case in group['cases']:
            self.assertEqual(case['source']['shape'], [9, 7, 7])
            self.assertEqual(case['background'], {'kind': 'dipole', 'parameters': {'epoch': 100.}})

    def test_simulation_viewer_keeps_reader_options_in_recipe(self):
        for args, expected in (
            (['--xmf', 'my run/step.xmf', '--h5', 'heavy.h5'],
             dict(kind='xdmf', path=str(Path('my run/step.xmf').resolve()),
                  options={'h5_file': str(Path('heavy.h5').resolve()), 'stride': 4})),
            (['--vtk', 'step.vti'], dict(kind='vtk', path=str(Path('step.vti').resolve()),
                                       options={'stride': 4})),
            (['--h5', 'field.h5', '--origin', '-2', '-3', '-4', '--spacing', '.5', '1', '2'],
             dict(kind='hdf5', path=str(Path('field.h5').resolve()),
                  options={'stride': 4, 'origin': [-2., -3., -4.], 'spacing': [.5, 1., 2.]})),
        ):
            with self.subTest(args=args):
                group = self.run_viewer('geometry_viewer_simulation.py', *args, '--stride', '4',
                                        '--component', 'delta_g', '--slice', 'z',
                                        '--slice-origin', '-6', '0', '0', '--slice-only')
                self.assertEqual(group['cases'][0]['source'], expected)
                self.assertEqual(group['view']['component'], 'delta_g')
                self.assertEqual(group['view']['normal'], [0., 0., 1.])
                self.assertEqual(group['view']['origin'], [-6., 0., 0.])
                self.assertEqual(group['view']['layout'], 'slice')
                self.assertEqual(group['analysis']['evaluation'], 'grid')
                self.assertIsNone(group['analysis']['delta'])

    def test_file_background_uses_same_reader_stride(self):
        for extension, kind in (('xmf', 'xdmf'), ('vtk', 'vtk')):
            with self.subTest(extension=extension):
                group = self.run_viewer('geometry_viewer_simulation.py',
                                        '--' + extension, 'total.' + extension,
                                        '--background-' + extension, 'background.' + extension,
                                        '--stride', '3', '--component', 'gamma')
                background = group['cases'][0]['background']
                self.assertEqual(background['kind'], kind)
                self.assertEqual(background['path'], str(Path('background.' + extension).resolve()))
                self.assertEqual(background['options']['stride'], 3)
                self.assertEqual(group['analysis']['kind'], 'attribution')

    def test_simulation_viewer_rejects_missing_or_invalid_sources(self):
        for args, message in (
            ([], 'specify a snapshot file'),
            (['--xmf', 'step.xmf', '--vtk', 'step.vti'], 'not allowed'),
            (['--vtk', 'step.vti', '--h5', 'field.h5'], '--h5 cannot be combined with --vtk'),
            (['--h5', 'field.h5'], 'direct HDF5 needs --origin and --spacing'),
            (['--vtk', 'step.vti', '--stride', '0'], '--stride must be positive'),
            (['--vtk', 'step.vti', '--evaluation', 'direct'], 'require a model source'),
        ):
            with self.subTest(args=args), \
                    patch.object(sys, 'stderr', new_callable=io.StringIO) as stderr, \
                    patch('mageometry.gui.app.run') as launch:
                with self.assertRaises(SystemExit) as error:
                    self.run_viewer('geometry_viewer_simulation.py', *args)
                self.assertEqual(error.exception.code, 2)
                self.assertIn(message, stderr.getvalue())
                launch.assert_not_called()

    def test_model_preset_preserves_field_metadata_and_trace_defaults(self):
        from mageometry import geopack, geopack_field
        from mageometry.session.presets import model_snapshot
        grid, options = model_snapshot(shape=(9, 7, 7))
        ps = geopack.recalc(100.)
        field = geopack_field('t96', 'dip', [2., -20., 0., -5., 0, 0, 0, 0, 0, 0], ps)
        coords = np.meshgrid(grid.x, grid.y, grid.z, indexing='ij')
        with np.errstate(divide='ignore', invalid='ignore'):
            expected = np.stack([np.where(options['mask'](*coords), np.nan, b)
                                 for b in field(*coords)], axis=-1)
        np.testing.assert_allclose(grid.b, expected, rtol=1e-13)
        self.assertEqual(grid.metadata['model'], 'T96 + dipole')
        self.assertEqual(grid.metadata['parameters'], {
            'Pdyn [nPa]': 2., 'Dst [nT]': -20., 'IMF By [nT]': 0., 'IMF Bz [nT]': -5.,
            'Dipole tilt [rad]': ps, 'Epoch [Unix s]': 100.})
        point = (-6., .3, 1.)
        expected_total = field(*point)
        expected_dipole = geopack_field(None, 'dip', ps=ps)(*point)
        geopack.recalc(1609459200.)
        np.testing.assert_allclose(options['field'](*point), expected_total, rtol=1e-13)
        geopack.recalc(1609459200.)
        np.testing.assert_allclose(options['background_choices']['Dipole'](*point),
                                   expected_dipole, rtol=1e-13)
        self.assertEqual(options['delta'], .002)
        self.assertEqual(options['trace_kwargs'], {'r0': 2.5, 'ds': .15, 'max_steps': 350})
        self.assertNotIn('seeds', options)


if __name__ == '__main__':
    unittest.main()
