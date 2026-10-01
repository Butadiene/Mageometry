"""Model-specific recipes, parameter mapping, provenance and Qt source forms."""

from copy import deepcopy
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mageometry import geopack, field_aligned_current_density
from mageometry.geopack import models
from mageometry.gui.cli import session_from_args
from mageometry.session import SessionEngine, load_session, model_session, save_session
from mageometry.session.sources import fingerprint, load_source, model_field
from mageometry.session.specs import MODEL_KINDS, model_source, validate_source


PARAMETERS = {
    't89': {'iopt': 4},
    't96': {'pdyn': 3., 'dst': -30., 'by': -4., 'bz': -10.},
    't01': {'pdyn': 3., 'dst': -30., 'by': -4., 'bz': -10., 'g1': 1.2, 'g2': .7},
    't04': {'pdyn': 3., 'dst': -30., 'by': -4., 'bz': -10.,
            'w1': .1, 'w2': .2, 'w3': .3, 'w4': .4, 'w5': .5, 'w6': .6},
}
PARMOD = {
    't89': 4,
    't96': [3., -30., -4., -10., 0., 0., 0., 0., 0., 0.],
    't01': [3., -30., -4., -10., 1.2, .7, 0., 0., 0., 0.],
    't04': [3., -30., -4., -10., .1, .2, .3, .4, .5, .6],
}


def small_source(model):
    source = model_source(model=model, parameters=PARAMETERS[model])
    source.update(bounds=[[-8., -4.], [1., 3.], [.5, 1.5]], shape=[5, 5, 5])
    return source


class TestSessionModels(unittest.TestCase):
    def test_all_models_match_scalar_reference_and_restore_bound_epoch(self):
        points = np.array([[-6., 2., 1.], [-5., 1., .7]])
        for name in MODEL_KINDS:
            with self.subTest(model=name):
                source = small_source(name)
                source['parameters']['epoch'] = 1.5e9
                field, ps = model_field(source)
                expected = [np.array(getattr(models, name)(PARMOD[name], ps, *point))
                            + np.array(geopack.dip(*point)) for point in points]
                source['parameters'].update(PARAMETERS[name])
                source['parameters']['epoch'] = 100.
                geopack.recalc(100.)
                np.testing.assert_allclose(np.array(field(*points.T)).T, expected, rtol=1e-10, atol=1e-7)
                self.assertEqual(fingerprint(source), [])

    def test_grids_metadata_and_parameter_effects(self):
        for name in MODEL_KINDS:
            with self.subTest(model=name):
                grid, field = load_source(small_source(name), mask_radius=6.)
                coords = np.meshgrid(grid.x, grid.y, grid.z, indexing='ij')
                valid = sum(c*c for c in coords) >= 36.
                np.testing.assert_allclose(grid.b[valid], np.array(field(*(c[valid] for c in coords))).T)
                self.assertTrue(np.all(np.isnan(grid.b[~valid])))
                self.assertEqual(grid.metadata['model'], name.upper() + ' + dipole')
                self.assertEqual(grid.metadata['field_unit'], 'nT')
                self.assertEqual(grid.metadata['length_unit'], 'Re')
                if name == 't89':
                    self.assertNotIn('IMF By [nT]', grid.metadata['parameters'])
                default, _ = model_field(model_source(model=name))
                self.assertGreater(np.linalg.norm(np.array(field(-6., 2., 1.))
                                                  - default(-6., 2., 1.)), .1)

    def test_t01_t04_outside_domain_is_undefined_without_clipped_gradients(self):
        for name in ('t01', 't04'):
            with self.subTest(model=name):
                field, _ = model_field(small_source(name))
                with patch('builtins.print') as notice:
                    values = field(np.array([[-16.], [-15.], [-6.], [np.nan]]), [1., 2.], 1.)
                    fac = field_aligned_current_density(field, [-15., -14.9], 2., 1., delta=.002)
                notice.assert_not_called()
                for value in values:
                    self.assertEqual(value.shape, (4, 2))
                    self.assertTrue(np.all(np.isnan(value[[0, 3]])))
                    self.assertTrue(np.all(np.isfinite(value[1:3])))
                self.assertTrue(np.isnan(fac[0]))
                self.assertTrue(np.isfinite(fac[1]))
                self.assertTrue(all(isinstance(v, float) and np.isnan(v) for v in field(-16., 2., 1.)))

    def test_direct_grid_attribution_and_session_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in MODEL_KINDS:
                session = model_session(model=name)
                group = session['groups'][0]
                group['cases'][0]['source'] = small_source(name)
                group['cases'][0]['background'] = dict(kind='dipole', parameters={'epoch': 100.})
                group['analysis'].update(seeds=[], max_points=125)
                for evaluation in ('direct', 'grid'):
                    with self.subTest(model=name, evaluation=evaluation):
                        group['analysis'].update(evaluation=evaluation, kind='field')
                        engine = SessionEngine(group)
                        result = engine.prepare(dict(group['view'], component='fac'))
                        self.assertTrue(np.any(np.isfinite(result['values'])))
                        if evaluation == 'direct':
                            field, _ = model_field(small_source(name))
                            expected = field_aligned_current_density(field, -6., 2., 1., delta=.002) * .125
                            self.assertAlmostEqual(result['values'][2, 2, 2], expected)
                        group['analysis']['kind'] = 'attribution'
                        result = SessionEngine(group).prepare(dict(group['view'], component='gamma',
                                                                   contribution='residual'))
                        self.assertTrue(np.any(np.isfinite(result['values'])))
                path = Path(directory) / (name + '.json')
                save_session(session, path)
                self.assertEqual(load_session(path)['groups'], session['groups'])

    def test_invalid_model_parameters(self):
        for name, key, value in (
            ('t89', 'iopt', 0), ('t89', 'iopt', 8), ('t89', 'iopt', 2.5), ('t89', 'iopt', True),
            ('t89', 'by', 1.), ('t96', 'g1', 1.), ('t01', 'w1', 1.), ('t04', 'g1', 1.),
            ('t96', 'pdyn', 0.), ('t01', 'g1', -1.), ('t04', 'w6', np.nan),
        ):
            with self.subTest(model=name, key=key, value=value), self.assertRaises(ValueError):
                model_source(model=name, parameters={key: value})
        for name, key in (('t89', 'iopt'), ('t01', 'g2'), ('t04', 'w3')):
            source = small_source(name)
            del source['parameters'][key]
            with self.assertRaises(ValueError):
                validate_source(source)

    def test_cli_model_parameters_and_rejection(self):
        for args, model, expected in (
            (['--model', 't89', '--iopt', '4'], 't89', {'iopt': 4}),
            (['--model', 't01', '--g1', '1.2', '--g2', '.7'], 't01', {'g1': 1.2, 'g2': .7}),
            (['--model', 't04', '--w', '.1', '.2', '.3', '.4', '.5', '.6'], 't04',
             {f'w{i}': i/10 for i in range(1, 7)}),
        ):
            with self.subTest(model=model):
                session, _ = session_from_args(args + ['--epoch', '1234'])
                source = session['groups'][0]['cases'][0]['source']
                self.assertEqual(source['kind'], model)
                self.assertEqual(source['parameters']['epoch'], 1234.)
                for key, value in expected.items():
                    self.assertEqual(source['parameters'][key], value)
        for args in (['--model', 't89', '--by', '0'], ['--model', 't89', '--dst', '-30'],
                     ['--model', 't96', '--g1', '1'], ['--model', 't01', '--g2', '-1'],
                     ['--model', 't04', '--w', '1', '2'], ['--model', 't89', '--iopt', '8'],
                     ['--model', 't01', '--vtk', 'data.vti']):
            with self.subTest(args=args), patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit):
                session_from_args(args)
        session, _ = session_from_args(['--dst', '-40', '--bz', '-8'],
                                        default_model_parameters={'dst': -30., 'bz': -10.})
        self.assertEqual(session['groups'][0]['cases'][0]['source']['parameters']['dst'], -40.)

    def test_by_comparisons_keep_model_parameters_and_independent_fields(self):
        from mageometry.session.presets import make_cases, make_fields
        for model in ('t01', 't04'):
            with self.subTest(model=model):
                parameters = {k: v for k, v in PARAMETERS[model].items() if k != 'by'}
                cases = make_cases((-5., 5.), (5, 5, 5), model=model, parameters=parameters)
                fields = make_fields((-5., 5.), model=model, parameters=parameters)
                evaluated = []
                for label, grid in cases.items():
                    self.assertTrue(label.startswith(model.upper()))
                    coords = np.meshgrid(grid.x, grid.y, grid.z, indexing='ij')
                    valid = np.all(np.isfinite(grid.b), axis=-1)
                    expected = np.array(fields[label](*(c[valid] for c in coords))).T
                    np.testing.assert_allclose(grid.b[valid], expected, rtol=1e-12)
                    evaluated.append(fields[label](-6., 2., 1.))
                self.assertGreater(np.linalg.norm(np.array(evaluated[0]) - evaluated[1]), 1.)

    def test_standalone_dispatches_selected_model_and_indices(self):
        from mageometry.viz3d import cli
        with patch.object(cli, 'model_snapshot', return_value=(object(), {})) as snapshot, \
                patch.object(cli.viz3d, 'geometry_view'), patch('builtins.print'):
            cli.main(['--model', 't89', '--iopt', '5'])
        snapshot.assert_called_once_with(model='t89', parameters={'iopt': 5})
        with patch.object(cli, 'make_cases', return_value={'case': object()}) as cases, \
                patch.object(cli, 'make_fields') as fields, \
                patch.object(cli.viz3d, 'compare_geometry') as render, patch('builtins.print'):
            cli.main(['--model', 't01', '--g1', '2', '--g2', '3', '--by', '-5', '5', '--initial-by', '5'])
        self.assertEqual(cases.call_args.kwargs, {'model': 't01', 'parameters': {'g1': 2., 'g2': 3.}})
        self.assertEqual(fields.call_args.kwargs, cases.call_args.kwargs)
        self.assertEqual(render.call_args.kwargs['initial_case'], 'T01: IMF By = +5 nT')


@unittest.skipUnless(importlib.util.find_spec('PySide6'), 'Qt unavailable')
class TestModelSourceDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_model_forms_round_trip_and_show_only_supported_inputs(self):
        from mageometry.gui.forms import SourceDialog
        for name in MODEL_KINDS:
            with self.subTest(model=name):
                dialog = SourceDialog(source=small_source(name))
                try:
                    self.assertEqual(dialog.kind.currentText(), name)
                    for key, visible in (('iopt', name == 't89'), ('by', name != 't89'),
                                          ('g1', name == 't01'), ('g2', name == 't01'), ('w', name == 't04')):
                        self.assertEqual(not dialog.fields[key].isHidden(), visible, key)
                    dialog.accept_source()
                    self.assertEqual(dialog.error.text(), '')
                    self.assertEqual(dialog.sources, [small_source(name)])
                finally:
                    dialog.close()

    def test_model_switching_and_by_scan_preserve_explicit_inputs(self):
        from mageometry.gui.forms import SourceDialog
        dialog = SourceDialog()
        try:
            dialog.kind.setCurrentText('t01')
            dialog.fields['g1'].setText('2.5')
            dialog.fields['by'].setText('-5 5')
            dialog.accept_source()
            self.assertEqual([s['parameters']['by'] for s in dialog.sources], [-5., 5.])
            self.assertTrue(all(s['parameters']['g1'] == 2.5 for s in dialog.sources))
            dialog.kind.setCurrentText('t89')
            dialog.accept_source()
            self.assertEqual(len(dialog.sources), 1)
            self.assertEqual(dialog.sources[0]['parameters'], {'epoch': 100., 'iopt': 2})
            dialog.kind.setCurrentText('t04')
            dialog.fields['w'].setText('1 2 3')
            dialog.accept_source()
            self.assertIn('Expected 6', dialog.error.text())
        finally:
            dialog.close()


if __name__ == '__main__':
    unittest.main()
