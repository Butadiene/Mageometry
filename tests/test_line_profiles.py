"""Profiles preserve field orientation, invariant validity and native samples."""

from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from mageometry import GriddedField, trace_field_lines, field_line_profile
from mageometry.geometry.line_profiles import _LineEvaluator
from mageometry.session import SessionEngine, model_session, validate_session, save_session, load_session
from mageometry.session.profiles import export_profile
from mageometry.session.jobs import JobRunner


def varying_twist(x, y, z):
    x, y, z = np.broadcast_arrays(x, y, z)
    return -y*z, x*z, np.ones_like(z)


def helix(x, y, z):
    x, y, z = np.broadcast_arrays(x, y, z)
    return -y, x, np.ones_like(z)


def profile_group():
    group = model_session()['groups'][0]
    group['cases'][0]['source'].update(bounds=[[-2., 2.]] * 3, shape=[5] * 3)
    group['analysis'].update(mask_radius=0., geometry_delta=.001, seeds=[[0., 0., 0.], [.8, 0., 0.]],
                              trace=dict(direction='both', ds=.1, max_steps=10))
    return group


def load_profile_source(source, mask_radius, check):
    check()
    if source['kind'] == 'dipole':
        return None, lambda x, y, z: (np.zeros_like(x), np.zeros_like(y), np.ones_like(z))
    axes = [np.linspace(*bounds, size) for bounds, size in zip(source['bounds'], source['shape'])]
    grid = GriddedField(*axes, *varying_twist(*np.meshgrid(*axes, indexing='ij')),
                        metadata={'field_unit': 'nT', 'length_unit': 'Re'})
    return grid, varying_twist


class TestLineProfiles(unittest.TestCase):
    def test_signed_distance_and_endpoints_in_all_directions(self):
        for direction in (1, -1, 'both'):
            with self.subTest(direction=direction):
                trace = trace_field_lines(varying_twist, 0., 0., 1., direction=direction, ds=.1, max_steps=5)
                original = trace.s.copy()
                result = field_line_profile(varying_twist, trace, quantities=('alpha',))
                self.assertTrue(np.all(np.diff(result['s']) > 0))
                np.testing.assert_allclose(result['points'][:, 2], 1 + result['s'], atol=1e-14)
                np.testing.assert_allclose(result['values']['alpha'], 2 * result['points'][:, 2], atol=1e-12)
                self.assertEqual(result['s'][result['seed_index']], 0.)
                self.assertEqual(np.count_nonzero(result['s'] == 0), 1)
                self.assertEqual(result['status_minus'], None if direction == 1 else 2)
                self.assertEqual(result['status_plus'], None if direction == -1 else 2)
                np.testing.assert_array_equal(trace.s, original)

    def test_straight_line_keeps_invariants_and_masks_only_undefined_quantities(self):
        trace = trace_field_lines(varying_twist, 0., 0., 0., direction='both', max_steps=4)
        result = field_line_profile(varying_twist, trace,
                                    quantities=('mu0J_T', 'alpha', 'gamma', 'beta_g', 'delta_g', 'eta', 'curvature'),
                                    delta=.001)
        values = result['values']
        np.testing.assert_allclose(values['alpha'], 2*result['s'], atol=1e-12)
        np.testing.assert_allclose(values['gamma'], 0, atol=1e-12)
        np.testing.assert_allclose(values['curvature'], 0)
        self.assertTrue(np.all(np.isnan(values['mu0J_T'])))
        for key in ('beta_g', 'delta_g'):
            self.assertFalse(np.any(result['valid'][key]))
        self.assertFalse(result['valid']['eta'][result['seed_index']])
        self.assertTrue(np.all(result['valid']['alpha']))

    def test_helix_alpha_is_constant_without_display_grid_interpolation(self):
        trace = trace_field_lines(helix, .8, 0., 0., direction='both', ds=.02, err=1e-7, max_steps=100)
        result = field_line_profile(helix, trace, quantities=('alpha', 'bmag'), delta=.001)
        np.testing.assert_allclose(result['values']['alpha'], 2/1.64, rtol=1e-9)
        np.testing.assert_allclose(result['values']['bmag'], np.sqrt(1.64), rtol=1e-9)

    def test_quantity_family_is_cached_and_invalid_names_rejected(self):
        from mageometry.geometry import field_line_transverse_geometry
        evaluator = _LineEvaluator(helix, np.array([[1., 0., 0.]]), .001)
        with patch('mageometry.geometry.line_profiles.field_line_transverse_geometry', wraps=field_line_transverse_geometry) as calculate:
            evaluator.get('alpha')
            evaluator.get('gamma')
            evaluator.get('eta')
        self.assertEqual(calculate.call_count, 1)
        with self.assertRaises(ValueError):
            evaluator.get('unknown')

    def test_basic_geometry_keeps_primary_api_stencils_regardless_of_quantity_order(self):
        from mageometry.geometry import field_line_curvature, field_line_torsion
        trace = trace_field_lines(varying_twist, .8, .1, 1., direction='both', ds=.05, max_steps=8)
        first = field_line_profile(varying_twist, trace, quantities=('alpha', 'curvature', 'torsion'), delta=.02)
        second = field_line_profile(varying_twist, trace, quantities=('curvature', 'alpha'), delta=.02)
        np.testing.assert_allclose(first['values']['curvature'], second['values']['curvature'])
        np.testing.assert_allclose(first['values']['curvature'],
                                   field_line_curvature(varying_twist, *first['points'].T, delta=.02))
        np.testing.assert_allclose(first['values']['torsion'],
                                   field_line_torsion(varying_twist, *first['points'].T, delta=.02))

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'matplotlib unavailable')
    def test_static_profiles_use_canonical_alpha_and_signed_distance(self):
        from mageometry import viz
        from matplotlib import pyplot as plt
        trace = trace_field_lines(varying_twist, 0., 0., 1., direction=-1, max_steps=5)
        axes = viz.plot_line_profiles(trace, varying_twist, quantities=('alpha', 'beta_g'), delta=.001)
        self.addCleanup(plt.close, axes[0].figure)
        s, alpha = axes[0].lines[0].get_data()
        self.assertTrue(np.all(s <= 0))
        np.testing.assert_allclose(alpha, 2*(1+s), atol=1e-12)
        self.assertTrue(np.all(np.isnan(axes[1].lines[0].get_ydata())))


class TestSessionProfiles(unittest.TestCase):
    def setUp(self):
        loader = patch('mageometry.session.engine.load_source', side_effect=load_profile_source)
        loader.start()
        self.addCleanup(loader.stop)
        self.group = profile_group()
        self.engine = SessionEngine(self.group)
        self.result = self.engine.prepare(self.group['view'])
        self.identity = self.result['trace_records'][0]['seed_id']

    def test_pointwise_values_units_and_cache_without_retracing(self):
        with patch('mageometry.session.engine.trace_field_lines', side_effect=AssertionError('Retraced')):
            data = self.engine.prepare_profile(self.group['view'], self.identity, ['fac', 'alpha', 'bmag'])
        np.testing.assert_allclose(data['native_values']['fac'], 2*data['s'], atol=1e-12)
        np.testing.assert_allclose(data['values']['fac'], .125*2*data['s'], atol=1e-12)
        np.testing.assert_allclose(data['values']['alpha'], 2*data['s'], atol=1e-12)
        self.assertEqual(data['labels']['bmag'], '|B| [nT]')
        self.assertIn('nA/m^2', data['labels']['fac'])
        self.assertEqual(data['evaluation'], 'pointwise direct model')
        with patch('mageometry.geometry.line_profiles.field_line_transverse_geometry', side_effect=AssertionError('Recomputed family')):
            self.engine.prepare_profile(self.group['view'], self.identity, ['gamma', 'eta'])

    def test_grid_mode_uses_preview_field_and_records_steps(self):
        self.group['analysis']['evaluation'] = 'grid'
        engine = SessionEngine(self.group)
        result = engine.prepare(self.group['view'])
        data = engine.prepare_profile(self.group['view'], result['trace_records'][0]['seed_id'], ['alpha', 'fac'])
        np.testing.assert_allclose(data['values']['alpha'], 2*data['s'], atol=1e-10)
        self.assertEqual(data['evaluation'], 'pointwise preview grid interpolant')
        self.assertEqual(data['fac_delta'], .001)
        self.assertEqual(data['preview_shape'], [5, 5, 5])

    def test_background_branches_retain_total_path_and_magnitude(self):
        self.group['analysis']['kind'] = 'attribution'
        self.group['cases'][0]['background'] = dict(kind='dipole', epoch=100.)
        engine = SessionEngine(self.group)
        result = engine.prepare(self.group['view'])
        seed_id = result['trace_records'][1]['seed_id']
        data = {}
        for branch in ('total', 'background', 'residual'):
            view = dict(self.group['view'], contribution=branch)
            data[branch] = engine.prepare_profile(view, seed_id, ['alpha', 'gamma', 'bmag'])
        for branch in ('background', 'residual'):
            np.testing.assert_array_equal(data[branch]['points'], data['total']['points'])
            np.testing.assert_allclose(data[branch]['values']['bmag'], data['total']['values']['bmag'])
            self.assertEqual(data[branch]['labels']['bmag'], 'Total |B| [nT]')
        np.testing.assert_allclose(data['background']['values']['alpha'], 0, atol=1e-12)
        np.testing.assert_allclose(data['residual']['values']['alpha'], data['total']['values']['alpha'])

    def test_export_keeps_all_samples_and_nan_validity_with_metadata(self):
        data = self.engine.prepare_profile(self.group['view'], self.identity, ['alpha', 'beta_g', 'fac'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'line.csv'
            export_profile(data, path)
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), len(data['s']))
            self.assertTrue(all(row['beta_g_valid'] == '0' and row['alpha_valid'] == '1' for row in rows))
            self.assertAlmostEqual(float(rows[0]['fac_display']), .125*float(rows[0]['fac_native']))
            metadata = json.loads(path.with_suffix('.profile.json').read_text())
            self.assertEqual(metadata['trace_options']['direction'], 'both')
            self.assertEqual(metadata['source']['source'], self.group['cases'][0]['source'])
            self.assertEqual(metadata['sample_count'], len(rows))

    def test_profile_settings_migrate_validate_and_round_trip(self):
        session = model_session()
        profile = session['groups'][0]['view'].pop('profile')
        restored = validate_session(session)
        self.assertEqual(restored['groups'][0]['view']['profile'], profile)
        profile.update(visible=True, seed_id=self.identity, quantities=['alpha', 'bmag'],
                       xlim=[-.5, .5], ylims={'alpha': [-2., 2.]}, cursor_s=.1)
        session['groups'][0]['view']['profile'] = profile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'session.json'
            save_session(session, path)
            self.assertEqual(load_session(path)['groups'][0]['view']['profile'], profile)
        for key, value in (('visible', 1), ('quantities', ['unknown']), ('quantities', ['alpha']*2),
                           ('xlim', [1, -1]), ('ylims', {'alpha': [0, float('nan')]}), ('cursor_s', float('inf'))):
            bad = deepcopy(session)
            bad['groups'][0]['view']['profile'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_session(bad)


class TestProfileWorker(unittest.TestCase):
    def test_profile_requests_have_independent_generations_and_leave_scene_jobs_intact(self):
        runner = JobRunner()
        self.addCleanup(runner.close)
        group = model_session()['groups'][0]
        group['cases'][0]['source']['shape'] = [5, 5, 5]
        group['analysis']['trace']['max_steps'] = 6
        token = runner.submit(group)
        records = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and records is None:
            for current, kind, value in runner.poll():
                self.assertEqual(current, token)
                self.assertNotEqual(kind, 'error', value)
                if kind == 'traces':
                    records = value['records']
            time.sleep(.01)
        self.assertTrue(records)
        first = runner.submit_profile(group, records[0]['seed_id'], ['alpha', 'bmag'])
        latest = runner.submit_profile(group, records[1]['seed_id'], ['gamma', 'eta'])
        self.assertEqual(first[0], token)
        self.assertEqual(latest[0], token)
        self.assertGreater(latest[1], first[1])
        result = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and result is None:
            for current, kind, value in runner.poll():
                self.assertEqual(current, token)
                self.assertNotIn(kind, ('result', 'traces', 'error', 'profile_error'), value)
                if kind == 'profile' and value['id'] == latest[1]:
                    result = value['data']
            time.sleep(.01)
        self.assertIsNotNone(result)
        self.assertEqual(result['seed_id'], records[1]['seed_id'])
        self.assertEqual(list(result['values']), ['gamma', 'eta'])


if __name__ == '__main__':
    unittest.main()
