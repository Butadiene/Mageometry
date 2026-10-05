"""Portable recipes, combined comparison/attribution, and serial job behavior."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from mageometry import GriddedField, geopack
from mageometry.session import model_session, empty_session, validate_session, SessionEngine, save_session, load_session
from mageometry.session.sources import model_field, load_source
from mageometry.session.jobs import JobRunner
from mageometry.viz3d._current import COMPONENTS, TRANSVERSE_COMPONENTS
from mageometry.viz3d._overview_data import _OverviewData
from mageometry.viz3d._contribution_data import _ContributionData


def small_session():
    session = model_session((-1., 1.))
    group = session['groups'][0]
    for case in group['cases']:
        case['source']['shape'] = [5, 5, 5]
    group['analysis']['seeds'] = []
    return session


class TestSessionRecipes(unittest.TestCase):
    def test_gamma_eta_colouring_defaults_validates_and_round_trips(self):
        session = small_session()
        view = session['groups'][0]['view']
        del view['gamma_eta']
        self.assertFalse(validate_session(session)['groups'][0]['view']['gamma_eta'])
        view.update(component='gamma', gamma_eta=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gamma-eta.json'
            save_session(session, path)
            self.assertEqual(load_session(path)['groups'], session['groups'])
        for value in (None, 0, 1, 'true', []):
            view['gamma_eta'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Gamma colouring'):
                validate_session(session)

    def test_optional_ranges_and_trace_setting_preserve_legacy_defaults(self):
        session = small_session()
        group = session['groups'][0]
        del group['analysis']['trace_enabled']
        for key in ('threshold_modes', 'value_intervals', 'slice_color_ranges', 'slice_extent'):
            del group['view'][key]
        restored = validate_session(session)['groups'][0]
        self.assertTrue(restored['analysis']['trace_enabled'])
        self.assertEqual(restored['view']['threshold_modes'], {})
        self.assertEqual(restored['view']['value_intervals'], {})
        self.assertEqual(restored['view']['slice_color_ranges'], {})
        self.assertIsNone(restored['view']['slice_extent'])

    def test_ranges_and_disabled_tracing_round_trip_and_validate(self):
        session = small_session()
        group = session['groups'][0]
        group['analysis'].update(trace_enabled=False, seeds=None)
        group['view'].update(threshold_modes={'field:alpha': 'interval'},
                             value_intervals={'field:alpha': [-.1, .2]},
                             slice_color_ranges={'field:alpha': [-.3, .5]},
                             slice_extent=[-4., 2., -1., 3.])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ranges.json'
            save_session(session, path)
            self.assertEqual(load_session(path)['groups'], session['groups'])
        for key, value in (('threshold_modes', {'field:alpha': 'invalid'}),
                           ('threshold_modes', {'field:fac': 'interval'}),
                           ('value_intervals', []), ('value_intervals', {'field:alpha': [1, 1]}),
                           ('value_intervals', {'field:alpha': [float('nan'), 1]}),
                           ('slice_color_ranges', None), ('slice_color_ranges', {'field:alpha': [2, -1]}),
                           ('slice_color_ranges', {'field:alpha': [0, float('inf')]}),
                           ('slice_extent', [0, 1, 2]), ('slice_extent', [0, 1, 2, 2])):
            invalid = deepcopy(session)
            invalid['groups'][0]['view'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_session(invalid)
        for value in (None, 0, 1, 'false'):
            group['analysis']['trace_enabled'] = value
            with self.subTest(trace_enabled=value), self.assertRaises(ValueError):
                validate_session(session)

    def test_legacy_steps_are_preserved_when_loading_and_saving(self):
        for fac, geometry in ((.03, None), ([.03, .02, .04], None), (.03, .01)):
            with self.subTest(fac=fac, geometry=geometry), tempfile.TemporaryDirectory() as directory:
                session = small_session()
                session['schema_version'] = 1
                session['groups'][0]['analysis'].update(delta=fac, geometry_delta=geometry)
                before = deepcopy(session)
                path = Path(directory) / 'legacy.json'
                path.write_text(json.dumps(session))
                migrated = load_session(path)
                self.assertEqual(migrated['schema_version'], 2)
                analysis = migrated['groups'][0]['analysis']
                self.assertEqual(analysis['delta'], fac)
                self.assertEqual(analysis['geometry_delta'], geometry or np.min(fac))
                save_session(migrated, path)
                self.assertEqual(load_session(path)['groups'], migrated['groups'])
                self.assertEqual(session, before)

    def test_slice_normal_is_normalized_without_changing_input(self):
        for normal in ([2., 0., 0.], [1.e308, 1.e308, 0.]):
            session = small_session()
            session['groups'][0]['view']['normal'] = normal
            restored = validate_session(session)
            self.assertAlmostEqual(np.linalg.norm(restored['groups'][0]['view']['normal']), 1.)
            self.assertEqual(session['groups'][0]['view']['normal'], normal)
        session['groups'][0]['view']['normal'] = [0., 0., 0.]
        with self.assertRaisesRegex(ValueError, 'nonzero'):
            validate_session(session)

    def test_optional_threshold_slider_limits_default_to_automatic(self):
        session = small_session()
        del session['groups'][0]['view']['threshold_slider_limits']
        restored = validate_session(session)
        self.assertEqual(restored['groups'][0]['view']['threshold_slider_limits'], {})
        self.assertNotIn('threshold_slider_limits', session['groups'][0]['view'])
        for limits in ([], None, {'field:alpha': 0}, {'field:alpha': -1},
                       {'field:alpha': True}, {'field:alpha': '1'}, {'field:alpha': float('inf')}):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                session['groups'][0]['view']['threshold_slider_limits'] = limits
                validate_session(session)

    def test_optional_value_sign_defaults_to_both_and_rejects_invalid_modes(self):
        session = small_session()
        del session['groups'][0]['view']['value_sign']
        restored = validate_session(session)
        self.assertEqual(restored['groups'][0]['view']['value_sign'], 'both')
        self.assertNotIn('value_sign', session['groups'][0]['view'])
        for mode in ('both', 'positive', 'negative'):
            session['groups'][0]['view']['value_sign'] = mode
            self.assertEqual(validate_session(session)['groups'][0]['view']['value_sign'], mode)
        for mode in ('', 'all', None, True, 1, []):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'Value sign'):
                session['groups'][0]['view']['value_sign'] = mode
                validate_session(session)

    def test_validation_rejects_invalid_numerics_and_selection(self):
        for key, value in [('delta', 0), ('geometry_delta', float('nan')),
                           ('max_points', 26), ('cache_size', 0), ('current_scale', -1),
                           ('mask_radius', -1), ('seeds', [[1, 2]])]:
            session = small_session()
            session['groups'][0]['analysis'][key] = value
            with self.subTest(key=key), self.assertRaises((ValueError, TypeError)):
                validate_session(session)
        session = small_session()
        session['groups'][0]['analysis']['kind'] = 'attribution'
        with self.assertRaisesRegex(ValueError, 'background'):
            validate_session(session)
        session['schema_version'] = 99
        with self.assertRaisesRegex(ValueError, 'schema'):
            validate_session(session)

    def test_persistence_preserves_selection_view_and_relative_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'session.json'
            session = empty_session()
            group = session['groups'][0]
            group['cases'] = [dict(id='case-a', label='Run A', background=None,
                                   source=dict(kind='xdmf', path=str(Path(directory) / 'a.xmf'),
                                               options={'h5_file': str(Path(directory) / 'heavy.h5')}))]
            group['reference'] = group['view']['case'] = 'case-a'
            group['view'].update(layout='three_d_slice', previous_layout='three_d_slice',
                                 panels_hidden=True, normal=[0., 1., 0.], value_sign='negative',
                                 origin=[-6., 2., 1.], thresholds={'field:alpha': .125},
                                 threshold_slider_limits={'field:alpha': .5})
            save_session(session, path)
            raw = json.loads(path.read_text())
            source = raw['groups'][0]['cases'][0]['source']
            self.assertEqual(source['path'], 'a.xmf')
            self.assertEqual(source['options']['h5_file'], 'heavy.h5')
            restored = load_session(path)
            self.assertEqual(restored['groups'], session['groups'])
            # Reading a recipe does not require files or execute their loaders.
            self.assertFalse((Path(directory) / 'a.xmf').exists())

    def test_model_epoch_is_restored_for_retained_total_and_background(self):
        source = small_session()['groups'][0]['cases'][0]['source']
        field, _ = model_field(source)
        first = field(-6., 2., 1.)
        other = deepcopy(source)
        other['parameters']['epoch'] = 1.5e9
        second, _ = model_field(other)
        second(-6., 2., 1.)
        geopack.recalc(1.e9)
        np.testing.assert_allclose(field(-6., 2., 1.), first, rtol=1e-12)

    def test_import_gui_does_not_load_qt(self):
        import subprocess
        import sys
        command = "import sys; import mageometry.gui, mageometry.session; assert not any(k.startswith(('PySide6', 'pyvistaqt')) for k in sys.modules)"
        subprocess.run([sys.executable, '-c', command], check=True, capture_output=True)


class TestSessionPreparation(unittest.TestCase):
    def test_disabled_tracing_skips_seeds_and_integrator_without_changing_values(self):
        group = small_session()['groups'][0]
        baseline = SessionEngine(group).prepare(group['view'])
        for seeds in (None, [[-6., 2., 1.]]):
            group['analysis'].update(trace_enabled=False, seeds=seeds)
            engine = SessionEngine(group)
            with patch.object(engine, '_resolve_seeds') as resolve, \
                    patch('mageometry.session.engine.trace_field_lines') as trace:
                result = engine.prepare(group['view'])
                resolve.assert_not_called()
                trace.assert_not_called()
            self.assertEqual(result['paths'], [])
            self.assertIsNone(result['resolved']['seeds'])
            self.assertEqual(result['analysis']['seeds'], seeds)
            np.testing.assert_allclose(result['values'], baseline['values'], equal_nan=True)
            self.assertEqual(result['scale'], baseline['scale'])
        # Re-enabling uses the retained seed configuration.
        group['analysis']['trace_enabled'] = True
        group['analysis']['trace']['max_steps'] = 3
        result = SessionEngine(group).prepare(group['view'])
        self.assertEqual(len(result['paths']), 1)
        self.assertEqual(result['resolved']['seeds'], seeds)

    def test_geometry_is_the_shared_step_and_fac_override_is_independent(self):
        def field(x, y, z):
            return -np.sin(y), np.sin(x), np.ones_like(z)

        axis = np.linspace(-2., 2., 5)
        grid = GriddedField(axis, axis, axis,
                            *field(*np.meshgrid(axis, axis, axis, indexing='ij')))
        for geometry, override in ((.1, None), (.2, None), (.2, .3),
                                    (.2, [.1, .3, .4]), (None, .3)):
            with self.subTest(geometry=geometry, override=override):
                group = model_session()['groups'][0]
                group['analysis'].update(geometry_delta=geometry, delta=override,
                                         mask_radius=0., current_scale=1., seeds=[])
                with patch('mageometry.session.engine.load_source', return_value=(grid, field)):
                    engine = SessionEngine(group)
                geom_step = .002 if geometry is None else geometry
                fac_step = geom_step if override is None else override
                steps = np.broadcast_to(fac_step, (3,))
                fac = engine.prepare(dict(group['view'], component='fac'))
                alpha = engine.prepare(dict(group['view'], component='alpha'))
                self.assertAlmostEqual(fac['values'][2, 2, 2],
                                       np.sin(steps[0])/steps[0] + np.sin(steps[1])/steps[1])
                self.assertAlmostEqual(alpha['values'][2, 2, 2], 2*np.sin(geom_step)/geom_step)
                self.assertEqual(fac['resolved']['fac_delta'], fac_step)
                self.assertEqual(alpha['resolved']['geometry_delta'], geom_step)

    def test_every_diagnostic_matches_existing_direct_comparison(self):
        group = small_session()['groups'][0]
        engine = SessionEngine(group)
        baseline = _OverviewData(engine.grids, fields=engine.fields, comparison=True, **engine.options)
        for component in COMPONENTS:
            with self.subTest(component=component):
                view = dict(group['view'], component=component)
                result = engine.prepare(view)
                expected = baseline.prepare(view['case'], component)
                np.testing.assert_allclose(result['values'], expected.values, equal_nan=True)
                if component == 'gamma':
                    eta, _ = baseline.values(baseline.get(view['case']), 'eta')
                    np.testing.assert_allclose(result['eta_values'], eta, equal_nan=True)
                self.assertEqual(result['scale']['limit'], expected.scale.limit)
                if result['basis'] is not None:
                    np.testing.assert_allclose(result['basis'], expected.basis, equal_nan=True)

    def test_case_and_contribution_have_common_scales_and_total_traces(self):
        group = small_session()['groups'][0]
        group['analysis']['kind'] = 'attribution'
        group['analysis']['cache_size'] = 1
        group['analysis']['seeds'] = [[-6., 2., 1.]]
        group['analysis']['trace']['max_steps'] = 3
        for case in group['cases']:
            case['background'] = dict(kind='dipole', parameters={'epoch': 100.})
        engine = SessionEngine(group)
        for component in TRANSVERSE_COMPONENTS:
            limits, percentiles = [], []
            for case in group['cases']:
                options = dict(engine.options, field=engine.fields[case['id']])
                expected = _ContributionData(engine.grids[case['id']], engine.backgrounds[case['id']], **options)
                paths = None
                for branch in ('total', 'background', 'residual'):
                    result = engine.prepare(dict(group['view'], case=case['id'], component=component, contribution=branch))
                    values, _ = expected.values(expected.get(branch), component)
                    np.testing.assert_allclose(result['values'], values, equal_nan=True)
                    if component == 'gamma':
                        eta, _ = expected.values(expected.get(branch), 'eta')
                        np.testing.assert_allclose(result['eta_values'], eta, equal_nan=True)
                    finite = np.abs(values[np.isfinite(values)])
                    if len(finite):
                        percentiles.append(np.percentile(finite, 98))
                    limits.append(result['scale']['limit'])
                    if paths is not None:
                        np.testing.assert_allclose(paths[0], result['paths'][0], equal_nan=True)
                    paths = result['paths']
            self.assertEqual(len(set(limits)), 1)
            self.assertAlmostEqual(limits[0], 1. if component == 'eta' else max(percentiles))

    def test_grid_mode_uses_preview_spacing_and_requires_matching_axes(self):
        group = small_session()['groups'][0]
        group['analysis'].update(evaluation='grid', delta=None, geometry_delta=None, max_points=27)
        engine = SessionEngine(group)
        result = engine.prepare(group['view'])
        self.assertEqual(result['resolved']['preview_shape'], [3, 3, 3])
        self.assertEqual(result['resolved']['geometry_delta'], 8.)
        group['cases'][1]['source']['bounds'][0] = [-14, 5]
        with self.assertRaisesRegex(ValueError, 'different axes'):
            SessionEngine(group)

    @unittest.skipUnless(importlib.util.find_spec('h5py'), 'h5py unavailable')
    def test_file_identity_change_requires_explicit_revision(self):
        import h5py
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'field.h5'
            with h5py.File(path, 'w') as stream:
                for name in ('BX', 'BY', 'BZ'):
                    stream[name] = np.ones((3, 3, 3))
            session = empty_session()
            group = session['groups'][0]
            group['cases'] = [dict(id='a', label='A', background=None,
                                   source=dict(kind='hdf5', path=str(path),
                                               options={'origin': [0, 0, 0], 'spacing': [1, 1, 1]}))]
            group['reference'] = group['view']['case'] = 'a'
            group['analysis']['seeds'] = []
            engine = SessionEngine(group)
            group['resolved'] = engine.prepare(group['view'])['resolved']
            with h5py.File(path, 'a') as stream:
                stream['BX'][0, 0, 0] = 2.
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                engine.prepare(group['view'])
            with self.assertRaisesRegex(ValueError, 'input content changed'):
                SessionEngine(group)
            group.pop('resolved')
            SessionEngine(group).prepare(group['view'])


class TestSessionJobs(unittest.TestCase):
    def test_worker_publishes_field_then_paths_and_reuses_inputs_on_trace_edit(self):
        runner = JobRunner()
        try:
            group = small_session()['groups'][0]
            group['analysis'].update(seeds=[[-6., 2., 1.], [-8., 2., 1.]])
            group['analysis']['trace']['max_steps'] = 5
            for repeat in range(2):
                group['analysis']['trace']['ds'] = .1 + .05 * repeat
                token = runner.submit(group)
                received, messages = [], []
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    for current, kind, value in runner.poll():
                        self.assertEqual(current, token)
                        self.assertNotEqual(kind, 'error', value)
                        if kind == 'progress':
                            messages.append(value)
                        else:
                            received.append((kind, value))
                    if received and received[-1][0] == 'traces':
                        break
                    time.sleep(.01)
                self.assertEqual([kind for kind, _ in received], ['result', 'traces'])
                self.assertEqual(received[0][1]['trace_status'], 'pending')
                self.assertEqual(received[0][1]['paths'], [])
                self.assertEqual(len(received[1][1]), 2)
                if repeat:
                    self.assertFalse(any(message.startswith('Loading') for message in messages))
                    self.assertFalse(any(message.startswith('Preparing') for message in messages))
                else:
                    self.assertTrue(any(message.startswith('Loading') for message in messages))
        finally:
            runner.close()

    def test_cancelled_request_cannot_replace_latest_result(self):
        runner = JobRunner()
        try:
            group = small_session()['groups'][0]
            old = runner.submit(group)
            runner.cancel()
            group['view']['case'] = group['cases'][1]['id']
            latest = runner.submit(group)
            deadline = time.monotonic() + 30
            results = []
            while time.monotonic() < deadline:
                events = runner.poll()
                for token, kind, value in events:
                    self.assertEqual(token, latest)
                    self.assertNotEqual(token, old)
                    self.assertNotEqual(kind, 'error', value)
                    if kind == 'result':
                        results.append(value)
                if results:
                    break
                time.sleep(.03)
            self.assertTrue(results, 'Worker did not return a result.')
            self.assertEqual(results[0]['case'], group['cases'][1]['id'])
        finally:
            runner.close()


if __name__ == '__main__':
    unittest.main()
