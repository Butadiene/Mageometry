"""Portable recipes, combined comparison/attribution, and serial job behavior."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from mageometry import geopack
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
            group['view'].update(layout='slice', panels_hidden=True, normal=[0., 1., 0.],
                                 origin=[-6., 2., 1.], thresholds={'field:alpha': .125})
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
        group['analysis'].update(evaluation='grid', delta=None, max_points=27)
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
