"""Input, diagnostic and trace reuse across staged numerical requests."""

from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from mageometry.io import GriddedField
from mageometry.session import SessionEngine, model_session
from mageometry.session.jobs import Cancelled
from mageometry.tracing import trace_field_lines


def field(x, y, z):
    return -np.asarray(y), np.asarray(x), np.ones_like(z)


def sample_source(source, mask_radius, check):
    check()
    axes = [np.linspace(*bounds, n) for bounds, n in zip(source['bounds'], source['shape'])]
    return GriddedField(*axes, *field(*np.meshgrid(*axes, indexing='ij'))), field


def group():
    result = model_session()['groups'][0]
    result['cases'][0]['source'].update(bounds=[[-2., 2.]] * 3, shape=[5, 5, 5])
    result['analysis'].update(mask_radius=0., seeds=[[1., 0., 0.], [.5, .5, 0.]],
                              trace=dict(ds=.1, max_steps=8))
    return result


class TestSessionReuse(unittest.TestCase):
    def setUp(self):
        loader = patch('mageometry.session.engine.load_source', side_effect=sample_source)
        self.load = loader.start()
        self.addCleanup(loader.stop)

    def test_trace_edits_reuse_inputs_and_diagnostics_and_match_fresh_result(self):
        recipe = group()
        first = SessionEngine(recipe)
        before = first.prepare(recipe['view'])
        recipe['analysis']['trace']['ds'] = .05
        with patch('mageometry.session.engine.fingerprint', side_effect=AssertionError('Rehashed input')):
            second = SessionEngine(recipe, previous=first)
        self.assertIs(second.total, first.total)
        self.assertEqual(self.load.call_count, 1)
        actual = second.prepare(recipe['view'])
        expected = SessionEngine(recipe).prepare(recipe['view'])
        np.testing.assert_allclose(actual['values'], before['values'], equal_nan=True)
        self.assertEqual(actual['scale'], before['scale'])
        for a, b in zip(actual['paths'], expected['paths']):
            np.testing.assert_allclose(a, b)
        self.assertFalse(np.array_equal(actual['paths'][0], before['paths'][0]))

    def test_derivative_edits_reuse_inputs_and_paths_but_rebuild_diagnostics(self):
        recipe = group()
        first = SessionEngine(recipe)
        before = first.prepare(recipe['view'])
        recipe['analysis']['geometry_delta'] *= 2
        second = SessionEngine(recipe, previous=first)
        self.assertIsNot(second.total, first.total)
        self.assertEqual(self.load.call_count, 1)
        with patch('mageometry.session.engine.trace_field_lines', side_effect=AssertionError('Retraced')):
            actual = second.prepare(recipe['view'], include_traces=False)
        self.assertEqual(actual['trace_status'], 'ready')
        self.assertIs(actual['paths'], before['paths'])
        expected = SessionEngine(recipe).prepare(recipe['view'])
        np.testing.assert_allclose(actual['values'], expected['values'], equal_nan=True)

    def test_disabled_tracing_retains_cached_paths_for_reenable(self):
        recipe = group()
        first = SessionEngine(recipe)
        before = first.prepare(recipe['view'])
        recipe['analysis']['trace_enabled'] = False
        second = SessionEngine(recipe, previous=first)
        result = second.prepare(recipe['view'], include_traces=False)
        self.assertEqual(result['trace_status'], 'disabled')
        self.assertEqual(result['paths'], [])
        recipe['analysis']['trace_enabled'] = True
        third = SessionEngine(recipe, previous=second)
        with patch('mageometry.session.engine.trace_field_lines', side_effect=AssertionError('Retraced')):
            self.assertIs(third.prepare(recipe['view'])['paths'], before['paths'])
        self.assertIs(third.total, first.total)
        self.assertEqual(self.load.call_count, 1)

    def test_resolved_auto_seeds_becoming_explicit_preserve_prepared_work(self):
        recipe = group()
        recipe['analysis'].update(seeds=None, n_lines=2)
        first = SessionEngine(recipe)
        before = first.prepare(recipe['view'])
        self.assertTrue(before['resolved']['seeds'])
        recipe['analysis']['seeds'] = before['resolved']['seeds']
        second = SessionEngine(recipe, previous=first)
        with patch('mageometry.session.engine.trace_field_lines', side_effect=AssertionError('Retraced')):
            actual = second.prepare(recipe['view'], include_traces=False)
        self.assertIs(second.total, first.total)
        self.assertIs(actual['paths'], before['paths'])
        self.assertEqual(self.load.call_count, 1)

    def test_seed_and_preview_changes_invalidate_paths(self):
        original = group()
        first = SessionEngine(original)
        first.prepare(original['view'])
        for key, value in (('seeds', [[1.5, 0., 0.]]), ('max_points', 27), ('evaluation', 'grid')):
            recipe = deepcopy(original)
            recipe['analysis'][key] = value
            second = SessionEngine(recipe, previous=first)
            result = second.prepare(recipe['view'], include_traces=False)
            self.assertEqual(result['trace_status'], 'pending', key)
        self.assertEqual(self.load.call_count, 1)

    def test_source_edit_and_explicit_reload_discard_affected_work(self):
        recipe = group()
        other = deepcopy(recipe['cases'][0])
        other.update(id='other', label='Other')
        other['source']['parameters']['by'] = 1.
        recipe['cases'].append(other)
        first = SessionEngine(recipe)
        first.prepare(recipe['view'])
        self.assertEqual(self.load.call_count, 2)
        recipe['cases'][1]['source']['parameters']['by'] = 2.
        second = SessionEngine(recipe, previous=first)
        self.assertEqual(self.load.call_count, 3)
        key = recipe['cases'][0]['id']
        self.assertIs(second.grids[key], first.grids[key])
        self.assertIsNot(second.total, first.total)
        recipe['revision'] = 1
        third = SessionEngine(recipe, previous=second)
        self.assertEqual(self.load.call_count, 5)
        self.assertIsNot(third.grids[key], second.grids[key])
        self.assertFalse(third.traces)

    def test_reused_file_identity_is_checked_before_accepting_saved_expectations(self):
        recipe = group()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.bin'
            path.write_bytes(b'original')
            recipe['cases'][0]['source'].update(kind='hdf5', path=str(path))
            first = SessionEngine(recipe)
            recipe['resolved'] = first.prepare(recipe['view'])['resolved']
            bad = deepcopy(recipe)
            key = recipe['cases'][0]['id']
            bad['resolved']['inputs'][key][0]['sha256'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'input content changed'):
                SessionEngine(bad, previous=first)
            path.write_bytes(b'changed contents')
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                SessionEngine(recipe, previous=first)
            recipe.pop('resolved')
            recipe['revision'] = 1
            fresh = SessionEngine(recipe, previous=first)
            self.assertNotEqual(fresh.inputs[key], first.inputs[key])

    def test_staged_result_precedes_trace_and_retains_identical_values(self):
        recipe = group()
        engine = SessionEngine(recipe)
        with patch('mageometry.session.engine.trace_field_lines', side_effect=AssertionError('Early tracing')):
            result = engine.prepare(recipe['view'], include_traces=False)
        self.assertEqual(result['trace_status'], 'pending')
        self.assertEqual(result['paths'], [])
        paths = engine.prepare_traces(recipe['view'])
        # The published packet is not mutated while Queue serializes it.
        self.assertEqual(result['paths'], [])
        self.assertEqual(result['trace_status'], 'pending')
        final = engine.prepare(recipe['view'], include_traces=False)
        self.assertEqual(final['trace_status'], 'ready')
        self.assertIs(final['paths'], paths)
        np.testing.assert_allclose(final['values'], result['values'], equal_nan=True)
        self.assertEqual(final['scale'], result['scale'])

    def test_cancel_inside_batched_integrator_does_not_cache_partial_paths(self):
        recipe = group()
        engine = SessionEngine(recipe)
        engine.prepare(recipe['view'], include_traces=False)
        checks = []

        def cancel(message):
            checks.append(message)
            if len(checks) == 7:
                raise Cancelled()

        engine.progress = cancel
        with self.assertRaises(Cancelled):
            engine.prepare_traces(recipe['view'])
        self.assertFalse(engine.traces)
        engine.progress = lambda message: None
        self.assertEqual(len(engine.prepare_traces(recipe['view'])), 2)

    def test_large_seed_sets_are_bounded_and_match_individual_traces(self):
        recipe = group()
        recipe['analysis']['seeds'] = [[1., y, 0.] for y in np.linspace(-.5, .5, 65)]
        engine = SessionEngine(recipe)
        with patch('mageometry.session.engine.trace_field_lines', wraps=trace_field_lines) as trace:
            result = engine.prepare(recipe['view'])
        self.assertEqual([len(call.args[1]) for call in trace.call_args_list], [32, 32, 1])
        prepared, _, _ = engine._get(recipe['view']['case'], 'total', 'alpha')
        for seed, path in zip(recipe['analysis']['seeds'], result['paths']):
            expected = trace_field_lines(prepared.field, *seed, direction='both',
                                        bounds=prepared.preview.bounds, **recipe['analysis']['trace'])
            np.testing.assert_allclose(path, np.column_stack(expected.path(0)), rtol=1e-12, atol=1e-12)


class TestBatchedModelTraces(unittest.TestCase):
    def test_all_models_match_individual_integration_with_unchanged_options(self):
        for model in ('t89', 't96', 't01', 't04'):
            with self.subTest(model=model):
                recipe = model_session(model=model)['groups'][0]
                recipe['cases'][0]['source']['shape'] = [5, 5, 5]
                recipe['analysis']['trace']['max_steps'] = 8
                engine = SessionEngine(recipe)
                result = engine.prepare(recipe['view'])
                prepared, _, _ = engine._get(recipe['view']['case'], 'total', 'alpha')
                for seed, actual in zip(recipe['analysis']['seeds'], result['paths']):
                    expected = trace_field_lines(prepared.field, *seed, direction='both',
                                                bounds=prepared.preview.bounds, **recipe['analysis']['trace'])
                    np.testing.assert_allclose(actual, np.column_stack(expected.path(0)),
                                               rtol=1e-10, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
