"""Shared attribution scales, sampling and interactive total-field context."""

import unittest
from unittest.mock import patch

import numpy as np

from mageometry import GriddedField, viz3d, trace_field_lines
from mageometry.geometry import field_line_transverse_decomposition
from mageometry.viz3d._contribution_data import _ContributionData
from test_geometry_comparison import HAVE_PV, camera_state, click, press
from test_transverse_decomposition import affine, TOTAL, BACKGROUND


def grid(field, metadata=None):
    axis = np.linspace(-1., 1., 9)
    return GriddedField(axis, axis, axis,
                        *field(*np.meshgrid(axis, axis, axis, indexing='ij')), metadata=metadata)


class TestContributionData(unittest.TestCase):
    def setUp(self):
        self.total = affine(TOTAL, [0., 0., 3.])
        self.background = affine(BACKGROUND, [1., 0., 0.])
        self.grid = grid(self.total, {'parameters': {'Run': 'test'}})

    def test_direct_sampling_and_shared_context(self):
        data = _ContributionData(self.grid, self.background, field=self.total,
                                 delta=.002, geometry_delta=.003, current_scale=100.,
                                 background_label='Reference run')
        expected = field_line_transverse_decomposition(self.total, self.background,
                                                       0., 0., 0., delta=.003)
        selections = [data.prepare(key, 'gamma') for key in data.labels]
        for selection in selections:
            self.assertAlmostEqual(selection.values[4, 4, 4], expected[selection.case]['gamma'])
            self.assertEqual(selection.scale, selections[0].scale)
            self.assertIs(selection.prepared.field, selections[0].prepared.field)
            self.assertIs(selection.prepared.preview, selections[0].prepared.preview)
        magnitudes = [np.abs(s.values[np.isfinite(s.values)]) for s in selections]
        self.assertEqual(selections[0].scale.limit, max(np.percentile(v, 98) for v in magnitudes))
        self.assertEqual(selections[0].scale.threshold, np.percentile(magnitudes[0], 90))
        self.assertEqual(data.prepare('residual', 'eta').scale.limit, 1.)
        self.assertEqual(self.grid.metadata, {'parameters': {'Run': 'test'}})
        self.assertEqual(data.metadata('residual')['parameters']['Background'], 'Reference run')

    def test_matching_grid_background_and_downsampling(self):
        background = grid(self.background)
        data = _ContributionData(self.grid, background, max_points=125)
        result = data.prepare('residual', 'eta')
        expected = field_line_transverse_decomposition(self.total, self.background, 0., 0., 0.)
        self.assertAlmostEqual(result.values[2, 2, 2], expected['residual']['eta'])
        self.assertEqual(result.values.shape, (5, 5, 5))
        self.assertTrue(np.all(np.isnan(result.values[0])))
        pure = _ContributionData(self.grid, self.grid)
        self.assertTrue(np.all(np.isnan(pure.prepare('residual', 'eta').values)))
        self.assertEqual(np.nanmax(pure.prepare('residual', 'gamma').values), 0.)

    def test_mask_and_background_nan_preserve_total(self):
        def background(x, y, z):
            self.assertTrue(np.all(x*x+y*y+z*z >= .1**2))
            return tuple(np.where(x > .5, np.nan, c) for c in self.background(x, y, z))
        data = _ContributionData(self.grid, background, field=self.total, delta=.002,
                                 mask=lambda x, y, z: x*x+y*y+z*z < .1**2)
        total = data.prepare('total', 'gamma').values
        residual = data.prepare('residual', 'gamma').values
        self.assertTrue(np.isnan(total[4, 4, 4]))
        self.assertTrue(np.isnan(residual[4, 4, 4]))
        self.assertTrue(np.isfinite(total[7, 4, 4]))
        self.assertTrue(np.isnan(residual[7, 4, 4]))
        self.grid.b[:] = np.nan
        empty = _ContributionData(self.grid, self.background)
        self.assertTrue(np.all(np.isnan(empty.prepare('total', 'eta').values)))

    def test_validation(self):
        for key in ('coordinate_system', 'length_unit', 'field_unit'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                _ContributionData(grid(self.total, {key: 'a'}), grid(self.background, {key: 'b'}))
        for options in ({'color_limits': {'fac': 1.}}, {'background_label': ''}):
            with self.assertRaises(ValueError):
                _ContributionData(self.grid, self.background, **options)
        with self.assertRaises(TypeError):
            _ContributionData(self.grid, object())
        for options in ({'component': 'fac'}, {'contribution': 'unknown'}):
            with self.assertRaises(ValueError):
                viz3d.transverse_contribution_view(self.grid, self.background, show=False, **options)

    @unittest.skipUnless(HAVE_PV, 'pyvista not installed')
    def test_real_switch_preserves_lines_camera_scale_and_focused_slice(self):
        with patch('mageometry.viz3d.fac.trace_field_lines', wraps=trace_field_lines) as trace:
            plotter = viz3d.transverse_contribution_view(
                self.grid, self.background, field=self.total, delta=.002,
                slice_panel=True, slice_normal='x', slice_origin=(0., 0., 0.),
                seeds=[[.5, .5, .5]], n_lines=1, trace_kwargs={'max_steps': 5}, show=False)
            try:
                plotter.render()
                initial_calls = trace.call_count
                self.assertGreater(initial_calls, 0)
                initial_cameras = [camera_state(r) for r in plotter.renderers]
                click(plotter, 'geometry-dataset-value')
                click(plotter, 'geometry-dataset-option-residual')
                self.assertEqual(trace.call_count, initial_calls)
                for renderer, camera in zip(plotter.renderers, initial_cameras):
                    np.testing.assert_allclose(camera_state(renderer), camera)
                actors = plotter.renderers[0].actors
                self.assertIn('not standalone field geometry', actors['fac-component-description'].GetInput())
                self.assertNotIn('current-component-option-fac', actors)
                self.assertEqual(plotter.renderers[1].actors['fac-projection'].mapper.scalar_range, (-1., 1.))
                press(plotter, 'F4')
                press(plotter, 'F7')
                self.assertTrue(actors['geometry-dataset-option-background'].GetTextProperty().GetBold())
                self.assertIn('Background gradient', plotter.renderers[4].actors['fac-panel-title'].GetInput())
                press(plotter, 'F6')
                self.assertEqual(trace.call_count, initial_calls)
            finally:
                plotter.close()


if __name__ == '__main__':
    unittest.main()
