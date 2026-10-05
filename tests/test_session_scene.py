"""Prepared-array scene layouts preserve scientific and camera state."""

from copy import deepcopy
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from mageometry.session.specs import default_analysis, default_view
from mageometry.viz3d.scene import GeometryScene, slice_axes


def packet():
    axis = np.linspace(-2, 2, 7)
    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
    values = x + 2*y - z
    values[0] = np.nan
    return dict(case='a', case_label='Synthetic', component='alpha', contribution='total',
                kind='field', axes=(axis, axis, axis), values=values, basis=None, paths=[],
                metadata={}, scale={'limit': 5., 'peak': 8., 'threshold': 2.},
                label='alpha [1 / grid unit]', analysis=default_analysis())


def gamma_packet():
    result = packet()
    x, y, z = np.meshgrid(*result['axes'], indexing='ij')
    result.update(component='gamma', values=3. + x + .1*y, eta_values=(y - x) / 4.,
                  label='Gamma [1 / grid unit]')
    return result


@unittest.skipUnless(importlib.util.find_spec('pyvista'), 'pyvista unavailable')
class TestSessionScene(unittest.TestCase):
    def setUp(self):
        import pyvista as pv
        self.plotter = pv.Plotter(shape='1|4', off_screen=True, border=False)
        self.scene = GeometryScene(self.plotter)
        self.view = default_view()
        self.view['origin'] = [0., 0., 0.]
        self.scene.set_result(packet(), self.view)

    def tearDown(self):
        self.plotter.close()

    def test_nonnegative_scales_start_at_zero_and_preserve_slice_overrides(self):
        for kind in ('field', 'attribution'):
            for component in ('gamma', 'gamma_over_abs_alpha', 'B_kappa', 'alpha'):
                if kind == 'attribution' and component == 'B_kappa':
                    continue
                with self.subTest(kind=kind, component=component):
                    result = packet()
                    result.update(kind=kind, component=component, values=np.abs(result['values']))
                    if component == 'B_kappa':
                        result['basis'] = np.broadcast_to([0., 0., 1.], result['values'].shape + (3,))
                    key = kind + ':' + component
                    self.view['color_limits'][key] = 4.
                    self.scene.set_result(result, self.view)
                    expected = (0. if component != 'alpha' else -4., 4.)
                    for index, renderer in enumerate(self.plotter.renderers):
                        actor = renderer.actors['slice' if index < 2 else 'projection']
                        self.assertEqual(actor.mapper.scalar_range, expected)
                        self.assertEqual(actor.mapper.lookup_table.scalar_range, expected)
                        self.assertEqual(actor.mapper.lookup_table.cmap.name,
                                         'RdBu_r' if component == 'alpha' else 'Reds')
                        bar = self.plotter.scalar_bars[self.scene._bar_name(index)]
                        self.assertEqual(bar.GetLookupTable().GetRange(), expected)
                    if component == 'B_kappa':
                        self.assertEqual(self.plotter.renderers[0].actors['arrows'].mapper.scalar_range, expected)
                    actors = [dict(renderer.actors) for renderer in self.plotter.renderers]
                    cameras = self.scene.camera_state()
                    self.view['color_limits'][key] = 3.
                    self.view['slice_color_ranges'][key] = [-.2, 2.]
                    self.scene.update_colors()
                    for index, renderer in enumerate(self.plotter.renderers):
                        limits = (-.2, 2.) if index < 2 else (0. if component != 'alpha' else -3., 3.)
                        bar = self.plotter.scalar_bars[self.scene._bar_name(index)]
                        self.assertEqual(bar.GetLookupTable().GetRange(), limits)
                        self.assertEqual(dict(renderer.actors), actors[index])
                    self.assertEqual(self.scene.camera_state(), cameras)
                    self.view['slice_color_ranges'].clear()

    def test_gamma_selects_regions_and_eta_colours_without_sign_filtering_eta(self):
        result = gamma_packet()
        original = deepcopy(result)
        self.view.update(gamma_eta=True, value_sign='positive')
        self.view['thresholds']['field:gamma'] = 2.1
        self.scene.set_result(result, self.view)
        self.plotter.renderers[0].actors['positive'].mapper.Update()
        region = self.plotter.renderers[0].actors['positive'].mapper.dataset
        self.assertGreaterEqual(region['value'].min(), 2.1 - 1e-6)
        self.assertEqual(self.plotter.renderers[0].actors['positive'].mapper.array_name, 'eta')
        np.testing.assert_allclose(region['eta'], (region.points[:, 1] - region.points[:, 0]) / 4., atol=1e-6)
        self.assertLess(region['eta'].min(), 0.)
        self.assertGreater(region['eta'].max(), 0.)
        sliced = self.plotter.renderers[1].actors['slice'].mapper.dataset
        np.testing.assert_allclose(sliced['value'], sliced.points[:, 1] / 4., atol=1e-6)
        self.assertEqual(self.scene.limit, 1.)
        # The filter belongs to Gamma even when every eta magnitude is < 1.
        self.view['threshold_modes']['field:gamma'] = 'interval'
        self.view['value_intervals']['field:gamma'] = [2.2, 2.4]
        self.scene.update_display()
        self.plotter.renderers[0].actors['positive'].mapper.Update()
        region = self.plotter.renderers[0].actors['positive'].mapper.dataset
        np.testing.assert_allclose([region['value'].min(), region['value'].max()], [2.2, 2.4], atol=1e-6)
        self.view['threshold_modes'].clear()
        self.view['thresholds']['field:gamma'] = 100.
        self.scene.update_display()
        self.assertNotIn('positive', self.plotter.renderers[0].actors)
        np.testing.assert_allclose(self.plotter.renderers[1].actors['slice'].mapper.dataset['value'],
                                   sliced['value'], equal_nan=True)
        for key in ('values', 'eta_values'):
            np.testing.assert_allclose(result[key], original[key], equal_nan=True)

    def test_gamma_peaks_use_eta_at_same_location_and_preserve_undefined_colours(self):
        result = gamma_packet()
        result['values'][:, 0, 0] = np.nan
        result['eta_values'][-1, 1, 1] = np.nan
        self.view['gamma_eta'] = True
        self.scene.set_result(result, self.view)
        for interval in (None, [2.1, 3.1], [20., 30.]):
            self.view['threshold_modes']['field:gamma'] = 'absolute' if interval is None else 'interval'
            if interval is not None:
                self.view['value_intervals']['field:gamma'] = interval
            self.scene.update_display()
            gamma = result['values']
            selected = gamma >= 2. if interval is None else (gamma >= interval[0]) & (gamma <= interval[1])
            for axis, renderer in enumerate(self.plotter.renderers[2:]):
                index = np.argmax(np.where(selected, gamma, -np.inf), axis=axis)
                eta = np.take_along_axis(result['eta_values'], np.expand_dims(index, axis), axis=axis).squeeze(axis)
                expected = np.where(np.any(selected, axis=axis), eta, np.nan)
                np.testing.assert_allclose(renderer.actors['projection'].mapper.dataset['value'],
                                           expected.ravel(order='F'), equal_nan=True)
            if interval is None:
                yz = self.plotter.renderers[2].actors['projection'].mapper.dataset['value'].reshape((7, 7), order='F')
                self.assertTrue(np.isnan(yz[1, 1]))
                self.assertTrue(np.isnan(yz[0, 0]))
                actor = self.plotter.renderers[0].actors['positive']
                actor.mapper.Update()
                self.assertTrue(np.any(np.isnan(actor.mapper.dataset['eta'])))
                self.assertEqual(actor.mapper.lookup_table.GetNanColor()[3], 0.)

    def test_gamma_eta_colours_keep_independent_ranges_and_no_stale_legends(self):
        self.scene.set_result(gamma_packet(), self.view)
        cameras = self.scene.camera_state()
        self.view['color_limits']['field:gamma'] = 4.
        self.view['color_limits']['field:eta'] = .8
        self.view['slice_color_ranges']['field:eta'] = [-.2, .5]
        for enabled in (True, False, True):
            self.view['gamma_eta'] = enabled
            self.scene.update_display()
            self.assertEqual(len(self.plotter.scalar_bars), 5)
            self.assertEqual(self.scene.camera_state(), cameras)
            for name in self.plotter.scalar_bars.keys():
                self.assertTrue(name.startswith('eta' if enabled else 'Gamma'))
            actor = self.plotter.renderers[2].actors['projection']
            self.assertEqual(actor.mapper.scalar_range, (-.8, .8) if enabled else (0., 4.))
            self.assertEqual(actor.mapper.lookup_table.cmap.name, 'RdBu_r' if enabled else 'Reds')
        actors = [dict(renderer.actors) for renderer in self.plotter.renderers]
        self.view['color_limits']['field:eta'] = .6
        self.scene.update_colors()
        np.testing.assert_allclose(actors[0]['positive'].mapper.scalar_range, [-.6, .6])
        np.testing.assert_allclose(actors[0]['slice'].mapper.scalar_range, [-.2, .5])
        for index, limits in ((0, [-.6, .6]), (1, [-.2, .5]), (2, [-.6, .6])):
            bar = self.plotter.scalar_bars[self.scene._bar_name(index)]
            np.testing.assert_allclose(bar.GetLookupTable().GetRange(), limits)
        for index, renderer in enumerate(self.plotter.renderers):
            self.assertEqual(dict(renderer.actors), actors[index])
        # With a plane outside the data, the region still has its eta legend.
        self.view['origin'] = [20., 0., 0.]
        self.scene.update_slice()
        self.assertIn(self.scene._bar_name(0), self.plotter.scalar_bars)
        self.assertNotIn(self.scene._bar_name(1), self.plotter.scalar_bars)
        # The remembered option has no effect on unrelated diagnostics.
        self.scene.set_result(packet(), self.view)
        self.assertFalse(self.scene.eta_colors)
        self.assertEqual(self.scene.color_key, 'field:alpha')

    def test_trace_completion_updates_only_lines_and_honours_active_view(self):
        self.scene.set_trace_status('pending')
        self.scene.set_layout('slice')
        self.view['lines'] = False
        cameras = self.scene.camera_state()
        mesh, volume = self.scene.mesh, self.scene.volume
        actors = [dict(renderer.actors) for renderer in self.plotter.renderers]
        paths = [np.array([[-1., 0., 0.], [1., 0., 0.]])]
        with patch.object(self.plotter, 'render') as render:
            self.scene.update_traces(paths)
            render.assert_called_once_with()
        self.assertEqual(self.plotter.renderers.active_index, 1)
        self.assertEqual(self.scene.camera_state(), cameras)
        self.assertIs(self.scene.mesh, mesh)
        self.assertIs(self.scene.volume, volume)
        self.assertFalse(self.plotter.renderers[0].actors['lines'].visibility)
        self.assertNotIn('computing', self.plotter.renderers[0].actors['validity'].GetInput())
        for index, renderer in enumerate(self.plotter.renderers):
            for name, actor in actors[index].items():
                self.assertIs(renderer.actors[name], actor)
        self.scene.update_traces([])
        self.assertNotIn('lines', self.plotter.renderers[0].actors)
        self.assertEqual(self.scene.camera_state(), cameras)

    def test_layouts_preserve_cameras_values_selection_and_plane(self):
        self.plotter.renderers[0].camera.zoom(1.7)
        self.plotter.renderers[1].camera.zoom(1.3)
        cameras = self.scene.camera_state()
        old_values = self.scene.result['values'].copy()
        visibility = {'all': [True]*5, 'three_d_slice': [True, True, False, False, False],
                      'three_d': [True, False, False, False, False],
                      'slice': [False, True, False, False, False]}
        for mode in ('three_d_slice', 'slice', 'three_d_slice', 'three_d', 'all'):
            self.scene.set_layout(mode)
            self.assertEqual(cameras, self.scene.camera_state())
            self.assertEqual(self.view['origin'], [0., 0., 0.])
            np.testing.assert_allclose(self.scene.result['values'], old_values, equal_nan=True)
            visible = [bool(r.GetDraw()) for r in self.plotter.renderers]
            self.assertEqual(visible, visibility[mode])

    def test_combined_layout_fills_height_and_preserves_shared_slice(self):
        self.scene.set_layout('three_d_slice')
        self.assertEqual(self.plotter.renderers[0].GetViewport(), (0., 0., .55, 1.))
        self.assertEqual(self.plotter.renderers[1].GetViewport(), (.55, 0., 1., 1.))
        self.scene.widget.SetOrigin(.5, 0., 0.)
        self.scene.widget.InvokeEvent('EndInteractionEvent')
        np.testing.assert_allclose(self.view['origin'], [.5, 0., 0.])
        for renderer in self.plotter.renderers[:2]:
            np.testing.assert_allclose(renderer.actors['slice'].mapper.dataset.points[:, 0], .5)
        self.scene.set_layout('all')
        self.scene.update_display()
        for index in range(2, 5):
            self.assertIn('projection', self.plotter.renderers[index].actors)

    def test_threshold_changes_do_not_remove_slice_values(self):
        original = self.plotter.renderers[1].actors['slice'].mapper.dataset.copy()
        self.view['thresholds']['field:alpha'] = 1e6
        self.scene.update_display()
        current = self.plotter.renderers[1].actors['slice'].mapper.dataset
        np.testing.assert_allclose(current['value'], original['value'], equal_nan=True)
        self.assertNotIn('positive', self.plotter.renderers[0].actors)

    def test_value_intervals_clip_regions_and_select_peaks_before_projection(self):
        result = packet()
        result['basis'] = np.broadcast_to([1., 0., 0.], result['values'].shape + (3,)).copy()
        self.scene.set_result(result, self.view)
        original = self.plotter.renderers[1].actors['slice'].mapper.dataset['value'].copy()
        cameras = self.scene.camera_state()
        self.view['threshold_modes']['field:alpha'] = 'interval'
        self.view['thresholds']['field:alpha'] = 1e6
        for low, high in ((.4, 1.2), (-2.4, -.3), (-.5, 1.2), (20., 30.)):
            self.view['value_intervals']['field:alpha'] = [low, high]
            self.scene.update_display()
            actors = self.plotter.renderers[0].actors
            for name in ('positive', 'negative', 'arrows'):
                if name in actors:
                    actors[name].mapper.Update()
                    values = actors[name].mapper.dataset['value']
                    self.assertGreaterEqual(values.min(), low - 1e-6)
                    self.assertLessEqual(values.max(), high + 1e-6)
            if low > 0:
                self.assertNotIn('negative', actors)
            if high < 0:
                self.assertNotIn('positive', actors)
            values = result['values']
            selected = np.where((values >= low) & (values <= high), values, np.nan)
            for axis, renderer in enumerate(self.plotter.renderers[2:]):
                indices = np.argmax(np.where(np.isfinite(selected), np.abs(selected), -np.inf), axis=axis)
                expected = np.take_along_axis(selected, np.expand_dims(indices, axis), axis=axis).squeeze(axis)
                np.testing.assert_allclose(renderer.actors['projection'].mapper.dataset['value'],
                                           expected.ravel(order='F'), equal_nan=True)
            np.testing.assert_allclose(self.plotter.renderers[1].actors['slice'].mapper.dataset['value'],
                                       original, equal_nan=True)
            self.assertEqual(self.scene.camera_state(), cameras)
        self.view['value_intervals']['field:alpha'] = [-1., 1.]
        self.view['value_sign'] = 'negative'
        self.scene.update_display()
        self.assertNotIn('positive', self.plotter.renderers[0].actors)
        self.assertIn('negative', self.plotter.renderers[0].actors)

    def test_interval_interpolates_region_even_without_nodes_inside_bounds(self):
        result = packet()
        result['values'] = np.broadcast_to(result['axes'][0][:, None, None], result['values'].shape).copy()
        self.view['threshold_modes']['field:alpha'] = 'interval'
        self.view['value_intervals']['field:alpha'] = [.1, .2]
        self.scene.set_result(result, self.view)
        region = self.plotter.renderers[0].actors['positive'].mapper.dataset
        np.testing.assert_allclose([region['value'].min(), region['value'].max()], [.1, .2], atol=1e-6)
        np.testing.assert_allclose(region.points[:, 0], region['value'], atol=1e-6)
        for renderer in self.plotter.renderers[2:]:
            self.assertFalse(np.any(np.isfinite(renderer.actors['projection'].mapper.dataset['value'])))
        # Zero is a valid interval sample, but has no positive/negative region.
        result['values'][:] = 0.
        self.view['value_intervals']['field:alpha'] = [-1., 1.]
        self.scene.set_result(result, self.view)
        for name in ('positive', 'negative'):
            self.assertNotIn(name, self.plotter.renderers[0].actors)
        for renderer in self.plotter.renderers[2:]:
            np.testing.assert_allclose(renderer.actors['projection'].mapper.dataset['value'], 0.)

    def test_slice_colour_range_changes_both_bars_without_filtering_values(self):
        original = self.plotter.renderers[1].actors['slice'].mapper.dataset['value'].copy()
        cameras = self.scene.camera_state()
        self.view['slice_color_ranges']['field:alpha'] = [-.2, .8]
        self.scene.update_colors()
        for renderer in self.plotter.renderers[:2]:
            actor = renderer.actors['slice']
            np.testing.assert_allclose(actor.mapper.scalar_range, [-.2, .8])
            np.testing.assert_allclose(actor.mapper.dataset['value'], original, equal_nan=True)
        for renderer in self.plotter.renderers[2:]:
            np.testing.assert_allclose(renderer.actors['projection'].mapper.scalar_range, [-5., 5.])
        self.assertEqual(self.scene.camera_state(), cameras)
        self.view['slice_color_ranges'].clear()
        self.scene.update_colors()
        np.testing.assert_allclose(self.plotter.renderers[1].actors['slice'].mapper.scalar_range, [-5., 5.])

    def test_colour_updates_retain_geometry_and_sync_hidden_panels_and_legends(self):
        result = packet()
        result['basis'] = np.broadcast_to([1., 0., 0.], result['values'].shape + (3,)).copy()
        self.scene.set_result(result, self.view)
        self.scene.set_layout('slice')
        cameras = self.scene.camera_state()
        actors = [dict(renderer.actors) for renderer in self.plotter.renderers]
        bars = dict(self.plotter.scalar_bars)
        mappers = [actor.mapper for panel in actors for name, actor in panel.items()
                   if name in ('positive', 'negative', 'arrows', 'slice', 'projection')]
        originals = [(mapper, mapper.GetInputAlgorithm(), mapper.dataset['value'].copy())
                     for mapper in mappers]
        for limit, slice_range in ((2., None), (3., [-.2, .8]), (4., [-.2, .8]), (None, None)):
            self.view['color_limits'].clear()
            self.view['slice_color_ranges'].clear()
            if limit is not None:
                self.view['color_limits']['field:alpha'] = limit
            if slice_range is not None:
                self.view['slice_color_ranges']['field:alpha'] = slice_range
            with patch.object(self.plotter, 'render') as render:
                self.scene.update_colors()
                render.assert_called_once_with()
            shared = [-self.scene.limit, self.scene.limit]
            self.assertEqual(self.scene.limit, 5. if limit is None else limit)
            for index, renderer in enumerate(self.plotter.renderers):
                self.assertEqual(dict(renderer.actors), actors[index])
                limits = slice_range if index < 2 and slice_range is not None else shared
                name = 'slice' if index < 2 else 'projection'
                np.testing.assert_allclose(renderer.actors[name].mapper.scalar_range, limits)
                bar = self.plotter.scalar_bars[f"{result['label']} / shared [{index}]"]
                np.testing.assert_allclose(bar.GetLookupTable().GetRange(), limits)
                suffix = ' / slice range' if index < 2 and slice_range is not None else ' / shared'
                self.assertEqual(bar.GetTitle(), result['label'] + suffix)
            np.testing.assert_allclose(actors[0]['arrows'].mapper.scalar_range, shared)
            self.assertEqual(dict(self.plotter.scalar_bars), bars)
            self.assertEqual(self.scene.camera_state(), cameras)
            self.assertEqual(self.plotter.renderers.active_index, 1)
            for mapper, source, original in originals:
                self.assertIs(mapper.GetInputAlgorithm(), source)
                np.testing.assert_allclose(mapper.dataset['value'], original, equal_nan=True)

    def test_visibility_updates_retain_actors_and_respect_slice_focus(self):
        result = packet()
        result['basis'] = np.broadcast_to([1., 0., 0.], result['values'].shape + (3,)).copy()
        result['paths'] = [np.array([[-1., 0., 0.], [1., 0., 0.]])]
        self.scene.set_result(result, self.view)
        actors = [dict(renderer.actors) for renderer in self.plotter.renderers]
        cameras = self.scene.camera_state()
        for layout in ('all', 'slice', 'three_d_slice', 'three_d'):
            self.scene.set_layout(layout)
            for visible in (False, True):
                for key in ('regions', 'arrows', 'lines', 'plane'):
                    self.view[key] = visible
                with patch.object(self.plotter, 'render') as render:
                    self.scene.update_visibility()
                    render.assert_called_once_with()
                for name in ('positive', 'negative', 'arrows', 'lines', 'slice'):
                    self.assertEqual(actors[0][name].visibility, visible)
                self.assertTrue(actors[1]['slice'].visibility)
                self.assertEqual(bool(self.scene.widget.GetEnabled()), visible and layout != 'slice')
                if self.scene.widget.GetEnabled():
                    self.assertIs(self.scene.widget.GetCurrentRenderer(), self.plotter.renderers[0])
                self.assertEqual(self.scene.camera_state(), cameras)
                for index, renderer in enumerate(self.plotter.renderers):
                    for name in ('positive', 'negative', 'arrows', 'lines', 'slice', 'projection'):
                        if name in actors[index]:
                            self.assertIs(renderer.actors[name], actors[index][name])

    def test_style_updates_handle_missing_actors_and_new_results(self):
        self.view['thresholds']['field:alpha'] = 1e6
        self.view['origin'] = [20., 0., 0.]
        self.scene.update_display()
        self.view['color_limits']['field:alpha'] = 2.
        self.view['regions'] = False
        self.scene.update_colors()
        self.scene.update_visibility()
        for name in ('positive', 'negative', 'arrows', 'slice'):
            self.assertNotIn(name, self.plotter.renderers[0].actors)
        for index in (0, 1):
            self.assertNotIn(f"{self.scene.result['label']} / shared [{index}]", self.plotter.scalar_bars)
        self.view['thresholds'].clear()
        self.view['origin'] = [0., 0., 0.]
        self.scene.set_result(packet(), self.view)
        self.assertFalse(self.plotter.renderers[0].actors['positive'].visibility)
        np.testing.assert_allclose(self.plotter.renderers[1].actors['slice'].mapper.scalar_range, [-2., 2.])
        self.scene.clear()
        self.scene.update_colors()
        self.scene.update_visibility()

    def test_manual_slice_extent_clips_axis_and_oblique_planes_and_clears_empty_bar(self):
        extent = [-.7, .9, -.4, .8]
        self.view['slice_extent'] = extent
        for normal in ([1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]):
            self.scene._drag_plane(normal, [0., 0., 0.])
            horizontal, vertical = slice_axes(normal)
            for renderer in self.plotter.renderers[:2]:
                sliced = renderer.actors['slice'].mapper.dataset
                for axis, low, high in ((horizontal, *extent[:2]), (vertical, *extent[2:])):
                    coords = sliced.points @ axis
                    self.assertGreaterEqual(coords.min(), low - 1e-6)
                    self.assertLessEqual(coords.max(), high + 1e-6)
                x, y, z = sliced.points.T
                np.testing.assert_allclose(sliced['value'], x + 2*y - z, atol=5e-7)
            # Display changes retain the user's zoom after fitting the new extent.
            self.plotter.renderers[1].camera.zoom(1.5)
            cameras = self.scene.camera_state()
            self.scene.update_display()
            self.assertEqual(self.scene.camera_state(), cameras)
        self.view['slice_extent'] = [20., 30., 20., 30.]
        self.scene.update_slice()
        self.assertNotIn('slice', self.plotter.renderers[1].actors)
        self.assertIn('empty', self.plotter.renderers[1].actors)
        self.assertNotIn(f"{self.scene.result['label']} / shared [1]", self.plotter.scalar_bars)
        self.view['slice_extent'] = None
        self.scene.update_slice()
        self.assertIn('slice', self.plotter.renderers[1].actors)

    def test_sign_filter_updates_all_panels_and_preserves_prepared_data(self):
        result = packet()
        result['basis'] = np.broadcast_to([1., 0., 0.], result['values'].shape + (3,)).copy()
        result['paths'] = [np.array([[-1., 0., 0.], [1., 0., 0.]])]
        self.scene.set_result(result, self.view)
        original = result['values'].copy()
        original_slice = self.plotter.renderers[1].actors['slice'].mapper.dataset['value'].copy()
        cameras = self.scene.camera_state()
        cutoff = result['scale']['threshold']
        for mode, sign, excluded in (('positive', 1, 'negative'), ('negative', -1, 'positive')):
            with self.subTest(mode=mode):
                self.view['value_sign'] = mode
                self.scene.update_display()
                actors = self.plotter.renderers[0].actors
                self.assertIn(mode, actors)
                self.assertNotIn(excluded, actors)
                actors['arrows'].mapper.Update()
                self.assertTrue(np.all(sign * actors['arrows'].mapper.dataset['value'] >= cutoff))
                np.testing.assert_allclose(actors['arrows'].mapper.dataset['GlyphVector'],
                                           np.broadcast_to([sign, 0., 0.],
                                                           actors['arrows'].mapper.dataset['GlyphVector'].shape))
                self.assertTrue(actors['lines'].visibility)
                expected_slice = np.where(sign * original_slice > 0, original_slice, np.nan)
                for renderer in self.plotter.renderers[:2]:
                    np.testing.assert_allclose(renderer.actors['slice'].mapper.dataset['value'],
                                               expected_slice, equal_nan=True)
                for axis, renderer in enumerate(self.plotter.renderers[2:]):
                    # Select within the requested sign before finding the peak.
                    strongest = np.max(np.where(sign * original > 0, sign * original, -np.inf), axis=axis)
                    expected = np.where(strongest >= cutoff, sign * strongest, np.nan)
                    np.testing.assert_allclose(renderer.actors['projection'].mapper.dataset['value'],
                                               expected.ravel(order='F'), equal_nan=True)
                self.assertEqual(cameras, self.scene.camera_state())
                self.assertEqual(self.scene.limit, result['scale']['limit'])
                np.testing.assert_allclose(result['values'], original, equal_nan=True)
                np.testing.assert_allclose(self.scene.mesh['value'], original.ravel(order='F'), equal_nan=True)
        self.view['value_sign'] = 'both'
        self.scene.update_display()
        np.testing.assert_allclose(self.plotter.renderers[1].actors['slice'].mapper.dataset['value'],
                                   original_slice, equal_nan=True)
        self.assertIn('positive', self.plotter.renderers[0].actors)
        self.assertIn('negative', self.plotter.renderers[0].actors)

    def test_sign_filter_recomputes_peaks_within_selected_sign(self):
        result = packet()
        result['values'] = np.broadcast_to(np.array([-9., -6., -3., 0., 1., 2., 4.])[:, None, None],
                                          result['values'].shape).copy()
        self.scene.set_result(result, self.view)
        for mode, expected in (('both', -9.), ('positive', 4.), ('negative', -9.), ('both', -9.)):
            self.view['value_sign'] = mode
            self.scene.update_display()
            np.testing.assert_allclose(self.plotter.renderers[2].actors['projection'].mapper.dataset['value'],
                                       expected)

    def test_sign_filter_handles_zero_missing_and_absent_sign(self):
        for value in (0., 1., np.nan):
            with self.subTest(value=value):
                result = packet()
                result['values'][:] = value
                result['basis'] = np.broadcast_to([1., 0., 0.], result['values'].shape + (3,)).copy()
                self.view['value_sign'] = 'negative'
                self.scene.set_result(result, self.view)
                self.assertIn('empty', self.plotter.renderers[1].actors)
                for name in ('positive', 'negative', 'arrows'):
                    self.assertNotIn(name, self.plotter.renderers[0].actors)
                for renderer in self.plotter.renderers[2:]:
                    self.assertFalse(np.any(np.isfinite(renderer.actors['projection'].mapper.dataset['value'])))
        result['values'][:] = 0.
        self.view['value_sign'] = 'positive'
        self.scene.set_result(result, self.view)
        self.assertIn('empty', self.plotter.renderers[1].actors)
        self.view['value_sign'] = 'both'
        self.scene.update_display()
        self.assertNotIn('empty', self.plotter.renderers[1].actors)

    def test_sign_filter_applies_after_slice_interpolation_and_survives_dragging(self):
        self.view['value_sign'] = 'positive'
        self.view['thresholds']['field:alpha'] = 1e6
        self.scene._drag_plane([1., 1., 0.], [.25, 0., 0.])
        sliced = self.plotter.renderers[1].actors['slice'].mapper.dataset
        x, y, z = sliced.points.T
        expected = x + 2*y - z
        expected = np.where(expected > 0, expected, np.nan)
        # VTK stores the interpolated point coordinates at single precision.
        np.testing.assert_allclose(sliced['value'], expected, atol=2e-7, equal_nan=True)
        self.assertTrue(np.any(np.isfinite(sliced['value'])))

    def test_case_update_preserves_focus_and_cameras(self):
        self.scene.set_layout('slice')
        self.plotter.renderers[1].camera.zoom(1.5)
        cameras = self.scene.camera_state()
        result = packet()
        result['values'] *= -1
        result['case'] = 'b'
        self.scene.set_result(result, self.view)
        self.assertEqual(self.view['layout'], 'slice')
        self.assertEqual(cameras, self.scene.camera_state())

    def test_invalid_packet_leaves_previous_scene(self):
        result = self.scene.result
        actors = dict(self.plotter.renderers[0].actors)
        invalid = packet()
        invalid['values'] = np.zeros((2, 2, 2))
        with self.assertRaises(ValueError):
            self.scene.set_result(invalid, self.view)
        self.assertIs(self.scene.result, result)
        self.assertEqual(actors, dict(self.plotter.renderers[0].actors))

    def test_mouse_rotation_targets_visible_3d_after_layout_changes(self):
        self.view['plane'] = False
        self.plotter.show(auto_close=False, interactive=False)
        iren = self.plotter.iren
        width, height = self.plotter.window_size
        start = (int(.15 * width), int(.8 * height))
        end = (start[0] + 30, start[1] - 20)
        for mode in ('three_d_slice', 'slice', 'three_d_slice', 'three_d', 'three_d', 'all'):
            self.scene.set_layout(mode)
            if mode == 'slice':
                continue
            with self.subTest(mode=mode):
                before = self.scene.camera_state()
                iren._mouse_left_button_press(*start)
                iren._mouse_move(*end)
                iren._mouse_left_button_release(*end)
                after = self.scene.camera_state()
                self.assertNotEqual(before['0']['position'], after['0']['position'])
                for index in range(1, 5):
                    self.assertEqual(before[str(index)], after[str(index)])
                self.assertEqual(self.view['origin'], [0., 0., 0.])

    def test_focus_pan_and_zoom_do_not_change_hidden_cameras(self):
        self.view['plane'] = False
        self.plotter.show(auto_close=False, interactive=False)
        iren = self.plotter.iren
        width, height = self.plotter.window_size
        start = (int(.15 * width), int(.8 * height))
        for mode, selected in (('slice', '1'), ('three_d', '0')):
            self.scene.set_layout(mode)
            before = self.scene.camera_state()
            iren._mouse_middle_button_press(*start)
            iren._mouse_move(start[0] + 20, start[1] - 15)
            iren._mouse_middle_button_release()
            iren.interactor.MouseWheelForwardEvent()
            after = self.scene.camera_state()
            self.assertNotEqual(before[selected]['focal_point'], after[selected]['focal_point'])
            self.assertNotEqual(before[selected]['parallel_scale'], after[selected]['parallel_scale'])
            for key in before.keys() - {selected}:
                self.assertEqual(before[key], after[key])

    def test_3d_turntable_keeps_world_z_upright_without_moving_the_pivot(self):
        self.view['plane'] = False
        self.scene.set_layout('three_d')
        self.plotter.show(auto_close=False, interactive=False)
        iren = self.plotter.iren
        camera = self.plotter.renderers[0].camera
        width, height = self.plotter.window_size
        start = (int(.4 * width), int(.5 * height))
        for parallel in (True, False):
            for control in (0, 1):
                with self.subTest(parallel=parallel, control=control):
                    camera.parallel_projection = parallel
                    # A saved trackball camera can already contain roll.
                    camera.Roll(23.)
                    before = self.scene.camera_state()
                    distance = np.linalg.norm(np.subtract(camera.position, camera.focal_point))
                    iren.interactor.SetControlKey(control)
                    iren._mouse_left_button_press(*start)
                    self.assertEqual(before, self.scene.camera_state())
                    for dx, dy in ((30, -20), (0, -40), (-30, -20), (0, 0)) * 3:
                        iren._mouse_move(start[0] + dx, start[1] + dy)
                        np.testing.assert_allclose(camera.up, [0., 0., 1.], atol=1e-12)
                        transform = camera.GetViewTransformMatrix()
                        self.assertAlmostEqual(transform.GetElement(0, 2), 0.)
                        self.assertGreater(transform.GetElement(1, 2), 0.)
                        np.testing.assert_allclose(camera.focal_point, before['0']['focal_point'])
                        self.assertAlmostEqual(np.linalg.norm(
                            np.subtract(camera.position, camera.focal_point)), distance)
                        self.assertEqual(camera.parallel_scale, before['0']['parallel_scale'])
                    iren._mouse_left_button_release()
                    iren.interactor.SetControlKey(0)
                    after = self.scene.camera_state()
                    for key in before.keys() - {'0'}:
                        self.assertEqual(before[key], after[key])
                    self.assertEqual(iren.style.GetState(), 0)
                    iren._mouse_move(start[0] + 10, start[1] + 10)
                    self.assertEqual(after, self.scene.camera_state())

    def test_3d_turntable_leaves_axis_views_and_stops_before_the_poles(self):
        self.view['plane'] = False
        self.scene.set_layout('three_d')
        self.plotter.show(auto_close=False, interactive=False)
        iren = self.plotter.iren
        camera = self.plotter.renderers[0].camera
        width, height = self.plotter.window_size
        start = (int(.4 * width), int(.5 * height))
        for axis in ('view_xy', 'view_xz', 'view_yz'):
            for negative in (False, True):
                with self.subTest(axis=axis, negative=negative):
                    self.plotter.subplot(0)
                    getattr(self.plotter, axis)(negative=negative)
                    before = self.scene.camera_state()
                    iren._mouse_left_button_press(*start)
                    self.assertEqual(before, self.scene.camera_state())
                    # Exercise both elevation limits, then move away again.
                    for dx, dy in ((20, 20), (40, 100), (60, -100), (80, 0)) * 2:
                        iren._mouse_move(start[0] + dx, start[1] + dy)
                        np.testing.assert_allclose(camera.up, [0., 0., 1.])
                        transform = camera.GetViewTransformMatrix()
                        self.assertAlmostEqual(transform.GetElement(0, 2), 0.)
                        self.assertGreater(transform.GetElement(1, 2), 0.)
                        self.assertTrue(np.all(np.isfinite(camera.position)))
                        self.assertLess(abs(camera.direction[2]), 1.)
                    iren._mouse_left_button_release()
                    self.assertNotEqual(before['0']['position'], list(camera.position))
                    for key in before.keys() - {'0'}:
                        self.assertEqual(before[key], self.scene.camera_state()[key])

    def test_shift_pan_keeps_3d_axis_view_and_releases_before_next_orbit(self):
        self.view['plane'] = False
        self.scene.set_layout('three_d')
        self.plotter.subplot(0)
        self.plotter.view_xy()
        self.plotter.show(auto_close=False, interactive=False)
        iren = self.plotter.iren
        width, height = self.plotter.window_size
        start = (int(.4 * width), int(.5 * height))
        before = self.scene.camera_state()
        iren.interactor.SetShiftKey(1)
        iren._mouse_left_button_press(*start)
        iren._mouse_move(start[0] + 25, start[1] + 20)
        iren._mouse_left_button_release()
        iren.interactor.SetShiftKey(0)
        after = self.scene.camera_state()
        self.assertNotEqual(before['0']['focal_point'], after['0']['focal_point'])
        self.assertEqual(before['0']['up'], after['0']['up'])
        np.testing.assert_allclose(
            np.subtract(before['0']['position'], before['0']['focal_point']),
            np.subtract(after['0']['position'], after['0']['focal_point']))
        self.assertEqual(iren.style.GetState(), 0)
        iren._mouse_move(*start)
        self.assertEqual(after, self.scene.camera_state())
        iren._mouse_left_button_press(*start)
        iren._mouse_move(start[0] + 30, start[1] - 30)
        iren._mouse_left_button_release()
        self.assertNotEqual(after['0']['position'], self.scene.camera_state()['0']['position'])
        for key in before.keys() - {'0'}:
            self.assertEqual(before[key], self.scene.camera_state()[key])

    def test_wheel_zoom_round_trip_targets_pointer_in_both_projections(self):
        self.view['plane'] = False
        self.plotter.show(auto_close=False, interactive=False)
        iren = self.plotter.iren
        width, height = self.plotter.window_size
        for layout, index in (('three_d_slice', 0), ('slice', 1), ('three_d', 0), ('all', 3)):
            self.scene.set_layout(layout)
            renderer = self.plotter.renderers[index]
            x0, y0, x1, y1 = renderer.GetViewport()
            iren._mouse_move(int(.5*(x0 + x1)*width), int(.5*(y0 + y1)*height))
            for parallel in (True, False):
                with self.subTest(layout=layout, parallel=parallel):
                    camera = renderer.camera
                    camera.parallel_projection = parallel
                    before = self.scene.camera_state()
                    offset = np.subtract(camera.position, camera.focal_point)
                    scale = camera.parallel_scale
                    iren.interactor.MouseWheelForwardEvent()
                    np.testing.assert_allclose(
                        np.subtract(camera.position, camera.focal_point),
                        offset if parallel else offset / 1.21, atol=1e-12)
                    self.assertAlmostEqual(camera.parallel_scale, scale / 1.21 if parallel else scale)
                    iren.interactor.MouseWheelBackwardEvent()
                    np.testing.assert_allclose(camera.position, before[str(index)]['position'], atol=1e-12)
                    self.assertAlmostEqual(camera.parallel_scale, scale)
                    self.assertEqual(list(camera.focal_point), before[str(index)]['focal_point'])
                    self.assertEqual(list(camera.up), before[str(index)]['up'])
                    after = self.scene.camera_state()
                    for key in before.keys() - {str(index)}:
                        self.assertEqual(before[key], after[key])

    def test_flat_panels_pan_and_zoom_without_rotation_in_all_layouts(self):
        self.view['normal'] = [1., 2., 3.]
        self.scene.update_slice()
        self.plotter.show(auto_close=False, interactive=False)
        iren = self.plotter.iren
        width, height = self.plotter.window_size
        origin = list(self.view['origin'])
        for layout in ('three_d_slice', 'slice', 'all', 'three_d_slice'):
            self.scene.set_layout(layout)
            for index in range(1, 5):
                renderer = self.plotter.renderers[index]
                if not renderer.GetDraw():
                    continue
                x0, y0, x1, y1 = renderer.GetViewport()
                start = (int((x0 + .5*(x1 - x0))*width), int((y0 + .55*(y1 - y0))*height))
                end = (start[0] + 15, start[1] - 10)
                for control, shift in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    with self.subTest(layout=layout, index=index, control=control, shift=shift):
                        iren.interactor.SetControlKey(control)
                        iren.interactor.SetShiftKey(shift)
                        before = self.scene.camera_state()
                        iren._mouse_left_button_press(*start)
                        iren._mouse_move(*end)
                        iren._mouse_left_button_release()
                        iren.interactor.SetControlKey(0)
                        iren.interactor.SetShiftKey(0)
                        iren.interactor.MouseWheelForwardEvent()
                        after = self.scene.camera_state()
                        key = str(index)
                        np.testing.assert_allclose(
                            np.subtract(after[key]['position'], after[key]['focal_point']),
                            np.subtract(before[key]['position'], before[key]['focal_point']), atol=1e-12)
                        np.testing.assert_allclose(after[key]['up'], before[key]['up'], atol=1e-12)
                        self.assertNotEqual(before[key]['focal_point'], after[key]['focal_point'])
                        self.assertNotEqual(before[key]['parallel_scale'], after[key]['parallel_scale'])
                        for other in before.keys() - {key}:
                            self.assertEqual(before[other], after[other])
                        self.assertEqual(self.view['origin'], origin)
                        self.assertEqual(self.view['normal'], [1., 2., 3.])
                        # Hovering after release must not continue the drag.
                        iren._mouse_move(*start)
                        self.assertEqual(self.scene.camera_state(), after)

    def test_drag_across_3d_and_slice_keeps_the_original_camera_and_releases(self):
        self.view['plane'] = False
        self.scene.update_visibility()
        self.plotter.show(auto_close=False, interactive=False)
        iren = self.plotter.iren
        width, height = self.plotter.window_size
        left = (int(.2*width), int(.5*height))
        right = (int(.8*width), int(.5*height))
        for start, end, key in ((left, right, '0'), (right, left, '1')):
            with self.subTest(start=key):
                before = self.scene.camera_state()
                iren._mouse_left_button_press(*start)
                iren._mouse_move(*end)
                iren._mouse_left_button_release()
                after = self.scene.camera_state()
                self.assertNotEqual(before[key]['position'], after[key]['position'])
                for other in before.keys() - {key}:
                    self.assertEqual(before[other], after[other])
                np.testing.assert_allclose(
                    np.subtract(after['1']['position'], after['1']['focal_point']),
                    np.subtract(before['1']['position'], before['1']['focal_point']), atol=1e-12)
                self.assertEqual(iren.style.GetState(), 0)
                iren._mouse_move(*start)
                self.assertEqual(self.scene.camera_state(), after)

    def test_hidden_renderers_do_not_cover_any_window_corner(self):
        width, height = self.plotter.window_size
        for mode in ('three_d_slice', 'three_d', 'slice'):
            self.scene.set_layout(mode)
            for renderer in self.plotter.renderers:
                if not renderer.GetDraw():
                    for x, y in ((0, 0), (width - 1, height - 1), (width // 2, height // 2)):
                        self.assertFalse(renderer.IsInViewport(x, y))

    def test_plane_returns_to_3d_renderer_after_slice_focus(self):
        self.scene.set_layout('slice')
        # The cursor/active renderer may still belong to a companion panel.
        self.plotter.subplot(4)
        self.scene.set_layout('three_d')
        self.assertTrue(self.scene.widget.GetEnabled())
        self.assertIs(self.scene.widget.GetCurrentRenderer(), self.plotter.renderers[0])
        self.scene.widget.SetOrigin(.5, 0., 0.)
        self.scene.widget.InvokeEvent('EndInteractionEvent')
        np.testing.assert_allclose(self.view['origin'], [.5, 0., 0.])

    def test_region_shading_and_context_match_standalone_settings(self):
        result = packet()
        result['paths'] = [np.array([[-1., 0., 0.], [0., 1., 0.], [1., 0., 0.]])]
        self.scene.set_result(result, self.view)
        actors = self.plotter.renderers[0].actors
        for name in ('positive', 'negative'):
            self.assertIn('Normals', actors[name].mapper.dataset.point_data)
            self.assertEqual(actors[name].prop.opacity, .5)
        self.assertEqual(actors['lines'].prop.opacity, .45)
        self.assertAlmostEqual(actors['lines'].prop.line_width, 1.4)
        self.assertAlmostEqual(self.scene.widget.GetPlaneProperty().GetOpacity(), .08)
        before = np.asarray(self.plotter.renderers[0].bounds)
        self.scene.widget.SetEnabled(False)
        np.testing.assert_allclose(self.plotter.renderers[0].bounds, before)

    def test_lighting_survives_initial_render_and_result_replacements(self):
        # Plotter's default light kit must survive the initial set_result.
        self.assertTrue(self.plotter.renderers[0].lights)
        self.plotter.renderers[0].lights[0].intensity = .37
        lights = [list(renderer.lights) for renderer in self.plotter.renderers]
        for mode in ('three_d', 'slice', 'all'):
            self.scene.set_layout(mode)
            result = packet()
            result['values'] *= -1
            self.scene.set_result(result, self.view)
            for renderer, expected in zip(self.plotter.renderers, lights):
                self.assertEqual(renderer.lights, expected)
            self.assertAlmostEqual(self.plotter.renderers[0].lights[0].intensity, .37)


if __name__ == '__main__':
    unittest.main()
