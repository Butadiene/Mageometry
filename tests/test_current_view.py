"""Notebook current decomposition and component-switching regressions."""

import unittest

import numpy as np

from mageometry import (GriddedField, field_line_current_density,
                        field_line_frenet_frame, field_magnitude_derivatives, viz3d)
from mageometry.geometry import field_line_transverse_geometry
from mageometry.viz3d._current import RATE_COMPONENTS, COMPONENTS, _CurrentPreview, _component_label
from mageometry.viz3d.fac import _masked_field, _peak_projection, _region_seeds, _sample_fac
from mageometry.viz3d.mesh import to_rectilinear_grid
from mageometry.viz3d.slicer import _widget_state

try:
    import pyvista as pv
    pv.OFF_SCREEN = True
    HAVE_PV = True
except ImportError:
    HAVE_PV = False


def bipolar(x, y, z):
    return -y * z, x * z, np.ones_like(x)


def grid(field=bipolar):
    axis = np.linspace(-2, 2, 13)
    return GriddedField(axis, axis, axis,
                        *field(*np.meshgrid(axis, axis, axis, indexing='ij')))


def press(plotter, key):
    plotter.iren.interactor.SetKeyCode('\0')
    plotter.iren.interactor.SetKeySym(key)
    plotter.iren.interactor.InvokeEvent('KeyPressEvent')
    plotter.iren.interactor.InvokeEvent('CharEvent')


def click_text(plotter, name):
    renderer = next(r for r in plotter.renderers if name in r.actors)
    text = renderer.actors[name]
    x, y = text.GetPositionCoordinate().GetComputedDisplayValue(renderer)
    plotter.iren.interactor.SetEventPosition(x + 3, y)
    plotter.iren.interactor.InvokeEvent('LeftButtonPressEvent')
    plotter.iren.interactor.InvokeEvent('LeftButtonReleaseEvent')


def select(plotter, component):
    click_text(plotter, 'current-component-value')
    click_text(plotter, f'current-component-option-{component}')


class TestCurrentComponents(unittest.TestCase):

    def test_rates_do_not_evaluate_legacy_frame_derivatives(self):
        from unittest.mock import patch
        preview, fac = _sample_fac(grid(), field=bipolar, delta=0.002)
        cache = _CurrentPreview(preview, fac, bipolar, 0.002)
        with patch('mageometry.viz3d._current.field_line_current_density',
                   side_effect=AssertionError('Rates must not compute legacy currents')):
            for key in RATE_COMPONENTS:
                values, basis = cache.get(key)
                self.assertIsNone(basis)
                self.assertTrue(np.any(np.isfinite(values)))

    def test_notebook_values_bases_and_binormal_terms(self):
        preview, fac = _sample_fac(grid(), field=bipolar, delta=0.002)
        cache = _CurrentPreview(preview, fac, _masked_field(bipolar, None), 0.002)
        coords = np.meshgrid(preview.x, preview.y, preview.z, indexing='ij')
        expected = field_line_current_density(bipolar, *coords, delta=0.002)
        expected.update({k: v for k, v in field_line_transverse_geometry(
            bipolar, *coords, delta=0.002).items() if k in RATE_COMPONENTS})
        frame = field_line_frenet_frame(bipolar, *coords, delta=0.002)
        mag = field_magnitude_derivatives(bipolar, *coords, delta=0.002)
        for key in ('mu0J_T', 'B_dT_dn_b', 'B_dn_db_T', 'B_twist_diff', 'mu0J_n', 'mu0J_b',
                    'mu0J_x', 'mu0J_y', 'mu0J_z', 'alpha', 'sigma', 'q', 'gamma', 'omega_c'):
            actual, basis = cache.get(key)
            np.testing.assert_allclose(actual, expected[key], equal_nan=True)
            if key[-1] in 'xyz':
                np.testing.assert_array_equal(basis[5, 5, 5], np.eye(3)['xyz'.index(key[-1])])
        np.testing.assert_allclose(cache.get('mu0J_n')[1], np.stack(frame[3:6], axis=-1))
        np.testing.assert_allclose(cache.get('mu0J_b')[1], np.stack(frame[6:9], axis=-1))
        np.testing.assert_allclose(cache.get('minus_dB_dn')[0], -mag['dB_dn'])
        actual = cache.get('B_kappa')[0] + cache.get('minus_dB_dn')[0]
        valid = np.isfinite(expected['mu0J_b'])
        np.testing.assert_allclose(actual[valid], expected['mu0J_b'][valid])
        self.assertIsNone(cache.get('alpha')[1])
        # Independent Cartesian reference, away from undefined frames.
        for key, reference in zip(('mu0J_x', 'mu0J_y', 'mu0J_z'),
                                  (-coords[0], -coords[1], 2 * coords[2])):
            np.testing.assert_allclose(expected[key][valid], reference[valid], atol=1e-4)

    def test_caching_masks_and_no_calls_at_nonfinite_points(self):
        calls = []

        def field(x, y, z):
            self.assertTrue(np.all(np.isfinite([x, y, z])))
            self.assertTrue(np.all(x >= 0))
            calls.append(1)
            return bipolar(x, y, z)

        mask = lambda x, y, z: x < 0
        preview, fac = _sample_fac(grid(), field=field, mask=mask, delta=0.002)
        cache = _CurrentPreview(preview, fac, _masked_field(field, mask), 0.002)
        before = len(calls)
        cache.get('fac')
        self.assertEqual(len(calls), before)
        cache.get('mu0J_T')
        count = len(calls)
        self.assertGreater(count, before)
        for key in COMPONENTS:
            self.assertTrue(np.all(np.isnan(cache.get(key)[0][:6])))
        self.assertEqual(len(calls), count)

    def test_straight_field_and_missing_data_stay_undefined(self):
        field = lambda x, y, z: (np.sin(z), np.cos(z), np.zeros_like(z))
        preview, fac = _sample_fac(grid(field), field=field, delta=0.002)
        cache = _CurrentPreview(preview, fac, _masked_field(field, None), 0.002)
        np.testing.assert_allclose(cache.get('fac')[0], 1, atol=1e-6)
        for key in COMPONENTS:
            if key not in ('fac', 'alpha', 'gamma', 'omega_c'):
                self.assertTrue(np.all(np.isnan(cache.get(key)[0])), key)
        np.testing.assert_allclose(cache.get('alpha')[0], 1, atol=1e-6)
        np.testing.assert_allclose(cache.get('gamma')[0], 1, atol=1e-6)
        np.testing.assert_allclose(cache.get('omega_c')[0], 0, atol=2e-8)
        preview.b[:] = np.nan
        empty = _CurrentPreview(preview, np.full(preview.shape, np.nan),
                                lambda *args: self.fail('Missing nodes must not be evaluated'), 0.1)
        for key in COMPONENTS:
            self.assertTrue(np.all(np.isnan(empty.get(key)[0])))

    def test_grid_geometry_matches_cell_scale_notebook_evaluation(self):
        preview, fac = _sample_fac(grid(), max_points=1000)
        step = np.min(np.diff(preview.x))
        field = _masked_field(preview.field(), None)
        cache = _CurrentPreview(preview, fac, field, step)
        coords = np.meshgrid(preview.x, preview.y, preview.z, indexing='ij')
        expected = field_line_current_density(field, *coords, delta=step)
        np.testing.assert_allclose(cache.get('mu0J_b')[0], expected['mu0J_b'])
        self.assertTrue(np.any(np.isfinite(cache.get('mu0J_b')[0])))

    def test_labels_and_invalid_parameters(self):
        self.assertEqual(_component_label('alpha', 'nA/m^2', 'Re'), 'alpha [1 / Re]')
        self.assertEqual(_component_label('mu0J_n', 'nA/m^2', 'Re'), 'J_n [nA/m^2]')
        for key, number in (('B_dT_dn_b', 1), ('B_dn_db_T', 2)):
            self.assertEqual(_component_label(key, 'nA/m^2', 'Re'),
                             f'parallel term {number} [nA/m^2]')
            self.assertEqual(_component_label(key, None, 'Re'),
                             f'parallel term {number} [field unit / length unit]')
        self.assertEqual(_component_label('fac', None, 'Re', 'custom'), 'custom')
        self.assertEqual(_component_label('B_twist_diff', 'nA/m^2', 'Re'),
                         'parallel terms: 1 - 2 [nA/m^2]')
        with self.assertRaises(ValueError):
            viz3d.current_view(grid(), component='J_unknown', show=False)
        if HAVE_PV:
            for step in (0, -1, np.inf, np.nan, (0.1, 0.2)):
                with self.subTest(step=step), self.assertRaises(ValueError):
                    viz3d.current_view(grid(), geometry_delta=step, show=False)

    def test_dipole_curvature_and_pressure_cancel(self):
        def dipole(x, y, z):
            r5 = (x*x + y*y + z*z) ** 2.5
            return 3*x*z/r5, 3*y*z/r5, (2*z*z - x*x - y*y)/r5

        axis = np.linspace(2, 5, 5)
        data = GriddedField(axis, axis, axis,
                            *dipole(*np.meshgrid(axis, axis, axis, indexing='ij')))
        preview, fac = _sample_fac(data, field=dipole, delta=0.002)
        cache = _CurrentPreview(preview, fac, dipole, 0.002)
        tension = cache.get('B_kappa')[0]
        pressure = cache.get('minus_dB_dn')[0]
        self.assertTrue(np.all(tension > 0))
        self.assertTrue(np.all(pressure < 0))
        np.testing.assert_allclose((tension + pressure) / tension, 0, atol=3e-6)


@unittest.skipUnless(HAVE_PV, 'pyvista not installed')
class TestCurrentViewer(unittest.TestCase):

    def tearDown(self):
        pv.close_all()

    def test_switch_updates_regions_projections_arrows_and_thresholds(self):
        p = viz3d.current_view(grid(), field=bipolar, delta=0.002,
                               component='fac', n_lines=0, threshold=0.5, show=False)
        camera = p.camera.copy()
        p.screenshot()  # initialise native keyboard handlers
        select(p, 'mu0J_x')
        for name, sign in (('positive', 1), ('negative', -1)):
            surface = p.actors[f'fac-{name}'].mapper.dataset
            self.assertTrue(np.all(sign * surface['mu0J_x'] > 0))
        arrows = p.actors['fac-arrows'].mapper.dataset
        self.assertTrue(np.all(arrows['GlyphVector'][:, 0] * arrows['mu0J_x'] > 0))
        np.testing.assert_allclose(arrows['GlyphVector'][:, 1:], 0, atol=1e-7)
        cutoff = _widget_state(p).slider_widgets[0].GetRepresentation().GetValue()
        current = field_line_current_density(bipolar, *np.meshgrid(
            grid().x, grid().y, grid().z, indexing='ij'), delta=0.002)['mu0J_x']
        for axis, renderer in enumerate(p.renderers[1:]):
            actor = renderer.actors['fac-projection']
            peak = _peak_projection(current, axis).ravel(order='F')
            np.testing.assert_allclose(actor.mapper.dataset['mu0J_x'],
                                       np.where(np.abs(peak) >= cutoff, peak, np.nan))
            self.assertEqual(actor.mapper.array_name, 'mu0J_x')
        select(p, 'fac')
        self.assertEqual(_widget_state(p).slider_widgets[0].GetRepresentation().GetValue(), 0.5)
        press(p, 'F6')
        self.assertIn('mu0J_T', p.actors['fac-arrows'].mapper.dataset.point_data)
        press(p, 'F5')
        self.assertIn('fac', p.actors['fac-arrows'].mapper.dataset.point_data)
        np.testing.assert_array_equal(p.camera.position, camera.position)
        np.testing.assert_array_equal(p.camera.focal_point, camera.focal_point)
        self.assertEqual(p.camera.parallel_scale, camera.parallel_scale)

    def test_component_arrows_follow_the_signed_frame(self):
        p = viz3d.current_view(grid(), field=bipolar, delta=0.002, n_lines=0, show=False)
        preview, fac = _sample_fac(grid(), field=bipolar, delta=0.002)
        cache = _CurrentPreview(preview, fac, _masked_field(bipolar, None), 0.002)
        mesh = to_rectilinear_grid(preview, quantities=())
        for key in ('mu0J_n', 'mu0J_b', 'B_kappa', 'minus_dB_dn',
                    'B_dT_dn_b', 'B_dn_db_T'):
            select(p, key)
            cutoff = _widget_state(p).slider_widgets[0].GetRepresentation().GetValue()
            values, basis = cache.get(key)
            flat = values.ravel(order='F')
            vectors = basis.reshape((-1, 3), order='F')
            valid = np.where(np.all(np.isfinite(vectors), axis=-1), flat, np.nan)
            ids = _region_seeds(mesh.points, valid, cutoff, 32, 0.07 * mesh.length)
            expected = np.sign(flat[ids, None]) * vectors[ids]
            glyphs = p.actors['fac-arrows'].mapper.dataset
            actual = glyphs['GlyphVector'].reshape((len(ids), -1, 3))[:, 0]
            np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_slice_only_switch_preserves_position_camera_and_cached_field(self):
        calls = []

        def field(x, y, z):
            calls.append(1)
            return bipolar(x, y, z)

        p = viz3d.current_view(grid(), field=field, delta=0.002,
                               component='mu0J_n', slice_only=True, slice_normal='z',
                               slice_origin=(0, 0, 1), n_lines=0, show=False,
                               current_scale=0.125, current_unit='nA/m^2', length_unit='Re')
        count = len(calls)
        selector = p.actors['current-component-value']
        self.assertGreater(selector.GetPositionCoordinate().GetComputedDisplayValue(p.renderer)[0],
                           p.window_size[0] * 0.55)
        p.camera.zoom(1.3)
        camera = p.camera.copy()
        actor = p.actors['fac-slice']
        for component in COMPONENTS:
            select(p, component)
            self.assertIs(p.actors['fac-slice'], actor)
            self.assertEqual(actor.mapper.array_name, component)
            np.testing.assert_allclose(actor.mapper.dataset.points[:, 2], 1)
            self.assertEqual([r.GetDraw() for r in p.renderers], [1, 0, 0, 0])
            self.assertTrue(selector.GetVisibility())
            self.assertFalse(_widget_state(p).slider_widgets[0].GetEnabled())
            self.assertEqual(len(p.scalar_bars), 2)  # projection + isolated slice
            self.assertEqual(p.camera.parallel_scale, camera.parallel_scale)
            np.testing.assert_array_equal(p.camera.position, camera.position)
            np.testing.assert_array_equal(p.camera.focal_point, camera.focal_point)
        self.assertEqual(len(calls), count)
        select(p, 'alpha')
        self.assertNotIn('fac-arrows', p.actors)
        self.assertTrue(all('alpha [1 / Re]' in title for title in p.scalar_bars.keys()))
        points = actor.mapper.dataset.points
        expected = field_line_transverse_geometry(bipolar, *points.T, delta=0.002)['alpha']
        # This plane lies exactly on grid nodes, so no cross-plane interpolation.
        np.testing.assert_allclose(actor.mapper.dataset['alpha'], expected, atol=1e-10)
        for key in RATE_COMPONENTS:
            select(p, key)
            expected_rate = field_line_transverse_geometry(
                bipolar, *actor.mapper.dataset.points.T, delta=0.002)[key]
            # VTK slice coordinates may be float32; coiling magnifies rounding.
            np.testing.assert_allclose(actor.mapper.dataset[key], expected_rate, atol=3e-8)
            self.assertNotIn('fac-arrows', p.actors)
            self.assertTrue(all('1 / Re' in title for title in p.scalar_bars.keys()))
        self.assertGreater(np.std(p.screenshot()), 5)
        press(p, 'F4')
        self.assertEqual([r.GetDraw() for r in p.renderers], [1, 1, 1, 1])
        self.assertLess(selector.GetPositionCoordinate().GetComputedDisplayValue(p.renderer)[0],
                        p.window_size[0] * 0.55)
        self.assertEqual(len(p.scalar_bars), 1)
        select(p, 'mu0J_n')
        self.assertTrue(p.actors['fac-arrows'].visibility)

    def test_parallel_term_slices_share_units_mask_sum_and_difference(self):
        from mageometry.viz import QUANTITIES

        p = viz3d.current_view(grid(), field=bipolar, delta=0.002,
                               component='B_dT_dn_b', slice_only=True, slice_normal='z',
                               slice_origin=(0, 0, 1), n_lines=0, show=False,
                               current_scale=0.125, current_unit='nA/m^2', length_unit='Re')
        values = {}
        points = p.actors['fac-slice'].mapper.dataset.points.copy()
        expected = field_line_current_density(bipolar, *points.T, delta=0.002)
        for key in ('B_dT_dn_b', 'B_dn_db_T', 'mu0J_T', 'B_twist_diff'):
            select(p, key)
            mesh = p.actors['fac-slice'].mapper.dataset
            np.testing.assert_array_equal(mesh.points, points)
            values[key] = mesh[key].copy()
            np.testing.assert_allclose(values[key], 0.125 * expected[key], atol=1e-12)
            self.assertTrue(all('nA/m^2' in title for title in p.scalar_bars.keys()))
            # The same named terms also work in notebook geometry maps.
            np.testing.assert_allclose(QUANTITIES[key].evaluate(bipolar, *points.T, delta=0.002),
                                       expected[key], atol=1e-12)
        np.testing.assert_allclose(values['B_dT_dn_b'] + values['B_dn_db_T'],
                                   values['mu0J_T'], atol=1e-12)
        np.testing.assert_allclose(values['B_dT_dn_b'] - values['B_dn_db_T'],
                                   values['B_twist_diff'], atol=1e-12)

    def test_initial_difference_regions_and_projections_use_signed_subtraction(self):
        key = 'B_twist_diff'
        p = viz3d.current_view(grid(), field=bipolar, delta=0.002,
                               component=key, n_lines=0, threshold=0.1, show=False)
        current = field_line_current_density(bipolar, *np.meshgrid(
            grid().x, grid().y, grid().z, indexing='ij'), delta=0.002)
        difference = current['B_dT_dn_b'] - current['B_dn_db_T']
        for name, sign in (('positive', 1), ('negative', -1)):
            surface = p.actors[f'fac-{name}'].mapper.dataset
            self.assertTrue(np.all(sign * surface[key] >= 0.1 - 1e-6))
        for axis, renderer in enumerate(p.renderers[1:]):
            actor = renderer.actors['fac-projection']
            peak = _peak_projection(difference, axis).ravel(order='F')
            np.testing.assert_allclose(actor.mapper.dataset[key],
                                       np.where(np.abs(peak) >= 0.1, peak, np.nan))
        self.assertIn('1 - 2', p.actors['current-component-value'].GetInput())

    def test_empty_component_does_not_show_stale_slice_and_restores_fac(self):
        field = lambda x, y, z: (np.sin(z), np.cos(z), np.zeros_like(z))
        p = viz3d.current_view(grid(field), field=field, component='fac',
                               slice_only=True, n_lines=0, show=False)
        actor = p.actors['fac-slice']
        select(p, 'mu0J_T')
        self.assertFalse(actor.visibility)
        self.assertNotIn('fac-positive', p.actors)
        self.assertNotIn('fac-arrows', p.actors)
        self.assertEqual(len(p.scalar_bars), 1)
        select(p, 'fac')
        self.assertTrue(actor.visibility)
        self.assertEqual(len(p.scalar_bars), 2)

    def test_external_plotter_and_hidden_slice_use_new_component(self):
        p = pv.Plotter(shape=(1, 2), off_screen=True)
        p.subplot(0, 0)
        other = p.add_mesh(pv.Sphere(), name='other')
        p.subplot(0, 1)
        viz3d.current_view(grid(), field=bipolar, delta=0.002, component='fac',
                           plotter=p, n_lines=0, show=False, slice_normal='z',
                           slice_origin=(0, 0, 1))
        press(p, 'c')
        p.subplot(0, 0)
        select(p, 'mu0J_b')
        self.assertIs(p.renderer, p.renderers[1])
        self.assertTrue(other.visibility)
        self.assertFalse(p.actors['fac-slice'].visibility)
        press(p, 'c')
        self.assertTrue(p.actors['fac-slice'].visibility)
        self.assertEqual(p.actors['fac-slice'].mapper.array_name, 'mu0J_b')
        self.assertEqual(tuple(p.scalar_bars.keys()), ('Slice: mu0 J_b [field unit / length unit]',))
        press(p, 'F4')
        select(p, 'mu0J_n')
        press(p, 'F4')
        self.assertEqual(len(p.scalar_bars), 1)
        self.assertTrue(other.visibility)
        self.assertTrue(p.actors['fac-slice'].visibility)


if __name__ == '__main__':
    unittest.main()
