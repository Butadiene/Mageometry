"""Analytic FAC, preview sampling, and interactive viewer regression tests."""

import unittest

import numpy as np

from mageometry import GriddedField, field_aligned_current_density
from mageometry.viz3d.fac import _sample_fac, _peak_projection, _region_seeds

try:
    import pyvista as pv
    pv.OFF_SCREEN = True
    HAVE_PV = True
except ImportError:
    HAVE_PV = False


def bipolar_field(x, y, z):
    """Solenoidal field with curl=(-x,-y,2z), FAC=2z/|B|."""
    return -y * z, x * z, np.ones_like(x)


def grid_field(n=13, field=bipolar_field):
    axis = np.linspace(-2, 2, n)
    return GriddedField(axis, axis, axis,
                        *field(*np.meshgrid(axis, axis, axis, indexing='ij')))


class TestParallelCurrent(unittest.TestCase):

    def test_signed_cartesian_curl_and_broadcast(self):
        x = np.array([[0.0], [0.5]])
        z = np.array([-1., 0., 2.])
        result = field_aligned_current_density(bipolar_field, x, 0., z,
                                               delta=(0.01, 0.02, 0.03))
        np.testing.assert_allclose(result, 2 * z / np.sqrt(1 + (x * z) ** 2),
                                   atol=1e-12)
        self.assertIsInstance(field_aligned_current_density(bipolar_field, 0, 0, 1), float)

    def test_straight_force_free_field_keeps_nonzero_fac(self):
        # T is constant along each field line (fixed z), so the Frenet normal
        # is undefined, but curl B = B and the FAC is nonzero everywhere.
        def field(x, y, z):
            return np.sin(z), np.cos(z), np.zeros_like(z)
        actual = field_aligned_current_density(field, [0, 1], 0, [0.2, 1.3], delta=1e-3)
        np.testing.assert_allclose(actual, 1.0, atol=2e-7)

    def test_null_and_invalid_stencil_are_nan(self):
        null = lambda x, y, z: (x, y, z)
        self.assertTrue(np.isnan(field_aligned_current_density(null, 0, 0, 0)))

        def finite_domain(x, y, z):
            return tuple(np.where(x < 1, c, np.nan) for c in bipolar_field(x, y, z))
        actual = field_aligned_current_density(finite_domain, [0.5, 0.99], 0, 1, delta=0.02)
        self.assertTrue(np.isfinite(actual[0]))
        self.assertTrue(np.isnan(actual[1]))
        for delta in (0, -1, np.nan, np.inf, (1, 0, 1)):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                field_aligned_current_density(bipolar_field, 0, 0, 1, delta=delta)

    def test_current_free_dipole(self):
        def dipole(x, y, z):
            r5 = (x * x + y * y + z * z) ** 2.5
            return 3 * x * z / r5, 3 * y * z / r5, (2 * z * z - x * x - y * y) / r5
        actual = field_aligned_current_density(dipole, [2, 3], [0.3, -0.8], [1, -1], delta=1e-4)
        np.testing.assert_allclose(actual, 0, atol=1e-9)


class TestFACPreview(unittest.TestCase):

    def test_nonuniform_axes_and_invalid_boundaries(self):
        x = np.array([-2, -1.3, -0.4, 0., 1., 2.])
        y = np.linspace(-1, 1, 7)
        z = np.array([-2, -1.1, 0., 0.7, 1.5])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid = GriddedField(x, y, z, *bipolar_field(X, Y, Z))
        _, values = _sample_fac(grid)
        expected = 2 * Z / np.sqrt(1 + Z ** 2 * (X ** 2 + Y ** 2))
        np.testing.assert_allclose(values[1:-1, 1:-1, 1:-1], expected[1:-1, 1:-1, 1:-1], atol=1e-12)
        for axis in range(3):
            self.assertTrue(np.all(np.isnan(np.take(values, [0, -1], axis=axis))))

    def test_missing_samples_and_masks_do_not_become_zero(self):
        grid = grid_field()
        grid.b[6, 6, 6] = np.nan
        _, values = _sample_fac(grid)
        self.assertTrue(np.isnan(values[6, 6, 6]))
        self.assertTrue(np.isnan(values[7, 6, 6]))
        self.assertTrue(np.isnan(values[6, 5, 6]))
        _, analytic = _sample_fac(grid, field=bipolar_field, delta=0.05,
                                   mask=lambda x, y, z: x < 0)
        self.assertTrue(np.all(np.isnan(analytic[:7])))
        self.assertTrue(np.isfinite(analytic[8, 6, 8]))

    def test_masked_model_is_not_evaluated_in_excluded_region(self):
        def model(x, y, z):
            self.assertTrue(np.all(x >= 0))
            return bipolar_field(x, y, z)
        _, values = _sample_fac(grid_field(), field=model, delta=0.05,
                                  mask=lambda x, y, z: x < 0)
        self.assertTrue(np.all(np.isnan(values[:7])))
        self.assertTrue(np.isfinite(values[8, 6, 8]))

    def test_preview_cap_preserves_bounds_and_input(self):
        grid = grid_field(30)
        original = grid.b.copy()
        preview, values = _sample_fac(grid, max_points=1000)
        self.assertLessEqual(values.size, 1000)
        self.assertEqual(preview.bounds, grid.bounds)
        np.testing.assert_array_equal(grid.b, original)
        with self.assertRaises(ValueError):
            _sample_fac(grid, max_points=20)
        with self.assertRaises(ValueError):
            _sample_fac(grid, delta=0.1)

    def test_signed_peak_projection_and_missing_sightlines(self):
        values = np.array([[[1., np.nan], [-7., np.nan]],
                           [[-4., np.nan], [2., np.nan]]])
        np.testing.assert_allclose(_peak_projection(values, 0), [[-4, np.nan], [-7, np.nan]], equal_nan=True)
        np.testing.assert_allclose(_peak_projection(values, 1), [[-7, np.nan], [-4, np.nan]], equal_nan=True)

    def test_seeds_include_both_signs_and_respect_cutoff(self):
        points = np.column_stack([np.arange(8), np.zeros(8), np.zeros(8)])
        values = np.array([10, 9, 8, 7, -2, -3, 0, np.nan])
        ids = _region_seeds(points, values, 2, 2, 1)
        self.assertEqual(set(np.sign(values[ids])), {-1, 1})
        self.assertEqual(_region_seeds(points, values, 11, 2, 1).size, 0)


@unittest.skipUnless(HAVE_PV, 'pyvista not installed')
class TestFACViewer(unittest.TestCase):

    def tearDown(self):
        pv.close_all()

    def test_signed_regions_arrow_direction_and_threshold_callback(self):
        from mageometry import viz3d
        plotter = viz3d.fac_view(grid_field(), n_lines=0, show=False, threshold=0.6)
        self.assertEqual(len(plotter.renderers), 4)
        self.assertIs(plotter.renderer, plotter.renderers[0])
        for name, sign in (('positive', 1), ('negative', -1)):
            surface = plotter.actors[f'fac-{name}'].mapper.dataset
            self.assertTrue(np.all(sign * surface['fac'] >= 0.6 - 1e-6))
        glyphs = plotter.actors['fac-arrows'].mapper.dataset
        self.assertTrue(np.all(glyphs['GlyphVector'][:, 2] * glyphs['fac'] > 0))
        state = plotter.widgets if hasattr(plotter, 'widgets') else plotter
        widget = state.slider_widgets[-1]
        widget.GetRepresentation().SetValue(widget.GetRepresentation().GetMaximumValue())
        widget.InvokeEvent('EndInteractionEvent')
        self.assertNotIn('fac-positive', plotter.actors)
        self.assertNotIn('fac-negative', plotter.actors)
        self.assertNotIn('fac-arrows', plotter.actors)
        for renderer in plotter.renderers[1:]:
            self.assertTrue(np.all(np.isnan(renderer.actors['fac-projection'].mapper.dataset['fac'])))
        widget.GetRepresentation().SetValue(0.5)
        widget.InvokeEvent('EndInteractionEvent')
        self.assertIn('fac-positive', plotter.actors)
        plotter.iren.interactor.SetKeySym('s')
        plotter.iren.interactor.InvokeEvent('KeyPressEvent')
        self.assertFalse(plotter.actors['fac-positive'].visibility)
        widget.GetRepresentation().SetValue(0.7)
        widget.InvokeEvent('EndInteractionEvent')
        self.assertFalse(plotter.actors['fac-positive'].visibility)
        plotter.iren.interactor.SetKeySym('s')
        plotter.iren.interactor.InvokeEvent('KeyPressEvent')
        self.assertTrue(plotter.actors['fac-positive'].visibility)
        image = plotter.screenshot()
        self.assertEqual(image.shape[:2], (960, 1440))
        self.assertGreater(np.std(image), 5)

    def test_zero_missing_and_constant_current_fields(self):
        from mageometry import viz3d
        uniform = lambda x, y, z: (np.zeros_like(x), np.zeros_like(x), np.ones_like(x))
        grid = grid_field(field=uniform)
        for missing in (False, True):
            if missing:
                grid.b[:] = np.nan
            p = viz3d.fac_view(grid, n_lines=0, show=False)
            self.assertNotIn('fac-positive', p.actors)
            self.assertNotIn('fac-negative', p.actors)
            self.assertNotIn('fac-arrows', p.actors)
            p.close()
        force_free = lambda x, y, z: (np.sin(z), np.cos(z), np.zeros_like(z))
        p = viz3d.fac_view(grid_field(field=force_free), field=force_free,
                           threshold=0.5, n_lines=0, show=False)
        self.assertIn('fac-positive', p.actors)
        self.assertNotIn('fac-negative', p.actors)

    def test_external_plotter_lines_and_scaling(self):
        from mageometry import viz3d
        p = pv.Plotter(off_screen=True)
        viz3d.fac_view(grid_field(), plotter=p, n_lines=2,
                       current_scale=0.125, threshold=0.1, show=False)
        self.assertEqual(len(p.renderers), 1)
        self.assertIn('fac-lines', p.actors)
        self.assertLess(np.max(p.actors['fac-positive'].mapper.dataset['fac']), 0.5)
        with self.assertRaises(ValueError):
            viz3d.fac_view(grid_field(), plotter=p, front_view=True, show=False)


if __name__ == '__main__':
    unittest.main()
