"""Along-field alpha/FAC gradients: analytic references and viewer semantics."""

import unittest
from unittest.mock import patch

import numpy as np

from mageometry import GriddedField, field_aligned_current_derivatives
from mageometry.viz3d._current import ALONG_FIELD_COMPONENTS, _component_label
from mageometry.viz3d._overview_data import _OverviewData


def bipolar(x, y, z):
    return -y * z, x * z, np.ones_like(x)


class TestAlongFieldDerivatives(unittest.TestCase):
    def test_analytic_gradients_broadcast_and_scalar(self):
        x = np.array([[0.], [.5]])
        z = np.array([-1., 0., 3.])
        squared_magnitude = 1 + x*x*z*z
        expected = 2 * (1 - x*x*z*z) / squared_magnitude**2.5
        result = field_aligned_current_derivatives(bipolar, x, 0., z, delta=1e-4)
        np.testing.assert_allclose(result['dalpha_ds'], expected, atol=2e-8)
        np.testing.assert_allclose(result['dalpha_ds_over_B'],
                                   expected / np.sqrt(squared_magnitude), atol=2e-8)
        np.testing.assert_allclose(result['dfac_ds'], 2 / squared_magnitude**2, atol=2e-8)
        for value in field_aligned_current_derivatives(bipolar, 0., 0., 1.).values():
            self.assertIsInstance(value, float)
            self.assertAlmostEqual(value, 2.)

    def test_field_strength_and_direction(self):
        reference = field_aligned_current_derivatives(bipolar, .2, .3, .7, delta=1e-3)
        for scale in (3., -2.):
            def field(x, y, z):
                return tuple(scale * b for b in bipolar(x, y, z))
            result = field_aligned_current_derivatives(field, .2, .3, .7, delta=1e-3)
            for key, factor in (('dalpha_ds', np.sign(scale)),
                                ('dalpha_ds_over_B', 1 / scale), ('dfac_ds', scale)):
                self.assertAlmostEqual(result[key], reference[key] * factor, places=8)

    def test_fac_override_does_not_change_alpha_derivatives(self):
        def field(x, y, z):
            return -y * np.sin(z), x * np.sin(z), 1. + 0.2 * np.sin(x)
        normal = field_aligned_current_derivatives(field, .4, .3, 1., delta=.002)
        override = field_aligned_current_derivatives(field, .4, .3, 1., delta=.002,
                                                     fac_delta=(.4, .2, .3))
        self.assertEqual(normal['dalpha_ds'], override['dalpha_ds'])
        self.assertEqual(normal['dalpha_ds_over_B'], override['dalpha_ds_over_B'])
        self.assertGreater(abs(normal['dfac_ds'] - override['dfac_ds']), 1e-5)

    def test_zero_curvature_nulls_and_incomplete_stencils(self):
        def straight(x, y, z):
            return np.sin(z), np.cos(z), np.zeros_like(z)
        for value in field_aligned_current_derivatives(straight, .4, .2, .3).values():
            self.assertAlmostEqual(value, 0.)
        for value in field_aligned_current_derivatives(lambda x, y, z: (x, y, z), 0., 0., 0.).values():
            self.assertTrue(np.isnan(value))

        def finite_domain(x, y, z):
            self.assertTrue(np.all(np.isfinite(x) & np.isfinite(y) & np.isfinite(z)))
            return tuple(np.where(z < 1., b, np.nan) for b in bipolar(x, y, z))
        result = field_aligned_current_derivatives(finite_domain, [0., 0., np.nan], 0.,
                                                   [.5, .985, .5], delta=.01)
        for value in result.values():
            self.assertTrue(np.isfinite(value[0]))
            self.assertTrue(np.all(np.isnan(value[1:])))
        for delta in (0., -1., np.nan, np.inf, [0.1, 0.2]):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                field_aligned_current_derivatives(bipolar, 0., 0., 0., delta=delta)
        for step in (0., np.nan, [1., 0., 1.], [.1, .2]):
            with self.subTest(fac_delta=step), self.assertRaises(ValueError):
                field_aligned_current_derivatives(bipolar, 0., 0., 0., fac_delta=step)


class TestAlongFieldPreview(unittest.TestCase):
    def setUp(self):
        axis = np.linspace(-1., 1., 13)
        self.grid = GriddedField(axis, axis, axis,
                                 *bipolar(*np.meshgrid(axis, axis, axis, indexing='ij')),
                                 metadata={'field_unit': 'nT', 'length_unit': 'Re'})

    def test_scaling_lazy_cache_and_legacy_selection_order(self):
        data = _OverviewData({'A': self.grid}, field=bipolar, delta=.001,
                              current_scale=.125)
        cache = data.get('A').cache
        cache.get('mu0J_T')
        self.assertTrue(ALONG_FIELD_COMPONENTS.isdisjoint(cache.values))
        with patch('mageometry.viz3d._current.field_line_current_density',
                   side_effect=AssertionError('No Frenet derivatives needed')):
            for key in ALONG_FIELD_COMPONENTS:
                selection = data.prepare('A', key)
                self.assertIsNone(selection.basis)
                self.assertTrue(np.any(np.isfinite(selection.values)))
                factor = .125 if key == 'dfac_ds' else 1.
                self.assertAlmostEqual(selection.values[6, 6, 6], 2 * factor)
        with patch('mageometry.viz3d._current.field_aligned_current_derivatives',
                   side_effect=AssertionError('Derivatives must be cached')):
            for key in ALONG_FIELD_COMPONENTS:
                data.prepare('A', key)

    def test_grid_gradients_nonuniform_axes_boundaries_and_holes(self):
        axis = np.array([-1., -.7, -.4, 0., .2, .6, 1.])
        grid = GriddedField(axis, axis, axis,
                            *bipolar(*np.meshgrid(axis, axis, axis, indexing='ij')))
        data = _OverviewData({'A': grid}, geometry_delta=.1)
        for key in ALONG_FIELD_COMPONENTS:
            values = data.prepare('A', key).values
            np.testing.assert_allclose(values[3, 3, 2:-2], 2., atol=1e-12)
            for dim in range(3):
                self.assertTrue(np.all(np.isnan(np.take(values, [0, 1, -2, -1], axis=dim))))
        grid.b[3, 3, 3] = np.nan
        data = _OverviewData({'A': grid}, geometry_delta=.1)
        for key in ALONG_FIELD_COMPONENTS:
            values = data.prepare('A', key).values
            self.assertTrue(np.isnan(values[3, 3, 3]))
            self.assertTrue(np.isnan(values[3, 3, 2]))

    def test_grid_convergence_to_analytic_gradients(self):
        errors = []
        for size in (13, 25):
            axis = np.linspace(-1, 1, size)
            coords = np.meshgrid(axis, axis, axis, indexing='ij')
            data = _OverviewData({'A': GriddedField(axis, axis, axis, *bipolar(*coords))})
            exact = field_aligned_current_derivatives(bipolar, *coords, delta=1e-4)
            errors.append({key: np.nanmax(np.abs(data.prepare('A', key).values - exact[key]))
                           for key in ALONG_FIELD_COMPONENTS})
        for key in ALONG_FIELD_COMPONENTS:
            self.assertLess(errors[1][key], .4 * errors[0][key])

    def test_default_direct_fac_steps_and_units(self):
        def field(x, y, z):
            return -y * np.sin(z), x * np.sin(z), 1. + 0.2 * np.sin(x)
        data = _OverviewData({'A': self.grid}, field=field, geometry_delta=.003)
        prepared = data.get('A')
        np.testing.assert_allclose(prepared.cache.fac_delta, np.ones(3) / 12)
        self.assertEqual(_component_label('dalpha_ds', 'nA/m^2', 'Re'),
                         'dalpha/ds [1 / Re^2]')
        self.assertEqual(_component_label('dalpha_ds_over_B', 'nA/m^2', 'Re', field_unit='nT'),
                         '(dalpha/ds) / |B| [1 / (nT Re^2)]')
        self.assertEqual(_component_label('dfac_ds', 'nA/m^2', 'Re'),
                         'dJ parallel/ds [(nA/m^2) / Re]')
        self.assertEqual(_component_label('dfac_ds', None, 'Re', field_unit='nT'),
                         'd(mu0 J parallel)/ds [nT / Re^2]')


if __name__ == '__main__':
    unittest.main()
