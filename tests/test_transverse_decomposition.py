"""Common-frame attribution, additive tensors and background validity."""

import unittest

import numpy as np

from mageometry import field_line_transverse_decomposition as decompose
from mageometry.geometry import field_line_transverse_geometry


def affine(gradient, offset):
    gradient, offset = np.asarray(gradient), np.asarray(offset)

    def field(x, y, z):
        points = np.stack(np.broadcast_arrays(x, y, z), axis=-1)
        magnetic = points @ gradient.T + offset
        return tuple(np.moveaxis(magnetic, -1, 0))

    return field


TOTAL = np.array([[2., -1, 1], [2, -2, 0], [0, 0, .5]])
BACKGROUND = np.array([[-1., 3, 2], [3, 1, 0], [2, 0, -.5]])


class TestTransverseDecomposition(unittest.TestCase):
    def setUp(self):
        self.field = affine(TOTAL, [0., 0., 1.])
        self.background = affine(BACKGROUND, [3., 1., 0.])

    def test_analytic_residual_and_additivity(self):
        result = decompose(self.field, self.background, 0., 0., 0.)
        total, background, residual = (result[key] for key in ('total', 'background', 'residual'))
        for key in ('gradient', 'transverse', 'shear', 'trace', 'divergence',
                    'alpha', 'beta_g', 'delta_g'):
            np.testing.assert_allclose(total[key], background[key] + residual[key], atol=2e-13)
        np.testing.assert_allclose(residual['gradient'], TOTAL - BACKGROUND, atol=2e-13)
        self.assertAlmostEqual(residual['alpha'], 3.)
        self.assertAlmostEqual(residual['beta_g'], -5.)
        self.assertAlmostEqual(residual['delta_g'], 6.)
        self.assertAlmostEqual(residual['gamma'], np.sqrt(61.))
        self.assertAlmostEqual(residual['eta'], (9.-61.)/(9.+61.))
        self.assertNotAlmostEqual(residual['gamma'], total['gamma'] - background['gamma'])
        self.assertNotAlmostEqual(residual['eta'], total['eta'] - background['eta'])
        for key, value in field_line_transverse_geometry(self.field, 0., 0., 0.).items():
            actual = result['reference']['curvature'] if key == 'curvature' else total[key]
            self.assertAlmostEqual(actual, value)
        self.assertEqual(total['gradient'].shape, (3, 3))
        self.assertEqual(result['reference']['tangent'].shape, (3,))
        self.assertIsInstance(total['eta'], float)

    def test_background_magnitude_and_direction_do_not_set_reference(self):
        first = decompose(self.field, self.background, 0., 0., 0.)
        # Even a background null is allowed; only its gradient enters attribution.
        second = decompose(self.field, affine(BACKGROUND, [0., 0., 0.]), 0., 0., 0.)
        for branch in first:
            for key in first[branch]:
                np.testing.assert_allclose(first[branch][key], second[branch][key], atol=2e-13)

    def test_broadcast_shapes_and_rotational_covariance(self):
        x, y = np.zeros((2, 1)), np.zeros((1, 3))
        result = decompose(self.field, self.background, x, y, 0., delta=(.01, .02, .03))
        self.assertEqual(result['total']['gradient'].shape, (2, 3, 3, 3))
        self.assertEqual(result['reference']['normal'].shape, (2, 3, 3))
        self.assertEqual(result['residual']['eta'].shape, (2, 3))
        rotation, _ = np.linalg.qr(np.array([[1., 2, 3], [3, 1, 4], [2, 5, 1]]))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        rotated = decompose(affine(rotation @ TOTAL @ rotation.T, rotation @ [0., 0., 1.]),
                            affine(rotation @ BACKGROUND @ rotation.T, rotation @ [3., 1., 0.]),
                            0., 0., 0.)
        base = decompose(self.field, self.background, 0., 0., 0.)
        for branch in ('total', 'background', 'residual'):
            for key in ('alpha', 'beta_g', 'delta_g', 'gamma', 'omega_c', 'eta'):
                self.assertAlmostEqual(rotated[branch][key], base[branch][key], places=11)
            for key in ('gradient', 'transverse', 'shear'):
                np.testing.assert_allclose(rotated[branch][key],
                    rotation @ base[branch][key] @ rotation.T, atol=2e-13)

    def test_invalid_background_does_not_erase_total(self):
        for center_only in (True, False):
            def invalid(x, y, z):
                bad = (x == 0) if center_only else (x > 0)
                return np.where(bad, np.nan, 1.), 0.*y, 0.*z
            result = decompose(self.field, invalid, 0., 0., 0.)
            self.assertTrue(np.isfinite(result['total']['eta']))
            for branch in ('background', 'residual'):
                self.assertTrue(all(np.all(np.isnan(value)) for value in result[branch].values()))
        result = decompose(affine(TOTAL, [0., 0., 0.]), self.background, 0., 0., 0.)
        for branch in result.values():
            self.assertTrue(all(np.all(np.isnan(value)) for value in branch.values()))

    def test_pure_dipole_has_zero_residual_and_undefined_residual_eta(self):
        def dipole(x, y, z):
            r5 = (x*x+y*y+z*z)**2.5
            return 3*x*z/r5, 3*y*z/r5, (2*z*z-x*x-y*y)/r5
        result = decompose(dipole, dipole, 3., 0., 2., delta=1e-4)
        self.assertGreater(result['total']['gamma'], 0.)
        self.assertAlmostEqual(result['total']['eta'], -1.)
        self.assertEqual(result['residual']['gamma'], 0.)
        self.assertTrue(np.isnan(result['residual']['eta']))
        # A uniform perturbation changes the total frame despite zero gradient.
        def perturbed(x, y, z):
            bx, by, bz = dipole(x, y, z)
            return bx, by, bz + .01
        perturbed_result = decompose(perturbed, dipole, 3., 0., 2., delta=1e-4)
        self.assertNotAlmostEqual(perturbed_result['total']['gamma'], result['total']['gamma'])
        np.testing.assert_allclose(perturbed_result['residual']['gradient'], 0., atol=1e-13)

    def test_total_curvature_cutoff_and_weak_gradients(self):
        result = decompose(self.field, self.background, 0., 0., 0., curvature_tol=2.)
        for key in ('total', 'background', 'residual'):
            self.assertTrue(np.isnan(result[key]['beta_g']))
            self.assertTrue(np.isnan(result[key]['delta_g']))
            self.assertTrue(np.isfinite(result[key]['eta']))
        with np.errstate(all='raise'):
            result = decompose(affine(TOTAL*1e-100, [0., 0., 1.]),
                               affine(BACKGROUND*1e-100, [0., 0., 0.]), 0., 0., 0.)
        self.assertAlmostEqual(result['residual']['eta'], -52./70.)


if __name__ == '__main__':
    unittest.main()
