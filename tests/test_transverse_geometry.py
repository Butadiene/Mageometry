"""Analytic transverse maps, validity, broadcasting and rotational invariance."""

import unittest
import numpy as np
from mageometry.geometry import field_line_transverse_geometry as diagnostic


def linear(p=2., q=-1., a=0., d=0., k=1.):
    """At the origin T=z, n=x, b=y and M has rows (a, q), (p, d)."""
    return lambda x, y, z: (a*x+q*y+k*z, p*x+d*y, 1+0*z)


class TestTransverseGeometry(unittest.TestCase):
    def test_analytic_maps(self):
        for p, q, a, d in ((2, -2, 0, 0), (4, 0, 0, 0), (3, 3, 0, 0),
                           (0, 0, 2, -2), (1, -1, 2, -2)):
            r = diagnostic(linear(p, q, a, d), 0., 0., 0.)
            expected = dict(alpha=p-q, beta_g=p+q, delta_g=a-d,
                            gamma=np.hypot(p+q, a-d), curvature=1.)
            expected['omega_c'] = np.sign(p-q)*np.sqrt(max((p-q)**2-expected['gamma']**2, 0))/2
            expected['eta'] = ((p-q)**2 - expected['gamma']**2) / ((p-q)**2 + expected['gamma']**2)
            for key, value in expected.items():
                self.assertIsInstance(r[key], float)
                self.assertAlmostEqual(r[key], value)

    def test_diagnostic_keys_and_matrix_entries_are_distinct(self):
        for x in (0., np.zeros((2, 3))):
            result = diagnostic(linear(p=2, q=-1, a=2, d=-2), x, 0., 0.)
            self.assertEqual(set(result), {'alpha', 'beta_g', 'delta_g', 'gamma',
                                           'omega_c', 'eta', 'curvature'})
            np.testing.assert_allclose(result['alpha'], 3.)
            np.testing.assert_allclose(result['beta_g'], 1.)
            np.testing.assert_allclose(result['delta_g'], 4.)

    def test_current_free_dipole_has_nonzero_anisotropy(self):
        def dipole(x, y, z):
            r5 = (x*x + y*y + z*z)**2.5
            return 3*x*z/r5, 3*y*z/r5, (2*z*z-x*x-y*y)/r5

        r = 4.
        colatitude = np.array([.3, .7, 1.2, 2., 2.7])
        result = diagnostic(dipole, r*np.sin(colatitude), 0., r*np.cos(colatitude), delta=1e-4)
        expected = (3*np.abs(np.cos(colatitude))*np.sin(colatitude)**2
                    / (r*(1+3*np.cos(colatitude)**2)**1.5))
        np.testing.assert_allclose(result['alpha'], 0., atol=1e-8)
        np.testing.assert_allclose(result['beta_g'], 0., atol=1e-8)
        np.testing.assert_allclose(result['gamma'], expected, rtol=2e-7)
        np.testing.assert_allclose(result['eta'], -1., atol=1e-12)

    def test_eta_zero_denominator_and_weak_gradients(self):
        with np.errstate(all='raise'):
            self.assertTrue(np.isnan(diagnostic(linear(0, 0, k=0), 0., 0., 0.)['eta']))
            for strength in (1., 1e-100):
                for p, q, expected in ((2, -2, 1), (3, 3, -1), (4, 0, 0), (2, -1, 0.8)):
                    result = diagnostic(linear(p*strength, q*strength, k=0), 0., 0., 0.)
                    self.assertAlmostEqual(result['eta'], expected)

    def test_straight_null_invalid_and_cutoff(self):
        r = diagnostic(linear(k=0), 0., 0., 0.)
        self.assertTrue(np.isnan(r['beta_g']) and np.isnan(r['delta_g']))
        self.assertAlmostEqual(r['alpha'], 3.)
        self.assertAlmostEqual(r['gamma'], 1.)
        self.assertAlmostEqual(r['omega_c'], np.sqrt(2.))
        self.assertAlmostEqual(r['eta'], 0.8)
        r = diagnostic(linear(), 0., 0., 0., curvature_tol=2.)
        self.assertTrue(np.isnan(r['beta_g']))
        self.assertAlmostEqual(r['gamma'], 1.)
        for field in (lambda x, y, z: (0., 0., 0.),
                      lambda x, y, z: (np.where(x > 0, np.nan, 0.), 0., 1.)):
            self.assertTrue(all(np.isnan(v) for v in diagnostic(field, 0., 0., 0.).values()))
        for delta in (0, -1, np.nan, (1, 0, 1)):
            with self.assertRaises(ValueError):
                diagnostic(linear(), 0., 0., 0., delta=delta)

    def test_broadcast_and_rotation(self):
        base = linear(a=2, d=-2)
        r = diagnostic(base, np.zeros((2, 1)), np.zeros((1, 3)), 0., delta=(.01, .02, .03))
        for value in r.values():
            self.assertEqual(value.shape, (2, 3))
        rotation, _ = np.linalg.qr(np.array([[1., 2, 3], [3, 1, 4], [2, 5, 1]]))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1

        def rotated(x, y, z):
            coords = np.stack(np.broadcast_arrays(x, y, z), axis=-1) @ rotation
            b = np.stack(base(*np.moveaxis(coords, -1, 0)), axis=-1) @ rotation.T
            return tuple(np.moveaxis(b, -1, 0))

        actual = diagnostic(rotated, 0., 0., 0.)
        for key, value in diagnostic(base, 0., 0., 0.).items():
            self.assertAlmostEqual(actual[key], value, places=12)


if __name__ == '__main__':
    unittest.main()
