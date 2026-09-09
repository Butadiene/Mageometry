"""Analytic transverse maps, validity, broadcasting and rotational invariance."""

import unittest
import numpy as np
from mageometry.geometry import field_line_transverse_geometry as diagnostic


def linear(a=2., c=-1., u=0., v=0., k=1.):
    return lambda x, y, z: (u*x+c*y+k*z, a*x+v*y, 1+0*z)


class TestTransverseGeometry(unittest.TestCase):
    def test_analytic_maps(self):
        for a, c, u, v in ((2, -2, 0, 0), (4, 0, 0, 0), (3, 3, 0, 0), (1, -1, 2, -2)):
            r = diagnostic(linear(a, c, u, v), 0., 0., 0.)
            expected = dict(alpha=a-c, sigma=a+c, q=u-v,
                            gamma=np.hypot(a+c, u-v), curvature=1.)
            expected['omega_c'] = np.sign(a-c)*np.sqrt(max((a-c)**2-expected['gamma']**2, 0))/2
            for key, value in expected.items():
                self.assertIsInstance(r[key], float)
                self.assertAlmostEqual(r[key], value)

    def test_straight_null_invalid_and_cutoff(self):
        r = diagnostic(linear(k=0), 0., 0., 0.)
        self.assertTrue(np.isnan(r['sigma']) and np.isnan(r['q']))
        self.assertAlmostEqual(r['alpha'], 3.)
        self.assertAlmostEqual(r['gamma'], 1.)
        self.assertAlmostEqual(r['omega_c'], np.sqrt(2.))
        r = diagnostic(linear(), 0., 0., 0., curvature_tol=2.)
        self.assertTrue(np.isnan(r['sigma']))
        self.assertAlmostEqual(r['gamma'], 1.)
        for field in (lambda x, y, z: (0., 0., 0.),
                      lambda x, y, z: (np.where(x > 0, np.nan, 0.), 0., 1.)):
            self.assertTrue(all(np.isnan(v) for v in diagnostic(field, 0., 0., 0.).values()))
        for delta in (0, -1, np.nan, (1, 0, 1)):
            with self.assertRaises(ValueError):
                diagnostic(linear(), 0., 0., 0., delta=delta)

    def test_broadcast_and_rotation(self):
        base = linear(u=2, v=-2)
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
