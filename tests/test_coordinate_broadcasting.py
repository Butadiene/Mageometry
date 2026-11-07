"""Scalar and NumPy broadcasting contracts for transforms and IGRF wrappers."""

import unittest

import numpy as np

from mageometry.geopack import geopack as scalar
from mageometry.geopack.vectorized import coordinates, coordinates_complex, igrf


class TestCoordinateBroadcasting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scalar.recalc(100., vxgse=-400., vygse=15., vzgse=20.)

    def assert_pointwise(self, vector_function, scalar_function, arguments, extra=()):
        broadcast = np.broadcast_arrays(*arguments)
        expected = np.array([scalar_function(*point, *extra)
                             for point in zip(*(c.ravel() for c in broadcast))])
        expected = expected.T.reshape((3,) + broadcast[0].shape)
        actual = vector_function(*arguments, *extra)
        for component in actual:
            self.assertEqual(component.shape, broadcast[0].shape)
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-8)

    def test_cartesian_transforms_broadcast_each_component_in_both_directions(self):
        names = ('gsmgse', 'geigeo', 'magsm', 'smgsm', 'geomag', 'geogsm', 'gswgsm')
        cases = [(np.array([1., 2.]), 3., 4.),
                 (1., np.array([2., 3.]), 4.),
                 (1., 2., np.array([3., 4.])),
                 (np.array([1., 2.])[:, None], np.array([2., 3., 4.]), 4.)]
        for name in names:
            for direction in (-1, 1):
                for coords in cases:
                    with self.subTest(transform=name, direction=direction,
                                      shapes=tuple(np.shape(c) for c in coords)):
                        self.assert_pointwise(getattr(coordinates, name), getattr(scalar, name),
                                              coords, (direction,))
                with self.subTest(transform=name, direction=direction, scalar=True):
                    result = getattr(coordinates, name)(1., 2., 3., direction)
                    self.assertTrue(all(isinstance(c, float) for c in result))
                    np.testing.assert_allclose(result, getattr(scalar, name)(1., 2., 3., direction))

    def test_spherical_coordinate_transforms_broadcast(self):
        cases = (
            ((5., np.array([.5, 1.]), .3), 1),
            ((np.array([4., 5.]), .5, .3), 1),
            ((5., .5, np.array([.3, .6])), 1),
            ((1., np.array([2., 3.]), 4.), -1),
            ((np.array([1., 2.])[:, None], 2., np.array([3., 4., 5.])), -1),
        )
        for coords, direction in cases:
            with self.subTest(direction=direction, shapes=tuple(np.shape(c) for c in coords)):
                self.assert_pointwise(coordinates_complex.sphcar, scalar.sphcar, coords, (direction,))

    def test_magnetic_component_transforms_broadcast_every_input(self):
        for name, base in (('bspcar', [.7, .3, 4., 2., 1.]),
                           ('bcarsp', [4., 2., 1., 3., 2., 1.])):
            for index in range(len(base)):
                arguments = list(base)
                arguments[index] = base[index] + np.array([0., .1])
                with self.subTest(transform=name, vector_input=index):
                    self.assert_pointwise(getattr(coordinates_complex, name), getattr(scalar, name),
                                          arguments)
            result = getattr(coordinates_complex, name)(*base)
            self.assertTrue(all(isinstance(c, float) for c in result))

    def test_igrf_cartesian_wrappers_broadcast(self):
        for name in ('igrf_gsm', 'igrf_gsw'):
            for coords in ((5., np.array([1., 2.]), 1.),
                           (np.array([4., 5.])[:, None], np.array([1., 2., 3.]), 1.)):
                with self.subTest(wrapper=name, shapes=tuple(np.shape(c) for c in coords)):
                    self.assert_pointwise(getattr(igrf, name), getattr(scalar, name), coords)
            result = getattr(igrf, name)(5., 1., 1.)
            self.assertTrue(all(isinstance(c, float) for c in result))


if __name__ == '__main__':
    unittest.main()
