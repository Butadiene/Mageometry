"""Regressions independent of scalar/vectorized implementation agreement."""

import unittest

import numpy as np

from mageometry.geopack import models
from mageometry.geopack.vectorized import models as vectorized


PARMOD = [2., -20., 2., -5., 1., 1., 1., 1., 1., 1.]


class TestModelInputRegressions(unittest.TestCase):
    def test_nan_coordinates_return_nan_for_scalar_inputs(self):
        for name in ('t01', 't96'):
            for axis in range(3):
                point = [-6., 1., 2.]
                point[axis] = np.nan
                with self.subTest(model=name, nan_axis=axis), np.errstate(invalid='ignore'):
                    expected = getattr(models, name)(PARMOD, .2, *point)
                    actual = getattr(vectorized, name)(PARMOD, .2, *point)
                    self.assertTrue(np.all(np.isnan(expected)))
                    self.assertTrue(np.all(np.isnan(actual)))
                    self.assertTrue(all(isinstance(c, float) for c in actual))

    def test_nan_coordinates_preserve_valid_points_in_multidimensional_batches(self):
        valid_points = np.array([[-6., 1., 2.], [10.0655881232758, 2., 1.], [15., 3., 1.]])
        for name in ('t01', 't96'):
            expected = np.full((3, len(valid_points), 2), np.nan)
            expected[:, :, 0] = np.array([
                getattr(models, name)(PARMOD, .2, *point) for point in valid_points
            ]).T
            self.assertTrue(np.all(np.isfinite(expected[:, :, 0])))
            for axis in range(3):
                points = np.repeat(valid_points[:, None, :], 2, axis=1)
                points[:, 1, axis] = np.nan
                with self.subTest(model=name, nan_axis=axis), np.errstate(invalid='ignore'):
                    actual = getattr(vectorized, name)(PARMOD, .2, *np.moveaxis(points, -1, 0))
                    self.assertEqual(np.shape(actual), expected.shape)
                    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-6,
                                               equal_nan=True)

    def test_t01_imf_interconnection_preserves_clock_angle(self):
        # With G1/G2 held fixed, the IMF dependence is exactly the prescribed
        # T01 interconnection field, including outside the magnetopause.
        for function in (models.t01, vectorized.t01):
            # The middle point is in the magnetopause transition layer
            # (sigma approximately 1.196 for Pdyn=2 and tilt=.2).
            for point in ((-6., 2., 1.), (10.0655881232758, 2., 1.), (15., 3., 1.)):
                base = list(PARMOD)
                base[2:4] = [0., 0.]
                baseline = np.array(function(base, .2, *point))
                for by in (-5., -1e-8, -0., 0., 1e-8, 5.):
                    for bz in (-5., 0., 5.):
                        with self.subTest(api=function.__module__, point=point, by=by, bz=bz):
                            parameters = list(PARMOD)
                            parameters[2:4] = [by, bz]
                            theta = np.arctan2(by, bz)
                            factor = .05887 + .57691 * np.sin(theta / 2.)**2
                            expected = baseline + factor * np.array([0., by, bz])
                            actual = function(parameters, .2, *point)
                            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)

    def test_integer_coordinates_match_scalar_fields_in_all_regions(self):
        points = np.array([[5, 1, 1], [10, 0, -5], [15, 1, 1]])
        for name in ('t01', 't96'):
            scalar, batched = getattr(models, name), getattr(vectorized, name)
            expected = np.array([scalar(PARMOD, .4, *point.astype(float)) for point in points]).T
            with self.subTest(model=name, mode='batch'):
                np.testing.assert_allclose(batched(PARMOD, .4, *points.T), expected,
                                           rtol=1e-10, atol=1e-6)
            for point, field in zip(points, expected.T):
                with self.subTest(model=name, point=point):
                    actual = batched(PARMOD, .4, *(int(c) for c in point))
                    np.testing.assert_allclose(actual, field, rtol=1e-10, atol=1e-6)
                    self.assertTrue(all(isinstance(c, float) for c in actual))

    def test_t01_broadcast_and_mesh_inputs_match_pointwise_fields(self):
        points = (
            (5., np.array([1., 2.]), 1.),
            (np.array([5., 10.])[:, None], np.array([1., 2., 3.]), -5.),
            np.meshgrid([5., 10.], [1., 2.], [-5., 1.], indexing='ij'),
        )
        for coords in points:
            broadcast = np.broadcast_arrays(*coords)
            expected = np.array([models.t01(PARMOD, .4, *point)
                                 for point in zip(*(c.ravel() for c in broadcast))])
            expected = expected.T.reshape((3,) + broadcast[0].shape)
            with self.subTest(shape=broadcast[0].shape):
                actual = vectorized.t01(PARMOD, .4, *coords)
                for component in actual:
                    self.assertEqual(component.shape, broadcast[0].shape)
                np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-6)

    def test_t01_array_layouts_preserve_inputs_and_shape(self):
        points = (
            (np.array([-6., 99., 10., 99., 15., 99.])[::2], 2., 1.),
            (np.asfortranarray([[-6., 5.], [10., 15.]]), 2., 1.),
            (np.array(10.), np.array(2.), np.array(1.)),
        )
        for coords in points:
            originals = [np.array(c, copy=True) for c in coords]
            broadcast = np.broadcast_arrays(*(np.atleast_1d(c) for c in coords))
            expected = np.array([models.t01(PARMOD, .4, *point)
                                 for point in zip(*(c.ravel() for c in broadcast))])
            expected = expected.T.reshape((3,) + broadcast[0].shape)
            with self.subTest(shape=broadcast[0].shape):
                actual = vectorized.t01(PARMOD, .4, *coords)
                self.assertEqual(np.shape(actual), np.shape(expected))
                np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-6)
                for coord, original in zip(coords, originals):
                    np.testing.assert_array_equal(coord, original)

    def test_coordinate_dtype_does_not_limit_field_precision(self):
        for name in ('t01', 't96'):
            points = np.array([[5, 1, 1], [10, 0, -5], [15, 1, 1]])
            expected = np.array([getattr(models, name)(PARMOD, .4, *point.astype(float))
                                 for point in points]).T
            for dtype in (np.int32, np.float32, np.float64):
                with self.subTest(model=name, dtype=dtype):
                    actual = getattr(vectorized, name)(PARMOD, .4, *points.astype(dtype).T)
                    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-6)
                    self.assertTrue(all(c.dtype == np.float64 for c in actual))


if __name__ == '__main__':
    unittest.main()
