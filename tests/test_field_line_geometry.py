"""
Tests for the field line geometry API (mageometry.geometry) and the
geopack_field adapter (mageometry.fields).

The geometry functions take a generic ``field(x, y, z) -> (bx, by, bz)``
callable; these tests exercise that interface with geopack-backed fields and
validate the results against analytic and internal-consistency checks.
"""

import unittest

import numpy as np

from mageometry import (
    geopack,
    geopack_field,
    field_line_curvature,
    field_line_frenet_frame,
    field_line_geometry_complete,
    field_line_directional_derivatives,
    verify_unit_vectors,
    verify_antisymmetry_relations,
    get_curvature_torsion_from_derivatives,
)

UT = 100  # epoch used for recalc in all tests
T96_PARMOD = np.array([2.0, -20.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0])


class TestFieldLineGeometry(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ps = geopack.recalc(UT)

    def test_dipole_curvature_matches_analytic(self):
        """For a pure dipole, curvature on the magnetic equator is 3/r.

        Points on the GSM y-axis lie in the magnetic equatorial plane for any
        dipole tilt (the tilted dipole axis has no y-component).
        """
        field = geopack_field(external=None, internal='dip')
        r = np.array([4.0, 5.0, 6.0, 7.0])
        zeros = np.zeros_like(r)
        kappa = field_line_curvature(field, zeros, r, zeros, delta=1e-3)
        np.testing.assert_allclose(kappa, 3.0 / r, rtol=1e-5)

    def test_frenet_frame_orthonormality(self):
        field = geopack_field('t96', 'dip', T96_PARMOD, self.ps)
        x = np.array([-6.0, -7.0, 5.0, 6.0])
        zeros = np.zeros_like(x)
        frame = field_line_frenet_frame(field, x, zeros, zeros, delta=1e-3)
        errors = verify_unit_vectors(*frame[:9])
        for name, err in errors.items():
            self.assertLess(np.max(np.abs(err)), 1e-6, msg=name)

    def test_directional_derivatives_antisymmetry(self):
        field = geopack_field('t96', 'dip', T96_PARMOD, self.ps)
        x = np.array([-6.0, -7.0, 5.0, 6.0])
        zeros = np.zeros_like(x)
        derivs = field_line_directional_derivatives(field, x, zeros, zeros, delta=1e-3)
        errors = verify_antisymmetry_relations(derivs)
        for name, err in errors.items():
            self.assertLess(np.max(np.abs(err)), 1e-6, msg=name)

    def test_derivative_curvature_consistent_with_direct(self):
        field = geopack_field('t96', 'dip', T96_PARMOD, self.ps)
        x = np.array([-6.0, 5.0])
        zeros = np.zeros_like(x)
        derivs = field_line_directional_derivatives(field, x, zeros, zeros, delta=1e-3)
        kappa_d, _ = get_curvature_torsion_from_derivatives(derivs)
        kappa = field_line_curvature(field, x, zeros, zeros, delta=1e-3)
        np.testing.assert_allclose(kappa_d, kappa, rtol=1e-4)

    def test_geometry_complete_shapes(self):
        field = geopack_field('t96', 'dip', T96_PARMOD, self.ps)
        x = np.array([-6.0, 5.0, 6.0])
        zeros = np.zeros_like(x)
        out = field_line_geometry_complete(field, x, zeros, zeros, delta=1e-3)
        self.assertEqual(len(out), 11)
        for component in out:
            self.assertEqual(np.shape(component), x.shape)

    def test_scalar_input_scalar_output(self):
        field = geopack_field('t96', 'dip', T96_PARMOD, self.ps)
        kappa = field_line_curvature(field, 6.0, 0.0, 0.0, delta=1e-3)
        self.assertIsInstance(kappa, float)
        derivs = field_line_directional_derivatives(field, 6.0, 0.0, 0.0, delta=1e-3)
        for name, value in derivs.items():
            self.assertIsInstance(value, float, msg=name)


class TestGeopackFieldAdapter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ps = geopack.recalc(UT)

    def test_matches_manual_closure(self):
        field = geopack_field('t96', 'dip', T96_PARMOD, self.ps)
        x = np.array([-6.0, -7.0, 5.0, 6.0])
        zeros = np.zeros_like(x)
        bx_d, by_d, bz_d = geopack.dip(x, zeros, zeros)
        bx_t, by_t, bz_t = geopack.t96_vectorized(T96_PARMOD, self.ps, x, zeros, zeros)
        bx, by, bz = field(x, zeros, zeros)
        np.testing.assert_array_equal(bx, bx_d + bx_t)
        np.testing.assert_array_equal(by, by_d + by_t)
        np.testing.assert_array_equal(bz, bz_d + bz_t)

    def test_all_external_models(self):
        for name, parmod in [('t89', 3), ('t96', T96_PARMOD),
                             ('t01', T96_PARMOD), ('t04', T96_PARMOD)]:
            field = geopack_field(name, 'dip', parmod, self.ps)
            bx, by, bz = field(5.0, 0.0, 0.0)
            self.assertTrue(np.all(np.isfinite([bx, by, bz])), msg=name)

    def test_internal_only_variants(self):
        for name in ['dip', 'igrf']:
            field = geopack_field(external=None, internal=name)
            bx, by, bz = field(3.0, 0.0, 0.0)
            self.assertTrue(np.all(np.isfinite([bx, by, bz])), msg=name)

    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            geopack_field(external=None, internal=None)
        with self.assertRaises(ValueError):
            geopack_field('t96')  # parmod/ps missing
        with self.assertRaises(ValueError):
            geopack_field('bogus', 'dip', T96_PARMOD, self.ps)
        with self.assertRaises(ValueError):
            geopack_field('t96', 'bogus', T96_PARMOD, self.ps)


if __name__ == '__main__':
    unittest.main()
