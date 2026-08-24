"""
Tests for the validity conventions of the geometry API: NaN for undefined
quantities, orthonormality by construction, the orthogonality tolerance,
and the finite-difference quality diagnostic.
"""

import unittest

import numpy as np

from mageometry import (
    field_line_tangent,
    field_line_curvature,
    field_line_normal,
    field_line_binormal,
    field_line_torsion,
    field_line_frenet_frame,
    field_line_geometry_complete,
    field_line_frame_quality,
    field_line_directional_derivatives,
    verify_unit_vectors,
    verify_antisymmetry_relations,
    GriddedField,
)
from mageometry.geometry import DEFAULT_ORTHOGONALITY_TOL


def dipole_b(x, y, z):
    r2 = x * x + y * y + z * z
    r5 = r2 ** 2.5
    return 3.0 * x * z / r5, 3.0 * y * z / r5, (3.0 * z * z - r2) / r5


def uniform_b(x, y, z):
    return np.zeros_like(x), np.zeros_like(y), np.ones_like(z)


def null_at_origin_b(x, y, z):
    """Linear X-type null: B = (x, -y, 0)."""
    return x, -y, np.zeros_like(z)


class TestNaNConvention(unittest.TestCase):

    def test_tangent_nan_where_field_undefined(self):
        # zero field
        tx, ty, tz = field_line_tangent(null_at_origin_b, np.array([0.0, 1.0]),
                                        np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        self.assertTrue(np.isnan(tx[0]) and np.isnan(ty[0]) and np.isnan(tz[0]))
        np.testing.assert_allclose((tx[1], ty[1], tz[1]), (1.0, 0.0, 0.0))
        # non-finite field (e.g. outside an interpolated domain)
        nan_field = lambda x, y, z: (np.full_like(x, np.nan),) * 3
        self.assertTrue(np.isnan(field_line_tangent(nan_field, 1.0, 2.0, 3.0)[0]))

    def test_no_absolute_field_threshold(self):
        """A tiny but finite field has a well-defined tangent (no unit-dependent cutoff)."""
        weak = lambda x, y, z: (1e-30 * np.ones_like(x), np.zeros_like(y), np.zeros_like(z))
        np.testing.assert_allclose(field_line_tangent(weak, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        self.assertAlmostEqual(field_line_curvature(weak, 0.0, 0.0, 0.0), 0.0)

    def test_straight_line_normal_is_nan_curvature_zero(self):
        self.assertEqual(field_line_curvature(uniform_b, 0.0, 0.0, 0.0), 0.0)
        n = field_line_normal(uniform_b, 0.0, 0.0, 0.0)
        b = field_line_binormal(uniform_b, 0.0, 0.0, 0.0)
        self.assertTrue(all(np.isnan(v) for v in n + b))
        self.assertTrue(np.isnan(field_line_torsion(uniform_b, 0.0, 0.0, 0.0)))
        d = field_line_directional_derivatives(uniform_b, 0.0, 0.0, 0.0)
        self.assertTrue(all(np.isnan(v) for v in d.values()))

    def test_nan_propagates_to_derivatives_only_where_invalid(self):
        x = np.array([3.0, 4.0, 0.0])  # third point: magnetic null
        y = np.zeros(3)
        z = np.zeros(3)
        d = field_line_directional_derivatives(null_at_origin_b, x, y, z, delta=0.01)
        for key, val in d.items():
            self.assertTrue(np.isnan(val[2]), key)
        # The X-line field has straight field lines on the axes -> zero
        # curvature -> undefined normal there too; use the dipole for finite
        # values with a NaN injected by an out-of-domain point.
        res = field_line_directional_derivatives(dipole_b, np.array([3.0, 4.0]),
                                                 np.array([0.0, 0.0]),
                                                 np.array([0.0, 0.0]))
        for key, val in res.items():
            self.assertTrue(np.all(np.isfinite(val)), key)

    def test_interpolated_domain_edge_is_nan(self):
        ax = np.linspace(-3.0, 3.0, 31)
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing='ij')
        with np.errstate(divide='ignore', invalid='ignore'):
            bx, by, bz = dipole_b(X, Y, Z)
        core = X ** 2 + Y ** 2 + Z ** 2 < 1.0
        for c in (bx, by, bz):
            c[core] = 0.0
        field = GriddedField(ax, ax, ax, bx, by, bz).field('linear')
        # inside; FD stencil crosses the z = 3 edge (tangent has tz ~ 0.6
        # at (2, 0, 2.95), so r + 0.2 T leaves the box); outside
        x = np.array([2.0, 2.0, 5.0])
        y = np.zeros(3)
        z = np.array([0.0, 2.95, 0.0])
        frame = field_line_frenet_frame(field, x, y, z, delta=0.2)
        self.assertTrue(np.all(np.isfinite([comp[0] for comp in frame])))
        self.assertTrue(np.isnan(frame[9][1]))   # curvature needs r ± δT
        self.assertTrue(np.isnan(frame[0][2]))   # tangent undefined outside


class TestOrthonormalityAndTolerance(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(3)
        n = 500
        r = rng.uniform(2.0, 10.0, n)
        th = rng.uniform(0.2, np.pi - 0.2, n)
        ph = rng.uniform(0.0, 2.0 * np.pi, n)
        self.x = r * np.sin(th) * np.cos(ph)
        self.y = r * np.sin(th) * np.sin(ph)
        self.z = r * np.cos(th)

    def test_frame_orthonormal_by_construction(self):
        """Even with a coarse delta the frame is orthonormal to round-off,
        because the normal is the projection of dT/ds perpendicular to T."""
        for delta in (0.01, 0.25):
            frame = field_line_frenet_frame(dipole_b, self.x, self.y, self.z, delta=delta)
            self.assertTrue(np.all(np.isfinite(frame[3])), f"delta={delta}")
            errors = verify_unit_vectors(*frame[:9])
            for name, err in errors.items():
                self.assertLess(np.max(np.abs(err)), 1e-12, f"{name} delta={delta}")

    def test_coarse_delta_keeps_normals(self):
        """At delta=0.25 the truncation-induced non-orthogonality of the raw
        finite difference is ~1e-3 (the old hard-coded cutoff zeroed ~40% of
        points); with the default tolerance all points stay valid."""
        q = field_line_frame_quality(dipole_b, self.x, self.y, self.z, delta=0.25)
        self.assertGreater(np.mean(q > 1e-3), 0.2)
        self.assertLess(np.max(q), DEFAULT_ORTHOGONALITY_TOL)
        n = field_line_normal(dipole_b, self.x, self.y, self.z, delta=0.25)
        self.assertTrue(np.all(np.isfinite(n[0])))

    def test_orthogonality_tol_masks_points(self):
        delta = 0.25
        q = field_line_frame_quality(dipole_b, self.x, self.y, self.z, delta=delta)
        tol = np.median(q)
        nx, ny, nz = field_line_normal(dipole_b, self.x, self.y, self.z,
                                       delta=delta, orthogonality_tol=tol)
        np.testing.assert_array_equal(np.isfinite(nx), q <= tol)
        # np.inf disables the check entirely.
        nx, _, _ = field_line_normal(dipole_b, self.x, self.y, self.z,
                                     delta=delta, orthogonality_tol=np.inf)
        self.assertTrue(np.all(np.isfinite(nx)))
        # Curvature is never masked by the tolerance.
        k = field_line_curvature(dipole_b, self.x, self.y, self.z, delta=delta)
        self.assertTrue(np.all(np.isfinite(k)))

    def test_frame_quality_scales_with_delta_squared(self):
        q1 = field_line_frame_quality(dipole_b, self.x, self.y, self.z, delta=0.02)
        q2 = field_line_frame_quality(dipole_b, self.x, self.y, self.z, delta=0.04)
        ratio = np.median(q2 / q1)
        self.assertAlmostEqual(ratio, 4.0, delta=0.3)

    def test_projection_does_not_change_dipole_curvature(self):
        # Equatorial curvature 3/r (y-axis points are on the magnetic equator).
        y = np.array([3.0, 5.0, 8.0])
        k = field_line_curvature(dipole_b, np.zeros(3), y, np.zeros(3), delta=0.01)
        # Central-difference truncation error scales as (kappa*delta)^2:
        # measured 2.8e-5 at r=3 (kappa=1), 6e-6 at r=5.
        np.testing.assert_allclose(k, 3.0 / y, rtol=5e-5)

    def test_geometry_complete_consistent_with_parts(self):
        out = field_line_geometry_complete(dipole_b, self.x, self.y, self.z, delta=0.05)
        self.assertEqual(len(out), 11)
        frame = field_line_frenet_frame(dipole_b, self.x, self.y, self.z, delta=0.05)
        for a, b in zip(out[:10], frame):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(
            out[10], field_line_torsion(dipole_b, self.x, self.y, self.z, delta=0.05))
        # Dipole field lines are planar: torsion vanishes.
        np.testing.assert_allclose(out[10], 0.0, atol=1e-6)

    def test_derivatives_match_frame_and_antisymmetry(self):
        d = field_line_directional_derivatives(dipole_b, self.x, self.y, self.z, delta=0.01)
        k = field_line_curvature(dipole_b, self.x, self.y, self.z, delta=0.01)
        valid = np.isfinite(d['dT_dT_n'])
        self.assertGreater(np.mean(valid), 0.95)
        np.testing.assert_allclose(d['dT_dT_n'][valid], k[valid], rtol=1e-3)
        # Residuals are finite-difference truncation at the high-curvature
        # end (r ~ 2): measured max 1.1e-4, identical to the previous
        # implementation; the T-direction relations are ~1e-14.
        for name, err in verify_antisymmetry_relations(d).items():
            self.assertLess(np.max(np.abs(err[valid])), 2e-4, name)

    def test_scalar_and_mixed_inputs(self):
        k = field_line_curvature(dipole_b, 0.0, 5.0, 0.0)
        self.assertIsInstance(k, float)
        self.assertAlmostEqual(k, 0.6, delta=2e-5)
        d = field_line_directional_derivatives(dipole_b, 0.0, 5.0, 0.0)
        self.assertIsInstance(d['dT_dT_n'], float)
        # scalar x with array y broadcasts
        k = field_line_curvature(dipole_b, 0.0, np.array([5.0, 6.0]), 0.0)
        self.assertEqual(k.shape, (2,))


if __name__ == '__main__':
    unittest.main()
