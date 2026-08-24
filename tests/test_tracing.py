"""
Tests for the generic field line tracer (mageometry.tracing).

The tracer takes any ``field(x, y, z) -> (bx, by, bz)`` callable. These
tests validate it against the analytic dipole (L-shell invariant, footpoint
latitude, curvature along the traced line), against the geopack reference
tracer for Tsyganenko fields, and on an interpolated grid (domain-edge
termination).
"""

import unittest

import numpy as np

from mageometry import (
    geopack,
    geopack_field,
    GriddedField,
    trace_field_lines,
    field_line_curvature,
)
from mageometry.tracing import (
    STATUS_INNER,
    STATUS_OUTER,
    STATUS_MAX_STEPS,
    STATUS_INVALID,
    STATUS_STOPPED,
)


def dipole_b(x, y, z, m=1.0):
    """Untilted dipole (moment along +z); returns (bx, by, bz)."""
    r2 = x * x + y * y + z * z
    r5 = r2 ** 2.5
    return 3.0 * m * x * z / r5, 3.0 * m * y * z / r5, m * (3.0 * z * z - r2) / r5


def dipole_curvature(r, lam):
    """Curvature of the dipole field line r = L cos^2(lam) at latitude lam.

    From the polar-curve curvature formula: kappa = 3 cos(lam) (1 + sin^2 lam)
    / (r (1 + 3 sin^2 lam)^(3/2)); equals 3/r on the equator.
    """
    s2 = np.sin(lam) ** 2
    return 3.0 * np.cos(lam) * (1.0 + s2) / (r * (1.0 + 3.0 * s2) ** 1.5)


class TestDipoleTracing(unittest.TestCase):
    """Analytic checks on an untilted dipole."""

    L = 5.0

    def test_both_directions_reach_inner_sphere(self):
        tr = trace_field_lines(dipole_b, self.L, 0.0, 0.0, direction='both',
                               ds=0.1, r0=1.0)
        self.assertEqual(tr.n_lines, 1)
        self.assertEqual(tr.status[0], STATUS_INNER)
        self.assertEqual(tr.status_backward[0], STATUS_INNER)

        x, y, z = tr.path(0)
        r = np.sqrt(x * x + y * y + z * z)
        # Endpoints lie exactly on the inner sphere.
        np.testing.assert_allclose(r[[0, -1]], 1.0, atol=1e-12)
        # Footpoint latitude: cos^2(lam) = r/L = 1/L  ->  |z| = sqrt(1 - 1/L).
        z_expected = np.sqrt(1.0 - 1.0 / self.L)
        np.testing.assert_allclose(np.abs(z[[0, -1]]), z_expected, atol=5e-3)
        # The two ends are in opposite hemispheres and the line stays in y=0.
        self.assertLess(z[0] * z[-1], 0.0)
        np.testing.assert_allclose(y, 0.0, atol=1e-12)

    def test_seed_position_and_arc_length(self):
        tr = trace_field_lines(dipole_b, self.L, 0.0, 0.0, direction='both',
                               ds=0.1, r0=1.0)
        i0 = tr.start_index[0]
        x, y, z = tr.path(0)
        s = tr.arc_length(0)
        self.assertEqual((x[i0], y[i0], z[i0]), (self.L, 0.0, 0.0))
        self.assertEqual(s[i0], 0.0)
        self.assertTrue(np.all(np.diff(s) > 0), "arc length must increase along the path")
        # Chord lengths approximate the accepted step lengths.
        chord = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
        np.testing.assert_allclose(chord, np.diff(s), rtol=2e-3)
        # Symmetric line: |s| at both ends agree.
        np.testing.assert_allclose(-s[0], s[-1], rtol=1e-6)
        self.assertEqual(tr.nsteps[0] + 1, x.size)

    def test_l_shell_invariant_along_path(self):
        """Interior points stay on r = L cos^2(lam) to integration accuracy."""
        tr = trace_field_lines(dipole_b, self.L, 0.0, 0.0, direction='both',
                               ds=0.1, r0=1.0)
        x, y, z = tr.path(0)
        r = np.sqrt(x * x + y * y + z * z)
        lam = np.arcsin(z / r)
        l_shell = r / np.cos(lam) ** 2
        # Interior points (measured ~1e-7); the endpoints are chord
        # intersections with the sphere and are only accurate to ~ds^2.
        np.testing.assert_allclose(l_shell[1:-1], self.L, atol=1e-6)
        np.testing.assert_allclose(l_shell[[0, -1]], self.L, atol=1e-2)

    def test_accuracy_improves_with_smaller_step(self):
        errs = []
        for ds in (0.2, 0.1, 0.05):
            tr = trace_field_lines(dipole_b, self.L, 0.0, 0.0, direction=1,
                                   ds=ds, r0=1.0)
            x, y, z = tr.path(0)
            r = np.sqrt(x * x + y * y + z * z)
            lam = np.arcsin(z / r)
            errs.append(np.max(np.abs(r[1:-1] / np.cos(lam[1:-1]) ** 2 - self.L)))
        self.assertTrue(errs[0] > errs[1] > errs[2], errs)

    def test_curvature_along_traced_line(self):
        """Trace + geometry: curvature along the path matches the analytic
        dipole curvature at the traced latitudes."""
        tr = trace_field_lines(dipole_b, self.L, 0.0, 0.0, direction='both',
                               ds=0.1, r0=1.2)
        x, y, z = tr.path(0)
        kappa = field_line_curvature(dipole_b, x, y, z, delta=0.01)
        r = np.sqrt(x * x + y * y + z * z)
        lam = np.arcsin(z / r)
        np.testing.assert_allclose(kappa, dipole_curvature(r, lam), rtol=1e-4)

    def test_batch_seeds_and_direction_sign(self):
        seeds = np.array([3.0, 4.0, 5.0, 6.0])
        zero = np.zeros_like(seeds)
        fwd = trace_field_lines(dipole_b, seeds, zero, zero, direction=1,
                                ds=0.1, r0=1.0)
        bwd = trace_field_lines(dipole_b, seeds, zero, zero, direction=-1,
                                ds=0.1, r0=1.0)
        self.assertEqual(fwd.n_lines, 4)
        np.testing.assert_array_equal(fwd.status, STATUS_INNER)
        np.testing.assert_array_equal(bwd.status, STATUS_INNER)
        xf, yf, zf = fwd.end
        xb, yb, zb = bwd.end
        # On the equator B points -z for a +z dipole moment: direction=+1
        # follows B into the southern hemisphere.
        self.assertTrue(np.all(zf < 0))
        self.assertTrue(np.all(zb > 0))
        np.testing.assert_allclose(zf, -zb, atol=1e-9)
        # Longer L-shells need more steps.
        self.assertTrue(np.all(np.diff(fwd.nsteps) > 0))
        # Padding beyond each line's end is NaN.
        for i in range(4):
            self.assertTrue(np.all(np.isnan(fwd.x[i, fwd.nsteps[i] + 1:])))

    def test_outer_sphere_and_max_steps_and_stop(self):
        tr = trace_field_lines(dipole_b, 2.0, 0.0, 1.0, direction=-1, ds=0.1,
                               rlim=3.0)
        self.assertEqual(tr.status[0], STATUS_OUTER)
        xe, ye, ze = tr.end
        np.testing.assert_allclose(np.hypot(xe, np.hypot(ye, ze)), 3.0, atol=1e-12)

        tr = trace_field_lines(dipole_b, 5.0, 0.0, 0.0, ds=0.1, max_steps=10)
        self.assertEqual(tr.status[0], STATUS_MAX_STEPS)
        self.assertEqual(tr.nsteps[0], 10)
        self.assertEqual(tr.x.shape, (1, 11))

        tr = trace_field_lines(dipole_b, 5.0, 0.0, 0.0, ds=0.1,
                               stop=lambda x, y, z: z < -1.0)
        self.assertEqual(tr.status[0], STATUS_STOPPED)
        self.assertLess(tr.end[2][0], -1.0)
        self.assertGreater(tr.end[2][0], -1.0 - 0.1)

    def test_seed_already_terminated(self):
        tr = trace_field_lines(dipole_b, [0.5, 12.0], [0.0, 0.0], [0.0, 0.0],
                               ds=0.1, r0=1.0, rlim=10.0)
        np.testing.assert_array_equal(tr.status, [STATUS_INNER, STATUS_OUTER])
        np.testing.assert_array_equal(tr.nsteps, [0, 0])
        np.testing.assert_array_equal(tr.end[0], [0.5, 12.0])

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            trace_field_lines(dipole_b, [1.0, 2.0], [0.0], [0.0])
        with self.assertRaises(ValueError):
            trace_field_lines(dipole_b, 5.0, 0.0, 0.0, ds=-0.1)
        with self.assertRaises(ValueError):
            trace_field_lines(dipole_b, 5.0, 0.0, 0.0, direction=2)
        with self.assertRaises(ValueError):
            trace_field_lines(dipole_b, 5.0, 0.0, 0.0, bounds=((0, 1), (0, 1)))
        with self.assertRaises(ValueError):
            trace_field_lines(dipole_b, 5.0, 0.0, 0.0, max_steps=0)


class TestAgainstGeopackTrace(unittest.TestCase):
    """The generic tracer reproduces the geopack reference tracer on a
    Tsyganenko field (same RK5 scheme; different boundary handling)."""

    UT = 100
    KP = 3

    @classmethod
    def setUpClass(cls):
        ps = geopack.recalc(cls.UT)
        cls.field = staticmethod(geopack_field('t89', 'igrf', cls.KP, ps))
        rng = np.random.default_rng(0)
        n = 30
        r = rng.uniform(3.0, 7.0, n)
        th = rng.uniform(0.3, np.pi - 0.3, n)
        ph = rng.uniform(0.0, 2.0 * np.pi, n)
        cls.xs = r * np.sin(th) * np.cos(ph)
        cls.ys = r * np.sin(th) * np.sin(ph)
        cls.zs = r * np.cos(th)

    def _compare(self, gp_dir):
        xf, yf, zf, st = geopack.trace_vectorized(
            self.xs, self.ys, self.zs, dir=gp_dir, rlim=10.0, r0=1.0,
            parmod=self.KP, exname='t89', inname='igrf', maxloop=1000,
            strict_scalar_models=False,
        )
        # geopack dir=+1 is antiparallel to B; the generic tracer's
        # direction=+1 is along B.
        tr = trace_field_lines(self.field, self.xs, self.ys, self.zs,
                               direction=-gp_dir, ds=0.5, r0=1.0, rlim=10.0)
        np.testing.assert_array_equal(tr.status, st)
        xe, ye, ze = tr.end
        dist = np.sqrt((xe - xf) ** 2 + (ye - yf) ** 2 + (ze - zf) ** 2)
        r_end = np.sqrt(xe * xe + ye * ye + ze * ze)

        inner = st == STATUS_INNER
        self.assertTrue(np.any(inner))
        # geopack stops at the first point inside r0 (its boundary
        # interpolation is a no-op) with steps of ~0.01 Re there, so its
        # footpoints overshoot by up to ~1e-2 Re (measured max 1.0e-2).
        self.assertLess(dist[inner].max(), 2e-2)
        np.testing.assert_allclose(r_end[inner], 1.0, atol=1e-12)

        outer = st == STATUS_OUTER
        if np.any(outer):
            # geopack overshoots rlim by up to one full step (0.5-1 Re).
            np.testing.assert_allclose(r_end[outer], 10.0, atol=1e-12)
            self.assertLess(dist[outer].max(), 1.2)

    def test_along_b(self):
        self._compare(-1.0)

    def test_against_b(self):
        self._compare(+1.0)


class TestGriddedFieldTracing(unittest.TestCase):
    """Tracing through an interpolated field: domain-edge termination."""

    @classmethod
    def setUpClass(cls):
        ax = np.linspace(-4.0, 4.0, 81)
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing='ij')
        with np.errstate(divide='ignore', invalid='ignore'):
            bx, by, bz = dipole_b(X, Y, Z)
        core = X ** 2 + Y ** 2 + Z ** 2 < 1.5 ** 2
        for comp in (bx, by, bz):
            comp[core] = 0.0
        cls.grid = GriddedField(ax, ax, ax, bx, by, bz)
        cls.field = staticmethod(cls.grid.field(method='linear'))

    def test_leaving_domain_is_invalid_without_bounds(self):
        # Seed on the L~5.4 line: it leaves the box through x = 4.
        tr = trace_field_lines(self.field, 3.5, 0.0, 2.0, direction='both', ds=0.1)
        self.assertEqual(tr.status[0], STATUS_INVALID)
        self.assertEqual(tr.status_backward[0], STATUS_INVALID)
        x, y, z = tr.path(0)
        self.assertTrue(np.all(np.isfinite(x)))
        # The line crept up to the domain edge (within ds * 2**-20 scale).
        self.assertLessEqual(x.max(), 4.0)
        self.assertGreater(x.max(), 4.0 - 1e-4)

    def test_leaving_domain_is_outer_with_bounds(self):
        tr = trace_field_lines(self.field, 3.5, 0.0, 2.0, direction='both',
                               ds=0.1, bounds=self.grid.bounds)
        x, y, z = tr.path(0)
        self.assertEqual(tr.status[0], STATUS_OUTER)
        self.assertEqual(x[-1], 4.0)  # snapped exactly onto the face
        # The other end runs into the zero-field core, not the box.
        self.assertEqual(tr.status_backward[0], STATUS_INVALID)
        self.assertLess(np.hypot(x[0], z[0]), 1.6)

    def test_path_matches_analytic_l_shell(self):
        tr = trace_field_lines(self.field, 3.0, 0.0, 0.0, direction=1, ds=0.1,
                               stop=lambda x, y, z: np.hypot(x, np.hypot(y, z)) < 2.0)
        self.assertEqual(tr.status[0], STATUS_STOPPED)
        x, y, z = tr.path(0)
        r = np.sqrt(x * x + y * y + z * z)
        lam = np.arcsin(z / r)
        # Linear interpolation on a 0.1 grid: measured max deviation ~4e-4.
        np.testing.assert_allclose(r / np.cos(lam) ** 2, 3.0, atol=2e-2)

    def test_seed_outside_domain(self):
        tr = trace_field_lines(self.field, 5.0, 0.0, 0.0, ds=0.1)
        self.assertEqual(tr.status[0], STATUS_INVALID)
        self.assertEqual(tr.nsteps[0], 0)


if __name__ == '__main__':
    unittest.main()
