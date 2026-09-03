"""
Tests for |B| magnitude derivatives and the Frenet-frame current density
(mageometry.geometry.field_line_current).

Physics-based checks:

- a dipole is current-free, so the twist vanishes and the two perpendicular
  terms (Bκ vs ∂B/∂n, and ∂B/∂b) must cancel/vanish — a nontrivial test
  coupling the frame, the curvature, and the |B| gradients;
- for T96+IGRF the assembled μ₀J must match a direct Cartesian
  finite-difference curl of B projected on the frame;
- the Frenet-frame divergence ∂B/∂T + B(dT_dn_n + dT_db_b) must match a
  direct Cartesian finite-difference ∇·B. (T96 is genuinely non-solenoidal
  in parts of its Birkeland-current module — |∇·B| ~ 1 nT/Re at some of the
  test points, in the scalar reference implementation as well — so the test
  compares the two ∇·B estimates instead of asserting zero.)
"""

import unittest

import numpy as np

from mageometry import (
    geopack,
    geopack_field,
    field_magnitude_derivatives,
    field_line_current_density,
    verify_divergence_identity,
)

UT = 100  # epoch used for recalc in all tests
T96_PARMOD = np.array([2.0, -20.0, 0.0, -5.0, 0, 0, 0, 0, 0, 0])
DELTA = 2e-3


def _curl_fd(field, x, y, z, h=2.5e-4):
    """Cartesian central-difference curl of the field, shape (3, N)."""
    def f(px, py, pz):
        return np.stack(field(px, py, pz))
    dfdx = (f(x + h, y, z) - f(x - h, y, z)) / (2 * h)
    dfdy = (f(x, y + h, z) - f(x, y - h, z)) / (2 * h)
    dfdz = (f(x, y, z + h) - f(x, y, z - h)) / (2 * h)
    return np.stack([dfdy[2] - dfdz[1], dfdz[0] - dfdx[2], dfdx[1] - dfdy[0]])


def _test_points():
    """Off-equatorial points at r = 4-8 Re on several meridians."""
    r = np.array([4.0, 5.0, 6.0, 7.0, 8.0, 4.5, 5.5, 6.5, 7.5, 5.0, 6.0, 7.0])
    lat = np.deg2rad(np.array([25, 30, 35, 40, 30, -25, -30, -35, -40, 35, 25, 30]))
    lon = np.deg2rad(np.array([0, 45, 90, 135, 180, 225, 270, 315, 30, 150, 210, 330]))
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return x, y, z


class TestFieldMagnitudeDerivatives(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ps = geopack.recalc(UT)

    def test_dipole_equator_analytic(self):
        """On the dipole magnetic equator, ∂B/∂n = 3B/r and ∂B/∂T = ∂B/∂b = 0.

        For a dipole, B ∝ r⁻³ on the equator and the principal normal points
        toward the planet, so ∂B/∂n = 3B/r; symmetry kills the other two.
        Points on the GSM y-axis lie in the magnetic equatorial plane for any
        dipole tilt.
        """
        field = geopack_field(external=None, internal='dip')
        r = np.array([4.0, 5.0, 6.0, 7.0])
        zeros = np.zeros_like(r)
        mag = field_magnitude_derivatives(field, zeros, r, zeros, delta=DELTA)
        np.testing.assert_allclose(mag['dB_dn'], 3.0 * mag['B'] / r, rtol=1e-4)
        scale = np.abs(mag['B']) / r
        self.assertLess(np.max(np.abs(mag['dB_dT']) / scale), 1e-6)
        self.assertLess(np.max(np.abs(mag['dB_db']) / scale), 1e-6)

    def test_scalar_input_returns_floats(self):
        field = geopack_field(external=None, internal='dip')
        mag = field_magnitude_derivatives(field, 0.0, 6.0, 0.0, delta=DELTA)
        for key, val in mag.items():
            self.assertIsInstance(val, float, msg=key)

    def test_nan_where_frame_undefined(self):
        """On the dipole axis the line is straight: dB_dT defined, dB_dn NaN."""
        field = geopack_field(external=None, internal='dip')
        # Move along the (tilted) dipole axis: use the field direction itself
        # at a polar point — simplest robust straight-line case is the
        # untilted-dipole z-axis in SM ≈ tilted axis; instead check via NaN
        # propagation contract: any point where the normal is undefined.
        from mageometry import field_line_normal
        x, y, z = 0.0, 0.0, 5.0
        nx, _, _ = field_line_normal(field, x, y, z, delta=DELTA)
        mag = field_magnitude_derivatives(field, x, y, z, delta=DELTA)
        if np.isnan(nx):
            self.assertTrue(np.isnan(mag['dB_dn']))
            self.assertTrue(np.isnan(mag['dB_db']))
            self.assertTrue(np.isfinite(mag['dB_dT']))
        else:  # tilted axis missed the pole: gradients must all be finite
            for key in ('dB_dT', 'dB_dn', 'dB_db'):
                self.assertTrue(np.isfinite(mag[key]), msg=key)


class TestFieldLineCurrentDensity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ps = geopack.recalc(UT)

    def test_dipole_is_current_free(self):
        """All three μ₀J components vanish against the Bκ term scale.

        The binormal component is the nontrivial cancellation Bκ = ∂B/∂n,
        which couples the frame, the curvature, and the |B| gradient.
        """
        field = geopack_field(external=None, internal='dip')
        x, y, z = _test_points()
        cur = field_line_current_density(field, x, y, z, delta=DELTA)
        scale = cur['B'] * cur['curvature']  # size of the cancelling terms
        for key in ('mu0J_T', 'mu0J_n', 'mu0J_b'):
            rel = np.abs(cur[key]) / scale
            self.assertLess(np.nanmax(rel), 1e-5, msg=key)

    def test_t96_matches_cartesian_curl(self):
        """μ₀J from the Frenet decomposition equals the FD curl of B.

        The comparison is restricted to points where the reference curl is
        step-converged: T96's Birkeland-current module has surfaces with
        discontinuous derivatives, where finite differences with different
        stencils legitimately disagree at the percent level.
        """
        field = geopack_field('t96', 'igrf', T96_PARMOD, self.ps)
        x, y, z = _test_points()
        cur = field_line_current_density(field, x, y, z, delta=DELTA)
        curl = _curl_fd(field, x, y, z)
        curl_mag = np.sqrt((curl ** 2).sum(axis=0))

        curl2 = _curl_fd(field, x, y, z, h=5e-4)
        smooth = (np.sqrt(((curl - curl2) ** 2).sum(axis=0)) / curl_mag) < 1e-5

        j_frenet = np.stack([cur['mu0J_x'], cur['mu0J_y'], cur['mu0J_z']])
        err = np.sqrt(((j_frenet - curl) ** 2).sum(axis=0)) / curl_mag
        use = smooth & np.isfinite(err)
        self.assertGreater(use.sum(), len(x) * 0.4)
        self.assertLess(np.max(err[use]), 1e-3)
        self.assertLess(np.median(err[use]), 1e-4)

    def test_parallel_current_is_pure_twist(self):
        """μ₀j∥ = B·T·(∇×T): alpha equals T·(∇×B)/B from the FD curl."""
        field = geopack_field('t96', 'igrf', T96_PARMOD, self.ps)
        x, y, z = _test_points()
        cur = field_line_current_density(field, x, y, z, delta=DELTA)
        curl = _curl_fd(field, x, y, z)
        bx, by, bz = field(x, y, z)
        B = np.sqrt(bx * bx + by * by + bz * bz)
        alpha_ref = (curl[0] * bx + curl[1] * by + curl[2] * bz) / B ** 2

        curl2 = _curl_fd(field, x, y, z, h=5e-4)
        curl_mag = np.sqrt((curl ** 2).sum(axis=0))
        smooth = (np.sqrt(((curl - curl2) ** 2).sum(axis=0)) / curl_mag) < 1e-5
        use = smooth & np.isfinite(cur['alpha'])
        self.assertGreater(use.sum(), len(x) * 0.4)
        np.testing.assert_allclose(cur['alpha'][use], alpha_ref[use],
                                   rtol=0, atol=1e-4 * np.max(np.abs(alpha_ref[use])))

    def test_scalar_input_returns_floats(self):
        field = geopack_field('t96', 'dip', T96_PARMOD, self.ps)
        cur = field_line_current_density(field, -6.0, 1.0, 2.0, delta=DELTA)
        for key, val in cur.items():
            self.assertIsInstance(val, float, msg=key)


def _div_fd(field, x, y, z, h=2.5e-4):
    """Cartesian central-difference divergence of the field."""
    def f(px, py, pz):
        return np.stack(field(px, py, pz))
    return ((f(x + h, y, z)[0] - f(x - h, y, z)[0])
            + (f(x, y + h, z)[1] - f(x, y - h, z)[1])
            + (f(x, y, z + h)[2] - f(x, y, z - h)[2])) / (2 * h)


class TestDivergenceIdentity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ps = geopack.recalc(UT)

    def test_dipole_divergence_vanishes(self):
        """∂B/∂T + B(dT_dn_n + dT_db_b) = ∇·B = 0 for the dipole."""
        field = geopack_field(external=None, internal='dip')
        x, y, z = _test_points()
        div = verify_divergence_identity(field, x, y, z, delta=DELTA)
        cur = field_line_current_density(field, x, y, z, delta=DELTA)
        rel = np.abs(div) / (cur['B'] * cur['curvature'])
        valid = np.isfinite(rel)
        self.assertGreater(valid.sum(), len(x) * 0.8)
        self.assertLess(np.max(rel[valid]), 1e-5)

    def test_t96_matches_cartesian_divergence(self):
        """The Frenet-frame ∇·B equals the Cartesian FD ∇·B.

        T96 is genuinely non-solenoidal at some test points (its
        Birkeland-current module), so both estimates must agree on the
        same nonzero value.
        """
        field = geopack_field('t96', 'igrf', T96_PARMOD, self.ps)
        x, y, z = _test_points()
        div = verify_divergence_identity(field, x, y, z, delta=DELTA)
        div_ref = _div_fd(field, x, y, z)
        cur = field_line_current_density(field, x, y, z, delta=DELTA)
        scale = cur['B'] * cur['curvature']
        smooth = np.abs(div_ref - _div_fd(field, x, y, z, h=5e-4)) / scale < 1e-5
        rel = np.abs(div - div_ref) / scale
        use = smooth & np.isfinite(rel)
        self.assertGreater(use.sum(), len(x) * 0.4)
        self.assertLess(np.max(rel[use]), 1e-4)
        # ... and the nonzero divergence itself is real: it must exceed the
        # finite-difference floor somewhere (T96's non-solenoidal region).
        self.assertGreater(np.nanmax(np.abs(div) / scale), 1e-3)


if __name__ == '__main__':
    unittest.main()
