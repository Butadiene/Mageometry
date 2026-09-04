"""
Tests for mageometry.viz (requires matplotlib; skipped otherwise).

Plots are rendered on the Agg backend. Tests check that the functions
accept the library's objects, produce the expected artists, honour NaN
(blank cells / dropped arrows), and handle 2D and 3D axes.
"""

import unittest

import numpy as np

from mageometry import trace_field_lines

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


def mesh_values(mesh):
    """Values of a QuadMesh as a plain array with NaN where blank/masked."""
    return np.ma.filled(np.ma.asarray(mesh.get_array(), dtype=float), np.nan)


def dipole_b(x, y, z):
    r2 = x * x + y * y + z * z
    r5 = r2 ** 2.5
    return 3.0 * x * z / r5, 3.0 * y * z / r5, (3.0 * z * z - r2) / r5


class TestResolveScale(unittest.TestCase):
    """Backend-neutral colour-scale limits (no matplotlib needed)."""

    @classmethod
    def setUpClass(cls):
        from mageometry.viz._quantities import Quantity
        from mageometry.viz._scales import resolve_scale
        cls.Quantity = Quantity
        cls.resolve = staticmethod(resolve_scale)

    def test_symmetric(self):
        q = self.Quantity(None, 'q', symmetric=True)
        vals = np.array([-4.0, -1.0, 0.0, 2.0, np.nan])
        lo, hi, use_log = self.resolve(vals, q)
        self.assertEqual((lo, hi, use_log), (-hi, np.percentile([4.0, 1.0, 0.0, 2.0], 98), False))
        lo, hi, use_log = self.resolve(vals, q, vmax=3.0)
        self.assertEqual((lo, hi, use_log), (-3.0, 3.0, False))

    def test_log_positive(self):
        q = self.Quantity(None, 'q', positive=True, log=True)
        vals = np.array([0.0, 1e-3, 1e-1, 10.0, np.nan])
        lo, hi, use_log = self.resolve(vals, q)
        self.assertTrue(use_log)
        self.assertGreater(lo, 0.0)
        self.assertGreater(hi, lo)
        # log=True with a single positive value: degenerate limits widened
        lo, hi, use_log = self.resolve(np.array([5.0]), q)
        self.assertEqual((lo, hi, use_log), (5.0, 50.0, True))

    def test_log_falls_back_to_linear_without_positive_values(self):
        q = self.Quantity(None, 'q', positive=True, log=True)
        lo, hi, use_log = self.resolve(np.array([0.0, 0.0]), q)
        self.assertFalse(use_log)
        self.assertEqual(lo, 0.0)   # positive quantity: linear scale starts at zero

    def test_linear_defaults_and_empty(self):
        q = self.Quantity(None, 'q')
        lo, hi, use_log = self.resolve(np.array([1.0, 2.0, 3.0]), q, log=False)
        self.assertFalse(use_log)
        self.assertLess(lo, hi)
        self.assertEqual(self.resolve(np.array([np.nan]), q), (0.0, 1.0, False))
        # degenerate constant values widened
        lo, hi, use_log = self.resolve(np.array([2.0, 2.0]), q, log=False)
        self.assertEqual((lo, hi), (2.0, 3.0))


@unittest.skipUnless(HAVE_MPL, "matplotlib not installed")
class TestColorNormAdapter(unittest.TestCase):
    """color_norm must keep returning matplotlib norms matching resolve_scale."""

    def test_matches_resolve_scale(self):
        import matplotlib.colors as mcolors
        from mageometry.viz._quantities import QUANTITIES
        from mageometry.viz._mpl import color_norm
        from mageometry.viz._scales import resolve_scale
        vals = np.array([1e-3, 0.5, 2.0, 40.0])
        norm = color_norm(vals, QUANTITIES['curvature'])
        self.assertIsInstance(norm, mcolors.LogNorm)
        lo, hi, use_log = resolve_scale(vals, QUANTITIES['curvature'])
        self.assertTrue(use_log)
        self.assertEqual((norm.vmin, norm.vmax), (lo, hi))
        signed = np.array([-3.0, 1.0])
        norm = color_norm(signed, QUANTITIES['torsion'])
        self.assertNotIsInstance(norm, mcolors.LogNorm)
        self.assertEqual(norm.vmin, -norm.vmax)


@unittest.skipUnless(HAVE_MPL, "matplotlib not installed")
class TestViz(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from mageometry import viz
        cls.viz = viz
        cls.trace = trace_field_lines(dipole_b, [3.0, 5.0], [0.0, 0.0], [0.0, 0.0],
                                      direction='both', ds=0.1, r0=1.0)

    def tearDown(self):
        plt.close('all')

    def test_plane_grid_and_project(self):
        H, V, x, y, z = self.viz.plane_grid('xz', (-2, 2, -1, 1), n=(5, 3), offset=0.5)
        self.assertEqual(H.shape, (3, 5))
        np.testing.assert_array_equal(x, H)
        np.testing.assert_array_equal(z, V)
        np.testing.assert_array_equal(y, 0.5)
        h, v = self.viz.project('yz', 1.0, 2.0, 3.0)
        self.assertEqual((h, v), (2.0, 3.0))
        with self.assertRaises(ValueError):
            self.viz.plane_axes('xx')

    def test_geometry_map_curvature(self):
        mask = lambda x, y, z: x * x + y * y + z * z < 1.5 ** 2
        mesh = self.viz.plot_geometry_map(dipole_b, 'curvature', plane='xz',
                                          extent=(-4, 4, -3, 3), n=(40, 30),
                                          delta=0.05, mask=mask, arrows=True, unit='Re')
        vals = mesh_values(mesh)
        self.assertEqual(vals.size, 40 * 30)
        # Masked core is NaN; outside it curvature is finite and positive.
        self.assertTrue(np.any(~np.isfinite(vals)))
        finite = vals[np.isfinite(vals)]
        self.assertTrue(np.all(finite > 0))
        # Equatorial value at x = 3 matches 3/r.
        H, V, x, y, z = self.viz.plane_grid('xz', (-4, 4, -3, 3), n=(40, 30))
        grid_vals = vals.reshape(H.shape)
        i = np.argmin(np.abs(V[:, 0]))
        j = np.argmin(np.abs(H[0] - 3.0))
        self.assertAlmostEqual(grid_vals[i, j] * H[0, j] / 3.0, 1.0, delta=0.05)
        self.assertEqual(mesh.axes.get_xlabel(), 'x (Re)')
        # log norm by default for curvature
        self.assertEqual(type(mesh.norm).__name__, 'LogNorm')

    def test_geometry_map_signed_and_callable(self):
        mesh = self.viz.plot_geometry_map(dipole_b, 'bz', extent=(-4, 4, -3, 3), n=20,
                                          mask=lambda x, y, z: x**2 + y**2 + z**2 < 2)
        self.assertAlmostEqual(mesh.norm.vmin, -mesh.norm.vmax)
        custom = lambda field, x, y, z: np.sqrt(x * x + y * y + z * z)
        mesh = self.viz.plot_geometry_map(dipole_b, custom, plane='xy', extent=(-2, 2, -2, 2),
                                          n=10, label='r', colorbar=False)
        np.testing.assert_allclose(np.nanmax(mesh_values(mesh)), np.sqrt(8), rtol=1e-6)
        with self.assertRaises(ValueError):
            self.viz.plot_geometry_map(dipole_b, 'no_such_quantity', n=4)

    def test_geometry_map_derivative_key(self):
        mesh = self.viz.plot_geometry_map(dipole_b, 'dT_dn_n', extent=(2, 6, -2, 2), n=12,
                                          delta=0.05)
        self.assertTrue(np.any(np.isfinite(mesh_values(mesh))))

    def test_field_direction(self):
        q = self.viz.plot_field_direction(dipole_b, plane='xz', extent=(-4, 4, -3, 3), n=8,
                                          mask=lambda x, y, z: x**2 + z**2 < 1)
        self.assertGreater(q.N, 0)
        self.assertLess(q.N, 64)     # masked points dropped

    def test_field_lines_2d_and_3d(self):
        lines = self.viz.plot_field_lines(self.trace, plane='xz', unit='Re')
        self.assertEqual(len(lines), 2)
        xdata = lines[0].get_xdata()
        self.assertTrue(np.all(np.isfinite(xdata)))   # NaN padding trimmed
        self.assertEqual(xdata.size, self.trace.nsteps[0] + 1)

        fig = plt.figure()
        ax3 = fig.add_subplot(111, projection='3d')
        lines3 = self.viz.plot_field_lines(self.trace, ax=ax3, color='k')
        self.assertEqual(len(lines3), 2)

    def test_field_lines_colored_by_quantity(self):
        coll = self.viz.plot_field_lines(self.trace, color='curvature', field=dipole_b,
                                         delta=0.05)
        n_seg = int(np.sum(self.trace.nsteps))
        self.assertEqual(coll.get_array().size, n_seg)
        self.assertTrue(np.all(coll.get_array() > 0))
        with self.assertRaises(ValueError):
            self.viz.plot_field_lines(self.trace, color='curvature')   # no field
        # explicit value array
        vals = self.trace.z.copy()
        coll = self.viz.plot_field_lines(self.trace, color=vals, label='z', colorbar=False)
        self.assertEqual(coll.get_array().size, n_seg)
        fig = plt.figure()
        ax3 = fig.add_subplot(111, projection='3d')
        coll3 = self.viz.plot_field_lines(self.trace, ax=ax3, color='torsion', field=dipole_b)
        self.assertEqual(coll3.get_array().size, n_seg)

    def test_axis_labels_are_kept_across_overlays(self):
        mesh = self.viz.plot_geometry_map(dipole_b, 'bmag', extent=(-4, 4, -3, 3), n=8, unit='Re')
        ax = mesh.axes
        self.viz.plot_field_lines(self.trace, ax=ax, color='w')      # no unit: keep labels
        self.assertEqual(ax.get_xlabel(), 'x (Re)')
        self.viz.plot_field_lines(self.trace, ax=ax, color='w', unit='km')  # explicit: override
        self.assertEqual(ax.get_ylabel(), 'z (km)')

    def test_line_profiles(self):
        axes = self.viz.plot_line_profiles(self.trace, dipole_b, delta=0.05, unit='Re')
        self.assertEqual(len(axes), 2)
        self.assertEqual(axes[0].get_yscale(), 'log')      # curvature
        self.assertEqual(axes[1].get_yscale(), 'linear')   # torsion
        self.assertEqual(len(axes[0].get_lines()), 2)
        s = axes[0].get_lines()[0].get_xdata()
        np.testing.assert_array_equal(s, self.trace.arc_length(0))
        with self.assertRaises(ValueError):
            self.viz.plot_line_profiles(self.trace, dipole_b, axes=[plt.gca()])

    def test_frenet_frame_2d_3d_and_nan(self):
        out = self.viz.plot_frenet_frame(dipole_b, [3.0, 4.0], [0.0, 0.0], [0.0, 1.0],
                                         delta=0.05, plane='xz')
        self.assertEqual(set(out), {'T', 'n', 'b'})
        self.assertEqual(out['T'].N, 2)
        fig = plt.figure()
        ax3 = fig.add_subplot(111, projection='3d')
        out3 = self.viz.plot_frenet_frame(dipole_b, 3.0, 0.0, 0.0, ax=ax3)
        self.assertEqual(set(out3), {'T', 'n', 'b'})
        # straight field: normal undefined -> only T drawn
        uniform = lambda x, y, z: (np.zeros_like(x), np.zeros_like(x), np.ones_like(x))
        out = self.viz.plot_frenet_frame(uniform, 0.0, 0.0, 0.0, legend=False)
        self.assertEqual(set(out), {'T'})
        # legend=False must not leave entries for a later legend call
        ax = out['T'].axes
        self.viz.plot_frenet_frame(dipole_b, 3.0, 0.0, 0.0, ax=ax, delta=0.05)
        self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()], ['T', 'n', 'b'])


if __name__ == '__main__':
    unittest.main()
