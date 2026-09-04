"""
Tests for mageometry.viz3d (requires pyvista; skipped otherwise).

Mesh-construction tests are pure data and never render. Rendering smoke
tests attempt an off-screen screenshot and skip themselves when no usable
GL context exists.
"""

import unittest

import numpy as np

from mageometry import GriddedField, trace_field_lines
from mageometry.tracing import FieldLineTrace

try:
    import pyvista as pv
    pv.OFF_SCREEN = True
    HAVE_PV = True
except ImportError:
    HAVE_PV = False


def dipole_b(x, y, z):
    r2 = x * x + y * y + z * z
    r5 = r2 ** 2.5
    return 3.0 * x * z / r5, 3.0 * y * z / r5, (3.0 * z * z - r2) / r5


def index_field():
    """A 3x4x5 GriddedField whose bx encodes the (i, j, k) node index."""
    x, y, z = np.arange(3.0), np.arange(4.0) * 2.0, np.arange(5.0) * 3.0
    i, j, k = np.meshgrid(np.arange(3), np.arange(4), np.arange(5), indexing='ij')
    bx = 100.0 * i + 10.0 * j + k
    return GriddedField(x, y, z, bx, bx + 0.5, np.zeros_like(bx))


def dipole_field(n=9):
    """The dipole sampled on a grid away from the origin."""
    x = np.linspace(2.0, 4.0, n)
    y = np.linspace(-1.0, 1.0, n - 1)
    z = np.linspace(-1.0, 1.0, n - 2)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return GriddedField(x, y, z, *dipole_b(X, Y, Z))


@unittest.skipUnless(HAVE_PV, "pyvista not installed")
class TestRectilinearGridBridge(unittest.TestCase):

    def test_transpose_and_bmag(self):
        from mageometry import viz3d
        gf = index_field()
        mesh = viz3d.to_rectilinear_grid(gf)
        self.assertEqual(mesh.dimensions, (3, 4, 5))
        B = np.asarray(mesh.point_data['B'])
        nx, ny = 3, 4
        for ix, iy, iz in [(0, 0, 0), (2, 0, 0), (0, 3, 4), (1, 2, 3)]:
            p = ix + nx * (iy + ny * iz)      # VTK point index, x fastest
            self.assertEqual(B[p, 0], gf.bx[ix, iy, iz])
            self.assertEqual(B[p, 1], gf.by[ix, iy, iz])
        expected = np.sqrt(gf.bx ** 2 + gf.by ** 2 + gf.bz ** 2).ravel(order='F')
        np.testing.assert_allclose(mesh.point_data['bmag'], expected)

    def test_nan_survives_and_components(self):
        from mageometry import viz3d
        gf = index_field()
        gf.b[1, 2, 3, :] = np.nan
        mesh = viz3d.to_rectilinear_grid(gf, quantities=('bx', 'bmag'))
        p = 1 + 3 * (2 + 4 * 3)
        self.assertTrue(np.isnan(mesh.point_data['bx'][p]))
        self.assertTrue(np.isnan(mesh.point_data['bmag'][p]))
        self.assertEqual(np.sum(~np.isfinite(mesh.point_data['bx'])), 1)

    def test_derivative_quantity_interior_finite_boundary_nan(self):
        from mageometry import viz3d
        gf = dipole_field()
        mesh = viz3d.to_rectilinear_grid(gf, quantities=('curvature',), delta=0.05)
        nx, ny, nz = gf.shape
        vals = np.asarray(mesh.point_data['curvature']).reshape((nx, ny, nz), order='F')
        # The finite-difference stencil leaves the grid on the outermost layer.
        self.assertTrue(np.isnan(vals[0, 0, 0]))
        self.assertTrue(np.all(np.isfinite(vals[2:-2, 2:-2, 2:-2])))
        self.assertTrue(np.all(vals[np.isfinite(vals)] > 0))

    def test_analytic_field_callable(self):
        from mageometry import viz3d
        gf = dipole_field()
        mesh = viz3d.to_rectilinear_grid(gf, quantities=('curvature',),
                                         field=dipole_b, delta=0.01)
        vals = np.asarray(mesh.point_data['curvature'])
        self.assertTrue(np.all(np.isfinite(vals)))   # analytic field: no boundary NaN


@unittest.skipUnless(HAVE_PV, "pyvista not installed")
class TestTracePolydata(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.trace = trace_field_lines(dipole_b, [3.0, 5.0], [0.0, 0.0], [0.0, 0.0],
                                      direction='both', ds=0.1, r0=1.0)

    def test_connectivity_and_arrays(self):
        from mageometry import viz3d
        poly = viz3d.trace_polydata(self.trace)
        n_expected = int(np.sum(self.trace.nsteps + 1))
        self.assertEqual(poly.n_points, n_expected)
        self.assertEqual(poly.n_cells, self.trace.n_lines)
        self.assertTrue(np.all(np.isfinite(poly.points)))   # NaN padding trimmed
        self.assertIsNone(poly.point_data.active_scalars_name)
        # per-line point counts and arc length (signed for direction='both')
        offset = 0
        for i in range(self.trace.n_lines):
            n = self.trace.nsteps[i] + 1
            cell = poly.get_cell(i)
            self.assertEqual(cell.n_points, n)
            np.testing.assert_array_equal(poly.point_data['s'][offset:offset + n],
                                          self.trace.arc_length(i))
            np.testing.assert_array_equal(poly.point_data['line_id'][offset:offset + n], i)
            offset += n
        self.assertTrue(np.any(poly.point_data['s'] < 0))
        self.assertTrue(np.any(poly.point_data['s'] > 0))

    def test_quantity_and_array_colouring(self):
        from mageometry import viz3d
        poly = viz3d.trace_polydata(self.trace, color='curvature', field=dipole_b,
                                    delta=0.05)
        self.assertEqual(poly.point_data.active_scalars_name, 'curvature')
        self.assertTrue(np.all(poly.point_data['curvature'] > 0))
        with self.assertRaises(ValueError):
            viz3d.trace_polydata(self.trace, color='curvature')   # no field
        # explicit value array is trimmed per line
        poly = viz3d.trace_polydata(self.trace, color=self.trace.z, label='z')
        self.assertEqual(poly.point_data.active_scalars_name, 'z')
        self.assertTrue(np.all(np.isfinite(poly.point_data['z'])))

    def test_short_lines_skipped(self):
        from mageometry import viz3d
        nan = np.nan
        trace = FieldLineTrace(
            x=np.array([[0.0, nan, nan], [0.0, 1.0, 2.0]]),
            y=np.zeros((2, 3)), z=np.zeros((2, 3)),
            s=np.array([[0.0, nan, nan], [0.0, 1.0, 2.0]]),
            nsteps=np.array([0, 2]), start_index=np.array([0, 0]),
            status=np.array([3, 2]))
        poly = viz3d.trace_polydata(trace)
        self.assertEqual(poly.n_cells, 1)
        self.assertEqual(poly.n_points, 3)
        np.testing.assert_array_equal(poly.point_data['line_id'], 1)


@unittest.skipUnless(HAVE_PV, "pyvista not installed")
class TestFaceCamera(unittest.TestCase):
    """Pure-logic tests for the face-on panel camera placement."""

    def test_axis_normals_snap_to_canonical_side(self):
        from mageometry.viz3d.slicer import _face_camera
        origin = np.array([1.0, 2.0, 3.0])
        for normal, direction, up in [
            ((1, 0, 0), (1, 0, 0), (0, 0, 1)),
            ((-5, 0, 0), (1, 0, 0), (0, 0, 1)),    # sign ignored: same side
            ((0, 1, 0), (0, -1, 0), (0, 0, 1)),
            ((0, -1, 0), (0, -1, 0), (0, 0, 1)),
            ((0, 0, 1), (0, 0, 1), (0, 1, 0)),
            ((0, 0, -2), (0, 0, 1), (0, 1, 0)),
        ]:
            pos, u = _face_camera(normal, origin, 10.0)
            np.testing.assert_allclose(pos, origin + 10.0 * np.asarray(direction))
            np.testing.assert_allclose(u, up)

    def test_oblique_normal(self):
        from mageometry.viz3d.slicer import _face_camera
        origin = np.zeros(3)
        n = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        pos, up = _face_camera(n, origin, 4.0)
        np.testing.assert_allclose(pos, 4.0 * n)
        np.testing.assert_allclose(up, [0.0, 0.0, 1.0])   # z already perp
        # near-z normal: up falls back to (orthogonalized) +y
        pos, up = _face_camera([0.05, 0.0, 1.0], origin, 4.0)
        self.assertAlmostEqual(np.linalg.norm(pos), 4.0)
        self.assertAlmostEqual(np.linalg.norm(up), 1.0)
        self.assertAlmostEqual(np.dot(up, pos), 0.0)
        self.assertGreater(up[1], 0.9)

    def test_degenerate_normal(self):
        from mageometry.viz3d.slicer import _face_camera
        pos, up = _face_camera([0.0, 0.0, 0.0], np.zeros(3), 2.0)
        np.testing.assert_allclose(pos, [0.0, -2.0, 0.0])
        np.testing.assert_allclose(up, [0.0, 0.0, 1.0])


@unittest.skipUnless(HAVE_PV, "pyvista not installed")
class TestRendering(unittest.TestCase):
    """Off-screen smoke tests; skipped when no GL context is available."""

    def _screenshot(self, plotter):
        try:
            img = plotter.screenshot()
        except Exception as exc:                       # no off-screen GL
            plotter.close()
            self.skipTest(f"no off-screen GL context: {exc}")
        plotter.close()
        self.assertGreater(np.asarray(img).size, 0)

    def test_slice_view_ortho(self):
        from mageometry import viz3d
        p = viz3d.slice_view(dipole_field(), show=False)
        self.assertGreater(len(p.actors), 0)
        self._screenshot(p)

    def test_slice_view_plane_and_dataset_input(self):
        from mageometry import viz3d
        gf = dipole_field()
        mesh = viz3d.to_rectilinear_grid(gf, quantities=('bz',))
        p = viz3d.slice_view(mesh, quantity='bz', mode='plane', normal='x',
                             show=False)
        self._screenshot(p)
        with self.assertRaises(ValueError):
            viz3d.slice_view(gf, mode='diagonal', show=False)

    def test_front_view_ortho_panels(self):
        from mageometry import viz3d
        p = viz3d.slice_view(dipole_field(), show=False)   # front_view default
        self.assertEqual(len(p.renderers), 4)
        for i in range(1, 4):
            self.assertGreater(len(p.renderers[i].actors), 0)
            self.assertTrue(p.renderers[i].camera.parallel_projection)
        # slice_view leaves the main 3D subplot active
        self.assertIs(p.renderer, p.renderers[0])
        self._screenshot(p)

    def test_front_view_ortho_panel_survives_drag(self):
        """Regression: a dragged plane left the face-on panel's clipping
        range, so the slice vanished from the panel."""
        from mageometry import viz3d
        from mageometry.viz3d.slicer import _widget_state
        gf = dipole_field()
        p = viz3d.slice_view(gf, show=False)
        state = _widget_state(p)
        widget = state.plane_widgets[0]              # the x-axis plane
        new_x = float(gf.x[-1]) - 0.1                # near the +x boundary
        _, oy, oz = widget.GetOrigin()
        widget.SetOrigin(new_x, oy, oz)
        widget.InvokeEvent('EndInteractionEvent')    # reslice (default event)
        widget.InvokeEvent('InteractionEvent')       # camera tracking
        # the sliced mesh moved ...
        self.assertAlmostEqual(state.plane_sliced_meshes[0].bounds[0], new_x)
        # ... the panel camera followed it ...
        cam = p.renderers[1].camera
        self.assertAlmostEqual(cam.focal_point[0], new_x)
        # ... and the slice sits inside the panel's clipping range.
        depth = np.linalg.norm(np.asarray(cam.position)
                               - np.asarray(cam.focal_point))
        self.assertLess(cam.clipping_range[0], depth)
        self.assertGreater(cam.clipping_range[1], depth)
        self._screenshot(p)

    def test_front_view_plane_camera_tracks_widget(self):
        from mageometry import viz3d
        p = viz3d.slice_view(dipole_field(), mode='plane', normal='y',
                             show=False)
        self.assertEqual(len(p.renderers), 2)
        cam = p.renderers[1].camera
        # initial normal 'y': canonical face-on view from -y, z up
        self.assertLess(cam.position[1], cam.focal_point[1])
        from mageometry.viz3d.slicer import _widget_state
        widget = _widget_state(p).plane_widgets[-1]
        widget.SetNormal(0.0, 0.0, 1.0)
        widget.InvokeEvent('InteractionEvent')
        view = np.asarray(cam.position) - np.asarray(cam.focal_point)
        np.testing.assert_allclose(view / np.linalg.norm(view), [0, 0, 1],
                                   atol=1e-12)
        np.testing.assert_allclose(cam.up, [0, 1, 0], atol=1e-12)
        self._screenshot(p)

    def test_front_view_rejects_external_plotter(self):
        from mageometry import viz3d
        p = pv.Plotter(off_screen=True)
        with self.assertRaises(ValueError):
            viz3d.slice_view(dipole_field(), front_view=True, plotter=p,
                             show=False)
        # front_view defaults off for an external plotter
        viz3d.slice_view(dipole_field(), plotter=p, show=False)
        self.assertEqual(len(p.renderers), 1)
        p.close()

    def test_explore_with_lines_and_frames(self):
        from mageometry import viz3d
        gf = dipole_field()
        p = viz3d.explore(gf, seeds=[[3.0, 0.0, 0.0]], line_color='bmag',
                          show=False)
        viz3d.add_frenet_frame(p, dipole_b, [3.0], [0.0], [0.0], delta=0.05,
                               length=0.3)
        self.assertEqual(len(p.renderers), 4)   # face-on panels by default
        self.assertGreater(len(p.actors), 1)    # lines+frames in the 3D view
        self._screenshot(p)

    def test_add_field_lines_solid_and_tube(self):
        from mageometry import viz3d
        trace = trace_field_lines(dipole_b, [3.0], [0.0], [0.0],
                                  direction='both', ds=0.1, r0=1.0)
        p = pv.Plotter(off_screen=True)
        viz3d.add_field_lines(p, trace, color='red')
        viz3d.add_field_lines(p, trace, color='torsion', field=dipole_b,
                              delta=0.05, tube_radius=0.05)
        self._screenshot(p)


if __name__ == '__main__':
    unittest.main()
