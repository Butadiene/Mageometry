"""FAC cross-sections: values, masks, camera preservation, and controls."""

import unittest

import numpy as np

from mageometry import GriddedField, viz3d
from mageometry.viz3d._fac_slice import _slice_settings
from mageometry.viz3d.slicer import _widget_state

try:
    import pyvista as pv
    pv.OFF_SCREEN = True
    HAVE_PV = True
except ImportError:
    HAVE_PV = False


def bipolar(x, y, z):
    return -y * z, x * z, np.ones_like(x)


def grid():
    axis = np.linspace(-2, 2, 13)
    return GriddedField(axis, axis, axis,
                        *bipolar(*np.meshgrid(axis, axis, axis, indexing='ij')))


def press(plotter, key):
    plotter.iren.interactor.SetKeyCode(key if len(key) == 1 else '\0')
    plotter.iren.interactor.SetKeySym(key)
    plotter.iren.interactor.InvokeEvent('KeyPressEvent')
    plotter.iren.interactor.InvokeEvent('CharEvent')


class TestSliceParameters(unittest.TestCase):

    def test_normal_and_origin_validation(self):
        np.testing.assert_array_equal(_slice_settings('Y', None)[0], [0, 1, 0])
        np.testing.assert_allclose(_slice_settings((2, 2, 0), None)[0],
                                   np.array([1, 1, 0]) / np.sqrt(2))
        for normal in ('invalid', (0, 0, 0), (1, 0), (np.nan, 0, 1)):
            with self.subTest(normal=normal), self.assertRaises(ValueError):
                _slice_settings(normal, None)
        for origin in ((1, 2), (0, 0, np.inf)):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                _slice_settings('x', origin)


@unittest.skipUnless(HAVE_PV, 'pyvista not installed')
class TestFACSlice(unittest.TestCase):

    def tearDown(self):
        pv.close_all()

    def test_slice_drag_samples_both_signs_without_changing_camera(self):
        p = viz3d.fac_view(grid(), n_lines=0, threshold=1.5, show=False,
                           slice_normal='z', slice_origin=(0, 0, 0))
        actor = p.actors['fac-slice']
        widget = _widget_state(p).plane_widgets[-1]
        # The z=0 plane has zero FAC: it must still be shown even though the
        # volume regions and peak maps are thresholded at 1.5.
        self.assertTrue(actor.visibility)
        np.testing.assert_array_equal(p.renderer.bounds, [-2, 2, -2, 2, -2, 2])
        np.testing.assert_allclose(actor.mapper.dataset['fac'], 0, atol=1e-12)
        camera = np.array(p.camera_position)
        scale = p.camera.parallel_scale
        for position, sign in ((0.5, 1), (-0.5, -1)):
            widget.SetOrigin(0, 0, position)
            widget.InvokeEvent('InteractionEvent')
            self.assertIs(p.actors['fac-slice'], actor)
            np.testing.assert_allclose(actor.mapper.dataset.points[:, 2], position)
            self.assertTrue(np.all(sign * actor.mapper.dataset['fac'] > 0))
        np.testing.assert_array_equal(p.camera_position, camera)
        self.assertEqual(p.camera.parallel_scale, scale)
        np.testing.assert_array_equal(p.renderer.bounds, [-2, 2, -2, 2, -2, 2])
        before = actor.mapper.dataset['fac'].copy()
        slider = _widget_state(p).slider_widgets[0]
        slider.GetRepresentation().SetValue(slider.GetRepresentation().GetMaximumValue())
        slider.InvokeEvent('EndInteractionEvent')
        np.testing.assert_array_equal(actor.mapper.dataset['fac'], before)
        self.assertTrue(actor.visibility)
        image = p.screenshot()
        self.assertGreater(np.std(image), 5)

    def test_toggle_align_and_oblique_plane_preserve_position(self):
        calls = []

        def field(x, y, z):
            calls.append(1)
            return bipolar(x, y, z)

        p = viz3d.fac_view(grid(), field=field, n_lines=0, show=False)
        self.assertNotIn('fac-slice', p.actors)
        self.assertEqual(len(_widget_state(p).plane_widgets), 0)
        initial_calls = len(calls)
        p.screenshot()  # initialise VTK's native interaction handlers too
        stereo = p.render_window.GetStereoRender()
        press(p, 'c')
        widget = _widget_state(p).plane_widgets[-1]
        self.assertEqual(widget.GetEnabled(), 1)
        widget.SetOrigin(0.1, 0.2, 0.3)
        widget.SetNormal(1, 1, 1)
        widget.InvokeEvent('InteractionEvent')
        points = p.actors['fac-slice'].mapper.dataset.points
        np.testing.assert_allclose((points - widget.GetOrigin()) @ widget.GetNormal(),
                                   0, atol=1e-6)
        press(p, 'c')
        self.assertFalse(p.actors['fac-slice'].visibility)
        self.assertEqual(widget.GetEnabled(), 0)
        for axis, key in enumerate(('F1', 'F2', 'F3')):
            press(p, key)
            np.testing.assert_allclose(widget.GetNormal(), np.eye(3)[axis])
            np.testing.assert_allclose(widget.GetOrigin(), [0.1, 0.2, 0.3])
            self.assertTrue(p.actors['fac-slice'].visibility)
            self.assertEqual(p.render_window.GetStereoRender(), stereo)
        self.assertEqual(len(_widget_state(p).plane_widgets), 1)
        self.assertEqual(len(calls), initial_calls)

    def test_masked_and_outside_planes_hide_without_stale_slice(self):
        p = viz3d.fac_view(grid(), mask=lambda x, y, z: x < 0, n_lines=0,
                           show=False, slice_normal='x', slice_origin=(1, 0, 0))
        actor = p.actors['fac-slice']
        widget = _widget_state(p).plane_widgets[-1]
        self.assertTrue(actor.visibility)
        for position in (-1, 5):
            widget.SetOrigin(position, 0, 0)
            widget.InvokeEvent('InteractionEvent')
            self.assertFalse(actor.visibility)
        widget.SetOrigin(1, 0, 0)
        widget.InvokeEvent('InteractionEvent')
        self.assertTrue(actor.visibility)
        gf = grid()
        gf.b[:] = np.nan
        empty = viz3d.fac_view(gf, slice_normal='y', n_lines=0, show=False)
        self.assertNotIn('fac-slice', empty.actors)
        press(empty, 'c')
        press(empty, 'c')
        self.assertNotIn('fac-slice', empty.actors)

    def test_external_plotter_slice_owns_only_its_renderer_and_legend(self):
        p = pv.Plotter(shape=(1, 2), off_screen=True)
        p.subplot(0, 0)
        p.add_mesh(pv.Sphere(), name='other-scene')
        p.subplot(0, 1)
        viz3d.fac_view(grid(), plotter=p, n_lines=0, show=False, slice_normal='x')
        renderer = p.renderer
        title = 'Slice: mu0 J parallel [field unit / length unit]'
        self.assertIn(title, p.scalar_bars)
        p.subplot(0, 0)
        widget = _widget_state(p).plane_widgets[-1]
        widget.SetOrigin(0.5, 0, 0)
        widget.InvokeEvent('InteractionEvent')
        self.assertIs(p.renderer, renderer)
        self.assertNotIn('fac-slice', p.renderers[0].actors)
        self.assertIn('other-scene', p.renderers[0].actors)
        press(p, 'c')
        self.assertNotIn(title, p.scalar_bars)
        press(p, 'c')
        self.assertIn(title, p.scalar_bars)


@unittest.skipUnless(HAVE_PV, 'pyvista not installed')
class TestSliceOnly(unittest.TestCase):

    def tearDown(self):
        pv.close_all()

    def test_isolation_restores_camera_layers_layout_and_interaction(self):
        p = viz3d.fac_view(grid(), n_lines=2, planet_radius=0.2,
                           slice_normal='x', show=False)
        press(p, 'a')  # intentionally hidden before entering the slice view
        p.camera.azimuth = 17
        p.camera.zoom(1.15)
        camera = p.camera.copy()
        style = p.iren.style
        viewport = p.renderer.GetViewport()
        visibility = {key: p.actors[key].visibility for key in
                      ('fac-positive', 'fac-negative', 'fac-lines', 'fac-arrows', 'fac-planet', 'fac-outline')}
        press(p, 'F4')
        self.assertEqual(p.renderer.GetViewport(), (0, 0, 1, 1))
        self.assertEqual([r.GetDraw() for r in p.renderers], [1, 0, 0, 0])
        self.assertFalse(_widget_state(p).plane_widgets[-1].GetEnabled())
        self.assertFalse(_widget_state(p).slider_widgets[0].GetEnabled())
        self.assertTrue(p.actors['fac-slice'].visibility)
        self.assertTrue(p.camera.parallel_projection)
        for key in visibility:
            self.assertFalse(p.actors[key].visibility, key)
        # Context shortcuts cannot accidentally reveal objects in this mode.
        press(p, 's')
        press(p, 'a')
        press(p, 'l')
        before = p.camera.copy()
        press(p, 'z')
        np.testing.assert_array_equal(p.camera.position, before.position)
        np.testing.assert_array_equal(p.camera.focal_point, before.focal_point)
        np.testing.assert_array_equal(p.camera.up, before.up)
        # Even a programmatic threshold change must not leak replaced actors.
        slider = _widget_state(p).slider_widgets[0]
        slider.GetRepresentation().SetValue(0.2)
        slider.InvokeEvent('EndInteractionEvent')
        for key in visibility:
            self.assertFalse(p.actors[key].visibility, key)
        press(p, 'F4')
        self.assertEqual(p.renderer.GetViewport(), viewport)
        self.assertEqual([r.GetDraw() for r in p.renderers], [1, 1, 1, 1])
        self.assertIs(p.iren.style, style)
        np.testing.assert_array_equal(p.camera.position, camera.position)
        np.testing.assert_array_equal(p.camera.focal_point, camera.focal_point)
        np.testing.assert_array_equal(p.camera.up, camera.up)
        self.assertEqual(p.camera.parallel_scale, camera.parallel_scale)
        self.assertEqual(p.camera.GetWindowCenter(), camera.GetWindowCenter())
        self.assertEqual(p.camera.clipping_range, camera.clipping_range)
        for key, visible in visibility.items():
            self.assertEqual(p.actors[key].visibility, visible, key)

    def test_scanning_preserves_pan_zoom_values_and_colour_scale(self):
        calls = []

        def field(x, y, z):
            calls.append(1)
            return bipolar(x, y, z)

        p = viz3d.fac_view(grid(), field=field, slice_normal='z', slice_only=True,
                           slice_origin=(0, 0, 0), n_lines=0, show=False)
        count = len(calls)
        actor = p.actors['fac-slice']
        np.testing.assert_allclose(actor.mapper.dataset['fac'], 0, atol=1e-12)
        limits = actor.mapper.scalar_range
        p.camera.zoom(1.3)
        p.camera.position = np.asarray(p.camera.position) + [0.2, -0.1, 0]
        p.camera.focal_point = np.asarray(p.camera.focal_point) + [0.2, -0.1, 0]
        camera = p.camera.copy()
        slider = _widget_state(p).slider_widgets[-1]
        slider.GetRepresentation().SetValue(0.5)
        slider.InvokeEvent('InteractionEvent')
        np.testing.assert_allclose(actor.mapper.dataset.points[:, 2], 0.5)
        self.assertTrue(np.all(actor.mapper.dataset['fac'] > 0))
        self.assertEqual(actor.mapper.scalar_range, limits)
        np.testing.assert_allclose(p.camera.position, np.asarray(camera.position) + [0, 0, 0.5])
        np.testing.assert_allclose(p.camera.focal_point, np.asarray(camera.focal_point) + [0, 0, 0.5])
        self.assertEqual(p.camera.parallel_scale, camera.parallel_scale)
        self.assertEqual(len(calls), count)
        for key, normal in (('F1', [1, 0, 0]), ('F2', [0, 1, 0]), ('F3', [0, 0, 1])):
            press(p, key)
            direction = np.asarray(p.camera.position) - p.camera.focal_point
            self.assertAlmostEqual(abs(np.dot(direction / np.linalg.norm(direction), normal)), 1)
            self.assertFalse(_widget_state(p).plane_widgets[-1].GetEnabled())
        self.assertGreater(np.std(p.screenshot()), 5)

    def test_oblique_and_near_axis_slices_are_exactly_face_on(self):
        for normal in ((1, 2, 1), (0.02, 1, 0)):
            with self.subTest(normal=normal):
                p = viz3d.fac_view(grid(), slice_normal=normal, slice_only=True,
                                   n_lines=0, show=False)
                direction = np.asarray(p.camera.position) - p.camera.focal_point
                n = np.asarray(normal) / np.linalg.norm(normal)
                self.assertAlmostEqual(abs(np.dot(direction / np.linalg.norm(direction), n)), 1, places=12)
                self.assertAlmostEqual(np.dot(p.camera.up, n), 0, places=12)
                p.close()

    def test_missing_data_and_repeated_entry_do_not_show_stale_surfaces(self):
        p = viz3d.fac_view(grid(), mask=lambda x, y, z: x < 0, n_lines=0,
                           slice_normal='x', slice_origin=(-1, 0, 0), slice_only=True, show=False)
        self.assertNotIn('fac-slice', p.actors)
        slider = _widget_state(p).slider_widgets[-1]
        slider.GetRepresentation().SetValue(1)
        slider.InvokeEvent('InteractionEvent')
        self.assertTrue(p.actors['fac-slice'].visibility)
        slider.GetRepresentation().SetValue(-1)
        slider.InvokeEvent('InteractionEvent')
        self.assertFalse(p.actors['fac-slice'].visibility)
        for _ in range(2):
            press(p, 'F4')
            press(p, 'F4')
        self.assertEqual(len(_widget_state(p).slider_widgets), 2)
        self.assertFalse(p.actors['fac-slice'].visibility)
        gf = grid()
        gf.b[:] = np.nan
        empty = viz3d.fac_view(gf, slice_only=True, n_lines=0, show=False)
        self.assertNotIn('fac-slice', empty.actors)
        self.assertGreater(np.std(empty.screenshot()), 0)
        press(empty, 'c')
        self.assertEqual(empty.renderer.GetViewport(), (0, 0, 0.7, 1))

    def test_external_subplots_and_initially_hidden_slice_are_restored(self):
        p = pv.Plotter(shape=(1, 2), off_screen=True)
        p.subplot(0, 0)
        other = p.add_mesh(pv.Sphere(), name='other-scene')
        other_camera = p.camera.copy()
        p.subplot(0, 1)
        viz3d.fac_view(grid(), plotter=p, n_lines=0, show=False)
        viewport = p.renderer.GetViewport()
        press(p, 'F4')
        self.assertEqual(p.renderer.GetViewport(), viewport)
        self.assertTrue(p.renderers[0].GetDraw())
        self.assertTrue(other.visibility)
        np.testing.assert_array_equal(p.renderers[0].camera.position, other_camera.position)
        press(p, 'F4')
        self.assertFalse(p.actors['fac-slice'].visibility)
        self.assertFalse(_widget_state(p).plane_widgets[-1].GetEnabled())
        self.assertEqual(len(p.scalar_bars), 0)


if __name__ == '__main__':
    unittest.main()
