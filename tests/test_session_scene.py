"""Prepared-array scene layouts preserve scientific and camera state."""

from copy import deepcopy
import importlib.util
import unittest

import numpy as np

from mageometry.session.specs import default_analysis, default_view
from mageometry.viz3d.scene import GeometryScene


def packet():
    axis = np.linspace(-2, 2, 7)
    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
    values = x + 2*y - z
    values[0] = np.nan
    return dict(case='a', case_label='Synthetic', component='alpha', contribution='total',
                kind='field', axes=(axis, axis, axis), values=values, basis=None, paths=[],
                metadata={}, scale={'limit': 5., 'peak': 8., 'threshold': 2.},
                label='alpha [1 / grid unit]', analysis=default_analysis())


@unittest.skipUnless(importlib.util.find_spec('pyvista'), 'pyvista unavailable')
class TestSessionScene(unittest.TestCase):
    def setUp(self):
        import pyvista as pv
        self.plotter = pv.Plotter(shape='1|4', off_screen=True, border=False)
        self.scene = GeometryScene(self.plotter)
        self.view = default_view()
        self.view['origin'] = [0., 0., 0.]
        self.scene.set_result(packet(), self.view)

    def tearDown(self):
        self.plotter.close()

    def test_layouts_preserve_cameras_values_selection_and_plane(self):
        self.plotter.renderers[0].camera.zoom(1.7)
        self.plotter.renderers[1].camera.zoom(1.3)
        cameras = self.scene.camera_state()
        old_values = self.scene.result['values'].copy()
        visibility = {'all': [True]*5, 'three_d_slice': [True, True, False, False, False],
                      'three_d': [True, False, False, False, False],
                      'slice': [False, True, False, False, False]}
        for mode in ('three_d_slice', 'slice', 'three_d_slice', 'three_d', 'all'):
            self.scene.set_layout(mode)
            self.assertEqual(cameras, self.scene.camera_state())
            self.assertEqual(self.view['origin'], [0., 0., 0.])
            np.testing.assert_allclose(self.scene.result['values'], old_values, equal_nan=True)
            visible = [bool(r.GetDraw()) for r in self.plotter.renderers]
            self.assertEqual(visible, visibility[mode])

    def test_combined_layout_fills_height_and_preserves_shared_slice(self):
        self.scene.set_layout('three_d_slice')
        self.assertEqual(self.plotter.renderers[0].GetViewport(), (0., 0., .55, 1.))
        self.assertEqual(self.plotter.renderers[1].GetViewport(), (.55, 0., 1., 1.))
        self.scene.widget.SetOrigin(.5, 0., 0.)
        self.scene.widget.InvokeEvent('EndInteractionEvent')
        np.testing.assert_allclose(self.view['origin'], [.5, 0., 0.])
        for renderer in self.plotter.renderers[:2]:
            np.testing.assert_allclose(renderer.actors['slice'].mapper.dataset.points[:, 0], .5)
        self.scene.set_layout('all')
        self.scene.update_display()
        for index in range(2, 5):
            self.assertIn('projection', self.plotter.renderers[index].actors)

    def test_threshold_changes_do_not_remove_slice_values(self):
        original = self.plotter.renderers[1].actors['slice'].mapper.dataset.copy()
        self.view['thresholds']['field:alpha'] = 1e6
        self.scene.update_display()
        current = self.plotter.renderers[1].actors['slice'].mapper.dataset
        np.testing.assert_allclose(current['value'], original['value'], equal_nan=True)
        self.assertNotIn('positive', self.plotter.renderers[0].actors)

    def test_case_update_preserves_focus_and_cameras(self):
        self.scene.set_layout('slice')
        self.plotter.renderers[1].camera.zoom(1.5)
        cameras = self.scene.camera_state()
        result = packet()
        result['values'] *= -1
        result['case'] = 'b'
        self.scene.set_result(result, self.view)
        self.assertEqual(self.view['layout'], 'slice')
        self.assertEqual(cameras, self.scene.camera_state())

    def test_invalid_packet_leaves_previous_scene(self):
        result = self.scene.result
        actors = dict(self.plotter.renderers[0].actors)
        invalid = packet()
        invalid['values'] = np.zeros((2, 2, 2))
        with self.assertRaises(ValueError):
            self.scene.set_result(invalid, self.view)
        self.assertIs(self.scene.result, result)
        self.assertEqual(actors, dict(self.plotter.renderers[0].actors))

    def test_mouse_rotation_targets_visible_3d_after_layout_changes(self):
        self.view['plane'] = False
        self.plotter.show(auto_close=False, interactive=False)
        iren = self.plotter.iren
        width, height = self.plotter.window_size
        start = (int(.15 * width), int(.8 * height))
        end = (start[0] + 30, start[1] - 20)
        for mode in ('three_d_slice', 'slice', 'three_d_slice', 'three_d', 'three_d', 'all'):
            self.scene.set_layout(mode)
            if mode == 'slice':
                continue
            with self.subTest(mode=mode):
                before = self.scene.camera_state()
                iren._mouse_left_button_press(*start)
                iren._mouse_move(*end)
                iren._mouse_left_button_release(*end)
                after = self.scene.camera_state()
                self.assertNotEqual(before['0']['position'], after['0']['position'])
                for index in range(1, 5):
                    self.assertEqual(before[str(index)], after[str(index)])
                self.assertEqual(self.view['origin'], [0., 0., 0.])

    def test_focus_pan_and_zoom_do_not_change_hidden_cameras(self):
        self.view['plane'] = False
        self.plotter.show(auto_close=False, interactive=False)
        iren = self.plotter.iren
        width, height = self.plotter.window_size
        start = (int(.15 * width), int(.8 * height))
        for mode, selected in (('slice', '1'), ('three_d', '0')):
            self.scene.set_layout(mode)
            before = self.scene.camera_state()
            iren._mouse_middle_button_press(*start)
            iren._mouse_move(start[0] + 20, start[1] - 15)
            iren._mouse_middle_button_release()
            iren.interactor.MouseWheelForwardEvent()
            after = self.scene.camera_state()
            self.assertNotEqual(before[selected]['focal_point'], after[selected]['focal_point'])
            self.assertNotEqual(before[selected]['parallel_scale'], after[selected]['parallel_scale'])
            for key in before.keys() - {selected}:
                self.assertEqual(before[key], after[key])

    def test_hidden_renderers_do_not_cover_any_window_corner(self):
        width, height = self.plotter.window_size
        for mode in ('three_d_slice', 'three_d', 'slice'):
            self.scene.set_layout(mode)
            for renderer in self.plotter.renderers:
                if not renderer.GetDraw():
                    for x, y in ((0, 0), (width - 1, height - 1), (width // 2, height // 2)):
                        self.assertFalse(renderer.IsInViewport(x, y))

    def test_plane_returns_to_3d_renderer_after_slice_focus(self):
        self.scene.set_layout('slice')
        # The cursor/active renderer may still belong to a companion panel.
        self.plotter.subplot(4)
        self.scene.set_layout('three_d')
        self.assertTrue(self.scene.widget.GetEnabled())
        self.assertIs(self.scene.widget.GetCurrentRenderer(), self.plotter.renderers[0])
        self.scene.widget.SetOrigin(.5, 0., 0.)
        self.scene.widget.InvokeEvent('EndInteractionEvent')
        np.testing.assert_allclose(self.view['origin'], [.5, 0., 0.])

    def test_region_shading_and_context_match_standalone_settings(self):
        result = packet()
        result['paths'] = [np.array([[-1., 0., 0.], [0., 1., 0.], [1., 0., 0.]])]
        self.scene.set_result(result, self.view)
        actors = self.plotter.renderers[0].actors
        for name in ('positive', 'negative'):
            self.assertIn('Normals', actors[name].mapper.dataset.point_data)
            self.assertEqual(actors[name].prop.opacity, .5)
        self.assertEqual(actors['lines'].prop.opacity, .45)
        self.assertAlmostEqual(actors['lines'].prop.line_width, 1.4)
        self.assertAlmostEqual(self.scene.widget.GetPlaneProperty().GetOpacity(), .08)
        before = np.asarray(self.plotter.renderers[0].bounds)
        self.scene.widget.SetEnabled(False)
        np.testing.assert_allclose(self.plotter.renderers[0].bounds, before)

    def test_lighting_survives_initial_render_and_result_replacements(self):
        # Plotter's default light kit must survive the initial set_result.
        self.assertTrue(self.plotter.renderers[0].lights)
        self.plotter.renderers[0].lights[0].intensity = .37
        lights = [list(renderer.lights) for renderer in self.plotter.renderers]
        for mode in ('three_d', 'slice', 'all'):
            self.scene.set_layout(mode)
            result = packet()
            result['values'] *= -1
            self.scene.set_result(result, self.view)
            for renderer, expected in zip(self.plotter.renderers, lights):
                self.assertEqual(renderer.lights, expected)
            self.assertAlmostEqual(self.plotter.renderers[0].lights[0].intensity, .37)


if __name__ == '__main__':
    unittest.main()
