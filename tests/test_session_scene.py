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
        for mode in ('three_d', 'slice', 'all', 'slice', 'all'):
            self.scene.set_layout(mode)
            self.assertEqual(cameras, self.scene.camera_state())
            self.assertEqual(self.view['origin'], [0., 0., 0.])
            np.testing.assert_allclose(self.scene.result['values'], old_values, equal_nan=True)
            visible = [bool(r.GetDraw()) for r in self.plotter.renderers]
            self.assertEqual(visible, [True]*5 if mode == 'all' else [i == (0 if mode == 'three_d' else 1) for i in range(5)])

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


if __name__ == '__main__':
    unittest.main()
