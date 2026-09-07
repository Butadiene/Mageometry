"""Real VTK mouse/keyboard interactions for the component dropdown."""

import unittest

import numpy as np

from mageometry.viz3d._dropdown import _Dropdown

try:
    import pyvista as pv
    pv.OFF_SCREEN = True
    HAVE_PV = True
except ImportError:
    HAVE_PV = False


@unittest.skipUnless(HAVE_PV, 'pyvista not installed')
class TestComponentDropdown(unittest.TestCase):

    def setUp(self):
        self.p = pv.Plotter(off_screen=True, window_size=(1000, 700))
        self.p.add_mesh(pv.Cube())
        self.calls = []

        def choose(key):
            self.calls.append(key)
            self.menu.set_selected(key)

        self.menu = _Dropdown(self.p, {'a': 'Alpha choice', 'b': 'Beta choice',
                                      'c': 'Gamma choice'}, 'a', choose)
        self.p.screenshot()  # initialise VTK's native camera interaction

    def tearDown(self):
        pv.close_all()

    def mouse(self, event, actor=None, position=None):
        if actor is not None:
            x, y = actor.GetPositionCoordinate().GetComputedDisplayValue(self.menu.renderer)
            position = (x + 3, y)
        if position is not None:
            self.p.iren.interactor.SetEventPosition(*position)
        self.p.iren.interactor.InvokeEvent(event)

    def click(self, actor=None, position=None):
        self.mouse('LeftButtonPressEvent', actor, position)
        self.mouse('LeftButtonReleaseEvent')

    def key(self, key):
        self.p.iren.interactor.SetKeySym(key)
        self.p.iren.interactor.SetKeyCode('\0')
        self.p.iren.interactor.InvokeEvent('KeyPressEvent')
        self.p.iren.interactor.InvokeEvent('CharEvent')

    def test_click_lists_named_choices_and_selects_without_camera_motion(self):
        camera = self.p.camera.copy()
        presses = []
        self.p.iren.interactor.AddObserver('LeftButtonPressEvent', lambda *_: presses.append(1), 0.9)
        self.click(self.menu.value)
        self.assertTrue(self.menu.opened)
        self.assertTrue(all(actor.GetVisibility() for row in self.menu.rows for actor in row))
        self.assertEqual(self.menu.rows[1][1].GetInput(), 'Beta choice')
        self.mouse('MouseMoveEvent', self.menu.rows[1][1])
        self.assertEqual(self.menu.highlight, 1)
        self.click(self.menu.rows[1][1])
        self.assertEqual(self.calls, ['b'])
        self.assertEqual(self.menu.value.GetInput(), 'Beta choice')
        self.assertFalse(self.menu.opened)
        self.assertTrue(all(not actor.GetVisibility() for row in self.menu.rows for actor in row))
        self.assertEqual(presses, [])
        np.testing.assert_array_equal(self.p.camera.position, camera.position)
        np.testing.assert_array_equal(self.p.camera.focal_point, camera.focal_point)
        self.assertEqual(self.p.camera.parallel_scale, camera.parallel_scale)
        # Menu event suppression must not leak into subsequent scene gestures.
        self.click(position=(50, 100))
        self.assertEqual(presses, [1])
        self.mouse('LeftButtonPressEvent', position=(100, 100))
        self.mouse('MouseMoveEvent', position=(180, 140))
        self.mouse('LeftButtonReleaseEvent')
        self.assertFalse(np.array_equal(self.p.camera.position, camera.position))

    def test_outside_escape_and_wheel_do_not_select_or_move_camera(self):
        camera = self.p.camera.copy()
        self.click(self.menu.value)
        self.mouse('MouseWheelForwardEvent')
        self.mouse('MouseWheelBackwardEvent')
        self.click(position=(50, 100))
        self.assertFalse(self.menu.opened)
        self.click(self.menu.value)
        self.key('Down')
        self.key('Escape')
        self.assertFalse(self.menu.opened)
        self.assertEqual(self.menu.selected, 'a')
        self.assertEqual(self.calls, [])
        np.testing.assert_array_equal(self.p.camera.position, camera.position)

    def test_keyboard_navigation_selection_and_other_shortcuts(self):
        self.click(self.menu.value)
        self.key('End')
        self.key('Up')
        self.key('Return')
        self.assertEqual(self.calls, ['b'])
        self.assertFalse(self.menu.opened)
        self.click(self.menu.value)
        self.key('Home')
        self.key('Down')
        self.key('Down')
        self.key('KP_Enter')
        self.assertEqual(self.calls, ['b', 'c'])
        callbacks = []
        self.p.add_key_event('F4', lambda: callbacks.append(1))
        self.click(self.menu.value)
        self.key('F4')
        self.assertEqual(callbacks, [1])
        self.assertFalse(self.menu.opened)

    def test_resize_viewport_and_image_style_preserve_hit_testing(self):
        for viewport, size in (((0., 0., 0.7, 1.), (1440, 960)),
                               ((0., 0., 1., 1.), (1440, 960)),
                               ((0.5, 0., 1., 1.), (1200, 800))):
            with self.subTest(viewport=viewport):
                self.p.renderer.SetViewport(viewport)
                self.p.window_size = size
                self.p.enable_image_style()
                self.p.render()
                camera = self.p.camera.copy()
                self.click(self.menu.value)
                self.assertTrue(self.menu.opened)
                self.click(self.menu.rows[2][1])
                self.assertEqual(self.menu.selected, 'c')
                self.assertFalse(self.menu.opened)
                np.testing.assert_array_equal(self.p.camera.position, camera.position)

    def test_drag_release_does_not_latch_camera_or_reselect(self):
        camera = self.p.camera.copy()
        self.mouse('LeftButtonPressEvent', self.menu.value)
        self.mouse('MouseMoveEvent', self.menu.rows[2][1])
        self.mouse('LeftButtonReleaseEvent')
        self.assertEqual(self.calls, [])
        self.assertTrue(self.menu.opened)
        self.click(self.menu.rows[2][1])
        self.assertEqual(self.calls, ['c'])
        np.testing.assert_array_equal(self.p.camera.position, camera.position)


if __name__ == '__main__':
    unittest.main()
