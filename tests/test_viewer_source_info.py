"""Generic provenance, dataset synchronization, and annotation layout."""

import unittest
from unittest.mock import patch

import numpy as np

from mageometry import GriddedField, viz3d
from mageometry.viz3d._source_info import _source_lines

try:
    import pyvista as pv
    pv.OFF_SCREEN = True
    HAVE_PV = True
except ImportError:
    HAVE_PV = False


def grid(metadata):
    axis = np.linspace(-2, 2, 9)
    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
    return GriddedField(axis, axis, axis, -y*z, x*z, np.ones_like(z), metadata=metadata)


def press(plotter, key):
    plotter.iren.interactor.SetKeyCode('\0')
    plotter.iren.interactor.SetKeySym(key)
    plotter.iren.interactor.InvokeEvent('KeyPressEvent')


class TestSourceMetadata(unittest.TestCase):
    def test_generic_labels_and_absent_provenance(self):
        self.assertEqual(_source_lines({}), [])
        self.assertEqual(_source_lines({'imf_by': 5}), [])
        text = '\n'.join(_source_lines(dict(source='run/step.vti', time=2.5,
                                            parameters={'Resistivity [native]': 0.01})))
        self.assertIn('Source: run/step.vti', text)
        self.assertIn('Time: 2.5', text)
        self.assertIn('Resistivity [native] = 0.01', text)
        self.assertNotIn('Model:', text)


@unittest.skipUnless(HAVE_PV, 'pyvista not installed')
class TestSourceInfoViewer(unittest.TestCase):
    def tearDown(self):
        pv.close_all()

    def test_metadata_follows_selected_case_in_both_modes_and_failed_selection(self):
        cases = {'analytic': grid(dict(model='Custom field', parameters={'Amplitude': 2.})),
                 'simulation': grid(dict(source='step.vti', time=3., parameters={'Step': 20})),
                 'unknown': grid({})}
        p = viz3d.compare_geometry(cases, component='eta', n_lines=0, show=False,
                                  slice_panel=True, slice_normal='z', slice_origin=(0, 0, 1))
        info = p.actors['geometry-source-info']
        p.screenshot()
        self.assertIn('Model: Custom field', info.GetInput())
        press(p, 'F4')
        press(p, 'F8')
        self.assertTrue(info.GetVisibility())
        self.assertIn('Source: step.vti', info.GetInput())
        self.assertIn('Step = 20', info.GetInput())
        self.assertNotIn('Custom field', info.GetInput())
        before = info.GetInput()
        callback = p.iren._key_press_event_callbacks['F8'][0]
        with patch('mageometry.viz3d.fac._OverviewData.prepare', side_effect=ValueError('bad data')):
            with self.assertRaises(ValueError):
                callback()
        self.assertEqual(info.GetInput(), before)
        press(p, 'F4')
        self.assertTrue(info.GetVisibility())
        press(p, 'F8')
        self.assertFalse(info.GetVisibility())
        press(p, 'F8')
        self.assertTrue(info.GetVisibility())
        self.assertIn('Amplitude = 2', info.GetInput())

    def test_source_and_headers_fit_on_resize_and_focus(self):
        metadata = dict(model='Custom analytic model', source='long-directory/' * 15 + 'step.vti',
                        coordinate_system='Cartesian', length_unit='km', field_unit='nT',
                        parameters={f'Parameter {i} [unit]': i / 3 for i in range(6)})
        p = viz3d.geometry_view(grid(metadata), component='eta', n_lines=0, show=False,
                               slice_panel=True, slice_normal='z', slice_origin=(0, 0, 1))
        info = p.actors['geometry-source-info']
        for size in ((1920, 960), (1280, 720)):
            p.window_size = size
            for focused in (False, True):
                with self.subTest(size=size, focused=focused):
                    if focused:
                        press(p, 'F4')
                    p.screenshot()
                    viewport = p.renderer.GetViewport()
                    width = size[0] * (viewport[2] - viewport[0])
                    height = size[1] * (viewport[3] - viewport[1])
                    measured = [0., 0.]
                    info.GetSize(p.renderer, measured)
                    self.assertLessEqual(measured[0], .93 * width)
                    self.assertLessEqual(measured[1], .11 * height)
                    header = p.actors['fac-focus-status' if focused else 'fac-slice-status']
                    self.assertLess(info.GetPosition()[1], header.GetPosition()[1])
                    description = p.actors['fac-component-description']
                    description.GetSize(p.renderer, measured)
                    self.assertLessEqual(measured[0], .93 * width)
                    for name, fraction in (('geometry-background', .285),
                                           ('geometry-dataset', .28),
                                           ('current-component', .32)):
                        title = p.actors[name + '-value']
                        title.GetSize(p.renderer, measured)
                        self.assertLessEqual(measured[0], fraction * width - 42)
                    if focused:
                        # The fitted plane lies below the reserved metadata
                        # area, rather than being covered by the text box.
                        for point in p.actors['fac-slice'].mapper.dataset.points:
                            p.renderer.SetWorldPoint(*point, 1.)
                            p.renderer.WorldToDisplay()
                            self.assertLessEqual(p.renderer.GetDisplayPoint()[1], .69 * height)
                        press(p, 'F4')


if __name__ == '__main__':
    unittest.main()
