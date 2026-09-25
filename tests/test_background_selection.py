"""GUI background selection, transactional failure and in-viewer file browsing."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mageometry import viz3d, trace_field_lines
from mageometry.viz3d._background import _load_background, _BackgroundChoices
from mageometry.viz3d._current import COMPONENTS
from mageometry.viz3d._contribution_data import _ContributionData
from test_geometry_comparison import HAVE_PV, camera_state, click, press
from test_transverse_contribution_view import grid
from test_transverse_decomposition import affine, TOTAL, BACKGROUND


class TestBackgroundSources(unittest.TestCase):
    def test_file_formats_and_stride(self):
        for suffix, loader in (('.xmf', 'load_xdmf'), ('.xdmf', 'load_xdmf'),
                               ('.vti', 'load_vtk'), ('.vtr', 'load_vtk')):
            path = Path('background' + suffix)
            with patch('mageometry.viz3d._background.' + loader) as load:
                self.assertIs(_load_background(path, stride=3), load.return_value)
                load.assert_called_once_with(path, stride=3)
        with self.assertRaises(ValueError):
            _load_background('unknown.h5')

    def test_invalid_presets_and_duplicate_initial_object(self):
        for choices in ([], {'': lambda *args: None}, {'invalid': object()}):
            with self.assertRaises((TypeError, ValueError)):
                _BackgroundChoices(choices)
        field = lambda x, y, z: (x, y, z)
        choices = _BackgroundChoices({'Dipole': field}, initial=field)
        self.assertEqual(choices.selected, 'preset-0')
        self.assertEqual(set(choices.options()), {'none', 'preset-0', 'load'})


@unittest.skipUnless(HAVE_PV, 'pyvista not installed')
class TestBackgroundGUI(unittest.TestCase):
    def setUp(self):
        self.field = affine(TOTAL, [0., 0., 3.])
        self.background = affine(BACKGROUND, [1., 0., 0.])
        self.grid = grid(self.field)

    def viewer(self, **kwargs):
        options = dict(field=self.field, delta=.002, slice_panel=True,
                       slice_normal='x', slice_origin=(0., 0., 0.), n_lines=0, show=False,
                       background_choices={'Reference': self.background})
        options.update(kwargs)
        plotter = viz3d.geometry_view(self.grid, **options)
        self.addCleanup(plotter.close)
        plotter.render()
        return plotter

    def choose(self, plotter, key):
        click(plotter, 'geometry-background-value')
        click(plotter, 'geometry-background-option-' + key)

    def test_enable_disable_and_replace_preserve_total_context(self):
        with patch('mageometry.viz3d.fac.trace_field_lines', wraps=trace_field_lines) as trace:
            plotter = self.viewer(component='eta', seeds=[[.5, .5, .5]], n_lines=1,
                                  trace_kwargs={'max_steps': 5},
                                  background_choices={'First': self.background,
                                                      'Second': affine(BACKGROUND * 2, [1., 0., 0.])})
            calls = trace.call_count
            cameras = [camera_state(r) for r in plotter.renderers]
            projection = plotter.renderers[1].actors['fac-projection']
            initial = projection.mapper.dataset.point_data['eta'].copy()
            self.choose(plotter, 'preset-0')
            press(plotter, 'F7')  # total -> residual
            self.assertFalse(np.allclose(projection.mapper.dataset.point_data['eta'], initial, equal_nan=True))
            for renderer, camera in zip(plotter.renderers, cameras):
                np.testing.assert_allclose(camera_state(renderer), camera)
            self.assertEqual(trace.call_count, calls)
            self.choose(plotter, 'preset-1')
            self.assertTrue(plotter.renderers[0].actors['geometry-dataset-option-residual'].GetTextProperty().GetBold())
            self.choose(plotter, 'none')
            np.testing.assert_allclose(projection.mapper.dataset.point_data['eta'], initial)
            self.assertEqual(trace.call_count, calls)
            for component in COMPONENTS:
                self.assertIn('current-component-option-' + component, plotter.renderers[0].actors)
            self.assertNotIn('geometry-dataset-option-residual', plotter.renderers[0].actors)

    def test_legacy_component_changes_to_eta_when_background_enabled(self):
        plotter = self.viewer(component='fac')
        self.choose(plotter, 'preset-0')
        actors = plotter.renderers[0].actors
        self.assertTrue(actors['current-component-option-eta'].GetTextProperty().GetBold())
        self.assertNotIn('current-component-option-fac', actors)
        press(plotter, 'F6')
        self.assertTrue(actors['current-component-option-alpha'].GetTextProperty().GetBold())

    def test_sampling_failure_keeps_data_labels_and_scale(self):
        plotter = self.viewer(component='gamma')
        actor = plotter.renderers[1].actors['fac-projection']
        before = actor.mapper.dataset.point_data['gamma'].copy()
        scale = actor.mapper.scalar_range
        with patch.object(_ContributionData, 'prepare', side_effect=RuntimeError('Invalid background stencil')):
            self.choose(plotter, 'preset-0')
        np.testing.assert_allclose(actor.mapper.dataset.point_data['gamma'], before)
        self.assertEqual(actor.mapper.scalar_range, scale)
        actors = plotter.renderers[0].actors
        self.assertTrue(actors['geometry-background-option-none'].GetTextProperty().GetBold())
        self.assertIn('Invalid background stencil', actors['geometry-background-message'].GetInput())

    def test_file_navigation_cancel_and_accept_in_focused_slice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'snapshots'
            folder.mkdir()
            path = folder / 'background.vtr'
            viz3d.to_rectilinear_grid(grid(self.background), quantities=()).save(path)
            plotter = self.viewer(component='eta', background_directory=root)
            press(plotter, 'F4')
            before = camera_state(plotter.renderers[0])
            self.choose(plotter, 'load')
            click(plotter, 'geometry-background-option-entry-0')
            self.assertIn(str(folder), plotter.renderers[0].actors['geometry-background-value'].GetInput())
            press(plotter, 'Escape')
            self.assertTrue(plotter.renderers[0].actors['geometry-background-option-none'].GetTextProperty().GetBold())
            self.choose(plotter, 'load')
            click(plotter, 'geometry-background-option-entry-0')
            click(plotter, 'geometry-background-option-entry-0')
            actors = plotter.renderers[0].actors
            self.assertEqual(actors['geometry-background-value'].GetInput(), path.name)
            self.assertTrue(actors['geometry-background-option-file'].GetTextProperty().GetBold())
            np.testing.assert_allclose(camera_state(plotter.renderers[0]), before)
            press(plotter, 'F8')
            self.assertTrue(actors['geometry-dataset-option-background'].GetTextProperty().GetBold())
            press(plotter, 'F4')
            self.assertFalse(actors['geometry-background-option-file'].GetVisibility())

    def test_invalid_file_and_cancel_via_other_menu_leave_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'broken.vti').touch()
            def fail(path):
                raise ValueError('Background axes do not match')
            plotter = self.viewer(component='eta', background_directory=root, background_loader=fail)
            self.choose(plotter, 'load')
            click(plotter, 'geometry-background-option-entry-0')
            actors = plotter.renderers[0].actors
            self.assertTrue(actors['geometry-background-option-none'].GetTextProperty().GetBold())
            self.assertIn('Background axes do not match', actors['geometry-background-message'].GetInput())
            self.choose(plotter, 'load')
            click(plotter, 'current-component-value')
            actors = plotter.renderers[0].actors
            self.assertIn('geometry-background-option-none', actors)
            self.assertNotIn('geometry-background-option-entry-0', actors)
            self.assertTrue(actors['current-component-option-eta'].GetVisibility())

    def test_incompatible_grid_does_not_replace_active_background(self):
        different = grid(self.background)
        different.x[:] += 10.
        plotter = self.viewer(component='eta', background_choices={
            'Valid': self.background, 'Invalid axes': different})
        self.choose(plotter, 'preset-0')
        press(plotter, 'F7')
        actor = plotter.renderers[1].actors['fac-projection']
        before = actor.mapper.dataset.point_data['eta'].copy()
        self.choose(plotter, 'preset-1')
        actors = plotter.renderers[0].actors
        self.assertTrue(actors['geometry-background-option-preset-0'].GetTextProperty().GetBold())
        self.assertTrue(actors['geometry-dataset-option-residual'].GetTextProperty().GetBold())
        self.assertIn('different axes', actors['geometry-background-message'].GetInput())
        np.testing.assert_allclose(actor.mapper.dataset.point_data['eta'], before)

    def test_file_paging_and_parent_navigation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(12):
                (root / f'{index:02d}.vti').touch()
            plotter = self.viewer(background_directory=root)
            self.choose(plotter, 'load')
            click(plotter, 'geometry-background-option-next')
            actors = plotter.renderers[0].actors
            self.assertIn('2 of 2', actors['geometry-background-caption'].GetInput())
            click(plotter, 'geometry-background-option-previous')
            self.assertIn('1 of 2', actors['geometry-background-caption'].GetInput())
            click(plotter, 'geometry-background-option-parent')
            self.assertEqual(actors['geometry-background-value'].GetInput(), str(root.parent))
            click(plotter, 'geometry-background-option-cancel')
            self.assertTrue(plotter.renderers[0].actors['geometry-background-option-none'].GetTextProperty().GetBold())


if __name__ == '__main__':
    unittest.main()
