"""Capture total, dipole-gradient and residual-gradient views on shared scales.

Run ``python benchmark/transverse_contribution_screenshots.py`` after an
editable install with ``.[viz3d]``. Uses the actual dropdown controls and
retains one total-field trace, slice, camera and per-diagnostic scale.
"""

from pathlib import Path
import sys

import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples'))

from fac_viewer import model_snapshot
from readme_screenshots import click_text, select
from mageometry import viz3d


def main():
    pv.OFF_SCREEN = True
    grid, options = model_snapshot()
    plotter = viz3d.geometry_view(
        grid, component='eta',
        slice_normal='x', slice_origin=(-6., 0., 0.), slice_panel=True,
        max_points=120000, show=False, **options)
    try:
        plotter.render()
        click_text(plotter, 'geometry-background-value')
        plotter.screenshot(ROOT / 'docs' / 'images' / 'viewer-background-menu.png')
        click_text(plotter, 'geometry-background-option-preset-0')
        assert plotter.renderers[0].actors['geometry-background-option-preset-0'].GetTextProperty().GetBold()
        for component in ('eta', 'gamma'):
            select(plotter, component)
            for contribution in ('total', 'background', 'residual'):
                click_text(plotter, 'geometry-dataset-value')
                click_text(plotter, f'geometry-dataset-option-{contribution}')
                actor = plotter.renderers[0].actors[f'geometry-dataset-option-{contribution}']
                assert actor.GetTextProperty().GetBold()
                plotter.render()
                path = ROOT / 'docs' / 'images' / f'contribution-{component}-{contribution}.png'
                plotter.screenshot(path)
                print(f'Saved {path}', flush=True)
    finally:
        plotter.close()


if __name__ == '__main__':
    main()
