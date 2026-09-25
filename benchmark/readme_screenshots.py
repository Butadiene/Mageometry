"""Regenerate the README's model-viewer screenshots with the current UI.

Run from the repository after installing ``.[viz3d]``::

    python benchmark/readme_screenshots.py

Uses the same T96 + dipole snapshot, derivative step, units, preview budget,
and window size as examples/fac_viewer.py. Component changes use the actual
dropdown so the cached values, slice position, and magnetic lines are shared.
"""

import argparse
from pathlib import Path
import sys

import pyvista as pv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples'))

from fac_viewer import model_snapshot
from mageometry import viz3d


def press(plotter, key):
    interactor = plotter.iren.interactor
    interactor.SetKeyCode(key if len(key) == 1 else '\0')
    interactor.SetKeySym(key)
    interactor.InvokeEvent('KeyPressEvent')
    interactor.InvokeEvent('CharEvent')


def click_text(plotter, name):
    renderer = next(r for r in plotter.renderers if name in r.actors)
    actor = renderer.actors[name]
    x, y = actor.GetPositionCoordinate().GetComputedDisplayValue(renderer)
    interactor = plotter.iren.interactor
    interactor.SetEventPosition(x + 3, y)
    interactor.InvokeEvent('LeftButtonPressEvent')
    interactor.InvokeEvent('LeftButtonReleaseEvent')


def select(plotter, component):
    click_text(plotter, 'current-component-value')
    click_text(plotter, f'current-component-option-{component}')
    renderer = plotter.renderers[0]
    assert renderer.actors[f'current-component-option-{component}'].GetTextProperty().GetBold()
    assert not renderer.actors[f'current-component-option-{component}'].GetVisibility()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'docs' / 'images')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pv.OFF_SCREEN = True
    print('Preparing the T96 + dipole viewer.', flush=True)
    grid, options = model_snapshot()
    plotter = viz3d.geometry_view(
        grid, component='fac', slice_normal='x', slice_origin=(-6., 0., 0.),
        slice_panel=True, max_points=120000, show=False, **options)

    def capture(filename):
        plotter.render()
        path = args.output_dir / filename
        plotter.screenshot(path)
        print(f'Saved {path}', flush=True)

    try:
        press(plotter, 'c')
        capture('fac-overview.png')
        press(plotter, 'c')
        capture('fac-slice.png')
        press(plotter, 'F4')
        capture('fac-slice-only.png')

        select(plotter, 'mu0J_n')
        press(plotter, 'F4')
        capture('current-components.png')
        press(plotter, 'F4')
        select(plotter, 'mu0J_b')
        capture('current-components-slice.png')
        select(plotter, 'B_dT_dn_b')
        capture('parallel-current-terms.png')
        select(plotter, 'B_twist_diff')
        capture('parallel-current-difference.png')
        click_text(plotter, 'current-component-value')
        capture('current-component-menu.png')
    finally:
        plotter.close()

    # Capture eta as a fresh CLI launch, with its own automatic context seeds.
    plotter = viz3d.geometry_view(
        grid, component='eta', slice_normal='x', slice_origin=(-6., 0., 0.),
        slice_panel=True, max_points=120000, show=False, **options)
    try:
        capture('viewer-eta-overview.png')
        press(plotter, 'F4')
        capture('viewer-eta-focus.png')
    finally:
        plotter.close()


if __name__ == '__main__':
    main()
