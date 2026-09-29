"""Render the documented analytic beta_g slice with the standalone viewer.

Run after installing .[viz3d]:
    python benchmark/transverse_geometry_screenshot.py
"""

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

from mageometry import GriddedField, viz3d

ROOT = Path(__file__).resolve().parents[1]


def field(x, y, z):
    x, y, z = np.broadcast_arrays(x, y, z)
    return -y * z, x * z, np.ones_like(z)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'docs/images/transverse-beta-g.png')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pv.OFF_SCREEN = True
    axes = (np.linspace(-2., 2., 41),) * 3
    grid = GriddedField(*axes, *field(*np.meshgrid(*axes, indexing='ij')),
                        metadata={'model': 'Analytic field',
                                  'parameters': {'B(x,y,z)': '(-yz, xz, 1)'}})
    plotter = viz3d.geometry_view(
        grid, component='beta_g', field=field, delta=0.002,
        geometry_delta=0.002, max_points=None, n_lines=0,
        slice_panel=True, slice_only=True, slice_normal='x',
        slice_origin=(1., 0., 0.), show=False)
    try:
        plotter.window_size = (1440, 960)
        plotter.screenshot(args.output)
        print(f'Saved {args.output}')
    finally:
        plotter.close()


if __name__ == '__main__':
    main()
