"""Explore magnetic currents and transverse geometry for a model or snapshot.

Examples (run from the repository after an editable install):
    python examples/fac_viewer.py
    python examples/fac_viewer.py --xmf run000.xmf --stride 4
    python examples/fac_viewer.py --xmf run000.xmf --h5 renamed-heavy.h5
    python examples/fac_viewer.py --h5 data.h5 --origin -20 -10 -10 --spacing .2 .2 .2
    python examples/fac_viewer.py --vtk run000.vti
    python examples/fac_viewer.py --slice y
    python examples/fac_viewer.py --slice x --slice-origin -6 0 0
    python examples/fac_viewer.py --slice x --slice-origin -6 0 0 --slice-only
    python examples/fac_viewer.py --component mu0J_n --slice x --slice-only
    python examples/fac_viewer.py --component mu0J_b --geometry-delta 0.002
    python examples/fac_viewer.py --component B_dT_dn_b --slice x --slice-only
    python examples/fac_viewer.py --component B_twist_diff --slice x --slice-only
    python examples/fac_viewer.py --screenshot /tmp/fac-preview.png

Choose a component with the top dropdown or F5/F6 (previous/next).
Red/blue: positive/negative in that component's basis; grey: magnetic lines.
Move the threshold slider to isolate stronger values. Geometry rates have no arrows.
Right-hand maps are signed peak projections, not slices or integrated current.
Press c for a draggable slice in the main view, or F1/F2/F3 for YZ/XZ/XY planes.
The centre panel always shows the current face-on slice.
F4 enlarges it with a dedicated position slider.
"""

import argparse

import numpy as np

from mageometry.viz3d._current import COMPONENTS, _component_name

from mageometry import GriddedField, geopack, geopack_field, load_hdf5, load_vtk, load_xdmf, viz3d


def model_snapshot():
    """A T96 + dipole magnetosphere; coordinates in Re and field in nT."""
    epoch = 100.
    ps = geopack.recalc(epoch)
    parmod = [2., -20., 0., -5., 0, 0, 0, 0, 0, 0]
    field = geopack_field('t96', 'dip', parmod, ps)
    axes = (np.linspace(-15, 5, 65), np.linspace(-8, 8, 49), np.linspace(-8, 8, 49))
    coords = np.meshgrid(*axes, indexing='ij')
    mask = lambda x, y, z: x * x + y * y + z * z < 2.5 ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        b = tuple(np.where(mask(*coords), np.nan, c) for c in field(*coords))
    metadata = dict(model='T96 + dipole', coordinate_system='GSM',
                    length_unit='Re', field_unit='nT',
                    parameters={'Pdyn [nPa]': parmod[0], 'Dst [nT]': parmod[1],
                                'IMF By [nT]': parmod[2], 'IMF Bz [nT]': parmod[3],
                                'Dipole tilt [rad]': float(ps), 'Epoch [Unix s]': epoch})
    grid = GriddedField(*axes, *b, metadata=metadata)
    return grid, dict(field=field, delta=0.002, mask=mask, planet_radius=1.,
                      length_unit='Re', current_scale=0.125, current_unit='nA/m^2',
                      trace_kwargs={'r0': 2.5, 'ds': 0.15, 'max_steps': 350})


def main(default_component="fac", description=None, require_source=False):
    parser = argparse.ArgumentParser(description=description or __doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument('--xmf', help='XDMF metadata for a simulation snapshot')
    sources.add_argument('--vtk', help='VTK ImageData or RectilinearGrid snapshot')
    parser.add_argument('--h5', help='HDF5 data, or heavy-file override with --xmf')
    parser.add_argument('--origin', type=float, nargs=3, help='x y z origin for direct HDF5')
    parser.add_argument('--spacing', type=float, nargs=3, help='dx dy dz for direct HDF5')
    parser.add_argument('--stride', type=int, default=1, help='read every nth grid node (default: 1)')
    parser.add_argument('--component', default=default_component,
                         type=_component_name,
                         choices=tuple(COMPONENTS),
                         help='initial component (F5/F6)')
    parser.add_argument('--threshold', type=float, help='absolute cutoff for the initial component')
    parser.add_argument('--geometry-delta', type=float,
                         help='derivative step for geometry diagnostics')
    parser.add_argument('--max-points', type=int, default=120000, help='preview node budget')
    parser.add_argument('--slice', choices=('x', 'y', 'z'), dest='slice_normal',
                         help='show a component slice initially, normal to this axis')
    parser.add_argument('--slice-origin', type=float, nargs=3,
                         help='initial point on the slice (x y z, in grid coordinates)')
    parser.add_argument('--slice-only', action='store_true',
                         help='start with only the slice, viewed face-on (F4 to return)')
    parser.add_argument('--screenshot', help='save an off-screen PNG instead of opening a window')
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('--stride must be positive')
    if args.vtk and args.h5:
        parser.error('--h5 cannot be combined with --vtk')
    if require_source and not (args.xmf or args.vtk or args.h5):
        parser.error('specify a snapshot file with --xmf, --vtk, or --h5')
    options = {}
    if args.xmf:
        grid = load_xdmf(args.xmf, h5_file=args.h5, stride=args.stride)
    elif args.vtk:
        grid = load_vtk(args.vtk, stride=args.stride)
    elif args.h5:
        if args.origin is None or args.spacing is None:
            parser.error('direct HDF5 needs --origin and --spacing to define grid coordinates')
        grid = load_hdf5(args.h5, origin=tuple(args.origin), spacing=tuple(args.spacing),
                         stride=args.stride)
    else:
        print('Model: T96 + dipole; native nT/Re converted to nA/m^2.', flush=True)
        grid, options = model_snapshot()
    if args.screenshot:
        import pyvista as pv
        pv.OFF_SCREEN = True
    print(f'Preparing magnetic geometry ({args.component}): {grid}', flush=True)
    plotter = viz3d.geometry_view(grid, component=args.component,
                            threshold=args.threshold, max_points=args.max_points,
                            geometry_delta=args.geometry_delta,
                            slice_normal=args.slice_normal, slice_origin=args.slice_origin,
                            slice_only=args.slice_only, slice_panel=True,
                            show=False, **options)
    if args.screenshot:
        plotter.screenshot(args.screenshot)
        plotter.close()
        print(f'Saved {args.screenshot}')
    else:
        plotter.show()


if __name__ == '__main__':
    main()
