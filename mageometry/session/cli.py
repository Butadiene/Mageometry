"""Source and display arguments shared by desktop and standalone launchers."""

import argparse

import numpy as np

from .presets import _validate_by, _validate_shape
from .specs import VIEW_LAYOUTS
from ..viz3d._current import COMPONENTS, TRANSVERSE_COMPONENTS, _component_name


def viewer_parser(description, default_component, *, desktop=False):
    parser = argparse.ArgumentParser(description=description or __doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument('--xmf', help='XDMF metadata for a simulation snapshot')
    sources.add_argument('--vtk', help='VTK ImageData or RectilinearGrid snapshot')
    parser.add_argument('--h5', help='HDF5 data, or heavy-file override with --xmf')
    parser.add_argument('--origin', type=float, nargs=3, help='x y z origin for direct HDF5')
    parser.add_argument('--spacing', type=float, nargs=3, help='dx dy dz for direct HDF5')
    parser.add_argument('--stride', type=int, default=1, help='read every nth grid node (default: 1)')
    model = parser.add_argument_group('model and comparison')
    model.add_argument('--by', type=float, nargs='+', help='compare these IMF By inputs in nT')
    model.add_argument('--initial-by', type=float, help='initial case, among --by values')
    model.add_argument('--shape', type=int, nargs=3, metavar=('NX', 'NY', 'NZ'),
                       help='model grid nodes (default: 65 49 49)')
    model.add_argument('--evaluation', choices=('direct', 'grid'),
                       help='derivative/tracing source (default: direct for models, grid for files)')
    if desktop:
        model.add_argument('--fac-delta', '--delta', dest='delta', type=float,
                           help='optional direct FAC step override (default: Geometry step)')
    else:
        model.add_argument('--delta', type=float, help='direct-model difference step in Re (default: 0.002)')
    model.add_argument('--cache-size', type=int, help='retained case previews (default: 2)' if desktop
                       else 'retained comparison previews (default: 2; requires --by)')
    model.add_argument('--color-limit', type=float, help='symmetric colour limit' if desktop
                       else 'shared symmetric colour limit (requires --by)')
    parser.add_argument('--component', type=_component_name, choices=tuple(COMPONENTS),
                         help=f'initial diagnostic (default: {default_component}, or eta with a background)')
    backgrounds = parser.add_mutually_exclusive_group()
    backgrounds.add_argument('--background', choices=('dipole',), help='dipole background for the model')
    backgrounds.add_argument('--background-xmf', help='background XDMF snapshot on the same axes')
    backgrounds.add_argument('--background-vtk', help='background VTK snapshot on the same axes')
    parser.add_argument('--contribution', choices=('total', 'background', 'residual'),
                         help='initial gradient contribution (requires a background)')
    parser.add_argument('--threshold', type=float, help='absolute cutoff for the initial diagnostic')
    parser.add_argument('--geometry-delta', type=float,
                         help='shared derivative step (default: 0.002 for direct models, preview spacing for grids)'
                         if desktop else 'derivative step for geometry diagnostics')
    parser.add_argument('--max-points', type=int, default=120000, help='preview node budget per case')
    parser.add_argument('--slice', choices=('x', 'y', 'z'), dest='slice_normal',
                         help='initial slice normal (default: x)' if desktop
                         else 'initial slice normal (default: y; x for --by)')
    parser.add_argument('--slice-origin', type=float, nargs=3, help='point on the slice in grid coordinates')
    display = parser.add_mutually_exclusive_group()
    display.add_argument('--slice-only', action='store_true', help='start with the enlarged face-on slice')
    if desktop:
        display.add_argument('--layout', choices=VIEW_LAYOUTS, help='initial plot layout (default: all)')
        startup = parser.add_mutually_exclusive_group()
        startup.add_argument('--session', help='open a saved JSON session')
        startup.add_argument('--empty', action='store_true', help='start with an empty file group')
    parser.add_argument('--screenshot', help='save an off-screen PNG instead of opening a window')
    return parser


def validate_viewer_options(parser, args, require_source, *, desktop=False):
    has_file = bool(args.xmf or args.vtk or args.h5)
    comparing = args.by is not None
    has_background = bool(args.background or args.background_xmf or args.background_vtk)
    if args.stride < 1:
        parser.error('--stride must be positive')
    if args.vtk and args.h5:
        parser.error('--h5 cannot be combined with --vtk')
    if require_source and not has_file:
        parser.error('specify a snapshot file with --xmf, --vtk, or --h5')
    if has_file and (comparing or args.shape is not None or args.delta is not None
                     or args.evaluation == 'direct'):
        parser.error('--by, --shape, --delta and direct evaluation require a model source')
    if args.h5 and not args.xmf and (args.origin is None or args.spacing is None):
        parser.error('direct HDF5 needs --origin and --spacing to define grid coordinates')
    if (args.origin is not None or args.spacing is not None) and not (args.h5 and not args.xmf):
        parser.error('--origin and --spacing are only used with direct HDF5 input')
    if args.contribution and not has_background:
        parser.error('--contribution requires a background')
    if args.background and has_file:
        parser.error('--background dipole is only for the built-in model; supply a background snapshot')
    if comparing and has_background and not desktop:
        parser.error('combined case/background comparison requires the desktop workspace: python -m mageometry.gui')
    if has_background and args.component not in TRANSVERSE_COMPONENTS:
        parser.error('background contributions require a transverse diagnostic')
    if not desktop and not comparing and any(value is not None for value in
                              (args.initial_by, args.cache_size, args.color_limit)):
        parser.error('--initial-by, --cache-size and --color-limit require --by')
    if args.initial_by is not None and (args.by is None or args.initial_by not in args.by):
        parser.error('--initial-by must be one of the --by values')
    args.evaluation = args.evaluation or ('grid' if has_file else 'direct')
    if args.evaluation == 'grid' and args.delta is not None:
        parser.error('--delta requires --evaluation direct; grid FAC uses axis spacing')
    for name in ('delta', 'geometry_delta', 'color_limit', 'threshold'):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value < 0 or
                                   (name != 'threshold' and value == 0)):
            qualifier = 'nonnegative' if name == 'threshold' else 'positive'
            parser.error(f'--{name.replace("_", "-")} must be {qualifier} and finite')
    if args.max_points < 27:
        parser.error('--max-points must be at least 27')
    if args.cache_size is not None and args.cache_size < 1:
        parser.error('--cache-size must be positive')
    try:
        if comparing:
            _validate_by(args.by)
        if args.shape is not None:
            _validate_shape(args.shape)
    except ValueError as error:
        parser.error(str(error))
