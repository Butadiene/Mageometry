"""Explore a magnetic model, simulation snapshot, or IMF By comparison.

Examples:
    python -m mageometry.viz3d --component gamma
    python -m mageometry.viz3d --xmf snapshot.xmf --stride 4
    python -m mageometry.viz3d --by -5 -3 -1 1 3 5 --slice-only
    python -m mageometry.viz3d --background dipole --contribution residual
    python -m mageometry.viz3d --component fac --screenshot preview.png

For interactive source editing and session save/restore, use mageometry.gui.
"""

from functools import partial
from pathlib import Path

from .. import load_hdf5, load_vtk, load_xdmf, viz3d
from ..session.presets import (DEFAULT_SHAPE, MODEL_DELTA, case_label, make_cases,
                               make_fields, model_snapshot, model_view_options)
from ._background import _load_background
from ..session.cli import viewer_parser as _parser, validate_viewer_options as _validate
from ..session.cli import model_overrides


def _single_snapshot(args):
    options = {}
    if args.xmf:
        grid = load_xdmf(args.xmf, h5_file=args.h5, stride=args.stride)
    elif args.vtk:
        grid = load_vtk(args.vtk, stride=args.stride)
    elif args.h5:
        grid = load_hdf5(args.h5, origin=tuple(args.origin), spacing=tuple(args.spacing),
                         stride=args.stride)
    else:
        print(f'Model: {(args.model or "t96").upper()} + dipole; native nT/Re converted to nA/m^2.', flush=True)
        overrides = model_overrides(args)
        if args.shape is not None:
            overrides['shape'] = tuple(args.shape)
        if args.evaluation != 'direct':
            overrides['evaluation'] = args.evaluation
        if args.delta is not None:
            overrides['delta'] = args.delta
        grid, options = model_snapshot(**overrides)
    options['background_loader'] = partial(_load_background, stride=args.stride)
    if args.xmf or args.vtk or args.h5:
        options['background_directory'] = Path(args.xmf or args.vtk or args.h5).resolve().parent
    viewer = viz3d.geometry_view
    if args.background or args.background_xmf or args.background_vtk:
        if args.background:
            background = options['background_choices']['Dipole']
            label = 'Dipole'
        elif args.background_xmf:
            background = load_xdmf(args.background_xmf, stride=args.stride)
            label = args.background_xmf
        else:
            background = load_vtk(args.background_vtk, stride=args.stride)
            label = args.background_vtk
        viewer = viz3d.transverse_contribution_view
        options.update(background=background, background_label=label,
                       contribution=args.contribution or 'total')
    return viewer, grid, options


def _comparison(args):
    shape = tuple(args.shape) if args.shape is not None else DEFAULT_SHAPE
    print(f'Sampling {(args.model or "t96").upper()} + dipole: IMF By={args.by} nT; grid {shape} in Re.', flush=True)
    overrides = model_overrides(args)
    cases = make_cases(args.by, shape, **overrides)
    fields = make_fields(args.by, **overrides) if args.evaluation == 'direct' else None
    options = model_view_options(comparison=True)
    options.update(fields=fields,
                   delta=(MODEL_DELTA if args.delta is None else args.delta) if fields is not None else None,
                   initial_case=None if args.initial_by is None else case_label(args.initial_by, args.model or 't96'),
                   color_limits=None if args.color_limit is None else {args.component: args.color_limit},
                   cache_size=2 if args.cache_size is None else args.cache_size)
    if args.slice_normal is None:
        args.slice_normal = 'x'
    if args.slice_origin is None:
        args.slice_origin = (-6., 0., 0.)
    return viz3d.compare_geometry, cases, options


def main(argv=None, *, default_component='alpha', description=None,
         require_source=False, default_by=None):
    """Run the shared CLI; keyword presets support the compatibility launchers."""
    parser = _parser(description or __doc__, default_component)
    args = parser.parse_args(argv)
    if args.by is None and args.model != 't89':
        args.by = default_by
    has_background = bool(args.background or args.background_xmf or args.background_vtk)
    args.component = args.component or ('eta' if has_background else default_component)
    _validate(parser, args, require_source)
    viewer, data, options = _comparison(args) if args.by is not None else _single_snapshot(args)
    if args.no_trace:
        options.update(seeds=[], n_lines=0)
    if args.screenshot:
        import pyvista as pv
        pv.OFF_SCREEN = True
    print(f'Preparing magnetic geometry ({args.component}, {args.evaluation} evaluation).', flush=True)
    plotter = viewer(data, component=args.component,
                     threshold=args.threshold, max_points=args.max_points,
                     geometry_delta=args.geometry_delta,
                     slice_normal=args.slice_normal, slice_origin=args.slice_origin,
                     slice_only=args.slice_only, slice_panel=True, show=False, **options)
    if args.screenshot:
        try:
            plotter.screenshot(args.screenshot)
            print(f'Saved {args.screenshot}')
        finally:
            plotter.close()
    else:
        plotter.show()
