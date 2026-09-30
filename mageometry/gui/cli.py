"""Build desktop session recipes from model, file and display arguments."""

from copy import deepcopy
from pathlib import Path

from ..session import empty_session, load_session, model_session, validate_session
from ..session.cli import viewer_parser, validate_viewer_options
from ..session.specs import new_id


def _file_source(kind, path, stride, **options):
    return dict(kind=kind, path=str(Path(path).resolve()),
                options=dict(stride=stride, **options))


def session_from_args(argv=None, *, default_component='alpha', description=None,
                      require_source=False, default_by=None):
    """Parse and validate a recipe without reading source arrays or importing Qt."""
    parser = viewer_parser(description or 'Unified magnetic geometry workspace',
                           default_component, desktop=True)
    args = parser.parse_args(argv)
    if args.session or args.empty:
        # Saved settings remain authoritative; only the layout may be overridden.
        allowed = {'session', 'empty', 'layout', 'slice_only', 'screenshot'}
        if any(value != parser.get_default(key) for key, value in vars(args).items()
               if key not in allowed):
            parser.error('--session and --empty only accept layout and screenshot options')
        if args.empty and require_source:
            parser.error('specify a snapshot file with --xmf, --vtk, or --h5, or a --session')
        session = load_session(args.session) if args.session else empty_session()
        group = next(g for g in session['groups'] if g['id'] == session['active_group'])
    else:
        has_file = bool(args.xmf or args.vtk or args.h5)
        if args.by is None and not has_file:
            args.by = default_by
        has_background = bool(args.background or args.background_xmf or args.background_vtk)
        args.component = args.component or ('eta' if has_background else default_component)
        validate_viewer_options(parser, args, require_source, desktop=True)
        session = empty_session() if has_file else model_session(args.by or (0.,))
        group = session['groups'][0]
        if has_file:
            if args.xmf:
                options = {'h5_file': str(Path(args.h5).resolve())} if args.h5 else {}
                source = _file_source('xdmf', args.xmf, args.stride, **options)
            elif args.vtk:
                source = _file_source('vtk', args.vtk, args.stride)
            else:
                source = _file_source('hdf5', args.h5, args.stride,
                                      origin=args.origin, spacing=args.spacing)
            case = dict(id=new_id(), label=Path(source['path']).name,
                        source=source, background=None)
            group['cases'] = [case]
            group['reference'] = group['view']['case'] = case['id']
        for case in group['cases']:
            if args.shape is not None:
                case['source']['shape'] = args.shape
            if args.background:
                case['background'] = dict(kind='dipole', parameters={
                    'epoch': case['source']['parameters']['epoch']})
            elif args.background_xmf:
                case['background'] = _file_source('xdmf', args.background_xmf, args.stride)
            elif args.background_vtk:
                case['background'] = _file_source('vtk', args.background_vtk, args.stride)
            if args.initial_by is not None and case['source']['parameters']['by'] == args.initial_by:
                group['view']['case'] = case['id']
        analysis = group['analysis']
        analysis.update(kind='attribution' if has_background else 'field',
                        evaluation=args.evaluation, max_points=args.max_points)
        if args.evaluation == 'grid':
            analysis['delta'] = None
            analysis['geometry_delta'] = None
        elif args.delta is not None:
            analysis['delta'] = args.delta
        for key in ('geometry_delta', 'cache_size'):
            if getattr(args, key) is not None:
                analysis[key] = getattr(args, key)
        view = group['view']
        view.update(component=args.component, contribution=args.contribution or 'total')
        if args.slice_normal:
            view['normal'] = [float(axis == args.slice_normal) for axis in 'xyz']
        if args.slice_origin is not None:
            view['origin'] = args.slice_origin
        key = analysis['kind'] + ':' + args.component
        if args.threshold is not None:
            view['thresholds'][key] = args.threshold
        if args.color_limit is not None:
            view['color_limits'][key] = args.color_limit
    layout = 'slice' if args.slice_only else args.layout
    if layout:
        group['view']['layout'] = layout
        if layout != 'slice':
            group['view']['previous_layout'] = layout
    if args.screenshot and not group['cases']:
        parser.error('a screenshot requires at least one model or file case')
    try:
        return validate_session(session), args.screenshot
    except ValueError as error:
        parser.error(str(error))


def export_screenshot(session, path):
    """Render the desktop plot layout and its matching recipe without Qt."""
    import pyvista as pv

    from ..session import SessionEngine, save_session
    from ..viz3d.scene import GeometryScene

    session = deepcopy(session)
    group = next(g for g in session['groups'] if g['id'] == session['active_group'])
    result = SessionEngine(group).prepare(group['view'])
    plotter = pv.Plotter(shape='1|4', off_screen=True, border=False,
                         window_size=(1400, 900))
    try:
        scene = GeometryScene(plotter)
        scene.set_result(result, group['view'])
        scene.screenshot(path)
        group['resolved'] = result['resolved']
        if group['analysis']['seeds'] is None:
            group['analysis']['seeds'] = result['resolved']['seeds']
        group['view']['cameras'] = scene.camera_state()
        save_session(session, Path(path).with_suffix('.session.json'))
    finally:
        plotter.close()


def main(argv=None, **presets):
    session, screenshot = session_from_args(argv, **presets)
    if screenshot:
        return export_screenshot(session, screenshot)
    from .app import run
    return run(session)
