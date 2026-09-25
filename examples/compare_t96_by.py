"""Compare six T96 + dipole fields with different IMF By inputs.

Run after an editable install with the viz3d extra:
    python examples/compare_t96_by.py
    python examples/compare_t96_by.py --by -10 -5 0 5 10 --component gamma
    python examples/compare_t96_by.py --initial-by 5 --slice-only --screenshot comparison.png
    python examples/compare_t96_by.py --evaluation grid

DATASET / F7-F8 selects a case; COMPONENT / F5-F6 selects a diagnostic.
Direct model derivatives and tracing are the default (delta=0.002 Re).
Grids define display coordinates; --evaluation grid uses their interpolation.
The viewer receives generic field callables and has no knowledge of this model.
No simulation files or intermediate data files are needed.
"""

import argparse

import numpy as np

from mageometry import GriddedField, geopack, geopack_field, viz3d
from mageometry.viz3d._current import COMPONENTS, _component_name


DEFAULT_BY = (-5., -3., -1., 1., 3., 5.)
DEFAULT_SHAPE = (65, 49, 49)
MODEL_DELTA = 0.002
EPOCH = 100.0  # Unix seconds, matching the single-snapshot model example.
INNER_RADIUS = 2.5
PDYN = 2.
DST = -20.
IMF_BZ = -5.


def inner_mask(x, y, z):
    """Exclude the unresolved inner magnetosphere in every case."""
    return x*x + y*y + z*z < INNER_RADIUS**2


def case_label(by):
    return f'IMF By = {by:+g} nT'


def _t96_field(by):
    """Bind one parameter array and re-establish its epoch on every evaluation.

    The example evaluates models serially: recalc and geopack model globals
    are not safe for concurrent threads, even with independent parameters.
    """
    epoch = EPOCH
    ps = geopack.recalc(epoch)
    parmod = np.array([PDYN, DST, by, IMF_BZ, 0., 0., 0., 0., 0., 0.])
    parmod.setflags(write=False)
    model = geopack_field('t96', 'dip', parmod, ps)

    def field(x, y, z):
        # A different case or unrelated recalc must not change this field.
        geopack.recalc(epoch)
        return model(x, y, z)

    return field


def make_fields(by_values=DEFAULT_BY):
    """Build independent direct T96 + dipole evaluators at the fixed epoch.

    Parameters
    ----------
    by_values : sequence of float, optional
        Distinct IMF By inputs in nT. All other model inputs stay fixed.

    Returns
    -------
    dict of str to callable
        Labelled field functions in GSM Re and nT, for serial evaluation.
    """
    by_values = np.asarray(by_values, dtype=float)
    if by_values.ndim != 1 or not by_values.size or not np.all(np.isfinite(by_values)):
        raise ValueError('by_values must be a nonempty sequence of finite numbers.')
    labels = [case_label(value) for value in by_values]
    if len(set(labels)) != len(labels):
        raise ValueError('IMF By values must have distinct display labels.')
    return {label: _t96_field(by) for label, by in zip(labels, by_values)}


def make_cases(by_values=DEFAULT_BY, shape=DEFAULT_SHAPE):
    """Materialize T96 + dipole snapshots at one epoch on a common GSM grid.

    Parameters
    ----------
    by_values : sequence of float, optional
        Distinct IMF By inputs in nT. All other model inputs stay fixed.
    shape : tuple of int, optional
        Grid node counts in x/y/z, each at least three. Coordinates are Re.

    Returns
    -------
    dict of str to GriddedField
        Independent magnetic arrays in nT, labelled by the IMF By input.
    """
    if len(shape) != 3 or any(not isinstance(n, (int, np.integer)) or
                              isinstance(n, bool) or n < 3 for n in shape):
        raise ValueError('shape must contain three integers >= 3.')
    fields = make_fields(by_values)
    axes = (np.linspace(-15, 5, shape[0]), np.linspace(-8, 8, shape[1]),
            np.linspace(-8, 8, shape[2]))
    coords = np.meshgrid(*axes, indexing='ij')
    valid = ~inner_mask(*coords)
    ps = geopack.recalc(EPOCH)
    cases = {}
    for by, (label, field) in zip(by_values, fields.items()):
        components = [np.full(shape, np.nan) for _ in range(3)]
        for output, sampled in zip(components, field(*(c[valid] for c in coords))):
            output[valid] = sampled
        metadata = dict(coordinate_system='GSM', length_unit='Re', field_unit='nT',
                        model='T96 + dipole', epoch=EPOCH, dipole_tilt=float(ps),
                        pdyn=PDYN, dst=DST, imf_by=float(by), imf_bz=IMF_BZ,
                        parameters={'Pdyn [nPa]': PDYN, 'Dst [nT]': DST,
                                    'IMF By [nT]': float(by), 'IMF Bz [nT]': IMF_BZ,
                                    'Dipole tilt [rad]': float(ps), 'Epoch [Unix s]': EPOCH})
        cases[label] = GriddedField(*axes, *components, metadata=metadata)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--by', type=float, nargs='+', default=DEFAULT_BY,
                         help='IMF By inputs in nT (default: -5 -3 -1 1 3 5)')
    parser.add_argument('--initial-by', type=float, help='initial case, among --by values')
    parser.add_argument('--shape', type=int, nargs=3, default=DEFAULT_SHAPE,
                         metavar=('NX', 'NY', 'NZ'), help='display grid nodes (default: 65 49 49)')
    parser.add_argument('--evaluation', choices=('direct', 'grid'), default='direct',
                         help='magnetic derivative/tracing source (default: direct)')
    parser.add_argument('--delta', type=float,
                         help='direct-model Cartesian FAC step in Re; also the default geometry step (default: 0.002)')
    parser.add_argument('--component', type=_component_name, choices=tuple(COMPONENTS),
                         default='alpha', help='diagnostic to display')
    parser.add_argument('--threshold', type=float, help='shared absolute cutoff for the initial diagnostic')
    parser.add_argument('--color-limit', type=float, help='shared symmetric colour limit for the initial diagnostic')
    parser.add_argument('--geometry-delta', type=float,
                         help='geometry step in Re (default: direct --delta, or grid preview spacing)')
    parser.add_argument('--max-points', type=int, default=120000, help='preview node budget per case')
    parser.add_argument('--cache-size', type=int, default=2, help='retained case previews (default: 2)')
    parser.add_argument('--slice', choices=('x', 'y', 'z'), default='x', dest='slice_normal')
    parser.add_argument('--slice-origin', type=float, nargs=3, default=(-6., 0., 0.))
    parser.add_argument('--slice-only', action='store_true', help='start with the enlarged slice')
    parser.add_argument('--screenshot', help='save an off-screen PNG instead of opening a window')
    args = parser.parse_args()
    if args.initial_by is not None and args.initial_by not in args.by:
        parser.error('--initial-by must be one of the --by values')
    if args.evaluation == 'grid' and args.delta is not None:
        parser.error('--delta requires --evaluation direct; grid FAC uses axis spacing')
    for name in ('delta', 'geometry_delta'):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error(f'--{name.replace("_", "-")} must be positive and finite')
    print(f'Sampling T96 + dipole: Pdyn={PDYN:g} nPa, Dst={DST:g} nT, IMF Bz={IMF_BZ:g} nT, '
          f'epoch={EPOCH:g} Unix seconds; grid {tuple(args.shape)} in Re.', flush=True)
    try:
        cases = make_cases(args.by, tuple(args.shape))
        fields = make_fields(args.by) if args.evaluation == 'direct' else None
    except ValueError as error:
        parser.error(str(error))
    if args.screenshot:
        import pyvista as pv
        pv.OFF_SCREEN = True
    delta = (MODEL_DELTA if args.delta is None else args.delta) if fields is not None else None
    step = args.geometry_delta if args.geometry_delta is not None else delta
    print(f'Evaluation: {args.evaluation}; geometry step: '
          f'{step if step is not None else "preview spacing"} Re.', flush=True)
    print(f'Preparing {args.component} with a shared scale over {len(cases)} cases. '
          'Dataset: F7/F8; diagnostic: F5/F6.', flush=True)
    # Seeds stay fixed in physical coordinates across both case and diagnostic
    # selection. Tracing uses the selected direct or interpolated field.
    seeds = np.array([[x, y, 1.] for x in (-6., -10.) for y in (-2., 0., 2.)])
    limits = None if args.color_limit is None else {args.component: args.color_limit}
    plotter = viz3d.compare_geometry(
        cases, component=args.component, fields=fields, delta=delta,
        initial_case=None if args.initial_by is None else case_label(args.initial_by),
        color_limits=limits, cache_size=args.cache_size, threshold=args.threshold,
        geometry_delta=args.geometry_delta, max_points=args.max_points,
        mask=inner_mask, seeds=seeds, planet_radius=1., length_unit='Re',
        current_scale=0.125, current_unit='nA/m^2',
        trace_kwargs=dict(r0=INNER_RADIUS, ds=0.15, max_steps=350),
        slice_normal=args.slice_normal, slice_origin=args.slice_origin,
        slice_panel=True, slice_only=args.slice_only, show=False)
    if args.screenshot:
        plotter.screenshot(args.screenshot)
        plotter.close()
        print(f'Saved {args.screenshot}')
    else:
        plotter.show()


if __name__ == '__main__':
    main()
