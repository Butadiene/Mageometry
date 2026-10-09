"""Declarative, JSON-compatible session recipes."""

from copy import deepcopy
import json
import uuid

import numpy as np

from ..viz3d._current import COMPONENTS, TRANSVERSE_COMPONENTS
from ..geometry.line_profiles import PROFILE_QUANTITIES
from .profiles import default_profile_view

SCHEMA_VERSION = 2
DEFAULT_GEOMETRY_DELTA = 0.002
MODEL_KINDS = ('t89', 't96', 't01', 't04')
MODEL_PARAMETERS = {
    't89': ('epoch', 'iopt'),
    't96': ('epoch', 'pdyn', 'dst', 'by', 'bz'),
    't01': ('epoch', 'pdyn', 'dst', 'by', 'bz', 'g1', 'g2'),
    't04': ('epoch', 'pdyn', 'dst', 'by', 'bz', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6'),
}
MODEL_DEFAULTS = dict(epoch=100., iopt=2, pdyn=2., dst=-20., by=0., bz=-5.,
                      g1=0., g2=0., w1=0., w2=0., w3=0., w4=0., w5=0., w6=0.)
SOURCE_KINDS = MODEL_KINDS + ('dipole', 'xdmf', 'hdf5', 'vtk')
VIEW_LAYOUTS = ('all', 'three_d_slice', 'three_d', 'slice')


def new_id():
    return uuid.uuid4().hex


def default_view():
    return dict(case=None, component='alpha', contribution='total',
                layout='three_d_slice', previous_layout='three_d_slice', panels_hidden=False,
                normal=[1., 0., 0.], origin=[-6., 0., 0.],
                thresholds={}, threshold_slider_limits={}, threshold_modes={}, value_intervals={},
                color_limits={}, slice_color_ranges={}, slice_extent=None, cameras={},
                lines=True, arrows=True, regions=True, plane=True, value_sign='both',
                gamma_eta=False, profile=default_profile_view())


def default_analysis(model=False):
    return dict(kind='field', evaluation='direct' if model else 'grid',
                delta=None, geometry_delta=DEFAULT_GEOMETRY_DELTA if model else None,
                max_points=120000, cache_size=2, percentile=90.,
                mask_radius=2.5 if model else 0.,
                planet_radius=1. if model else None,
                current_scale=0.125 if model else 1.,
                current_unit='nA/m^2' if model else None,
                length_unit='Re' if model else 'grid unit',
                seeds=[[x, y, 1.] for x in (-6., -10.) for y in (-2., 0., 2.)]
                if model else None, n_lines=10, trace_enabled=True,
                trace=dict(ds=0.15, max_steps=350, r0=2.5) if model else {})


def new_group(label='Snapshots', model=False):
    return dict(id=new_id(), label=label, cases=[], reference=None,
                analysis=default_analysis(model), view=default_view())


def empty_session():
    """Return an empty, portable session recipe."""
    group = new_group()
    group['view']['origin'] = None
    return dict(schema_version=SCHEMA_VERSION, active_group=group['id'], groups=[group])


def model_source(by=0., *, model='t96', parameters=None):
    if model not in MODEL_KINDS:
        raise ValueError(f'Unsupported model: {model!r}.')
    values = {key: MODEL_DEFAULTS[key] for key in MODEL_PARAMETERS[model]}
    if model != 't89':
        values['by'] = float(by)
    elif by != 0.:
        raise ValueError('T89 uses iopt, not IMF By.')
    values.update(parameters or {})
    source = dict(kind=model, parameters=values,
                  bounds=[[-15., 5.], [-8., 8.], [-8., 8.]], shape=[65, 49, 49])
    validate_source(source)
    return source


def model_case_label(source):
    kind, parameters = source['kind'], source['parameters']
    if kind == 't89':
        return f"T89 iopt = {parameters['iopt']}"
    prefix = '' if kind == 't96' else kind.upper() + ': '
    return prefix + f"IMF By = {parameters['by']:+g} nT"


def model_session(by_values=(0.,), *, model='t96', parameters=None):
    """Create model + dipole recipes with independently identified cases.

    Parameters
    ----------
    by_values : sequence of float
        IMF By inputs in nT. T89 accepts only the default single zero entry.
    model : {'t89', 't96', 't01', 't04'}, optional
        External field model. Default T96.
    parameters : dict, optional
        Shared model inputs, excluding By (use by_values). Defaults include
        iopt=2 for T89, G1=G2=0 for T01, and W1 through W6=0 for T04.
        The zero driving indices are demonstration inputs, not inferred history.

    Returns
    -------
    dict
        JSON-compatible session with common numerical and viewing conditions.
    """
    if parameters and 'by' in parameters:
        raise ValueError('Use by_values to specify IMF By.')
    if model == 't89' and tuple(by_values) != (0.,):
        raise ValueError('T89 uses iopt, not an IMF By comparison.')
    group = new_group(model.upper() + (' By comparison' if len(by_values) > 1 else ' snapshot'), True)
    for by in by_values:
        source = model_source(by, model=model, parameters=parameters)
        group['cases'].append(dict(id=new_id(), label=model_case_label(source),
                                    source=source, background=None))
    if group['cases']:
        group['reference'] = group['view']['case'] = group['cases'][0]['id']
    session = dict(schema_version=SCHEMA_VERSION, active_group=group['id'], groups=[group])
    return validate_session(session)


def finite(value, name, minimum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f'{name} must be a finite number.')
    if (minimum is not None and value < minimum) or (positive and value <= 0):
        raise ValueError(f'Invalid {name}: {value}.')


def vector(value, name, length=3):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f'{name} must contain {length} numbers.')
    for number in value:
        finite(number, name)


def validate_source(source):
    if not isinstance(source, dict) or source.get('kind') not in SOURCE_KINDS:
        raise ValueError('Unsupported source kind.')
    kind = source['kind']
    if kind in MODEL_KINDS or kind == 'dipole':
        parameters = source.get('parameters', {})
        finite(parameters.get('epoch'), 'epoch')
        if kind in MODEL_KINDS:
            if set(parameters) - set(MODEL_PARAMETERS[kind]):
                raise ValueError(f'Unsupported {kind.upper()} parameters: '
                                 f'{set(parameters) - set(MODEL_PARAMETERS[kind])}.')
            for key in MODEL_PARAMETERS[kind]:
                finite(parameters.get(key), key, positive=key == 'pdyn',
                       minimum=0 if key.startswith(('g', 'w')) else None)
            if kind == 't89' and (type(parameters['iopt']) is not int or not 1 <= parameters['iopt'] <= 7):
                raise ValueError('T89 iopt must be an integer from 1 to 7.')
            shape = source.get('shape', [])
            if len(shape) != 3 or any(type(n) is not int or n < 3 for n in shape):
                raise ValueError('Model shape must contain three integers >= 3.')
            bounds = source.get('bounds', [])
            if len(bounds) != 3:
                raise ValueError('Model bounds require three axis intervals.')
            for interval in bounds:
                vector(interval, 'bounds', 2)
                if interval[0] >= interval[1]:
                    raise ValueError('Bounds must increase.')
    else:
        if not isinstance(source.get('path'), str) or not source['path'].strip():
            raise ValueError('A file source requires a path.')
        options = source.get('options', {})
        allowed = {'stride', 'region'}
        allowed.update({'vtk': {'name'}, 'xdmf': {'components', 'h5_file'},
                        'hdf5': {'datasets', 'origin', 'spacing', 'zyx_order'}}[kind])
        if set(options) - allowed:
            raise ValueError(f'Unsupported reader options: {set(options) - allowed}.')
        stride = options.get('stride', 1)
        if type(stride) is not int or stride < 1:
            raise ValueError('Reader stride must be a positive integer.')
        if kind == 'hdf5':
            for name in ('origin', 'spacing'):
                vector(options.get(name), name)
            if any(s <= 0 for s in options['spacing']):
                raise ValueError('HDF5 spacing must be positive.')


def validate_session(session):
    """Validate and copy a versioned recipe without reading files or rendering."""
    if not isinstance(session, dict) or session.get('schema_version') not in (1, SCHEMA_VERSION):
        raise ValueError('Unsupported session schema version.')
    # Reject nonportable objects and NaN/Infinity before passing recipes to workers.
    json.dumps(session, allow_nan=False)
    result = deepcopy(session)
    groups = result.get('groups')
    if not isinstance(groups, list) or not groups:
        raise ValueError('A session requires at least one group.')
    ids = [g['id'] for g in groups]
    if len(set(ids)) != len(ids) or result.get('active_group') not in ids:
        raise ValueError('Group IDs must be distinct and the active group must exist.')
    for group in groups:
        cases = group['cases']
        case_ids = [case['id'] for case in cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError('Case IDs must be distinct within a group.')
        view = group['view']
        if cases and (group['reference'] not in case_ids or view['case'] not in case_ids):
            raise ValueError('Reference and selected case must belong to the group.')
        for case in cases:
            if not isinstance(case['label'], str) or not case['label'].strip():
                raise ValueError('Case labels must be nonempty.')
            validate_source(case['source'])
            if case.get('background') is not None:
                validate_source(case['background'])
        a = group['analysis']
        if type(a.setdefault('trace_enabled', True)) is not bool:
            raise ValueError('Trace enabled must be a boolean.')
        if a['kind'] not in ('field', 'attribution') or a['evaluation'] not in ('grid', 'direct'):
            raise ValueError('Invalid analysis kind or evaluation mode.')
        if a['evaluation'] == 'direct':
            if any(c['source']['kind'] not in MODEL_KINDS for c in cases):
                raise ValueError('Direct evaluation requires model sources for every case.')
            if a['delta'] is not None or result['schema_version'] == 1:
                steps = a['delta'] if isinstance(a['delta'], list) else [a['delta']]
                if len(steps) not in (1, 3):
                    raise ValueError('FAC delta must be a scalar or three steps.')
                for step in steps:
                    finite(step, 'FAC delta', positive=True)
                # Version 1 inherited geometry from FAC. Freeze that old
                # geometry step before reversing the inheritance direction.
                if result['schema_version'] == 1 and a['geometry_delta'] is None:
                    a['geometry_delta'] = min(steps)
        elif a['delta'] is not None:
            raise ValueError('Grid FAC uses axis spacing; set delta to null.')
        if a['geometry_delta'] is not None:
            finite(a['geometry_delta'], 'geometry delta', positive=True)
        for key, lower in (('max_points', 27), ('cache_size', 1), ('n_lines', 0)):
            if type(a[key]) is not int or a[key] < lower:
                raise ValueError(f'{key} must be an integer >= {lower}.')
        finite(a['mask_radius'], 'mask radius', minimum=0)
        finite(a['current_scale'], 'current scale', positive=True)
        finite(a['percentile'], 'percentile', minimum=0)
        if a['percentile'] > 100:
            raise ValueError('Percentile must be <= 100.')
        if a['planet_radius'] is not None:
            finite(a['planet_radius'], 'planet radius', positive=True)
        if a['seeds'] is not None:
            for seed in a['seeds']:
                vector(seed, 'seed')
        if set(a['trace']) - {'direction', 'ds', 'err', 'r0', 'rlim', 'bounds', 'max_steps'}:
            raise ValueError('Unsupported trace option.')
        for key in ('ds', 'err', 'r0', 'rlim'):
            if key in a['trace'] and a['trace'][key] is not None:
                finite(a['trace'][key], 'trace ' + key, positive=True)
        if 'max_steps' in a['trace'] and (type(a['trace']['max_steps']) is not int or a['trace']['max_steps'] < 1):
            raise ValueError('Trace max_steps must be a positive integer.')
        if a['trace'].get('direction', 'both') not in ('both', 1, -1):
            raise ValueError('Invalid trace direction.')
        if view['component'] not in COMPONENTS:
            raise ValueError('Unknown diagnostic.')
        if a['kind'] == 'attribution':
            if any(c.get('background') is None for c in cases):
                raise ValueError('Assign a background to every case before attribution.')
            if view['component'] not in TRANSVERSE_COMPONENTS:
                raise ValueError('Attribution requires a transverse diagnostic.')
        elif view['contribution'] != 'total':
            raise ValueError('Field analysis uses the total contribution.')
        if view['contribution'] not in ('total', 'background', 'residual'):
            raise ValueError('Unknown contribution.')
        if view['layout'] not in VIEW_LAYOUTS:
            raise ValueError('Unknown view layout.')
        if view.setdefault('value_sign', 'both') not in ('both', 'positive', 'negative'):
            raise ValueError('Value sign must be both, positive or negative.')
        if not isinstance(view.setdefault('gamma_eta', False), bool):
            raise ValueError('Gamma colouring by eta must be a boolean.')
        if view.get('previous_layout', 'all') not in VIEW_LAYOUTS[:-1]:
            raise ValueError('Previous layout must be a non-slice layout.')
        vector(view['normal'], 'slice normal')
        normal = np.asarray(view['normal'], dtype=float)
        magnitude = np.max(np.abs(normal))
        if magnitude == 0:
            raise ValueError('Slice normal must be nonzero.')
        normal /= magnitude
        view['normal'] = (normal / np.linalg.norm(normal)).tolist()
        if view['origin'] is not None:
            vector(view['origin'], 'slice origin')
        profile = view.setdefault('profile', default_profile_view())
        if not isinstance(profile, dict):
            raise ValueError('Profile settings must be a mapping.')
        for name, value in default_profile_view().items():
            profile.setdefault(name, value)
        if type(profile['visible']) is not bool:
            raise ValueError('Profile visibility must be a boolean.')
        if profile['seed_id'] is not None and not isinstance(profile['seed_id'], str):
            raise ValueError('Profile seed ID must be a string or null.')
        names = profile['quantities']
        if (not isinstance(names, list) or len(names) > 4 or
                any(not isinstance(name, str) or name not in PROFILE_QUANTITIES for name in names) or
                len(set(names)) != len(names)):
            raise ValueError('Profile quantities must contain up to four distinct diagnostic names.')
        finite(profile['cursor_s'], 'profile cursor')
        if not isinstance(profile['ylims'], dict) or any(name not in PROFILE_QUANTITIES for name in profile['ylims']):
            raise ValueError('Profile y ranges must use known diagnostic names.')
        ranges = list(profile['ylims'].values())
        if profile['xlim'] is not None:
            ranges.append(profile['xlim'])
        for bounds in ranges:
            vector(bounds, 'profile range', 2)
            if bounds[0] >= bounds[1]:
                raise ValueError('Profile range requires lower < upper.')
        slider_limits = view.setdefault('threshold_slider_limits', {})
        if not isinstance(slider_limits, dict):
            raise ValueError('Threshold slider limits must be a mapping.')
        for value in slider_limits.values():
            finite(value, 'threshold slider upper bound', positive=True)
        modes = view.setdefault('threshold_modes', {})
        intervals = view.setdefault('value_intervals', {})
        if not isinstance(modes, dict) or not isinstance(intervals, dict):
            raise ValueError('Threshold modes and value intervals must be mappings.')
        for key, mode in modes.items():
            if mode not in ('absolute', 'interval') or (mode == 'interval' and key not in intervals):
                raise ValueError('An interval filter requires a valid value interval.')
        for interval in intervals.values():
            vector(interval, 'value interval', 2)
            if interval[0] >= interval[1]:
                raise ValueError('Value interval requires lower < upper.')
        slice_ranges = view.setdefault('slice_color_ranges', {})
        if not isinstance(slice_ranges, dict):
            raise ValueError('Slice colour ranges must be a mapping.')
        for bounds in slice_ranges.values():
            vector(bounds, 'slice colour range', 2)
            if bounds[0] >= bounds[1]:
                raise ValueError('Slice colour range requires lower < upper.')
        extent = view.setdefault('slice_extent', None)
        if extent is not None:
            vector(extent, 'slice extent', 4)
            if extent[0] >= extent[1] or extent[2] >= extent[3]:
                raise ValueError('Slice extent requires horizontal/vertical min < max.')
        for values, positive in ((view['thresholds'], False), (view['color_limits'], True)):
            for value in values.values():
                finite(value, 'display limit', minimum=0, positive=positive)
    result['schema_version'] = SCHEMA_VERSION
    return result
