"""Declarative, JSON-compatible session recipes."""

from copy import deepcopy
import json
import uuid

import numpy as np

from ..viz3d._current import COMPONENTS, TRANSVERSE_COMPONENTS

SCHEMA_VERSION = 1
SOURCE_KINDS = ('t96', 'dipole', 'xdmf', 'hdf5', 'vtk')


def new_id():
    return uuid.uuid4().hex


def default_view():
    return dict(case=None, component='alpha', contribution='total',
                layout='all', previous_layout='all', panels_hidden=False,
                normal=[1., 0., 0.], origin=[-6., 0., 0.],
                thresholds={}, color_limits={}, cameras={},
                lines=True, arrows=True, regions=True, plane=True)


def default_analysis(model=False):
    return dict(kind='field', evaluation='direct' if model else 'grid',
                delta=0.002 if model else None, geometry_delta=None,
                max_points=120000, cache_size=2, percentile=90.,
                mask_radius=2.5 if model else 0.,
                planet_radius=1. if model else None,
                current_scale=0.125 if model else 1.,
                current_unit='nA/m^2' if model else None,
                length_unit='Re' if model else 'grid unit',
                seeds=[[x, y, 1.] for x in (-6., -10.) for y in (-2., 0., 2.)]
                if model else None, n_lines=10,
                trace=dict(ds=0.15, max_steps=350, r0=2.5) if model else {})


def new_group(label='Snapshots', model=False):
    return dict(id=new_id(), label=label, cases=[], reference=None,
                analysis=default_analysis(model), view=default_view())


def empty_session():
    """Return an empty, portable session recipe."""
    group = new_group()
    group['view']['origin'] = None
    return dict(schema_version=SCHEMA_VERSION, active_group=group['id'], groups=[group])


def model_source(by=0.):
    return dict(kind='t96', parameters=dict(epoch=100., pdyn=2., dst=-20.,
                                           by=float(by), bz=-5.),
                bounds=[[-15., 5.], [-8., 8.], [-8., 8.]], shape=[65, 49, 49])


def model_session(by_values=(0.,)):
    """Create a T96 + dipole recipe with independently identified By cases.

    Parameters
    ----------
    by_values : sequence of float
        IMF By inputs in nT.

    Returns
    -------
    dict
        JSON-compatible session with common numerical and viewing conditions.
    """
    group = new_group('T96 By comparison' if len(by_values) > 1 else 'T96 snapshot', True)
    group['cases'] = [dict(id=new_id(), label=f'IMF By = {by:+g} nT',
                           source=model_source(by), background=None) for by in by_values]
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
    if kind in ('t96', 'dipole'):
        parameters = source.get('parameters', {})
        finite(parameters.get('epoch'), 'epoch')
        if kind == 't96':
            for key in ('pdyn', 'dst', 'by', 'bz'):
                finite(parameters.get(key), key, positive=key == 'pdyn')
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
    if not isinstance(session, dict) or session.get('schema_version') != SCHEMA_VERSION:
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
        if a['kind'] not in ('field', 'attribution') or a['evaluation'] not in ('grid', 'direct'):
            raise ValueError('Invalid analysis kind or evaluation mode.')
        if a['evaluation'] == 'direct':
            if any(c['source']['kind'] != 't96' for c in cases):
                raise ValueError('Direct evaluation requires model sources for every case.')
            steps = a['delta'] if isinstance(a['delta'], list) else [a['delta']]
            if len(steps) not in (1, 3):
                raise ValueError('FAC delta must be a scalar or three steps.')
            for step in steps:
                finite(step, 'FAC delta', positive=True)
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
        if view['layout'] not in ('all', 'three_d', 'slice'):
            raise ValueError('Unknown view layout.')
        vector(view['normal'], 'slice normal')
        normal = np.asarray(view['normal'], dtype=float)
        magnitude = np.max(np.abs(normal))
        if magnitude == 0:
            raise ValueError('Slice normal must be nonzero.')
        normal /= magnitude
        view['normal'] = (normal / np.linalg.norm(normal)).tolist()
        if view['origin'] is not None:
            vector(view['origin'], 'slice origin')
        for values, positive in ((view['thresholds'], False), (view['color_limits'], True)):
            for value in values.values():
                finite(value, 'display limit', minimum=0, positive=positive)
    return result
