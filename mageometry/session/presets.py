"""Model demonstration data shared by the standalone CLI and capture scripts.

Source recipes and evaluation use the same implementation as the desktop
workspace. Model callables restore their epoch and must be evaluated serially.
"""

import numpy as np

from .sources import load_source, model_field
from .specs import default_analysis, model_case_label, model_session, model_source, validate_source


DEFAULT_BY = (-10., -5., 0., 5., 10.)
DEFAULT_SHAPE = tuple(model_source()['shape'])
MODEL_DELTA = default_analysis(model=True)['geometry_delta']
INNER_RADIUS = default_analysis(model=True)['mask_radius']
EPOCH, PDYN = (model_source()['parameters'][key] for key in ('epoch', 'pdyn'))
DST = -30.
IMF_BZ = -10.


def comparison_session(by_values=DEFAULT_BY, *, model='t96', parameters=None):
    """Create a desktop recipe with the shared IMF By comparison conditions.

    Parameters
    ----------
    by_values : sequence of float, optional
        IMF By inputs in nT; defaults to -10, -5, 0, 5, 10.
    model : {'t96', 't01', 't04'}, optional
        External model, default T96; T89 does not support IMF By scans.
    parameters : dict, optional
        Shared inputs overriding the comparison's Dst=-30 and IMF Bz=-10 nT.
        Specify By with by_values instead.

    Returns
    -------
    dict
        Session recipe with the same sources as make_cases and make_fields.
    """
    if model == 't89':
        raise ValueError('T89 does not support IMF By comparisons.')
    values = _validate_by(by_values)
    inputs = dict(dst=DST, bz=IMF_BZ)
    inputs.update(parameters or {})
    return model_session(values.tolist(), model=model, parameters=inputs)


def inner_mask(x, y, z):
    """Exclude the unresolved inner magnetosphere in every case."""
    return x*x + y*y + z*z < INNER_RADIUS**2


def case_label(by, model='t96'):
    return model_case_label(model_source(by, model=model))


def _validate_by(by_values):
    values = np.asarray(by_values, dtype=float)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError('by_values must be a nonempty sequence of finite numbers.')
    labels = [case_label(value) for value in values]
    if len(set(labels)) != len(labels):
        raise ValueError('IMF By values must have distinct display labels.')
    return values


def _validate_shape(shape):
    if len(shape) != 3 or any(not isinstance(n, (int, np.integer)) or
                              isinstance(n, bool) or n < 3 for n in shape):
        raise ValueError('shape must contain three integers >= 3.')
    return tuple(int(n) for n in shape)


def model_view_options(comparison=False):
    """Return shared model units, mask and tracing options for PyVista."""
    analysis = default_analysis(model=True)
    options = {key: analysis[key] for key in
               ('planet_radius', 'length_unit', 'current_scale', 'current_unit')}
    options.update(mask=inner_mask, trace_kwargs=analysis['trace'])
    if comparison:
        options['seeds'] = np.asarray(analysis['seeds'])
    return options


def _comparison_source(by, dst, bz, model, parameters):
    if parameters and 'by' in parameters:
        raise ValueError('Use by_values to specify IMF By.')
    source = model_source(by, model=model)
    if model != 't89':
        source['parameters'].update(dst=dst, bz=bz)
    source['parameters'].update(parameters or {})
    validate_source(source)
    return source


def make_fields(by_values=DEFAULT_BY, *, dst=DST, bz=IMF_BZ, model='t96', parameters=None):
    """Build independent direct model + dipole evaluators.

    Parameters
    ----------
    by_values : sequence of float, optional
        Distinct IMF By inputs in nT.
    dst, bz : float, optional
        Shared Dst and IMF Bz inputs in nT; defaults are -30 and -10.
    model : {'t96', 't01', 't04'}, optional
        External model, default T96; T89 does not support IMF By scans.
    parameters : dict, optional
        Shared model inputs, excluding By. Override defaults including dst/bz.

    Returns
    -------
    dict of str to callable
        Labelled field functions in GSM Re and nT, for serial evaluation.
    """
    return {case_label(by, model): model_field(_comparison_source(by, dst, bz, model, parameters))[0]
            for by in _validate_by(by_values)}


def make_cases(by_values=DEFAULT_BY, shape=DEFAULT_SHAPE, *, dst=DST, bz=IMF_BZ,
               model='t96', parameters=None):
    """Materialize model + dipole snapshots on a common GSM grid.

    Parameters
    ----------
    by_values : sequence of float, optional
        Distinct IMF By inputs in nT.
    shape : tuple of int, optional
        Grid node counts in x/y/z, each at least three. Coordinates are Re.
    dst, bz : float, optional
        Shared Dst and IMF Bz inputs in nT; defaults are -30 and -10.
    model : {'t96', 't01', 't04'}, optional
        External model, default T96; T89 does not support IMF By scans.
    parameters : dict, optional
        Shared model inputs, excluding By. Override defaults including dst/bz.

    Returns
    -------
    dict of str to GriddedField
        Independent magnetic arrays in nT, labelled by the IMF By input.
    """
    shape = _validate_shape(shape)
    cases = {}
    for by in _validate_by(by_values):
        source = _comparison_source(by, dst, bz, model, parameters)
        source['shape'] = list(shape)
        cases[case_label(by, model)] = load_source(source, mask_radius=INNER_RADIUS)[0]
    return cases


def model_snapshot(shape=DEFAULT_SHAPE, evaluation='direct', delta=None, *, model='t96', parameters=None):
    """Build a single model + dipole demonstration and its viewer options.

    Parameters
    ----------
    shape : tuple of int, optional
        Grid node counts in x/y/z, each at least three.
    evaluation : {'direct', 'grid'}, optional
        Use the analytic model or grid interpolation for derivatives/traces.
    delta : float, optional
        Direct-model difference step in Re; defaults to 0.002.
    model : {'t89', 't96', 't01', 't04'}, optional
        External field model. Default T96.
    parameters : dict, optional
        Model inputs overriding the demonstration defaults.

    Returns
    -------
    grid : GriddedField
        Snapshot in GSM Re and nT, with the inner region masked.
    options : dict
        Viewer arguments with units, trace settings and a dipole background.
        Single snapshots retain automatic trace seeds; comparisons fix seeds.
    """
    if evaluation not in ('direct', 'grid'):
        raise ValueError('evaluation must be direct or grid.')
    if delta is not None:
        if evaluation != 'direct' or not np.isfinite(delta) or delta <= 0:
            raise ValueError('delta must be positive and finite and requires direct evaluation.')
    source = model_source(model=model, parameters=parameters)
    source['shape'] = list(_validate_shape(shape))
    grid, field = load_source(source, mask_radius=INNER_RADIUS)
    background, _ = model_field(dict(kind='dipole', parameters=source['parameters']))
    options = model_view_options()
    options.update(field=field if evaluation == 'direct' else None,
                   delta=(MODEL_DELTA if delta is None else delta) if evaluation == 'direct' else None,
                   background_choices={'Dipole': background})
    return grid, options
