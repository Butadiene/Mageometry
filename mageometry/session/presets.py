"""T96 demonstration data shared by the standalone CLI and capture scripts.

Source recipes and evaluation use the same implementation as the desktop
workspace. Model callables restore their epoch and must be evaluated serially.
"""

import numpy as np

from .sources import load_source, model_field
from .specs import default_analysis, model_source


DEFAULT_BY = (-5., -3., -1., 1., 3., 5.)
DEFAULT_SHAPE = tuple(model_source()['shape'])
MODEL_DELTA = default_analysis(model=True)['geometry_delta']
INNER_RADIUS = default_analysis(model=True)['mask_radius']
EPOCH, PDYN, DST, IMF_BZ = (model_source()['parameters'][key]
                            for key in ('epoch', 'pdyn', 'dst', 'bz'))


def inner_mask(x, y, z):
    """Exclude the unresolved inner magnetosphere in every case."""
    return x*x + y*y + z*z < INNER_RADIUS**2


def case_label(by):
    return f'IMF By = {by:+g} nT'


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
    return {case_label(by): model_field(model_source(by))[0]
            for by in _validate_by(by_values)}


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
    shape = _validate_shape(shape)
    cases = {}
    for by in _validate_by(by_values):
        source = model_source(by)
        source['shape'] = list(shape)
        cases[case_label(by)] = load_source(source, mask_radius=INNER_RADIUS)[0]
    return cases


def model_snapshot(shape=DEFAULT_SHAPE, evaluation='direct', delta=None):
    """Build the single T96 + dipole demonstration and its viewer options.

    Parameters
    ----------
    shape : tuple of int, optional
        Grid node counts in x/y/z, each at least three.
    evaluation : {'direct', 'grid'}, optional
        Use the analytic model or grid interpolation for derivatives/traces.
    delta : float, optional
        Direct-model difference step in Re; defaults to 0.002.

    Returns
    -------
    grid : GriddedField
        By=0 snapshot in GSM Re and nT, with the inner region masked.
    options : dict
        Viewer arguments with units, trace settings and a dipole background.
        Single snapshots retain automatic trace seeds; comparisons fix seeds.
    """
    if evaluation not in ('direct', 'grid'):
        raise ValueError('evaluation must be direct or grid.')
    if delta is not None:
        if evaluation != 'direct' or not np.isfinite(delta) or delta <= 0:
            raise ValueError('delta must be positive and finite and requires direct evaluation.')
    source = model_source()
    source['shape'] = list(_validate_shape(shape))
    grid, field = load_source(source, mask_radius=INNER_RADIUS)
    background, _ = model_field(dict(kind='dipole', parameters=source['parameters']))
    options = model_view_options()
    options.update(field=field if evaluation == 'direct' else None,
                   delta=(MODEL_DELTA if delta is None else delta) if evaluation == 'direct' else None,
                   background_choices={'Dipole': background})
    return grid, options
