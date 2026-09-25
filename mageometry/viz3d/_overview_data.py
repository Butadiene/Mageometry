"""Case preparation, bounded diagnostic caches, and shared comparison scales."""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..io import GriddedField
from ._current import TRANSVERSE_COMPONENTS, _CurrentPreview, _component_name
from ._preview import _masked_field, _preview_indices, _sample_fac


@dataclass(frozen=True)
class _Scale:
    limit: float
    peak: float
    threshold: float


@dataclass(frozen=True)
class _Selection:
    case: str
    component: str
    prepared: object
    values: np.ndarray
    basis: object
    scale: _Scale


class _PreparedCase:
    def __init__(self, grid, field, delta, max_points, mask, geometry_delta):
        self.preview, fac = _sample_fac(grid, field, delta, max_points, mask)
        self.spacing = min(float(np.min(np.diff(a))) for a in
                           (self.preview.x, self.preview.y, self.preview.z))
        step = geometry_delta
        if step is None:
            step = float(np.min(delta)) if field is not None and delta is not None else self.spacing
        self.field = _masked_field(self.preview.field() if field is None else field, mask)
        self.cache = _CurrentPreview(self.preview, fac, self.field, step)


def _validate_cases(cases):
    if not isinstance(cases, Mapping) or not cases:
        raise ValueError('cases must be a nonempty mapping of labels to GriddedField objects.')
    cases = dict(cases)
    for label, grid in cases.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError('Case labels must be nonempty strings.')
        if not isinstance(grid, GriddedField):
            raise TypeError(f'Case {label!r} must be a GriddedField.')
    reference = next(iter(cases.values()))
    for label, grid in cases.items():
        if not all(np.array_equal(a, b) for a, b in zip(
                (reference.x, reference.y, reference.z), (grid.x, grid.y, grid.z))):
            raise ValueError(f'Case {label!r} has different axes; resample explicitly before comparison.')
    for key in ('coordinate_system', 'length_unit', 'field_unit'):
        declarations = [g.metadata[key] for g in cases.values() if key in g.metadata]
        if declarations and any(value != declarations[0] for value in declarations[1:]):
            raise ValueError(f'Conflicting {key} metadata across cases.')
    return cases


def _validate_fields(cases, fields, delta):
    if fields is None:
        if delta is not None:
            raise ValueError('delta requires per-case fields; grid FAC uses axis spacing.')
        return None
    if not isinstance(fields, Mapping) or set(fields) != set(cases):
        raise ValueError('fields must be a mapping with exactly the same labels as cases.')
    fields = dict(fields)
    if not all(callable(field) for field in fields.values()):
        raise ValueError('Every entry in fields must be callable.')
    if delta is None:
        raise ValueError('Direct field comparison requires an explicit shared delta.')
    steps = np.broadcast_to(np.asarray(delta, dtype=float), (3,))
    if not np.all(np.isfinite(steps) & (steps > 0)):
        raise ValueError('delta must contain positive finite step sizes.')
    return fields


class _OverviewData:
    """Own numerical state; scene changes commit a prepared selection explicitly.

    Source grids stay owned by the caller and must not be mutated while the
    viewer is open. Only derived case previews are managed by this LRU cache.
    """

    def __init__(self, cases, *, comparison=False, field=None, fields=None, delta=None,
                 max_points=120000, mask=None, geometry_delta=None,
                 current_scale=1.0, percentile=90.0, color_limits=None,
                 cache_size=2):
        self.cases = _validate_cases(cases)
        self.labels = tuple(self.cases)
        self.reference = self.labels[0]
        self.comparison = comparison
        if fields is not None and field is not None:
            raise ValueError('Use either field or per-case fields, not both.')
        self.fields = (_validate_fields(self.cases, fields, delta)
                       if comparison or fields is not None else None)
        if not isinstance(cache_size, (int, np.integer)) or isinstance(cache_size, bool) or cache_size < 1:
            raise ValueError('cache_size must be a positive integer.')
        self.cache_size = int(cache_size)
        self.options = dict(field=field, delta=delta, max_points=max_points,
                            mask=mask, geometry_delta=geometry_delta)
        _preview_indices(next(iter(self.cases.values())).shape, max_points)
        self.current_scale = current_scale
        self.percentile = percentile
        if color_limits is not None and not isinstance(color_limits, Mapping):
            raise ValueError('color_limits must be a mapping of diagnostics to limits.')
        self.limits = {}
        for key, limit in (color_limits or {}).items():
            key = _component_name(key)
            if not np.isscalar(limit) or not np.isfinite(limit) or limit <= 0:
                raise ValueError('color_limits must map diagnostics to positive finite limits.')
            self.limits[key] = limit
        self.cache = OrderedDict()
        self.scales = {}
        self.selection = None
        self.thresholds = {}

    def get(self, case):
        if case not in self.cases:
            raise ValueError(f'Unknown case {case!r}.')
        if case not in self.cache:
            options = dict(self.options)
            if self.fields is not None:
                options['field'] = self.fields[case]
            prepared = _PreparedCase(self.cases[case], **options)
            self.cache[case] = prepared
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(case)
        return self.cache[case]

    def values(self, prepared, component):
        component = _component_name(component)
        values, basis = prepared.cache.get(component)
        scale = 1.0 if component in TRANSVERSE_COMPONENTS else self.current_scale
        return values * scale, basis

    def statistics(self, component, progress=None):
        component = _component_name(component)
        if component in self.scales:
            return self.scales[component]
        limit, peak, threshold = 0.0, 0.0, 0.0
        labels = self.labels if self.comparison else (self.reference,)
        for index, case in enumerate(labels):
            if progress is not None:
                progress(f'Preparing {component}: {index + 1}/{len(labels)} - {case}')
            values, _ = self.values(self.get(case), component)
            finite = np.abs(values[np.isfinite(values)])
            if finite.size:
                limit = max(limit, float(np.percentile(finite, 98)))
                peak = max(peak, float(finite.max()))
                if case == self.reference:
                    threshold = float(np.percentile(finite, self.percentile))
        default_limit = 1.0 if component == 'eta' else limit or peak or 1.0
        result = _Scale(self.limits.get(component, default_limit), peak, threshold)
        self.scales[component] = result
        return result

    def prepare(self, case, component, progress=None):
        component = _component_name(component)
        if case not in self.cases:
            raise ValueError(f'Unknown case {case!r}.')
        scale = self.statistics(component, progress)
        prepared = self.get(case)
        values, basis = self.values(prepared, component)
        return _Selection(case, component, prepared, values, basis, scale)

    def commit(self, selection):
        self.selection = selection
