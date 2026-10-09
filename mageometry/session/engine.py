"""Render-independent preparation using the existing diagnostic implementations."""

from collections import OrderedDict
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from ..tracing import trace_field_lines
from ..geometry.line_profiles import profile_coordinates, _LineEvaluator, TRANSVERSE
from ..viz3d._overview_data import _OverviewData, _validate_cases
from ..viz3d._contribution_data import _ContributionData, CONTRIBUTIONS
from ..viz3d._current import COMPONENTS, _component_label
from ..viz3d.fac import _region_seeds
from .sources import load_source, fingerprint, source_label
from .specs import DEFAULT_GEOMETRY_DELTA
from .profiles import profile_label, profile_multiplier


def calculation_key(group):
    return json.dumps({k: group.get(k) for k in ('cases', 'reference', 'analysis', 'revision')}, sort_keys=True)


def _diagnostic_key(group):
    # Trace configuration and presentation labels do not change diagnostic arrays.
    ignored = {'seeds', 'n_lines', 'trace', 'trace_enabled',
               'current_unit', 'length_unit', 'planet_radius'}
    return calculation_key(dict(group, analysis={k: v for k, v in group['analysis'].items()
                                                 if k not in ignored}))


class SessionEngine:
    """Prepare one comparison group without Qt or live rendering objects.

    Parameters
    ----------
    group : dict
        Validated group description from a session recipe.
    progress : callable, optional
        Receives a short progress message. It may raise to cancel preparation.
    previous : SessionEngine, optional
        Reuse compatible inputs and derived arrays from the previous request.
    """

    def __init__(self, group, progress=None, previous=None):
        self.group = deepcopy(group)
        self.analysis = a = self.group['analysis']
        self.progress = progress or (lambda message: None)
        self.cases = {c['id']: c for c in group['cases']}
        if not self.cases:
            raise ValueError('Add a model or file case before preparing a view.')
        ordered = [group['reference']] + [key for key in self.cases if key != group['reference']]
        self.grids, fields, self.inputs = {}, {}, {}
        self.sources = {}
        reuse = (previous is not None and group['id'] == previous.group['id']
                 and group.get('revision') == previous.group.get('revision'))
        sources = previous.sources if reuse else {}
        expected = group.get('resolved', {}).get('inputs', {})
        for key in ordered:
            case = self.cases[key]
            self.grids[key], fields[key], self.inputs[key] = self._load(
                case['source'], sources, expected.get(key), case['label'])
        _validate_cases(self.grids)
        radius = a['mask_radius']
        self.mask = (lambda x, y, z: x*x + y*y + z*z < radius**2) if radius else None
        geometry_delta, fac_delta = a['geometry_delta'], None
        if a['evaluation'] == 'direct':
            if geometry_delta is None:
                geometry_delta = DEFAULT_GEOMETRY_DELTA
            fac_delta = geometry_delta if a['delta'] is None else a['delta']
        self.options = dict(delta=fac_delta, max_points=a['max_points'], mask=self.mask,
                            geometry_delta=geometry_delta, current_scale=a['current_scale'],
                            percentile=a['percentile'], cache_size=a['cache_size'])
        self.fields = fields if a['evaluation'] == 'direct' else None
        self.total = _OverviewData(self.grids, fields=self.fields, comparison=True, **self.options)
        self.backgrounds = {}
        self.contributions = OrderedDict()
        self.statistics = {}
        self.traces = OrderedDict()
        self.profiles = OrderedDict()
        self.seeds = None if a['seeds'] is None else np.array(a['seeds'], dtype=float).reshape(-1, 3)
        if a['kind'] == 'attribution':
            for key in ordered:
                case = self.cases[key]
                identity = key + ':background'
                grid, field, self.inputs[identity] = self._load(
                    case['background'], sources, expected.get(identity), case['label'] + ' background')
                if grid is not None:
                    _validate_cases({'total': self.grids[key], 'background': grid})
                # A dipole-only source has no sampling grid; preserve its analytic evaluator.
                self.backgrounds[key] = field if grid is None or self.fields is not None else grid
                if self.backgrounds[key] is None:
                    self.backgrounds[key] = grid
        if reuse:
            if _diagnostic_key(group) == _diagnostic_key(previous.group):
                self.total = previous.total
                self.contributions = previous.contributions.copy()
                self.statistics = previous.statistics.copy()
                if (self.seeds is None and previous.analysis['seeds'] is None
                        and a['n_lines'] == previous.analysis['n_lines']):
                    self.seeds = previous.seeds
            # Trace keys include effective integration options and seed coordinates.
            # Derivative-step edits can therefore retain paths independently.
            self.traces = OrderedDict((key, paths) for key, paths in previous.traces.items()
                                      if self.grids.get(key[0]) is previous.grids.get(key[0])
                                      and key[0] in self.grids)
            while len(self.traces) > a['cache_size']:
                self.traces.popitem(last=False)

    def _load(self, source, sources, expected, label):
        key = json.dumps([source, self.analysis['mask_radius']], sort_keys=True)
        cached = self.sources.get(key, sources.get(key))
        self.progress(f"{'Loading' if cached is None else 'Reusing'} {label}")
        if cached is None:
            records = fingerprint(source, self.check)
            self._verify_fingerprint(expected, records, label)
            grid, field = load_source(source, self.analysis['mask_radius'], self.check)
            cached = grid, field, records
        else:
            self._verify_records(cached[2])
            self._verify_fingerprint(expected, cached[2], label)
        self.sources[key] = cached
        return cached

    def check(self):
        self.progress('')

    @staticmethod
    def _verify_fingerprint(expected, actual, label):
        if expected is not None and ([x['sha256'] for x in expected] != [x['sha256'] for x in actual]):
            raise ValueError(f'{label}: input content changed since the saved result. '
                             'Use Reload sources to accept a new input revision.')

    def verify_inputs(self):
        for records in self.inputs.values():
            self._verify_records(records)

    @staticmethod
    def _verify_records(records):
        for record in records:
            stat = Path(record['path']).stat()
            if (stat.st_size, stat.st_mtime_ns) != (record['size'], record['mtime_ns']):
                raise ValueError(f"Source changed: {record['path']}. Use Reload sources.")

    def _data(self, key):
        if self.analysis['kind'] == 'field':
            return self.total
        if key not in self.contributions:
            options = dict(self.options)
            options['field'] = None if self.fields is None else self.fields[key]
            self.contributions[key] = _ContributionData(
                self.grids[key], self.backgrounds[key], background_label='Assigned background', **options)
            while len(self.contributions) > self.analysis['cache_size']:
                self.contributions.popitem(last=False)
        self.contributions.move_to_end(key)
        return self.contributions[key]

    def _get(self, key, branch, component):
        self.check()
        data = self._data(key)
        prepared = data.get(key if self.analysis['kind'] == 'field' else branch)
        values, basis = data.values(prepared, component)
        self.check()
        return prepared, values, basis

    def _scale(self, component):
        if component not in self.statistics:
            limit = peak = threshold = 0.
            branches = tuple(CONTRIBUTIONS) if self.analysis['kind'] == 'attribution' else ('total',)
            for key, case in self.cases.items():
                self.progress(f"Preparing {component}: {case['label']}")
                for branch in branches:
                    _, values, _ = self._get(key, branch, component)
                    finite = np.abs(values[np.isfinite(values)])
                    if finite.size:
                        limit = max(limit, float(np.percentile(finite, 98)))
                        peak = max(peak, float(finite.max()))
                        if key == self.group['reference'] and branch == 'total':
                            threshold = float(np.percentile(finite, self.analysis['percentile']))
            self.statistics[component] = dict(limit=1. if component == 'eta' else limit or peak or 1.,
                                               peak=peak, threshold=threshold)
        return self.statistics[component]

    def _resolve_seeds(self, component):
        if self.seeds is None:
            prepared, values, _ = self._get(self.group['reference'], 'total', component)
            grid = prepared.preview
            points = np.column_stack([a.ravel(order='F') for a in
                                      np.meshgrid(grid.x, grid.y, grid.z, indexing='ij')])
            length = float(np.linalg.norm([np.ptp(a) for a in (grid.x, grid.y, grid.z)]))
            indices = _region_seeds(points, values.ravel(order='F'),
                                    self._scale(component)['threshold'],
                                    self.analysis['n_lines'], .12 * length)
            self.seeds = points[indices]

    def _trace_options(self, key, prepared):
        options = dict(direction='both', ds=prepared.spacing / 2,
                       bounds=prepared.preview.bounds, max_steps=600)
        options.update(self.analysis['trace'])
        identity = json.dumps(dict(options=options, seeds=self.seeds.tolist(),
                                   evaluation=self.analysis['evaluation'],
                                   max_points=self.analysis['max_points'],
                                   mask_radius=self.analysis['mask_radius']), sort_keys=True)
        return (key, identity), options

    def _trace_paths(self, key, prepared):
        identity, options = self._trace_options(key, prepared)
        if identity not in self.traces:
            paths, records = [], []

            def field(x, y, z):
                # Check cancellation at every RK field evaluation, including retries.
                self.check()
                return prepared.field(x, y, z)

            for start in range(0, len(self.seeds), 32):
                self.progress(f"Tracing {self.cases[key]['label']}: {start}/{len(self.seeds)}")
                seeds = self.seeds[start:start + 32]
                trace = trace_field_lines(field, *seeds.T, **options)
                paths.extend(np.column_stack(trace.path(i)) for i in range(len(seeds)))
                for i in range(len(seeds)):
                    record = profile_coordinates(trace, i)
                    record['line_id'] = start + i
                    # Physical coordinates survive reordering and case changes.
                    record['seed_id'] = json.dumps(record['seed'].tolist())
                    records.append(record)
            self.check()
            self.traces[identity] = dict(paths=paths, records=records)
            while len(self.traces) > self.analysis['cache_size']:
                self.traces.popitem(last=False)
        self.traces.move_to_end(identity)
        return self.traces[identity]['paths']

    def trace_records(self, view):
        """Return completed oriented traces without initiating integration."""
        if not self.analysis.get('trace_enabled', True) or self.seeds is None:
            return []
        prepared = self.total.get(view['case'])
        identity, _ = self._trace_options(view['case'], prepared)
        return self.traces.get(identity, {}).get('records', [])

    def prepare_profile(self, view, seed_id, quantities):
        """Evaluate only the requested line, using the committed effective field."""
        self.verify_inputs()
        if not self.analysis.get('trace_enabled', True):
            raise ValueError('Enable tracing and Apply before requesting a profile.')
        names = tuple(dict.fromkeys(quantities))
        if not names or len(names) > 4:
            raise ValueError('Select between one and four profile quantities.')
        key, branch = view['case'], view['contribution']
        if self.analysis['kind'] == 'attribution' and any(n not in TRANSVERSE + ('bmag',) for n in names):
            raise ValueError('Attribution profiles support transverse diagnostics and total |B|.')
        self._resolve_seeds(view['component'])
        prepared = self.total.get(key)
        self._trace_paths(key, prepared)
        record = next((r for r in self.trace_records(view) if r['seed_id'] == seed_id), None)
        if record is None:
            raise ValueError('The selected seed is no longer in this result.')
        identity, options = self._trace_options(key, prepared)
        cache_key = identity, seed_id, branch
        if cache_key not in self.profiles:
            background = None
            if self.analysis['kind'] == 'attribution':
                background = self._data(key).profile_background()

            def field(x, y, z):
                self.check()
                return prepared.field(x, y, z)

            self.profiles[cache_key] = _LineEvaluator(
                field, record['points'], prepared.cache.delta,
                fac_delta=prepared.cache.fac_delta if self.fields is not None else prepared.cache.delta,
                background=background, contribution=branch)
        evaluator = self.profiles[cache_key]
        self.profiles.move_to_end(cache_key)
        while len(self.profiles) > self.analysis['cache_size'] * 4:
            self.profiles.popitem(last=False)
        native = {}
        for name in names:
            self.progress(f'Profile: {name}')
            native[name] = evaluator.get(name).copy()
        self.check()
        metadata = deepcopy(self.grids[key].metadata)
        return dict(record, native_values=native,
                    values={name: values * profile_multiplier(name, self.analysis) for name, values in native.items()},
                    labels={name: profile_label(name, self.analysis, metadata) for name in names},
                    multipliers={name: profile_multiplier(name, self.analysis) for name in names},
                    case=key, case_label=self.cases[key]['label'], contribution=branch,
                    metadata=metadata, analysis=deepcopy(self.analysis), source=deepcopy(self.cases[key]),
                    inputs=deepcopy(self.inputs), revision=self.group.get('revision'),
                    evaluation='pointwise ' + ('direct model' if self.fields is not None else 'preview grid interpolant'),
                    geometry_delta=evaluator.delta, fac_delta=evaluator.fac_delta,
                    preview_shape=list(prepared.preview.shape),
                    grid_spacing=[float(np.min(np.diff(axis))) for axis in
                                  (prepared.preview.x, prepared.preview.y, prepared.preview.z)],
                    trace_options=options)

    def prepare_traces(self, view):
        """Finish pending total-field paths after publishing diagnostic arrays."""
        self.verify_inputs()
        self._resolve_seeds(view['component'])
        prepared, _, _ = self._get(view['case'], view['contribution'], view['component'])
        return self._trace_paths(view['case'], prepared)

    def prepare(self, view, *, include_traces=True):
        """Return NumPy arrays, traces and provenance for a requested selection.

        Parameters
        ----------
        view : dict
            Case, diagnostic and contribution selection from the group view.
        include_traces : bool, optional
            Finish missing traces before returning (default). If False, return
            usable diagnostic arrays with ``trace_status='pending'`` first.

        Returns
        -------
        dict
            Serializable prepared result. No field callables or VTK objects.
        """
        self.verify_inputs()
        key, component, branch = view['case'], view['component'], view['contribution']
        if key not in self.cases or component not in COMPONENTS:
            raise ValueError('Unknown case or diagnostic.')
        if self.analysis['kind'] == 'field' and branch != 'total':
            raise ValueError('Field analysis requires the total branch.')
        scale = self._scale(component)
        tracing = self.analysis.get('trace_enabled', True)
        if tracing:
            self._resolve_seeds(component)
        prepared, values, basis = self._get(key, branch, component)
        # Gamma and eta share the cached transverse calculation and branch.
        # Prepare both so changing colours never submits numerical work.
        eta_values = self._get(key, branch, 'eta')[1] if component == 'gamma' else None
        paths, trace_status = [], 'disabled'
        if tracing:
            identity, _ = self._trace_options(key, prepared)
            if include_traces or identity in self.traces or not len(self.seeds):
                paths = self._trace_paths(key, prepared)
                trace_status = 'ready'
            else:
                trace_status = 'pending'
        self.check()
        grid = prepared.preview
        metadata = deepcopy(self.grids[key].metadata)
        if self.analysis['kind'] == 'attribution':
            metadata['parameters'] = dict(metadata.get('parameters', {}),
                                           Background=source_label(self.cases[key]['background']),
                                           Frame='Total field', Contribution=CONTRIBUTIONS[branch])
        return dict(case=key, case_label=self.cases[key]['label'], component=component,
                    contribution=branch, kind=self.analysis['kind'],
                    axes=(grid.x, grid.y, grid.z), values=values, basis=basis,
                    eta_values=eta_values,
                    paths=paths, trace_records=self.trace_records(view), trace_status=trace_status, metadata=metadata, scale=scale,
                    label=_component_label(component, self.analysis['current_unit'],
                                           self.analysis['length_unit'],
                                           field_unit=metadata.get('field_unit', 'field unit')),
                    analysis=deepcopy(self.analysis),
                    resolved=dict(inputs=deepcopy(self.inputs), seeds=self.seeds.tolist() if tracing else None,
                                  fac_delta=deepcopy(self.options['delta']),
                                  geometry_delta=float(prepared.cache.delta) if hasattr(prepared.cache, 'delta')
                                  else float(self.options['geometry_delta'] or prepared.spacing),
                                  preview_shape=list(grid.shape), scales=deepcopy(self.statistics)))
