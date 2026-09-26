"""Render-independent preparation using the existing diagnostic implementations."""

from collections import OrderedDict
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from ..tracing import trace_field_lines
from ..viz3d._overview_data import _OverviewData, _validate_cases
from ..viz3d._contribution_data import _ContributionData, CONTRIBUTIONS
from ..viz3d._current import COMPONENTS, _component_label
from ..viz3d.fac import _region_seeds
from .sources import load_source, fingerprint


def calculation_key(group):
    return json.dumps({k: group.get(k) for k in ('cases', 'reference', 'analysis', 'revision')}, sort_keys=True)


class SessionEngine:
    """Prepare one comparison group without Qt or live rendering objects.

    Parameters
    ----------
    group : dict
        Validated group description from a session recipe.
    progress : callable, optional
        Receives a short progress message. It may raise to cancel preparation.
    """

    def __init__(self, group, progress=None):
        self.group = deepcopy(group)
        self.analysis = a = self.group['analysis']
        self.progress = progress or (lambda message: None)
        self.cases = {c['id']: c for c in group['cases']}
        if not self.cases:
            raise ValueError('Add a model or file case before preparing a view.')
        ordered = [group['reference']] + [key for key in self.cases if key != group['reference']]
        self.grids, fields, self.inputs = {}, {}, {}
        expected = group.get('resolved', {}).get('inputs', {})
        for key in ordered:
            case = self.cases[key]
            self.progress(f"Loading {case['label']}")
            self.inputs[key] = fingerprint(case['source'], self.check)
            self._verify_fingerprint(expected.get(key), self.inputs[key], case['label'])
            self.grids[key], fields[key] = load_source(case['source'], a['mask_radius'], self.check)
        _validate_cases(self.grids)
        radius = a['mask_radius']
        self.mask = (lambda x, y, z: x*x + y*y + z*z < radius**2) if radius else None
        self.options = dict(delta=a['delta'], max_points=a['max_points'], mask=self.mask,
                            geometry_delta=a['geometry_delta'], current_scale=a['current_scale'],
                            percentile=a['percentile'], cache_size=a['cache_size'])
        self.fields = fields if a['evaluation'] == 'direct' else None
        self.total = _OverviewData(self.grids, fields=self.fields, comparison=True, **self.options)
        self.backgrounds = {}
        self.contributions = OrderedDict()
        self.statistics = {}
        self.traces = OrderedDict()
        self.seeds = None if a['seeds'] is None else np.array(a['seeds'], dtype=float).reshape(-1, 3)
        if a['kind'] == 'attribution':
            for key in ordered:
                case = self.cases[key]
                self.progress(f"Loading background for {case['label']}")
                identity = key + ':background'
                self.inputs[identity] = fingerprint(case['background'], self.check)
                self._verify_fingerprint(expected.get(identity), self.inputs[identity], case['label'] + ' background')
                grid, field = load_source(case['background'], a['mask_radius'], self.check)
                if grid is not None:
                    _validate_cases({'total': self.grids[key], 'background': grid})
                # A dipole-only source has no sampling grid; preserve its analytic evaluator.
                self.backgrounds[key] = field if grid is None or self.fields is not None else grid
                if self.backgrounds[key] is None:
                    self.backgrounds[key] = grid

    def check(self):
        self.progress('')

    @staticmethod
    def _verify_fingerprint(expected, actual, label):
        if expected is not None and ([x['sha256'] for x in expected] != [x['sha256'] for x in actual]):
            raise ValueError(f'{label}: input content changed since the saved result. '
                             'Use Reload sources to accept a new input revision.')

    def verify_inputs(self):
        for records in self.inputs.values():
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

    def prepare(self, view):
        """Return NumPy arrays, traces and provenance for a requested selection.

        Parameters
        ----------
        view : dict
            Case, diagnostic and contribution selection from the group view.

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
        self._resolve_seeds(component)
        prepared, values, basis = self._get(key, branch, component)
        if key not in self.traces:
            self.progress(f"Tracing {self.cases[key]['label']}")
            options = dict(direction='both', ds=prepared.spacing / 2,
                           bounds=prepared.preview.bounds, max_steps=600)
            options.update(self.analysis['trace'])
            paths = []
            for seed in self.seeds:
                self.check()
                trace = trace_field_lines(prepared.field, *seed, **options)
                paths.append(np.column_stack(trace.path(0)))
            self.traces[key] = paths
            while len(self.traces) > self.analysis['cache_size']:
                self.traces.popitem(last=False)
        self.traces.move_to_end(key)
        self.check()
        grid = prepared.preview
        metadata = deepcopy(self.grids[key].metadata)
        if self.analysis['kind'] == 'attribution':
            metadata['parameters'] = dict(metadata.get('parameters', {}),
                                           Background=self.cases[key]['background'].get('path', 'Dipole'),
                                           Frame='Total field', Contribution=CONTRIBUTIONS[branch])
        return dict(case=key, case_label=self.cases[key]['label'], component=component,
                    contribution=branch, kind=self.analysis['kind'],
                    axes=(grid.x, grid.y, grid.z), values=values, basis=basis,
                    paths=self.traces[key], metadata=metadata, scale=scale,
                    label=_component_label(component, self.analysis['current_unit'],
                                           self.analysis['length_unit']),
                    analysis=deepcopy(self.analysis),
                    resolved=dict(inputs=deepcopy(self.inputs), seeds=self.seeds.tolist(),
                                  geometry_delta=float(prepared.cache.delta) if hasattr(prepared.cache, 'delta')
                                  else float(self.analysis['geometry_delta'] or
                                             (np.min(self.analysis['delta']) if self.fields is not None
                                              else prepared.spacing)),
                                  preview_shape=list(grid.shape), scales=deepcopy(self.statistics)))
