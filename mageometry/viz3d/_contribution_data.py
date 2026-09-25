"""Gradient attribution with one total-field preview and shared display scales."""

from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np

from ..geometry import field_line_transverse_decomposition
from ..io import GriddedField
from ._current import TRANSVERSE_COMPONENTS
from ._overview_data import _OverviewData, _PreparedCase, _validate_cases
from ._preview import _masked_field, _preview_indices


CONTRIBUTIONS = {'total': 'Total field', 'background': 'Background gradient',
                 'residual': 'Residual gradient'}


def _transverse_component(component):
    if not isinstance(component, str) or component not in TRANSVERSE_COMPONENTS:
        raise ValueError('Contribution diagnostics must be alpha, beta_g, delta_g, gamma, omega_c or eta.')
    return component


class _ContributionCache:
    def __init__(self, values):
        self.values = values

    def get(self, component):
        return self.values[_transverse_component(component)], None


class _ContributionData(_OverviewData):
    def __init__(self, grid, background, *, background_label='Background', **kwargs):
        if isinstance(background, GriddedField):
            _validate_cases({'total': grid, 'background': background})
        elif not callable(background):
            raise TypeError('background must be a callable or GriddedField.')
        if not isinstance(background_label, str) or not background_label.strip():
            raise ValueError('background_label must be a nonempty string.')
        super().__init__({key: grid for key in CONTRIBUTIONS}, **kwargs)
        for component in self.limits:
            _transverse_component(component)
        self.comparison = True
        self.background = background
        self.background_label = background_label

    def metadata(self, case):
        metadata = dict(super().metadata(case))
        declared = metadata.get('parameters', {})
        parameters = dict(declared) if isinstance(declared, Mapping) else {}
        parameters.update({'Background': self.background_label,
                           'Frame and normalization': 'Total field',
                           'Contribution': CONTRIBUTIONS[case]})
        metadata['parameters'] = parameters
        return metadata

    def get(self, case):
        if case not in self.cases:
            raise ValueError(f'Unknown contribution {case!r}.')
        if not self.cache:
            total = _PreparedCase(self.cases[self.reference], **self.options)
            background = self.background
            if isinstance(background, GriddedField):
                indices = _preview_indices(background.shape, self.options['max_points'])
                axes = [axis[index] for axis, index in zip(
                    (background.x, background.y, background.z), indices)]
                components = [background.b[..., i][np.ix_(*indices)] for i in range(3)]
                background = GriddedField(*axes, *components).field()
            background = _masked_field(background, self.options['mask'])
            coords = np.meshgrid(total.preview.x, total.preview.y, total.preview.z, indexing='ij')
            valid = np.all(np.isfinite(total.preview.b), axis=-1)
            result = field_line_transverse_decomposition(
                total.field, background, *(axis[valid] for axis in coords),
                delta=total.cache.delta)
            for key in CONTRIBUTIONS:
                values = {}
                for component in TRANSVERSE_COMPONENTS:
                    values[component] = np.full(total.preview.shape, np.nan)
                    values[component][valid] = result[key][component]
                self.cache[key] = SimpleNamespace(preview=total.preview, field=total.field,
                                                  spacing=total.spacing,
                                                  cache=_ContributionCache(values))
        return self.cache[case]
