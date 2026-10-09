"""Profile presentation metadata and portable exports; no rendering imports."""

import csv
import json
import platform
from pathlib import Path

import numpy as np

from ..geometry.line_profiles import BASIC
from ..viz3d._current import COMPONENT_LABELS, UNSCALED_COMPONENTS, _component_label

PROFILE_LABELS = dict(COMPONENT_LABELS, bmag='|B|', bx='Bx', by='By', bz='Bz',
                      curvature='Curvature', torsion='Torsion', frame_quality='Frame quality')
STATUS_LABELS = {None: 'not traced', 0: 'inner boundary', 1: 'outer boundary',
                 2: 'step limit', 3: 'undefined field', 4: 'custom stop'}


def default_profile_view():
    return dict(visible=False, seed_id=None, quantities=[], xlim=None, ylims={}, cursor_s=0.)


def profile_label(name, analysis, metadata):
    length = analysis['length_unit']
    field = metadata.get('field_unit', 'field unit')
    if name in ('bmag', 'bx', 'by', 'bz'):
        label = 'Total |B|' if name == 'bmag' and analysis['kind'] == 'attribution' else PROFILE_LABELS[name]
        return f'{label} [{field}]'
    if name in ('curvature', 'torsion'):
        return f'{PROFILE_LABELS[name]} [1 / {length}]'
    if name == 'frame_quality':
        return 'Frame quality [dimensionless]'
    return _component_label(name, analysis['current_unit'], length, field_unit=field)


def profile_multiplier(name, analysis):
    return 1. if name in BASIC or name in UNSCALED_COMPONENTS else analysis['current_scale']


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def export_profile(data, path, figure=None, view=None):
    """Write complete samples or a figure, with matching provenance JSON."""
    from .. import __version__
    path = Path(path)
    if path.suffix.lower() == '.csv':
        names = list(data['values'])
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['s', 'x', 'y', 'z'] + [column for name in names
                             for column in (name + '_native', name + '_display', name + '_valid')])
            for i, distance in enumerate(data['s']):
                row = [distance, *data['points'][i]]
                for name in names:
                    row.extend([data['native_values'][name][i], data['values'][name][i],
                                int(np.isfinite(data['values'][name][i]))])
                writer.writerow(row)
    elif path.suffix.lower() in ('.png', '.svg', '.pdf') and figure is not None:
        figure.savefig(path, dpi=180, bbox_inches='tight')
    else:
        raise ValueError('Choose CSV, PNG, SVG or PDF for a profile export.')
    metadata = {key: value for key, value in data.items()
                if key not in ('points', 's', 'values', 'native_values')}
    metadata.update(sample_count=len(data['s']), sampling='adaptive trace points; no smoothing',
                    distance_convention='s = 0 at seed; increases along B', view=view,
                    saved_with=dict(mageometry=__version__, numpy=np.__version__, python=platform.python_version()))
    path.with_suffix('.profile.json').write_text(
        json.dumps(metadata, indent=2, default=_json_value, allow_nan=False) + '\n', encoding='utf-8')
