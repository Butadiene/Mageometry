"""Cached notebook-10 current components, independent of rendering."""

import numpy as np

from ..geometry import (field_line_current_density, field_line_frenet_frame,
                        field_magnitude_derivatives, field_line_transverse_geometry)


# key: (display symbol, basis, explanation). b is the Frenet binormal, not B.
COMPONENTS = {
    'fac': ('J parallel', 'T', 'FAC: Cartesian curl(B) dot T'),
    'mu0J_T': ('J_T', 'T', 'J_T: field-line twist (notebook 10)'),
    'B_dT_dn_b': ('parallel term 1', 'T', 'J_T term 1: B (dT/dn) dot b'),
    'B_dn_db_T': ('parallel term 2', 'T', 'J_T term 2: B (dn/db) dot T'),
    'B_twist_diff': ('parallel terms: 1 - 2', None,
                     'B(dT/dn).b - B(dn/db).T: signed difference, not total J_T'),
    'mu0J_n': ('J_n', 'n', 'J_n: d|B|/db (magnetic-pressure term)'),
    'mu0J_b': ('J_b', 'b', 'J_b: |B| kappa - d|B|/dn (curvature + pressure)'),
    'mu0J_x': ('J_x', 'x', 'J_x: Cartesian x component of the Frenet reconstruction'),
    'mu0J_y': ('J_y', 'y', 'J_y: Cartesian y component of the Frenet reconstruction'),
    'mu0J_z': ('J_z', 'z', 'J_z: Cartesian z component of the Frenet reconstruction'),
    'alpha': ('alpha', None, 'alpha = T dot curl(B) / |B|: twice the mean winding rate'),
    'B_kappa': ('curvature term', 'b', 'Binormal contribution: +|B| kappa'),
    'minus_dB_dn': ('pressure term', 'b', 'Binormal contribution: -d|B|/dn'),
}

RATE_COMPONENTS = frozenset(('alpha', 'sigma', 'q', 'gamma', 'omega_c'))

COMPONENTS.update({
    'sigma': ('sigma', None, 'sigma = a+c: signed transverse shear in the Frenet frame'),
    'q': ('q', None, 'q = u-v: transverse normal-strain difference'),
    'gamma': ('Gamma', None, 'Gamma = sqrt(sigma^2+q^2): basis-independent anisotropy'),
    'omega_c': ('omega_c', None, 'Signed local coiling rate; zero for shear-dominated structure'),
})

COMPONENT_LABELS = {
    'fac': 'FAC - Along B (Cartesian curl)',
    'mu0J_T': 'J_T - Parallel / twist',
    'B_dT_dn_b': 'J_T term 1: B(dT/dn).b',
    'B_dn_db_T': 'J_T term 2: B(dn/db).T',
    'B_twist_diff': 'J_T terms: 1 - 2 (difference)',
    'mu0J_n': 'J_n - Normal current',
    'mu0J_b': 'J_b - Binormal current',
    'mu0J_x': 'J_x - Cartesian x',
    'mu0J_y': 'J_y - Cartesian y',
    'mu0J_z': 'J_z - Cartesian z',
    'alpha': 'alpha - Twist (not current)',
    'B_kappa': 'Curvature term: +|B| kappa',
    'minus_dB_dn': 'Pressure term: -d|B|/dn',
}


COMPONENT_LABELS.update({key: COMPONENTS[key][0] + ' - Transverse geometry'
                         for key in COMPONENTS if key in RATE_COMPONENTS and key != 'alpha'})


def _component_name(component):
    if not isinstance(component, str) or component not in COMPONENTS:
        raise ValueError(f"Unknown current component {component!r}; choose {tuple(COMPONENTS)}.")
    return component


def _component_label(component, current_unit, length_unit, fac_label=None):
    if component in RATE_COMPONENTS:
        return f'{COMPONENTS[component][0]} [1 / {length_unit}]'
    if component == 'fac' and fac_label is not None:
        return fac_label
    symbol = COMPONENTS[component][0]
    if current_unit is None and (component == 'fac' or component.startswith('mu0J_')):
        symbol = f'mu0 {symbol}'
    return f'{symbol} [{current_unit or "field unit / length unit"}]'


class _CurrentPreview:
    """Evaluate the notebook APIs once, on demand, and retain native units.

    Transverse rates can be evaluated without computing notebook currents.
    Legacy current components retain the notebook Frenet-frame conventions.
    """

    def __init__(self, preview, fac, field, delta):
        self.preview = preview
        self.field = field
        self.delta = delta
        self.values = {'fac': fac}
        magnitude = np.linalg.norm(preview.b, axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.bases = {'T': preview.b / magnitude[..., None]}

    def get(self, component):
        _component_name(component)
        if component not in self.values:
            if component in RATE_COMPONENTS:
                self._transverse()
            else:
                self._geometry()
        basis = COMPONENTS[component][1]
        return self.values[component], self.bases.get(basis)

    def _transverse(self):
        coords = np.meshgrid(self.preview.x, self.preview.y, self.preview.z, indexing='ij')
        valid = np.all(np.isfinite(self.preview.b), axis=-1)
        rates = field_line_transverse_geometry(
            self.field, *(c[valid] for c in coords), delta=self.delta) if np.any(valid) else {}
        for key in RATE_COMPONENTS:
            values = np.full(self.preview.shape, np.nan)
            if key in rates:
                values[valid] = rates[key]
            self.values[key] = values

    def _geometry(self):
        coords = np.meshgrid(self.preview.x, self.preview.y, self.preview.z, indexing='ij')
        # A base node excluded by the supplied grid remains excluded even
        # if the analytic callable itself is defined there.
        valid = np.all(np.isfinite(self.preview.b), axis=-1)
        points = tuple(c[valid] for c in coords)
        shape = self.preview.shape
        if np.any(valid):
            current = field_line_current_density(self.field, *points, delta=self.delta)
            frame = field_line_frenet_frame(self.field, *points, delta=self.delta)
            mag = field_magnitude_derivatives(self.field, *points, delta=self.delta)
            current['B_kappa'] = np.where(np.isfinite(frame[3]),
                                          current['B'] * current['curvature'], np.nan)
            current['minus_dB_dn'] = -mag['dB_dn']
        else:
            current, frame = {}, None
        for key in COMPONENTS:
            if key == 'fac' or key in RATE_COMPONENTS:
                continue
            values = np.full(shape, np.nan)
            if key in current:
                values[valid] = current[key]
            self.values[key] = values
        if 'alpha' not in self.values:
            self._transverse()
        for basis, start in (('n', 3), ('b', 6)):
            vectors = np.full(shape + (3,), np.nan)
            if frame is not None:
                vectors[valid] = np.stack(frame[start:start + 3], axis=-1)
            self.bases[basis] = vectors
        for axis, basis in enumerate('xyz'):
            self.bases[basis] = np.broadcast_to(np.eye(3)[axis], shape + (3,))
