"""Pointwise diagnostics on traced lines, independent of plotting and sessions."""

import numpy as np

from .field_line_geometry import (field_line_curvature, field_line_torsion,
                                  field_line_frame_quality, field_line_frenet_frame)
from .field_line_current import (field_aligned_current_density, field_aligned_current_derivatives,
                                 field_line_current_density, field_magnitude_derivatives)
from .field_line_transverse import field_line_transverse_geometry, field_line_transverse_decomposition

TRANSVERSE = ('alpha', 'beta_g', 'delta_g', 'gamma', 'gamma_over_abs_alpha', 'omega_c', 'eta')
ALONG_FIELD = ('dalpha_ds', 'dalpha_ds_over_B', 'dfac_ds')
CURRENT = ('mu0J_T', 'mu0J_n', 'mu0J_b', 'mu0J_x', 'mu0J_y', 'mu0J_z',
           'B_dT_dn_b', 'B_dn_db_T', 'B_twist_diff', 'B_kappa', 'minus_dB_dn')
BASIC = ('bmag', 'bx', 'by', 'bz', 'curvature', 'torsion', 'frame_quality')
PROFILE_QUANTITIES = BASIC + TRANSVERSE + ('fac',) + CURRENT + ALONG_FIELD


def profile_coordinates(trace, line=0):
    """Return a line ordered along B, preserving its integrated arc length.

    Parameters
    ----------
    trace : FieldLineTrace
        Trace including the integration direction and termination codes.
    line : int, optional
        Index of the requested line.

    Returns
    -------
    dict
        ``points``, signed ``s``, ``seed``, ``seed_index``, ``direction`` and
        ``status_minus`` / ``status_plus`` (None on an untraced side).
        No changes are made to the input trace or its legacy distance convention.
    """
    if not isinstance(line, (int, np.integer)) or not 0 <= line < trace.n_lines:
        raise ValueError('Line index is outside the trace.')
    points = np.column_stack(trace.path(line)).copy()
    distance = trace.arc_length(line).copy()
    seed_index = int(trace.start_index[line])
    seed = points[seed_index].copy()
    direction = trace.direction
    minus = int(trace.status_backward[line]) if direction == 'both' else None
    plus = int(trace.status[line]) if direction != -1 else None
    if direction == -1:
        points, distance = points[::-1].copy(), -distance[::-1].copy()
        seed_index = len(points) - 1 - seed_index
        minus = int(trace.status[line])
    return dict(points=points, s=distance, seed=seed, seed_index=seed_index,
                direction=direction, status_minus=minus, status_plus=plus)


class _LineEvaluator:
    """Cache numerical families in native units for one immutable set of points."""

    def __init__(self, field, points, delta, fac_delta=None, background=None, contribution='total'):
        self.field = field
        self.points = np.asarray(points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError('Profile points must have shape (n, 3).')
        if not np.isscalar(delta) or not np.isfinite(delta) or delta <= 0:
            raise ValueError('Profile delta must be a positive finite scalar.')
        if contribution not in ('total', 'background', 'residual'):
            raise ValueError('Unknown gradient contribution.')
        if contribution != 'total' and background is None:
            raise ValueError('A background is required for gradient attribution.')
        self.delta, self.fac_delta = float(delta), fac_delta
        self.background, self.contribution = background, contribution
        self.values = {}

    def get(self, name):
        if name not in PROFILE_QUANTITIES:
            raise ValueError(f'Unknown profile quantity: {name}')
        if self.contribution != 'total' and name not in TRANSVERSE + ('bmag',):
            raise ValueError('Gradient attribution supports transverse diagnostics and total |B| only.')
        if name not in self.values:
            field, coords, delta = self.field, self.points.T, self.delta
            if name in ('bmag', 'bx', 'by', 'bz'):
                components = np.broadcast_arrays(*field(*coords), np.empty(len(self.points)))[:3]
                for key, values in zip(('bx', 'by', 'bz'), components):
                    self.values[key] = np.asarray(values, dtype=float)
                self.values['bmag'] = np.linalg.norm(components, axis=0)
            elif name in TRANSVERSE:
                if self.background is None:
                    result = field_line_transverse_geometry(field, *coords, delta=delta)
                    self.values.update({key: result[key] for key in TRANSVERSE})
                else:
                    result = field_line_transverse_decomposition(
                        field, self.background, *coords, delta=delta)
                    self.values.update({key: result[self.contribution][key] for key in TRANSVERSE})
            elif name == 'fac':
                self.values[name] = field_aligned_current_density(
                    field, *coords, delta=delta if self.fac_delta is None else self.fac_delta)
            elif name in ALONG_FIELD:
                self.values.update(field_aligned_current_derivatives(
                    field, *coords, delta=delta, fac_delta=self.fac_delta))
            elif name in CURRENT:
                result = field_line_current_density(field, *coords, delta=delta)
                # Never overwrite the canonical first-gradient alpha with the
                # legacy current reconstruction's Frenet-dependent alpha.
                self.values.update({key: result[key] for key in CURRENT if key in result})
                frame = field_line_frenet_frame(field, *coords, delta=delta)
                self.values['B_kappa'] = np.where(np.isfinite(frame[3]), result['B'] * frame[9], np.nan)
                self.values['minus_dB_dn'] = -field_magnitude_derivatives(field, *coords, delta=delta)['dB_dn']
            else:
                function = {'curvature': field_line_curvature, 'torsion': field_line_torsion,
                            'frame_quality': field_line_frame_quality}[name]
                self.values[name] = function(field, *coords, delta=delta)
        return np.asarray(self.values[name], dtype=float)


def field_line_profile(field, trace, line=0, quantities=('alpha', 'bmag'), *, delta=0.01,
                       fac_delta=None, background=None, contribution='total'):
    """Evaluate native-unit diagnostics at the adaptive points of a traced line.

    Parameters
    ----------
    field : callable
        The same effective magnetic field used for tracing, accepting arrays.
    trace : FieldLineTrace
        Trace with integrated distances and direction information.
    line : int, optional
        Line index, default 0.
    quantities : sequence of str, optional
        Diagnostic names from ``PROFILE_QUANTITIES``.
    delta : float, optional
        Positive spatial difference step, independent of the trace step.
    fac_delta : float or sequence of float, optional
        Cartesian FAC difference step(s), defaulting to delta.
    background : callable, optional
        Background for gradient attribution in the total-field frame.
    contribution : {'total', 'background', 'residual'}, optional
        Gradient branch. Non-total branches support transverse rates and |B|.

    Returns
    -------
    dict
        Oriented coordinates and termination metadata, plus ``values`` and
        per-quantity ``valid`` masks. Values are evaluated pointwise, not
        interpolated from displayed diagnostic arrays; no smoothing or current
        conversion is applied. Undefined geometry remains NaN.
    """
    result = profile_coordinates(trace, line)
    evaluator = _LineEvaluator(field, result['points'], delta, fac_delta, background, contribution)
    result['values'] = {key: evaluator.get(key).copy() for key in quantities}
    result['valid'] = {key: np.isfinite(values) for key, values in result['values'].items()}
    return result
