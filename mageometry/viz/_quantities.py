# mageometry/viz/_quantities.py
"""
Named scalar quantities that the plotting functions can evaluate on points.

A quantity is either a name from `QUANTITIES` or a callable
``quantity(field, x, y, z) -> values`` (1D arrays in, 1D array out). Names
cover the geometry API (curvature, torsion, frame quality, every
directional-derivative key, the Frenet-frame current density components
``mu0J_T``/``mu0J_n``/``mu0J_b`` and the twist ``alpha``) and the field
itself (|B| and components).
"""

import numpy as np

from ..geometry import (
    field_line_curvature,
    field_line_torsion,
    field_line_frame_quality,
    field_line_directional_derivatives,
    field_line_current_density,
)
from ..geometry.field_line_directional_derivatives import _PROJECTIONS

_DERIVATIVE_KEYS = tuple(key for group in _PROJECTIONS.values() for key, _, _ in group)


def _pretty_derivative(key):
    # 'dT_dn_b' -> '(∂T/∂n)·b'
    dv, du, w = key.split('_')
    return f"(∂{dv[1:]}/∂{du[1:]})·{w}"


class Quantity:
    """Specification of a plottable scalar: how to compute it and how to show it."""

    def __init__(self, func, label, symmetric=False, positive=False, log=False,
                 cmap=None):
        self.func = func
        self.label = label
        self.symmetric = symmetric   # diverging colormap centred on zero
        self.positive = positive     # non-negative; colour scale starts at zero
        self.log = log               # log10 colour scale by default
        self.cmap = cmap or ('RdBu_r' if symmetric else 'viridis')

    def evaluate(self, field, x, y, z, delta=0.01, **kw):
        return np.asarray(self.func(field, x, y, z, delta=delta, **kw), dtype=np.float64)


def _bmag(field, x, y, z, delta=None, **kw):
    bx, by, bz = field(x, y, z)
    return np.sqrt(np.asarray(bx) ** 2 + np.asarray(by) ** 2 + np.asarray(bz) ** 2)


def _component(i):
    def f(field, x, y, z, delta=None, **kw):
        return np.asarray(field(x, y, z)[i], dtype=np.float64)
    return f


def _derivative(key):
    def f(field, x, y, z, delta=0.01, **kw):
        return field_line_directional_derivatives(field, x, y, z, delta=delta, **kw)[key]
    return f


QUANTITIES = {
    'curvature': Quantity(lambda f, x, y, z, delta=0.01, **kw: field_line_curvature(f, x, y, z, delta=delta),
                          'curvature κ', positive=True, log=True, cmap='plasma'),
    'torsion': Quantity(field_line_torsion, 'torsion τ', symmetric=True),
    'frame_quality': Quantity(lambda f, x, y, z, delta=0.01, **kw: field_line_frame_quality(f, x, y, z, delta=delta),
                              'cos θ (frame quality)', positive=True, log=True, cmap='magma'),
    'bmag': Quantity(_bmag, '|B|', positive=True, log=True, cmap='viridis'),
    'bx': Quantity(_component(0), 'Bx', symmetric=True),
    'by': Quantity(_component(1), 'By', symmetric=True),
    'bz': Quantity(_component(2), 'Bz', symmetric=True),
}
for _key in _DERIVATIVE_KEYS:
    QUANTITIES[_key] = Quantity(_derivative(_key), _pretty_derivative(_key),
                                symmetric=(_key != 'dT_dT_n'),
                                positive=(_key == 'dT_dT_n'),
                                cmap='plasma' if _key == 'dT_dT_n' else None)
del _key


def _current(key):
    def f(field, x, y, z, delta=0.01, **kw):
        return field_line_current_density(field, x, y, z, delta=delta, **kw)[key]
    return f


_CURRENT_LABELS = {
    'mu0J_T': 'μ₀J·T (parallel current)',
    'mu0J_n': 'μ₀J·n',
    'mu0J_b': 'μ₀J·b',
    'alpha': 'α = μ₀j∥/B (twist)',
}
for _key, _label in _CURRENT_LABELS.items():
    QUANTITIES[_key] = Quantity(_current(_key), _label, symmetric=True)
del _key, _label


def resolve_quantity(quantity, label=None):
    """Return a `Quantity` for a name or a callable."""
    if isinstance(quantity, Quantity):
        return quantity
    if isinstance(quantity, str):
        try:
            return QUANTITIES[quantity]
        except KeyError:
            raise ValueError(
                f"Unknown quantity {quantity!r}; choose one of "
                f"{sorted(QUANTITIES)} or pass a callable quantity(field, x, y, z)."
            ) from None
    if callable(quantity):
        def func(field, x, y, z, delta=0.01, **kw):
            return quantity(field, x, y, z)
        return Quantity(func, label or getattr(quantity, '__name__', 'value'))
    raise TypeError("quantity must be a name, a Quantity, or a callable.")
