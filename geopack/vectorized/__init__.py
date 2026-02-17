# geopack/vectorized/__init__.py
"""
geopack.vectorized package initializer.

This module re-exports public vectorized functions and modules so that they can be
imported directly from `geopack.vectorized`.
"""

# vectorized modules
from . import (
    models,
    coordinates,
    coordinates_complex,
    igrf,
)

# Re-export external field model functions (vectorized)
from .models import t89, t96, t01, t04

# Re-export vectorized IGRF functions
from .igrf import (
    igrf_geo,
    igrf_gsm,
    igrf_gsw,
)

# Re-export vectorized coordinate transforms
from .coordinates import (
    gsmgse,
    geigeo,
    magsm,
    smgsm,
    geomag,
    geogsm,
    gswgsm,
)

# Re-export vectorized "complex" coordinate transforms
from .coordinates_complex import (
    sphcar,
    bspcar,
    bcarsp,
)

# Re-export extra exact implementations (vectorized utilities)
from .condip1_exact import condip1_exact

from .trace import trace

# Core functions (kept here assuming shared global state is intended)
from ..geopack import (
    recalc,
    dip,
    dip_gsw,
)

__all__ = [
    # vectorized modules
    "models",
    "coordinates",
    "coordinates_complex",
    "igrf",

    # vectorized models
    "t89",
    "t96",
    "t01",
    "t04",

    # vectorized IGRF
    "igrf_geo",
    "igrf_gsm",
    "igrf_gsw",

    # vectorized coordinate transforms
    "gsmgse",
    "geigeo",
    "magsm",
    "smgsm",
    "geomag",
    "geogsm",
    "gswgsm",

    # vectorized "complex" coordinate transforms
    "sphcar",
    "bspcar",
    "bcarsp",

    # vectorized utilities
    "condip1_exact",

    # vectorized trace function
    "trace",

    # core functions re-exported
    "recalc",
    "dip",
    "dip_gsw",
]
